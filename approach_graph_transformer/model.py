import torch
import torch.nn as nn
from utils import x_map, e_map
from torch_geometric.nn import GPSConv, GINEConv, global_mean_pool, global_add_pool
import torch.nn.functional as F
from dataclasses import dataclass


ATOM_FEATURE_DIMS = [
    len(x_map['atomic_num']),
    len(x_map['chirality']),
    len(x_map['degree']),
    len(x_map['formal_charge']),
    len(x_map['num_hs']),
    len(x_map['num_radical_electrons']),
    len(x_map['hybridization']),
    len(x_map['is_aromatic']),
    len(x_map['is_in_ring']),
]

BOND_FEATURE_DIMS = [
    len(e_map['bond_type']),
    len(e_map['stereo']),
    len(e_map['is_conjugated']),
]


@dataclass
class GraphEncoderConfig:
    hidden_dim = 256
    out_dim= 768  

    # GPS backbone
    num_layers= 4
    num_heads = 4
    dropout = 0.1
    attn_type = "multihead"

    # pooling + output normalization
    pool = "mean"
    normalize_out = True

    # small regularization in the categorical embedding sums
    feature_dropout = 0.0


class AtomEncoder(nn.Module):
    def __init__(self, hidden_dim, dropout = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embs = nn.ModuleList([nn.Embedding(dim, hidden_dim) for dim in ATOM_FEATURE_DIMS])
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for emb in self.embs:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def forward(self, x) :
        if x.dim() != 2 or x.size(1) != len(self.embs):
            raise ValueError(f"AtomEncoder expected x shape (N,{len(self.embs)}), got {tuple(x.shape)}")
        x = x.long()

        h = 0
        for i, emb in enumerate(self.embs):
            h = h + emb(x[:, i])
        return self.dropout(h)
    

class BondEncoder(nn.Module):
    def __init__(self, hidden_dim, dropout = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embs = nn.ModuleList([nn.Embedding(dim, hidden_dim) for dim in BOND_FEATURE_DIMS])
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for emb in self.embs:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def forward(self, edge_attr) :
        if edge_attr.dim() != 2 or edge_attr.size(1) != len(self.embs):
            raise ValueError(f"BondEncoder expected edge_attr shape (E,{len(self.embs)}), got {tuple(edge_attr.shape)}")
        edge_attr = edge_attr.long()

        e = 0
        for i, emb in enumerate(self.embs):
            e = e + emb(edge_attr[:, i])
        return self.dropout(e)
    


class GPSBackbone(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_layers,
        num_heads,
        dropout,
        attn_type,
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            local_mlp = nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            local_conv = GINEConv(local_mlp, train_eps=True, edge_dim=hidden_dim)

            # GPS layer combines local_conv + global attention
            self.layers.append(
                GPSConv(
                    channels=hidden_dim,
                    conv=local_conv,
                    heads=num_heads,
                    dropout=dropout,
                    attn_type=attn_type,
                )
            )

    def forward(self, h_nodes: torch.Tensor, edge_index: torch.Tensor, batch_vec: torch.Tensor, h_edges: torch.Tensor) -> torch.Tensor:
        h = h_nodes
        e = h_edges
        for layer in self.layers:
            # GPSConv signature: (x, edge_index, batch, edge_attr=...)
            h = layer(h, edge_index, batch_vec, edge_attr=e)
        return h
    


class GraphEncoder(nn.Module):
    def __init__(self, cfg: GraphEncoderConfig):
        super().__init__()
        self.cfg = cfg

        self.atom_encoder = AtomEncoder(cfg.hidden_dim, dropout=cfg.feature_dropout)
        self.bond_encoder = BondEncoder(cfg.hidden_dim, dropout=cfg.feature_dropout)

        self.backbone = GPSBackbone(
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            attn_type=cfg.attn_type,
        )

        # pooling
        if cfg.pool == "mean":
            self.pool = global_mean_pool
        elif cfg.pool == "add":
            self.pool = global_add_pool
        else:
            raise ValueError("cfg.pool must be 'mean' or 'add'")

        # projection head
        self.proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.out_dim),
        )

    def forward(self, batch) -> torch.Tensor:
        # Ensure we have a batch vector even for single-graph Data
        if not hasattr(batch, "batch") or batch.batch is None:
            batch.batch = torch.zeros(batch.x.size(0), dtype=torch.long, device=batch.x.device)

        # 1) categorical -> continuous
        h_nodes = self.atom_encoder(batch.x)         
        h_edges = self.bond_encoder(batch.edge_attr) 

        # 2) GPS node-level reasoning
        h_nodes = self.backbone(h_nodes, batch.edge_index, batch.batch, h_edges)

        # 3) pool -> graph embedding
        g = self.pool(h_nodes, batch.batch)          

        # 4) projection -> out space
        z = self.proj(g)                             

        # 5) normalize for cosine similarity
        if self.cfg.normalize_out:
            z = F.normalize(z, dim=-1)

        return z



if __name__ == "__main__":
    cfg = GraphEncoderConfig(hidden_dim=256, out_dim=768, num_layers=4, num_heads=4, dropout=0.1)
    model = GraphEncoder(cfg)
    print(model)














class BondEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
    def forward(self, x):
        pass

class AtomEncoder(nn.Module):
    def forward(self, x):
        pass


class GraphEncoder(nn.Module):
    def forward(self, x):
        pass
    

