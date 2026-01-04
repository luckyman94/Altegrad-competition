import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pathlib import Path

from model import GraphEncoder, GraphEncoderConfig
from loss import contrastive_clip_loss
from utils import PreprocessedGraphDataset,load_id2emb, collate_fn
import argparse


def train_epoch(
    graph_encoder,
    dataloader,
    optimizer,
    device,
):
    graph_encoder.train()
    total_loss = 0.0
    total = 0

    for graphs, z_text in dataloader:
        graphs = graphs.to(device)
        z_text = F.normalize(z_text.to(device), dim=-1)

        z_graph = graph_encoder(graphs)

        loss = contrastive_clip_loss(z_graph, z_text)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(graph_encoder.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * graphs.num_graphs
        total += graphs.num_graphs

    return total_loss / total




@torch.no_grad()
def evaluate_retrieval(
    graph_encoder,
    dataloader,
    device,
):
    graph_encoder.eval()

    all_graph, all_text = [], []

    for graphs, z_text in dataloader:
        graphs = graphs.to(device)
        z_text = F.normalize(z_text.to(device), dim=-1)

        z_graph = graph_encoder(graphs)

        all_graph.append(z_graph)
        all_text.append(z_text)

    G = torch.cat(all_graph)
    T = torch.cat(all_text)

    sims = T @ G.T
    ranks = sims.argsort(dim=-1, descending=True)

    gt = torch.arange(T.size(0), device=device)
    pos = (ranks == gt[:, None]).nonzero()[:, 1] + 1

    return {
        "MRR": (1.0 / pos.float()).mean().item(),
        "R@1": (pos <= 1).float().mean().item(),
        "R@5": (pos <= 5).float().mean().item(),
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--train_emb", type=str, required=True)
    parser.add_argument("--val_emb", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)

    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--out_dim", type=int, default=768)
    parser.add_argument("--attn_type", type=str, default="multihead")
    parser.add_argument("--pool", type=str, default="mean")

    parser.add_argument("--out_ckpt", type=str, default="graph_encoder.pt")
    parser.add_argument("--patience", type=int, default=5)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --------------------------------------------------
    # Paths
    # --------------------------------------------------
    data_dir = Path(args.data_dir)
    train_graphs = data_dir / "train_graphs.pkl"
    val_graphs = data_dir / "validation_graphs.pkl"

    # --------------------------------------------------
    # Load text embeddings
    # --------------------------------------------------
    train_text_emb = load_id2emb(args.train_emb)
    val_text_emb = load_id2emb(args.val_emb)

    out_dim = len(next(iter(train_text_emb.values())))

    # --------------------------------------------------
    # Datasets / loaders
    # --------------------------------------------------
    train_ds = PreprocessedGraphDataset(train_graphs, train_text_emb)
    val_ds = PreprocessedGraphDataset(val_graphs, val_text_emb)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    cfg = GraphEncoderConfig(
        hidden_dim=args.hidden_dim,
        out_dim=out_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        attn_type=args.attn_type,
        pool=args.pool,
    )

    graph_encoder = GraphEncoder(cfg).to(device)

    print(f"Trainable params: {sum(p.numel() for p in graph_encoder.parameters() if p.requires_grad):,}")

    # --------------------------------------------------
    # Optimizer / scheduler
    # --------------------------------------------------
    optimizer = torch.optim.AdamW(
        graph_encoder.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    best_mrr = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(graph_encoder, train_dl, optimizer, device)


        metrics = evaluate_retrieval(graph_encoder, val_dl, device)


        scheduler.step()

        print(
            f"Epoch {epoch:03d} | "
            f"loss={train_loss:.4f} | "
            f"MRR={metrics['MRR']:.4f} | "
            f"R@1={metrics['R@1']:.4f} | "
            f"R@5={metrics['R@5']:.4f}"
        )

        # ----------------------------------------------
        # Early stopping / checkpoint
        # ----------------------------------------------
        if metrics["MRR"] > best_mrr:
            best_mrr = metrics["MRR"]
            patience_counter = 0
            torch.save(graph_encoder.state_dict(), args.out_ckpt)
            print(f"  → New best MRR: {best_mrr:.4f} | saved {args.out_ckpt}")
        else:
            patience_counter += 1
            print(f"  → No improvement ({patience_counter}/{args.patience})")

        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break

    print(f"\nTraining finished. Best MRR = {best_mrr:.4f}")


if __name__ == "__main__":
    main()
