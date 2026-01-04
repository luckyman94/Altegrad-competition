import torch
import torch.nn.functional as F
import json
from tqdm import tqdm
from pathlib import Path

from model import GraphEncoder, GraphEncoderConfig
from utils import (
    load_id2emb,
    load_descriptions_from_graphs,
    PreprocessedGraphDataset,
    collate_fn,
)
from torch.utils.data import DataLoader


def build_prompt(neighbor_descs):
    prompt = (
        "You are a chemist.\n\n"
        "Below are descriptions of molecules similar to the target molecule:\n\n"
    )
    for i, d in enumerate(neighbor_descs, 1):
        prompt += f"Example {i}:\n{d}\n\n"

    prompt += "Write a clear and factual description of the target molecule."
    return prompt


@torch.no_grad()
def main():
    DATA = "/content/drive/MyDrive/molecule-captioning/data"
    TRAIN_GRAPHS = f"{DATA}/train_graphs.pkl"
    TRAIN_EMB = "/content/drive/MyDrive/molecule-captioning/embeddings/train_embeddings_sentence-transformers_all-mpnet-base-v2.csv"
    CKPT = "/content/drive/MyDrive/molecule-captioning/checkpoints/gps_mpnet.pt"

    OUT_JSONL = "sft_train.jsonl"
    K = 5
    BATCH = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------
    # Load text embeddings
    # ------------------------
    train_emb = load_id2emb(TRAIN_EMB)
    train_ids = list(train_emb.keys())
    train_embs = torch.stack([train_emb[i] for i in train_ids]).to(device)
    train_embs = F.normalize(train_embs, dim=-1)

    # ------------------------
    # Load descriptions
    # ------------------------
    id2desc = load_descriptions_from_graphs(TRAIN_GRAPHS)
    train_descs = [id2desc[i] for i in train_ids]

    # ------------------------
    # Load graph encoder
    # ------------------------
    out_dim = train_embs.size(1)
    cfg = GraphEncoderConfig(out_dim=out_dim)
    encoder = GraphEncoder(cfg).to(device)
    encoder.load_state_dict(torch.load(CKPT, map_location=device))
    encoder.eval()

    # ------------------------
    # Encode train graphs
    # ------------------------
    ds = PreprocessedGraphDataset(TRAIN_GRAPHS)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=False, collate_fn=collate_fn)

    graph_embs = []
    for graphs in tqdm(dl, desc="Encoding train graphs"):
        graphs = graphs.to(device)
        z = encoder(graphs)
        z = F.normalize(z, dim=-1)
        graph_embs.append(z)

    graph_embs = torch.cat(graph_embs, dim=0)

    # ------------------------
    # Build SFT pairs
    # ------------------------
    sims = graph_embs @ train_embs.T
    topk = sims.topk(K + 1, dim=-1).indices.cpu()

    with open(OUT_JSONL, "w") as f:
        for i, idxs in tqdm(enumerate(topk), total=len(topk)):
            neigh = []
            for j in idxs.tolist():
                if j != i and len(neigh) < K:
                    neigh.append(train_descs[j])

            prompt = build_prompt(neigh)
            target = train_descs[i]

            f.write(json.dumps({
                "prompt": prompt,
                "completion": target
            }) + "\n")

    print(f"Saved SFT dataset to {OUT_JSONL}")


if __name__ == "__main__":
    main()
