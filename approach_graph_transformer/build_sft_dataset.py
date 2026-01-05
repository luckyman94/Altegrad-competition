#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import GraphEncoder, GraphEncoderConfig
from utils import (
    load_id2emb,
    load_descriptions_from_graphs,
    PreprocessedGraphDataset,
    collate_fn,
)


# --------------------------------------------------
# Prompt builder
# --------------------------------------------------
def build_prompt(neighbor_descs):
    prompt = (
        "You are a domain expert in chemistry and molecular biology.\n"
        "Your task is to write a factual, concise description of a target molecule.\n\n"

        "You are given descriptions of molecules that are chemically or functionally similar.\n"
        "Use them ONLY as stylistic and structural references.\n"
        "Do NOT copy sentences or specific values.\n"
        "Do NOT invent properties that are not supported by the examples.\n\n"

        "[REFERENCE DESCRIPTIONS]\n"
    )

    for i, d in enumerate(neighbor_descs, 1):
        prompt += f"Reference {i}:\n{d.strip()}\n\n"

    prompt += (
        "[INSTRUCTIONS]\n"
        "- Write a single coherent paragraph.\n"
        "- Use precise scientific language.\n"
        "- Focus on chemical structure, molecular class, and known functional roles if evident.\n"
        "- If information is uncertain, remain general.\n"
        "- Do NOT mention the references.\n\n"
        "Target molecule description:"
    )

    return prompt



# --------------------------------------------------
# Main
# --------------------------------------------------
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument("--train_graphs", required=True, type=str)
    parser.add_argument("--train_emb", required=True, type=str)
    parser.add_argument("--model_ckpt", required=True, type=str)
    parser.add_argument("--out_jsonl", required=True, type=str)

    # Params
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_graphs = Path(args.train_graphs)
    train_emb_path = Path(args.train_emb)
    model_ckpt = Path(args.model_ckpt)
    out_jsonl = Path(args.out_jsonl)

    # ------------------------
    # Load text embeddings
    # ------------------------
    train_emb = load_id2emb(train_emb_path)
    train_ids = list(train_emb.keys())
    train_embs = torch.stack([train_emb[i] for i in train_ids]).to(device)
    train_embs = F.normalize(train_embs, dim=-1)

    # ------------------------
    # Load descriptions
    # ------------------------
    id2desc = load_descriptions_from_graphs(train_graphs)
    train_descs = [id2desc[i] for i in train_ids]

    # ------------------------
    # Load graph encoder
    # ------------------------
    out_dim = train_embs.size(1)
    cfg = GraphEncoderConfig(out_dim=out_dim)
    encoder = GraphEncoder(cfg).to(device)
    encoder.load_state_dict(torch.load(model_ckpt, map_location=device))
    encoder.eval()

    # ------------------------
    # Encode train graphs
    # ------------------------
    ds = PreprocessedGraphDataset(train_graphs)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    graph_embs = []
    for graphs in tqdm(dl, desc="Encoding train graphs"):
        graphs = graphs.to(device)
        z = encoder(graphs)
        z = F.normalize(z, dim=-1)
        graph_embs.append(z)

    graph_embs = torch.cat(graph_embs, dim=0)

    # ------------------------
    # Build SFT pairs (leave-one-out)
    # ------------------------
    sims = graph_embs @ train_embs.T
    topk = sims.topk(args.k + 1, dim=-1).indices.cpu()

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with open(out_jsonl, "w") as f:
        for i, idxs in tqdm(enumerate(topk), total=len(topk)):
            neigh = []
            for j in idxs.tolist():
                if j != i and len(neigh) < args.k:
                    neigh.append(train_descs[j])

            prompt = build_prompt(neigh)
            target = train_descs[i]

            f.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "completion": target,
                    }
                )
                + "\n"
            )

    print(f"✅ Saved SFT dataset to {out_jsonl}")


if __name__ == "__main__":
    main()
