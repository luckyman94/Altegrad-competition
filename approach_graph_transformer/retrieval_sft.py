#!/usr/bin/env python3
# retrieval_gpt2.py
# --------------------------------------------------
# Graph retrieval + GPT-2 generation
# --------------------------------------------------

import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from model import GraphEncoder, GraphEncoderConfig
from utils import (
    PreprocessedGraphDataset,
    collate_fn,
    load_id2emb,
    load_descriptions_from_graphs,
)

# --------------------------------------------------
# Load trained GraphEncoder
# --------------------------------------------------
def load_graph_encoder(ckpt_path: str, out_dim: int, device: str) -> GraphEncoder:
    cfg = GraphEncoderConfig(out_dim=out_dim)
    model = GraphEncoder(cfg)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# --------------------------------------------------
# Encode graphs → embeddings
# --------------------------------------------------
@torch.no_grad()
def encode_graphs(
    model: GraphEncoder,
    graph_pkl: str,
    device: str,
    batch_size: int,
):
    ds = PreprocessedGraphDataset(graph_pkl)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_embs = []
    all_ids = []

    ptr = 0
    for graphs in tqdm(dl, desc="Encoding graphs"):
        graphs = graphs.to(device)
        z = model(graphs)
        z = F.normalize(z, dim=-1)
        all_embs.append(z)

        bs = graphs.num_graphs
        all_ids.extend(ds.ids[ptr:ptr + bs])
        ptr += bs

    return torch.cat(all_embs, dim=0), all_ids


# --------------------------------------------------
# kNN retrieval
# --------------------------------------------------
@torch.no_grad()
def knn_retrieval(
    query_embs: torch.Tensor,
    train_embs: torch.Tensor,
    train_descs: List[str],
    k: int,
):
    sims = query_embs @ train_embs.T
    topk = sims.topk(k, dim=-1).indices

    retrieved = []
    for idxs in topk:
        retrieved.append([train_descs[i] for i in idxs.tolist()])
    return retrieved


# --------------------------------------------------
# Prompt (BLEU-friendly)
# --------------------------------------------------
def build_prompt(neighbor_descs: List[str]) -> str:
    prompt = "Examples of molecule descriptions:\n\n"
    for d in neighbor_descs:
        prompt += d.strip() + "\n\n"
    prompt += "Description:\n"
    return prompt




# --------------------------------------------------
# GPT-2 generation
# --------------------------------------------------
def clean_generation(text: str) -> str:
    BAD_PREFIXES = [
        "Description:",
        "Examples",
        "Example",
        "You are",
        "Write a",
    ]
    for p in BAD_PREFIXES:
        if p in text:
            text = text.split(p)[0].strip()
    return text

def postprocess_description(text: str) -> str:
    text = text.strip()

    # Split si GPT-2 génère plusieurs descriptions
    if text.count("The molecule is") > 1:
        text = text.split("The molecule is", 2)[1]
        text = "The molecule is" + text

    # Coupe aux doubles sauts de ligne
    text = text.split("\n\n")[0]

    # Coupe phrases incomplètes
    if not text.endswith("."):
        text = text.rsplit(".", 1)[0] + "."

    return text.strip()


@torch.no_grad()
def generate_descriptions(
    prompts: List[str],
    model_name: str,
    device: str,
    max_new_tokens: int,
    batch_size: int,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # ✅ IMPORTANT pour GPT-2

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    # ✅ Limite GPT-2
    MAX_CONTEXT = model.config.n_positions  # 1024
    MAX_PROMPT_LEN = MAX_CONTEXT - max_new_tokens
    if MAX_PROMPT_LEN <= 0:
        raise ValueError("max_new_tokens is too large for GPT-2 context window")

    outputs = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch = prompts[i:i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_PROMPT_LEN,  # ✅ FIX CRITIQUE
        ).to(device)

        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        for j in range(len(batch)):
            prompt_len = inputs["input_ids"][j].shape[0]
            gen = out[j][prompt_len:]
            text = tokenizer.decode(gen, skip_special_tokens=True)
            text = postprocess_description(text)
            outputs.append(text.strip())

    return outputs



# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--train_emb", type=str, required=True)
    parser.add_argument("--graph_ckpt", type=str, required=True)

    parser.add_argument("--llm", type=str, default="gpt2")
    parser.add_argument("--k", type=int, default=5)

    parser.add_argument("--graph_batch_size", type=int, default=64)
    parser.add_argument("--gen_batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=120)

    parser.add_argument("--out_csv", type=str, default="submission.csv")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = Path(args.data_dir)
    train_graphs = str(data_dir / "train_graphs.pkl")
    test_graphs = str(data_dir / "test_graphs.pkl")

    # -------------------------------
    # Load text embeddings + desc
    # -------------------------------
    train_text_emb = load_id2emb(args.train_emb)
    train_ids = list(train_text_emb.keys())
    train_embs = torch.stack([train_text_emb[i] for i in train_ids]).to(device)
    train_embs = F.normalize(train_embs, dim=-1)

    train_id2desc = load_descriptions_from_graphs(train_graphs)
    train_descs = [train_id2desc[i] for i in train_ids]

    print(f"Loaded {len(train_embs)} training text embeddings")

    # -------------------------------
    # Load graph encoder
    # -------------------------------
    out_dim = train_embs.size(1)
    graph_encoder = load_graph_encoder(args.graph_ckpt, out_dim, device)

    # -------------------------------
    # Encode test graphs
    # -------------------------------
    query_embs, query_ids = encode_graphs(
        graph_encoder,
        test_graphs,
        device,
        batch_size=args.graph_batch_size,
    )

    print("Encoded test graphs:", query_embs.shape)

    # -------------------------------
    # Retrieval
    # -------------------------------
    retrieved_descs = knn_retrieval(
        query_embs,
        train_embs,
        train_descs,
        k=args.k,
    )

    prompts = [build_prompt(r) for r in retrieved_descs]

    # -------------------------------
    # Generation
    # -------------------------------
    generations = generate_descriptions(
        prompts,
        model_name=args.llm,
        device=device,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.gen_batch_size,
    )

    # -------------------------------
    # Save
    # -------------------------------
    df = pd.DataFrame({
        "ID": query_ids,
        "description": generations,
    })
    df.to_csv(args.out_csv, index=False)
    print(f"Saved submission to {args.out_csv}")


if __name__ == "__main__":
    main()
