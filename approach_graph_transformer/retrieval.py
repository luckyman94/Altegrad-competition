#!/usr/bin/env python3
# retrieval.py
# --------------------------------------------------
# Simple semantic retrieval + generation
# GPS graph encoder → text kNN → LLM generation
# --------------------------------------------------

import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    GenerationConfig,
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
def load_graph_encoder(ckpt_path: str, device: str) -> GraphEncoder:
    cfg = GraphEncoderConfig()
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
def encode_graphs(model, graph_pkl, device, batch_size=64):
    ds = PreprocessedGraphDataset(graph_pkl)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_embs = []
    all_ids = []

    ptr = 0
    for graphs in dl:
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
    train_text_embs: torch.Tensor,
    train_descriptions: List[str],
    k: int,
):
    sims = query_embs @ train_text_embs.T
    topk = sims.topk(k, dim=-1).indices

    retrieved = []
    for idxs in topk:
        retrieved.append([train_descriptions[i] for i in idxs.tolist()])
    return retrieved


# --------------------------------------------------
# Prompt construction
# --------------------------------------------------
def build_prompt(retrieved_texts: List[str]) -> str:
    prompt = (
        "You are a chemist.\n\n"
        "Below are descriptions of molecules similar to the target molecule:\n\n"
    )

    for i, txt in enumerate(retrieved_texts, 1):
        prompt += f"Example {i}:\n{txt}\n\n"

    prompt += (
        "Based only on these examples, write a clear and factual description "
        "of the target molecule."
    )
    return prompt


# --------------------------------------------------
# Generation
# --------------------------------------------------
@torch.no_grad()
def generate_descriptions(
    prompts,
    model_name,
    max_new_tokens=128,
):
    config = AutoConfig.from_pretrained(model_name)

    if config.is_encoder_decoder:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, dtype=torch.float16, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, device_map="auto"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" if config.is_encoder_decoder else "left"

    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    outputs = []

    for p in prompts:
        inputs = tokenizer(
            p,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(model.device)

        out = model.generate(**inputs, generation_config=gen_cfg)

        if config.is_encoder_decoder:
            text = tokenizer.decode(out[0], skip_special_tokens=True)
        else:
            gen = out[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(gen, skip_special_tokens=True)

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
    parser.add_argument("--llm", type=str, default="QizhiPei/biot5-plus-base")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out_csv", type=str, default="retrieval_results.csv")

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
    # Encode test graphs
    # -------------------------------
    graph_encoder = load_graph_encoder(args.graph_ckpt, device)

    query_embs, query_ids = encode_graphs(
        graph_encoder,
        test_graphs,
        device,
        batch_size=args.batch_size,
    )

    print("Loaded and encoded test graphs")

    # -------------------------------
    # Retrieval
    # -------------------------------
    retrieved = knn_retrieval(
        query_embs,
        train_embs,
        train_descs,
        k=args.k,
    )

    print("Completed retrieval")

    prompts = [build_prompt(r) for r in retrieved]

    # -------------------------------
    # Generation
    # -------------------------------
    generations = generate_descriptions(
        prompts,
        model_name=args.llm,
    )

    print("Completed generation")

    # -------------------------------
    # Save
    # -------------------------------
    import pandas as pd

    df = pd.DataFrame({
        "ID": query_ids,
        "description": generations,
    })
    df.to_csv(args.out_csv, index=False)
    print(f"Saved predictions to {args.out_csv}")


if __name__ == "__main__":
    main()
