import argparse
import pickle
import os
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, T5EncoderModel


# -----------------------------------------------------
# Pooling strategies
# -----------------------------------------------------
def pool_embeddings(outputs, attention_mask, model_type, pooling):
    hidden = outputs.last_hidden_state

    if pooling == "cls":
        return hidden[:, 0]

    elif pooling == "mean":
        mask = attention_mask.unsqueeze(-1).float()
        summed = torch.sum(hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    elif pooling == "last":
        lengths = attention_mask.sum(dim=1) - 1
        return hidden[torch.arange(hidden.size(0)), lengths]

    else:
        raise ValueError(f"Unknown pooling: {pooling}")

def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")

# -----------------------------------------------------
# Main
# -----------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="embeddings")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--pooling", type=str, choices=["cls", "mean", "last"], default="cls")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------------------------------
    # Load tokenizer + model
    # -------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "t5" in args.model.lower():
        model = T5EncoderModel.from_pretrained(args.model)
    else:
        model = AutoModel.from_pretrained(args.model)

    model = model.to(device)
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)

    # -------------------------------------------------
    # Process splits
    # -------------------------------------------------
    for split in args.splits:
        print(f"\nProcessing {split} split")

        pkl_path = Path(args.data_dir) / f"{split}_graphs.pkl"
        with open(pkl_path, "rb") as f:
            graphs = pickle.load(f)

        texts = [g.description for g in graphs]
        ids = [g.id for g in graphs]

        all_embeddings = []

        for i in tqdm(range(0, len(texts), args.batch_size)):
            batch_texts = texts[i : i + args.batch_size]

            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            emb = pool_embeddings(
                outputs,
                inputs["attention_mask"],
                model.config.model_type,
                args.pooling,
            )

            all_embeddings.append(emb.cpu())

        all_embeddings = torch.cat(all_embeddings, dim=0)

        df = pd.DataFrame({
            "ID": ids,
            "embedding": [
                ",".join(map(str, e.tolist())) for e in all_embeddings
            ]
        })

        model_tag = sanitize_model_name(args.model)
        out_path = Path(args.out_dir) / f"{split}_embeddings_{model_tag}.csv"

        df.to_csv(out_path, index=False)
        print(f"Saved embeddings to {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
