# knn_generate.py
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

from model import GraphEncoder, GraphEncoderConfig
from utils import (
    PreprocessedGraphDataset,
    collate_fn,
    load_id2emb,
    load_descriptions_from_graphs,
)


@torch.no_grad()
def encode_graphs(model, graphs_pkl, device, batch_size=64):
    ds = PreprocessedGraphDataset(graphs_pkl)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_embs = []
    all_ids = []

    for graphs in tqdm(dl, desc="Encoding graphs"):
        graphs = graphs.to(device)
        z = model(graphs)
        z = F.normalize(z, dim=-1)
        all_embs.append(z.cpu())
        all_ids.extend(ds.ids[: graphs.num_graphs])

    return torch.cat(all_embs), all_ids


def knn_topk(query_embs, train_embs, k):
    sims = query_embs @ train_embs.T
    top_sims, top_idx = torch.topk(sims, k=k, dim=-1)
    return top_idx, top_sims

def build_prompt(query_desc, neighbor_descs):
    prompt = (
        "You are a biomedical expert.\n\n"
        "[SIMILAR DESCRIPTIONS]\n"
    )
    for i, d in enumerate(neighbor_descs, 1):
        prompt += f"{i}. {d}\n"

    prompt += (
        "\n[INSTRUCTION]\n"
        "Generate a concise and factual description of the query protein graph.\n\n"
        "Description:"
    )
    return prompt


class KNNSFTDataset(torch.utils.data.Dataset):
    def __init__(self, prompts, references, tokenizer, max_len=1024):
        self.prompts = prompts
        self.references = references
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        ref = self.references[idx]

        full = prompt + " " + ref + self.tokenizer.eos_token

        enc = self.tokenizer(
            full,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        input_ids = enc.input_ids.squeeze()
        labels = input_ids.clone()

        prompt_ids = self.tokenizer(
            prompt, add_special_tokens=False
        ).input_ids

        labels[: len(prompt_ids)] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": enc.attention_mask.squeeze(),
        }


def finetune_gpt2(prompts, refs, output_dir):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained("gpt2")

    lora_cfg = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = KNNSFTDataset(prompts, refs, tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        max_steps=800,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_steps=200,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


@torch.no_grad()
def generate(model_dir, prompts, out_csv):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float16
    ).cuda()
    model.eval()

    rows = []

    for p in tqdm(prompts, desc="Generating"):
        inputs = tokenizer(p, return_tensors="pt").to("cuda")
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )
        gen = out[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen, skip_special_tokens=True)
        rows.append(text)

    import pandas as pd
    pd.DataFrame({"description": rows}).to_csv(out_csv, index=False)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir")
    parser.add_argument("--train_emb")
    parser.add_argument("--model_ckpt")
    parser.add_argument("--out_dir")
    args = parser.parse_args()

    device = "cuda"

    # Load graph encoder
    cfg = GraphEncoderConfig()
    gnn = GraphEncoder(cfg).to(device)
    gnn.load_state_dict(torch.load(args.model_ckpt))
    gnn.eval()

    # Load data
    train_graphs = Path(args.data_dir) / "train_graphs.pkl"
    id2desc = load_descriptions_from_graphs(train_graphs)

    train_embs = load_id2emb(args.train_emb)
    train_ids = list(train_embs.keys())
    train_text_embs = torch.stack([train_embs[i] for i in train_ids])

    # Encode graphs
    g_embs, g_ids = encode_graphs(gnn, train_graphs, device)

    # kNN
    top_idx, _ = knn_topk(g_embs, train_text_embs, k=3)

    prompts, refs = [], []
    for i, qid in enumerate(g_ids):
        neigh_ids = [train_ids[j] for j in top_idx[i]]
        neigh_descs = [id2desc[n] for n in neigh_ids]
        prompts.append(build_prompt(id2desc[qid], neigh_descs))
        refs.append(id2desc[qid])

    finetune_gpt2(prompts, refs, args.out_dir)
