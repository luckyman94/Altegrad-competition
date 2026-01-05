#!/usr/bin/env python3
import argparse
import torch
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model


# --------------------------------------------------
# Tokenization
# --------------------------------------------------
def build_tokenize_fn(tokenizer, max_len):
    def tokenize(example):
        prompt = example["prompt"]
        completion = example["completion"]

        full_text = prompt + "\n" + completion + tokenizer.eos_token

        enc = tokenizer(
            full_text,
            truncation=True,
            max_length=max_len,
            padding="max_length",
        )

        labels = enc["input_ids"].copy()

        # Mask prompt
        prompt_ids = tokenizer(
            prompt,
            truncation=True,
            max_length=max_len,
            add_special_tokens=False,
        )["input_ids"]

        labels[: len(prompt_ids)] = [-100] * len(prompt_ids)
        enc["labels"] = labels
        return enc

    return tokenize


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    # Data / model
    parser.add_argument("--model_name", default="gpt2")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--val_file", default=None)
    parser.add_argument("--out_dir", required=True)

    # Training
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=5)

    # Checkpoints / eval
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)

    # Early stopping
    parser.add_argument("--early_stopping_patience", type=int, default=2)

    # LoRA
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------
    # Tokenizer
    # --------------------------------------------------
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    model.gradient_checkpointing_enable()

    # LoRA
    if args.use_lora:
        lora_alpha = args.lora_alpha or (2 * args.lora_r)
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    model.to(device)

    # Optional torch.compile (safe)
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception:
            pass

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------
    train_ds = load_dataset("json", data_files=args.train_file)["train"]

    tokenize_fn = build_tokenize_fn(tokenizer, args.max_len)
    train_ds = train_ds.map(
        tokenize_fn,
        remove_columns=["prompt", "completion"],
        desc="Tokenizing train",
    )

    eval_ds = None
    if args.val_file is not None:
        eval_ds = load_dataset("json", data_files=args.val_file)["train"]
        eval_ds = eval_ds.map(
            tokenize_fn,
            remove_columns=["prompt", "completion"],
            desc="Tokenizing val",
        )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # --------------------------------------------------
    # Training arguments
    # --------------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        fp16=torch.cuda.is_available(),

        evaluation_strategy="steps" if eval_ds else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        load_best_model_at_end=bool(eval_ds),
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        logging_steps=100,
        report_to=[],
        remove_unused_columns=False,
    )

    callbacks = []
    if eval_ds:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    trainer.train()

    # --------------------------------------------------
    # Save final model
    # --------------------------------------------------
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)


if __name__ == "__main__":
    main()
