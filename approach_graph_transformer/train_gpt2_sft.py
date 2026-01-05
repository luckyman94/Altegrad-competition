import json
import torch
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

MODEL_NAME = "gpt2"
DATA_FILE = "sft_train.jsonl"
OUT_DIR = "./gpt2-sft"
MAX_LEN = 512

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

dataset = load_dataset("json", data_files=DATA_FILE)["train"]


def tokenize(example):
    prompt = example["prompt"]
    completion = example["completion"]

    full_text = prompt + "\n" + completion

    enc = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )

    labels = enc["input_ids"].copy()

    # Mask du prompt
    prompt_ids = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_LEN,
        add_special_tokens=False,
    )["input_ids"]

    labels[:len(prompt_ids)] = [-100] * len(prompt_ids)

    enc["labels"] = labels
    return enc



dataset = dataset.map(tokenize, remove_columns=["prompt", "completion"])

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    num_train_epochs=3,
    fp16=True,
    logging_steps=100,
    save_strategy="epoch",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=collator,
)

trainer.train()
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
