import argparse
from typing import Dict

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

SUPPORTED_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen2.5-7B-Instruct",
]


def load_model_and_tokenizer(model_name: str,
                             use_lora: bool = True,
                             load_in_4bit: bool = False):
    """Load a causal LM and tokenizer with optional LoRA/QLoRA."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict = {"device_map": "auto"}
    if load_in_4bit:
        model_kwargs.update({
            "load_in_4bit": True,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.float16,
        })
    else:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    if use_lora:
        lora = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)

    return model, tokenizer


def tokenize_example(example: Dict, tokenizer, max_length: int) -> Dict:
    prompt_ids = tokenizer.encode(example["prompt"], add_special_tokens=False)
    target_ids = tokenizer.encode(example["target"], add_special_tokens=False)
    input_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
    labels = [-100] * len(prompt_ids) + target_ids + [tokenizer.eos_token_id]
    return {
        "input_ids": input_ids[:max_length],
        "labels": labels[:max_length],
    }


class ConstantLengthDataset(torch.utils.data.Dataset):
    """Pack tokenized examples into constant-length sequences."""

    def __init__(self, dataset, tokenizer, seq_length: int):
        eos = tokenizer.eos_token_id
        self.seq_length = seq_length
        self.examples = []
        buffer_input, buffer_labels = [], []
        for ex in dataset:
            buffer_input.extend(ex["input_ids"] + [eos])
            buffer_labels.extend(ex["labels"] + [eos])
            while len(buffer_input) >= seq_length:
                self.examples.append({
                    "input_ids": buffer_input[:seq_length],
                    "labels": buffer_labels[:seq_length],
                    "attention_mask": [1] * seq_length,
                })
                buffer_input = buffer_input[seq_length:]
                buffer_labels = buffer_labels[seq_length:]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        return {k: torch.tensor(v) for k, v in item.items()}


def build_dataset(ds: Dataset, tokenizer, max_length: int, pack: bool = False):
    tokenized = ds.map(
        lambda x: tokenize_example(x, tokenizer, max_length),
        remove_columns=ds.column_names,
    )
    if pack:
        return ConstantLengthDataset(tokenized, tokenizer, max_length)
    tokenized.set_format(type="torch")
    return tokenized


def train(
    dataset: Dataset,
    model_name: str,
    output_dir: str,
    use_lora: bool = True,
    load_in_4bit: bool = False,
    max_length: int = 2048,
    packing: bool = False,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    num_train_epochs: int = 1,
    learning_rate: float = 2e-4,
    lr_scheduler_type: str = "linear",
):
    model, tokenizer = load_model_and_tokenizer(
        model_name, use_lora=use_lora, load_in_4bit=load_in_4bit
    )
    train_ds = build_dataset(dataset, tokenizer, max_length, pack=packing)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        save_strategy="epoch",
        fp16=not load_in_4bit,
        logging_steps=10,
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds)
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LMs with LoRA or full FT")
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default=SUPPORTED_MODELS[0])
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    args = parser.parse_args()
    print(
        "Utility script; import `train` and provide a Dataset to run fine-tuning.",
        f"Selected model: {args.model}, output_dir: {args.output_dir}",
    )
