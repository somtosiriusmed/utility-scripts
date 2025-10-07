#!/usr/bin/env python
# train_lora.py

import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# ========= Environment Setup =========
os.environ["UNSLOTH_DISABLE_DYNAMO"] = "1"
os.environ["DISABLE_TRANSFORMERS_DYNAMO"] = "1"
torch._dynamo.config.disable = True
os.environ["ACCELERATE_DISABLE_FP16"] = "1"
torch.set_default_dtype(torch.float32)

# ========= Load Model =========
print(">>> Downloading and loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "unsloth/Phi-4-mini-instruct",
    max_seq_length = 2048,
    load_in_4bit   = False,
    device_map     = None,   # disables accelerate’s meta offload
)

# Ensure pad token exists
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# ========= Attach LoRA =========
print(">>> Attaching LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj",
    ],
    use_gradient_checkpointing=False,
)

# ========= Load Dataset =========
print(">>> Loading dataset...")
dataset = load_dataset("json", data_files="bbc_pidgin_dataset/bbc_pidgin_dataset.jsonl")

# Format function
def format_example(example):
    if example.get("input", ""):
        return {
            "text": f"### Instruction:\n{example['instruction']}\n\n"
                    f"### Input:\n{example['input']}\n\n"
                    f"### Response:\n{example['output']}"
        }
    else:
        return {
            "text": f"### Instruction:\n{example['instruction']}\n\n"
                    f"### Response:\n{example['output']}"
        }

dataset = dataset.map(format_example)

# Train / eval split
train_dataset = dataset["train"].shuffle(seed=42).select(
    range(int(0.9 * len(dataset["train"])))
)
eval_dataset = dataset["train"].shuffle(seed=42).select(
    range(int(0.9 * len(dataset["train"])), len(dataset["train"]))
)

# ========= Trainer Config =========
config = SFTConfig(
    output_dir="unsloth-lora-output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=1,
    save_strategy="epoch",
    fp16=False,
    bf16=False,
    fp16_full_eval=False,
    report_to="none",
    eval_strategy="steps",
    eval_steps=20,
    optim="adamw_torch",
)

# ========= Trainer =========
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    packing=False,
    args=config,
    compile=False,
)

# ========= Sanity Check =========
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable} / {total}")

# ========= Train =========
print(">>> Starting training...")
trainer.train()

# ========= Evaluate =========
print(">>> Evaluating model...")
results = trainer.evaluate()
print("Eval results:", results)

# ========= Save Final Model =========
print(">>> Saving model...")
trainer.save_model("unsloth-lora-output")
print("Training complete. Model saved at 'unsloth-lora-output'")

# Shut server down
os.system("sudo shutdown -h now")
