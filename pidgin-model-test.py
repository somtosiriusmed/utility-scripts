#!/usr/bin/env python
# test_lora_fixed.py
import os
import torch
from unsloth import FastLanguageModel
from peft import PeftModel

# Environment setup
os.environ["UNSLOTH_DISABLE_DYNAMO"] = "1"
torch._dynamo.config.disable = True

print("Loading base model...")
# Load base model first
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Phi-4-mini-instruct",  # Base model
    max_seq_length=2048,
    load_in_4bit=False,
    device_map="auto"
)

print("Loading LoRA adapter...")
# Load the LoRA adapter using PEFT
try:
    model = PeftModel.from_pretrained(model, "unsloth-lora-output")
    print("LoRA adapter loaded successfully!")
except Exception as e:
    print(f"Error loading LoRA: {e}")
    print("Trying alternative method...")
    
    # Alternative: recreate LoRA structure and load
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
    )
    model.load_adapter("unsloth-lora-output", "default")

# Enable inference mode
FastLanguageModel.for_inference(model)

def test_generation(prompt, max_tokens=100):
    """Test generation with multiple fallback methods"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Method 1: Try with cache disabled
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                use_cache=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Method 1 failed: {e}")
    
    # Method 2: Try with different generation config
    try:
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
                use_cache=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Method 2 failed: {e}")
    
    # Method 3: Manual forward pass (fallback)
    try:
        with torch.no_grad():
            # Simple greedy decoding
            generated = inputs.input_ids.clone()
            for _ in range(max_tokens):
                outputs = model(generated)
                next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
                generated = torch.cat([generated, next_token], dim=1)
                if next_token.item() == tokenizer.eos_token_id:
                    break
        return tokenizer.decode(generated[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Method 3 failed: {e}")
        return "All generation methods failed"

# Test cases
test_prompts = [
    "### Instruction:\nSay hello in Nigerian Pidgin\n\n### Response:\n",
    "### Instruction:\nTranslate 'Good morning' to Nigerian Pidgin\n\n### Response:\n",
    "### Instruction:\nHow are you in pidgin English?\n\n### Response:\n"
]

print("\n" + "="*50)
print("TESTING FINE-TUNED MODEL")
print("="*50)

for i, prompt in enumerate(test_prompts, 1):
    print(f"\nTest {i}:")
    print(f"Prompt: {prompt.strip()}")
    print("Response:")
    response = test_generation(prompt)
    print(response)
    print("-" * 30)

print("\nTesting complete!")
