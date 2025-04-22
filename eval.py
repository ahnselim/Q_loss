import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_name= "your_model_name"  # Replace with your model name
device = "cuda" if torch.cuda.is_available() else "cpu"
model_quantized="your_model_name"
model_restored="your_model_name"

# 1. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Load full-precision model
model_fp = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
model_fp.eval()
# Load quantized and restored models properly
model_quantized = AutoModelForCausalLM.from_pretrained(model_quantized, torch_dtype=torch.float16).to(device)
model_quantized.eval()

model_restored = AutoModelForCausalLM.from_pretrained(model_restored, torch_dtype=torch.float16).to(device)
model_restored.eval()


def generate_text(model, prompt, max_new_tokens=50):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def compute_ppl(model, prompt):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    return torch.exp(loss).item()

prompt = "Tell me a fun fact about the moon."

print("\n[Full-Precision Model Output]")
print(generate_text(model_fp, prompt))
print(f"PPL: {compute_ppl(model_fp, prompt):.2f}")

print("\n[4-bit Quantized Model Output]")
print(generate_text(model_quantized, prompt))
print(f"PPL: {compute_ppl(model_quantized, prompt):.2f}")

print("\n[Restored Model Output]")
print(generate_text(model_restored, prompt))
print(f"PPL: {compute_ppl(model_restored, prompt):.2f}")