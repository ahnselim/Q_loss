from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import datetime
import os

model_name = "your model"
model_basename = model_name.split("/")[-1]
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",  # or "fp4"
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)


save_path = f"your path_{model_basename}_quantized_{timestamp}"
os.makedirs(save_path, exist_ok=True)

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"save path: {save_path}")

print("success")
