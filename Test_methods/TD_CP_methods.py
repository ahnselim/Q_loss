import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from copy import deepcopy
from datetime import datetime
import os
import tensorly as tl
from tensorly.decomposition import parafac 

tl.set_backend('pytorch')

# 0. Load model 
model_name = "your_model_name"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Load full-precision model
model_fp = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
model_fp.eval()

# 3. Clone model for CP restoration
model_restored = deepcopy(model_fp)

# 4. Weight quantization (absmax 8bit) & CP restoration
def absmax_quantize(weight):
    max_val = weight.abs().max()
    scale = max_val / 127
    quant = torch.round(weight / scale).clamp(-127, 127)
    dequant = quant * scale
    return dequant, scale

def low_rank_restore_cp(fp_weight, quant_weight, rank=8):
    shape = fp_weight.shape
    if len(shape) < 2:
        raise ValueError("Tensor must be at least 2D for CP decomposition.")

    delta = fp_weight - quant_weight
    factors = parafac(delta, rank=rank, init='svd', n_iter_max=100)
    delta_approx = tl.cp_to_tensor(factors)
    W_approx = quant_weight + delta_approx
    return W_approx

# 5. Apply CP-restored weights
print("🔧 Restoring attention weights (CP Decomposition)...")
with torch.no_grad():
    for name, param in model_restored.named_parameters():
        if any(k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj"]) and "weight" in name:
            fp = param.data.detach().cpu().float()
            dequant, _ = absmax_quantize(fp)
            try:
                restored = low_rank_restore_cp(fp, dequant, rank=8)
                param.copy_(restored.to(param.device).to(param.dtype))
                print(f"✅ Restored: {name}")
            except Exception as e:
                print(f"❌ Skip {name}: {e}")

# 6. Text generation & PPL
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

# 7. Prompt test
prompt = "Tell me a fun fact about the moon."

print("\n🧪 [Full-Precision Model Output]")
print(generate_text(model_fp, prompt))
print(f"PPL: {compute_ppl(model_fp, prompt):.2f}")

print("\n🧪 [CP Restored Model Output]")
print(generate_text(model_restored, prompt))
print(f"PPL: {compute_ppl(model_restored, prompt):.2f}")

# 8. Save the restored model
model_basename = model_name.split("/")[-1]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path_restored = f"/home/caslab/asl/Q_loss/Q_loss/retore_checkpoint/{model_basename}_cp_restored_{timestamp}"
os.makedirs(save_path_restored, exist_ok=True)

model_restored.save_pretrained(save_path_restored)
tokenizer.save_pretrained(save_path_restored)

print(f"\n📦 Saved Restored Model : {save_path_restored}")
