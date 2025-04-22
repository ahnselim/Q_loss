import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from copy import deepcopy
from datetime import datetime
import os

# 0. Load model
model_name = "your_model_name"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Load full-precision model
model_fp = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
model_fp.eval()

# 3. Clone model for restoration
model_restored = deepcopy(model_fp)

# 4. Weight restoration function
def absmax_quantize(weight):
    max_val = weight.abs().max()
    scale = max_val / 127
    quant = torch.round(weight / scale).clamp(-127, 127)
    dequant = quant * scale
    return dequant, scale

def low_rank_restore(fp_weight, quant_weight, rank=8):
    if fp_weight.numel() < 2 or quant_weight.numel() < 2:
        raise ValueError("Too small tensor for SVD")

    shape = fp_weight.shape
    fp_matrix = fp_weight.view(shape[0], -1)
    quant_matrix = quant_weight.view(shape[0], -1)

    delta = fp_matrix - quant_matrix
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    U_r = U[:, :rank]
    S_r = torch.diag(S[:rank])
    V_r = Vh[:rank, :]
    delta_approx = U_r @ S_r @ V_r
    W_approx = quant_matrix + delta_approx

    return W_approx.view(shape)

# 5. Use restored weights
print(" Applying..")
with torch.no_grad():
    for name, param in model_restored.named_parameters():
        if any(k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj"]) and "weight" in name:
            fp = param.data.detach().cpu().float()
            q, _ = absmax_quantize(fp)
            try:
                restored = low_rank_restore(fp, q, rank=8)
                param.copy_(restored.to(param.device).to(param.dtype))
                print(f"Apply: {name}")
            except:
                print(f"Skip: {name}")

# 6. Generate text functions
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

# 7. Perplexity calculation function
def compute_ppl(model, prompt):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    return torch.exp(loss).item()

# 8. Prompt test
prompt = "Tell me a fun fact about the moon."

print("\n[Full-Precision Model Output]")
print(generate_text(model_fp, prompt))
print(f"PPL: {compute_ppl(model_fp, prompt):.2f}")

print("\n[Restored Model Output]")
print(generate_text(model_restored, prompt))
print(f"PPL: {compute_ppl(model_restored, prompt):.2f}")

# 9. Save the restored model
model_basename = model_name.split("/")[-1]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = f"/home/caslab/asl/Q_loss/Q_loss/retore_checkpoint/{model_name}_svd_restored_{timestamp}"
os.makedirs(save_path, exist_ok=True)

# 10. Save the restored model
model_restored.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"📦 Saved : {save_path}")


