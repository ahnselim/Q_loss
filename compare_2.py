import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from copy import deepcopy

def absmax_quantize(X):
    # Calculate scale
    scale = 127 / torch.max(torch.abs(X))

    # Quantize
    X_quant = (scale * X).round()

    # Dequantize
    X_dequant = X_quant / scale

    return X_quant.to(torch.int8), X_dequant

torch.manual_seed(0)
device = torch.device(f'cuda:{1}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) 

# Load model and tokenizer
model_id = "your model"
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

weights = model.model.layers[0].self_attn.q_proj.weight

# print("Original weights:")
# print(weights)
# print(weights.size())

# Quantize layer using absmax quantization
weights_abs_quant, _ = absmax_quantize(weights)
# print("\nAbsmax quantized weights:")
# print(weights_abs_quant)
# print(weights_abs_quant.size())

#W≈Wq​+LowRank(ΔW)

def low_rank_restore(fp_weight, quant_weight, rank=8):
    # Step 1: Compute error
    delta = weights - weights_abs_quant # ΔW
    
    # Step 2: SVD
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    
    # Step 3: Low-rank approximation
    U_r = U[:, :rank]
    S_r = torch.diag(S[:rank])
    V_r = Vh[:rank, :]
    delta_approx = U_r @ S_r @ V_r
    
    # Step 4: Add to quantized weight
    W_approx = weights_abs_quant + delta_approx
    return W_approx, delta_approx

# ---------- Step 1: 원본 weight 저장 ----------
weights_fp = [param.data.cpu().clone() for param in model.parameters()]

# ---------- Step 2: 양자화된 모델 생성 ----------
model_abs = deepcopy(model)
weights_abs = []

for param in model_abs.parameters():
    _, dequant = absmax_quantize(param.data.cpu())
    param.data.copy_(dequant.to(dtype=param.data.dtype))
    weights_abs.append(dequant.clone())

# ---------- Step 3: 복원 모델 생성 ----------
model_restored = deepcopy(model)

for i, param in enumerate(model_restored.parameters()):
    fp = weights_fp[i]
    quant = weights_abs[i]

    # 복원 수행
    restored, _ = low_rank_restore(fp, quant, rank=8)

    # 복원 weight를 모델에 반영
    param.data.copy_(restored.to(dtype=param.data.dtype))


def generate_text(model, input_text, max_length=50):
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    output = model.generate(inputs=input_ids,
                            max_length=max_length,
                            do_sample=True,
                            top_k=30,
                            pad_token_id=tokenizer.eos_token_id,
                            attention_mask=input_ids.new_ones(input_ids.shape))
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Generate text with original and quantized models
original_text = generate_text(model, "I have a dream")
absmax_text   = generate_text(model_abs, "I have a dream")
delta_text       = generate_text(model_restored, "I have a dream")

print(f"Original model:\n{original_text}")
print("-" * 50)
print(f"Absmax model:\n{absmax_text}")
print("-" * 50)
print(f"Restored model:\n{delta_text}")
print("-" * 50)