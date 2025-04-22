import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import tensorly as tl
from tensorly.decomposition import tucker
from copy import deepcopy
from datetime import datetime
import os

# Set backend for tensorly
tl.set_backend("pytorch")

# Model settings
model_name= "your_model_name"  # Replace with your model name
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_fp = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
model_fp.eval()
model_restored = deepcopy(model_fp)

# Quantization function (4-bit)
def absmax_quantize(weight):
    max_val = weight.abs().max()
    scale = max_val / 7  # for int4 range (-8 to 7)
    quant = torch.round(weight / scale).clamp(-8, 7)
    dequant = quant * scale
    return dequant, scale

# Tucker low-rank restoration
def low_rank_restore_tucker(fp_weight, quant_weight, ranks=(64, 64)):
    delta = fp_weight - quant_weight
    core, factors = tucker(delta, ranks=ranks)
    delta_approx = tl.tucker_to_tensor((core, factors))
    return quant_weight + delta_approx

# Apply Tucker to Q/V projections only if shape matches
print("\n🔧 Restoring Q/V projections using Tucker...")
with torch.no_grad():
    for i, layer in enumerate(model_restored.model.layers):
        try:
            q_weight = layer.self_attn.q_proj.weight.data.detach().cpu().float()
            v_weight = layer.self_attn.v_proj.weight.data.detach().cpu().float()

            if q_weight.shape != v_weight.shape:
                print(f"❌ Layer {i} skip: shape mismatch: {q_weight.shape} vs {v_weight.shape}")
                continue

            # Stack Q and V: shape [2, H, D]
            qv_concat = torch.stack([q_weight, v_weight], dim=0)
            qv_quant, _ = absmax_quantize(qv_concat)
            qv_restored = low_rank_restore_tucker(qv_concat, qv_quant, ranks=(2, 128, 128))

            # Restore back
            layer.self_attn.q_proj.weight.data.copy_(qv_restored[0].to(layer.self_attn.q_proj.weight.dtype).to(device))
            layer.self_attn.v_proj.weight.data.copy_(qv_restored[1].to(layer.self_attn.v_proj.weight.dtype).to(device))

            print(f"✅ Layer {i} Tucker Restored (Q/V)")

        except Exception as e:
            print(f"❌ Layer {i} skip: {e}")

# Save model
model_basename = model_name.split("/")[-1]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = f"/home/caslab/asl/Q_loss/Q_loss/retore_checkpoint/{model_basename}_tucker_qv_restored_{timestamp}"
os.makedirs(save_path, exist_ok=True)
model_restored.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"\n📦 Model saved at {save_path}")