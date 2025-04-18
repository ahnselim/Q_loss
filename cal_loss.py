import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import bitsandbytes as bnb

fp_model_name = "your model"
quant_ckpt_path = "your model path"
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_models():
    print("Loading full-precision model...")
    model_fp = AutoModelForCausalLM.from_pretrained(
        fp_model_name,
        torch_dtype=torch.float16
    ).to(device)

    print("Loading quantized model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_enable_fp32_cpu_offload=True, 
        bnb_4bit_compute_dtype=torch.float16
    )
    model_q = AutoModelForCausalLM.from_pretrained(
        quant_ckpt_path,
        quantization_config=bnb_config,
        device_map="auto"  
    )

    return model_fp, model_q

def print_tensor(name, tensor, max_weights=5):
    print(f"  - {name:<15}: {tensor.view(-1)[:max_weights]}")

def compare_weights(model_fp, model_q, max_weights=5):
    total_mse = 0.0
    compared_layers = 0

    fp_modules = dict(model_fp.named_modules())

    print("🔎 Comparing weights between FP and Quantized model...\n")

    for name, module_q in model_q.named_modules():
        if not isinstance(module_q, bnb.nn.Linear4bit):
            continue

        module_fp = fp_modules.get(name, None)
        if module_fp is None or not hasattr(module_fp, "weight"):
            print(f"[{name}] ❌ No matching FP module.")
            continue

        try:
            fp_weight = module_fp.weight.detach().cpu().float()
            q_weight = module_q.weight.detach().cpu().float()  # already dequantized float

            if fp_weight.shape != q_weight.shape:
                print(f"[{name}] ⚠️ Shape mismatch: {fp_weight.shape} vs {q_weight.shape}")
                continue

            mse = torch.mean((fp_weight - q_weight) ** 2).item()
            print(f"[{name}] ✅ Float MSE (dequantized): {mse:.6f}")
            print_tensor("FP Weight", fp_weight, max_weights)
            print_tensor("Dequantized Weight", q_weight, max_weights)

            total_mse += mse
            compared_layers += 1

        except Exception as e:
            print(f"[{name}] ❌ Error during comparison: {e}")

    avg_mse = total_mse / compared_layers if compared_layers > 0 else 0.0
    print(f"\n📊 평균 MSE Loss (available layers): {avg_mse:.6f} across {compared_layers} layers.")


if __name__ == "__main__":
    model_fp, model_q = load_models()
    compare_weights(model_fp, model_q)
