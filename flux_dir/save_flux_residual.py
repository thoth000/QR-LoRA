import torch
from diffusers import FluxPipeline
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_scripts.custom_delta_r_lora import inject_delta_r_lora_layer, DeltaRLoraLayer
import argparse
from safetensors.torch import save_file

def parse_args():
    parser = argparse.ArgumentParser(description="Save residual weights for Flux model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save residual weights")
    parser.add_argument("--rank", type=int, default=64, help="Rank for LoRA")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16"])
    return parser.parse_args()

def save_residual_weights(
    pipeline: FluxPipeline,
    output_dir: str,
    rank: int = 64,
    dtype: torch.dtype = torch.float32,
) -> None:
    """save residual weights (Q&R)"""
    transformer = pipeline.transformer
    transformer = transformer.to(dtype)
    
    target_modules = [
        "attn.to_k",
        "attn.to_q", 
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_k_proj",
        "attn.add_q_proj", 
        "attn.add_v_proj",
        "attn.to_add_out",
        "ff.net.0.proj",
        "ff.net.2",
        "ff_context.net.0.proj",
        "ff_context.net.2",
    ]
    
    residual_dict = {}
    original_weights = {}
    
    for name, module in transformer.named_modules():
        module_parts = name.split(".")
        for target in target_modules:
            target_parts = target.split(".")
            if len(module_parts) >= len(target_parts):
                if all(t == m for t, m in zip(target_parts, module_parts[-len(target_parts):])):
                    if hasattr(module, "weight"):
                        print(f"Found target module: {name}")
                        original_weights[name] = module.weight.data.cpu()
    
    transformer = inject_delta_r_lora_layer(
        transformer,
        target_modules=target_modules,
        rank=rank,
        alpha=rank,
    )
    
    for name, module in transformer.named_modules():
        if isinstance(module, DeltaRLoraLayer):
            matching_keys = [
                k for k in original_weights.keys()
                if k.startswith(name) or
                (k.replace("single_transformer_blocks", "transformer_blocks") == name) or
                any(k.endswith(target) and name in k for target in target_modules)
            ]
            
            if not matching_keys:
                print(f"Warning: Could not find original weights for {name}")
                print(f"Available keys: {list(original_weights.keys())}")
                continue
            
            original_weight = original_weights[matching_keys[0]]
            q_weight = module.frozen_Q["default"].cpu()
            base_r_weight = module.lora_base["default"].cpu()
            
            w_res = original_weight - torch.mm(q_weight, base_r_weight)
            residual_dict[f"{name}.residual.weight"] = w_res
    
    os.makedirs(output_dir, exist_ok=True)
    save_file(residual_dict, os.path.join(output_dir, "flux_residual_weights.safetensors"))
    print(f"Residual weights saved to {output_dir}")

def main():
    args = parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    
    pipe = FluxPipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
    ).to(device)
    
    save_residual_weights(
        pipe,
        args.output_dir,
        args.rank,
        dtype
    )

if __name__ == "__main__":
    main() 