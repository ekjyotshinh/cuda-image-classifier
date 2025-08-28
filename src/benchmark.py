import torch
import time
from src.model import SimpleCNN
from src.utils import try_patch_model_for_infer

@torch.inference_mode()
def benchmark(batch_sizes=[32, 128, 512, 1024], image_sizes=[32, 64, 128], n_warm=10, n_iters=50):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Print table header
    print(f"{'Batch':>6} | {'H/W':>5} | {'Baseline (ms)':>12} | {'Custom (ms)':>11} | {'Δ (%)':>6}")
    print("-"*52)
    
    for bs in batch_sizes:
        for H in image_sizes:
            W = H
            x = torch.randn(bs, 3, H, W, device=device)

            # Baseline model
            base = SimpleCNN().to(device).eval()
            for _ in range(n_warm): _ = base(x)
            torch.cuda.synchronize() if device=='cuda' else None
            t0 = time.time()
            for _ in range(n_iters): _ = base(x)
            torch.cuda.synchronize() if device=='cuda' else None
            base_t = (time.time()-t0)/n_iters

            # Custom CUDA kernel
            patched, mod = try_patch_model_for_infer(SimpleCNN().to(device).eval())
            if mod is None:
                print("CUDA kernel not available, skipping custom benchmark.")
                continue

            for _ in range(n_warm): _ = patched(x)
            torch.cuda.synchronize() if device=='cuda' else None
            t0 = time.time()
            for _ in range(n_iters): _ = patched(x)
            torch.cuda.synchronize() if device=='cuda' else None
            cust_t = (time.time()-t0)/n_iters

            delta = (base_t - cust_t)/base_t*100
            print(f"{bs:6d} | {H:5d} | {base_t*1000:12.2f} | {cust_t*1000:11.2f} | {delta:6.1f}")

if __name__ == "__main__":
    benchmark()
