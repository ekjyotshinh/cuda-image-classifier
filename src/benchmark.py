import torch, time
from src.model import SimpleCNN
from src.utils import try_patch_model_for_infer

@torch.inference_mode()
def run(n_warm=10, n_iters=50, batch_sizes=(16, 32, 64, 128), resolutions=(32, 64, 128)):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}\n")
    
    for H in resolutions:
        for bs in batch_sizes:
            x = torch.randn(bs, 3, H, H, device=device)

            # baseline output & warmup
            baseline_model = SimpleCNN().to(device).eval()
            for _ in range(n_warm): _ = baseline_model(x)
            # Add synchronization for accurate timing on CUDA
            if device == "cuda": torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n_iters): _ = baseline_model(x)
            if device == "cuda": torch.cuda.synchronize()
            y_baseline = baseline_model(x)
            base_t = (time.time() - t0) / n_iters

            # custom kernel output & warmup
            patched_model, mod = try_patch_model_for_infer(SimpleCNN().to(device).eval())
            if mod is None:
                print(f"Batch {bs}, Res {H}x{H} -> No CUDA. Baseline avg: {base_t*1000:.2f} ms/iter")
                continue

            for _ in range(n_warm): _ = patched_model(x)
            # Add synchronization for accurate timing on CUDA
            if device == "cuda": torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n_iters): _ = patched_model(x)
            if device == "cuda": torch.cuda.synchronize()
            y_custom = patched_model(x)
            cust_t = (time.time() - t0) / n_iters

            # correctness check
            if y_baseline.shape != y_custom.shape:
                correctness_msg = "❌ Shape mismatch"
                max_diff = None
            elif torch.allclose(y_baseline, y_custom, rtol=1e-3, atol=1e-5):
                correctness_msg = "✅ Correct"
                max_diff = (y_baseline - y_custom).abs().max().item()
            else:
                max_diff = (y_baseline - y_custom).abs().max().item()
                correctness_msg = f"❌ Max diff: {max_diff:.5f}"

            print(f"Batch {bs}, Res {H}x{H} -> "
                  f"Baseline: {base_t*1000:.2f} ms | "
                  f"Custom: {cust_t*1000:.2f} ms | "
                  f"Δ={(base_t-cust_t)/base_t*100:.1f}% | {correctness_msg}")


if __name__ == "__main__":
    run()
