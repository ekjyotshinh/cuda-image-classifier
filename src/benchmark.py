import torch, time
from src.model import SimpleCNN
from src.utils import try_patch_model_for_infer

@torch.inference_mode()
def run(
    batch_sizes=(16, 32, 64, 128),
    resolutions=((32, 32), (64, 64), (128, 128)),
    n_warm=10,
    n_iters=50,
    n_trials=3
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}\n")

    for H, W in resolutions:
        for bs in batch_sizes:
            x = torch.randn(bs, 3, H, W, device=device)

            # baseline
            base = SimpleCNN().to(device).eval()
            for _ in range(n_warm): _ = base(x)
            times = []
            for _ in range(n_trials):
                t0 = time.time()
                for _ in range(n_iters): _ = base(x)
                times.append((time.time() - t0) / n_iters)
            base_t = sum(times) / len(times)

            # custom kernel
            patched, mod = try_patch_model_for_infer(SimpleCNN().to(device).eval())
            if mod is None:
                print(f"No CUDA. Baseline avg: {base_t*1000:.2f} ms/iter")
                continue

            for _ in range(n_warm): _ = patched(x)
            times = []
            for _ in range(n_trials):
                t0 = time.time()
                for _ in range(n_iters): _ = patched(x)
                times.append((time.time() - t0) / n_iters)
            cust_t = sum(times) / len(times)

            delta = (base_t - cust_t) / base_t * 100
            print(f"Batch {bs}, Res {H}x{W} -> Baseline: {base_t*1000:.2f} ms | Custom: {cust_t*1000:.2f} ms | Δ={delta:.1f}%")

if __name__ == "__main__":
    run()
