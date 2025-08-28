import torch, time
from src.model import SimpleCNN
from src.utils import try_patch_model_for_infer

@torch.inference_mode()
def run(n_warm=10, n_iters=50, bs=128, H=32, W=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(bs, 3, H, W, device=device)

    # baseline
    base = SimpleCNN().to(device).eval()
    for _ in range(n_warm): _ = base(x)
    t0 = time.time()
    for _ in range(n_iters): _ = base(x)
    base_t = (time.time() - t0) / n_iters

    # custom kernel
    patched, mod = try_patch_model_for_infer(SimpleCNN().to(device).eval())
    if mod is None:
        print(f"No CUDA. Baseline avg: {base_t*1000:.2f} ms/iter")
        return

    for _ in range(n_warm): _ = patched(x)
    t0 = time.time()
    for _ in range(n_iters): _ = patched(x)
    cust_t = (time.time() - t0) / n_iters

    print(f"Baseline avg: {base_t*1000:.2f} ms | Custom avg: {cust_t*1000:.2f} ms | Δ={(base_t-cust_t)/base_t*100:.1f}%")

if __name__ == "__main__":
    run(bs=32, H=64, W=64)
