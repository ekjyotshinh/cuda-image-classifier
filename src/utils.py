from torch.utils.cpp_extension import load
import os

def load_custom():
    this_dir = os.path.dirname(__file__)
    mod = load(
        name="custom_conv",
        sources=[os.path.join(this_dir, "kernels/custom_conv.cpp"),
                 os.path.join(this_dir, "kernels/custom_conv_kernel.cu")],
        verbose=True,
    )
    return mod

def try_patch_model_for_infer(model):
    mod = load_custom()
    # Patch model forward for demonstration (use a 5x5 mask)
    mask = torch.ones(5,5, device="cuda")
    original_forward = model.forward
    model.forward = lambda x: mod.forward(x, mask)
    return model, mod
