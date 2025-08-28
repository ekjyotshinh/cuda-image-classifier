import torch
from torch.utils.cpp_extension import load
import os
from src.model import replace_with_custom

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
    if not torch.cuda.is_available():
        return model, None

    mod = load_custom()
    if mod is None: 
        return model, None

    # This function will be used to replace the original conv2
    def custom_fn(x, w, b):
        return mod.forward(x, w, b)

    model.eval()
    # Use the provided replace_with_custom function to patch the model
    model = replace_with_custom(model, custom_fn)
    return model, mod