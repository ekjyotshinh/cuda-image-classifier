import torch
from torch.utils.cpp_extension import load

def load_custom():
    if not torch.cuda.is_available():
        return None
    return load(
        name="custom_conv3x3",
        sources=[
            "src/kernels/custom_conv.cpp",
            "src/kernels/custom_conv_kernel.cu"
        ],
        verbose=True
    )
