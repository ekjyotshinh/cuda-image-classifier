from src.model import SimpleCNN, replace_with_custom

def try_patch_model_for_infer(model):
    import torch
    if not torch.cuda.is_available():
        return model, None
    from src.kernels.build import load_custom
    mod = load_custom()
    if mod is None: 
        return model, None

    def custom_fn(x, w, b):
        return mod.conv3x3_pad1_forward(x, w, b if b is not None else None)

    model.eval()
    model = replace_with_custom(model, custom_fn)
    return model, mod
