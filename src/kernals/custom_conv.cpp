#include <torch/extension.h>

torch::Tensor conv3x3_pad1_forward(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("conv3x3_pad1_forward", &conv3x3_pad1_forward, "Custom 3x3 conv forward (CUDA)");
}
