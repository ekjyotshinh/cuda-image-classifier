#include <torch/extension.h>

// Forward declaration of the CUDA kernel
void conv3x3_tiled_kernel(
    const float *__restrict__ input,
    const float *__restrict__ weight,
    const float *__restrict__ bias,
    float *__restrict__ output,
    int N, int C, int H, int W, int O);

// C++ wrapper function that will be called from Python
torch::Tensor custom_conv_forward(
    torch::Tensor input, 
    torch::Tensor weight, 
    c10::optional<torch::Tensor> bias)
{
    TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "Tensors must be on CUDA");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int O = weight.size(0);

    auto output = torch::empty({N, O, H, W}, input.options());

    // Set up grid and block dimensions for the tiled kernel
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16, N * O);

    // Launch the kernel
    conv3x3_tiled_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C, H, W, O);

    return output;
}

// Bind the C++ function to the Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &custom_conv_forward, "Custom 3x3 convolution forward");
}