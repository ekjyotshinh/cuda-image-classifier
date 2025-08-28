#include <torch/extension.h>

// Forward declaration of the CUDA kernel
void conv3x3_pad1_kernel(
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

    int N = input.size(0); // Batch size
    int C = input.size(1); // Input channels
    int H = input.size(2); // Height
    int W = input.size(3); // Width
    int O = weight.size(0); // Output channels

    // Create the output tensor with the correct shape
    auto output = torch::empty({N, O, H, W}, input.options());

    // Set up grid and block dimensions for the CUDA kernel
    int threads = 256;
    int blocks = (H * W + threads - 1) / threads;
    dim3 grid(blocks, O, N);

    // Launch the kernel
    conv3x3_pad1_kernel<<<grid, threads>>>(
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