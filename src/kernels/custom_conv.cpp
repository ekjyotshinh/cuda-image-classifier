#include <torch/extension.h>
#include "custom_conv_kernel.cu"

torch::Tensor custom_conv(torch::Tensor input, torch::Tensor mask)
{
    auto output = torch::zeros_like(input);

    dim3 grid(ceil(input.size(3) / 16.0), ceil(input.size(2) / 16.0));
    dim3 block(16, 16);

    convolution<<<grid, block>>>(
        input.data_ptr<float>(),
        mask.data_ptr<float>(),
        output.data_ptr<float>(),
        input.size(1), // numChannels
        input.size(3), // width
        input.size(2)  // height
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &custom_conv, "Custom convolution forward (CUDA)");
}
