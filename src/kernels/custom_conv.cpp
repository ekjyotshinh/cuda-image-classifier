#include <torch/extension.h>

// Declare kernel function from .cu
void launch_convolution(const float *input, const float *mask, float *output,
                        int channels, int width, int height);

torch::Tensor custom_conv(torch::Tensor input, torch::Tensor mask)
{
    auto output = torch::zeros_like(input);
    launch_convolution(input.data_ptr<float>(),
                       mask.data_ptr<float>(),
                       output.data_ptr<float>(),
                       input.size(1), // channels
                       input.size(2), // width
                       input.size(3)  // height
    );
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("custom_conv", &custom_conv, "Custom CUDA convolution");
}
