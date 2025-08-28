#include <torch/extension.h>

// Declare kernel function from .cu
void launch_convolution(const float *input, const float *mask, float *output,
                        int channels, int width, int height);

torch::Tensor custom_conv(torch::Tensor input, torch::Tensor mask)
{
    // The output shape should have the number of channels from the mask (weight)
    auto out_channels = mask.size(0);
    auto output = torch::zeros({input.size(0), out_channels, input.size(2), input.size(3)}, input.options());

    launch_convolution(input.data_ptr<float>(),
                       mask.data_ptr<float>(),
                       output.data_ptr<float>(),
                       input.size(1), // in_channels
                       input.size(2), // width
                       input.size(3)  // height
    );
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &custom_conv, "Custom convolution forward");
}
