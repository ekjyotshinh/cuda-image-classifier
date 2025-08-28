#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void conv3x3_pad1_kernel(
    const float *__restrict__ input,
    const float *__restrict__ weight,
    const float *__restrict__ bias,
    float *__restrict__ output,
    int N, int C, int H, int W, int O)
{
    int n = blockIdx.z;
    int o = blockIdx.y;
    int hw = blockIdx.x * blockDim.x + threadIdx.x;
    if (hw >= H * W)
        return;

    int h = hw / W;
    int w = hw % W;

    float acc = bias ? bias[o] : 0.f;

    for (int c = 0; c < C; ++c)
    {
        for (int kh = 0; kh < 3; ++kh)
        {
            int ih = h + kh - 1;
            if (ih < 0 || ih >= H)
                continue;
            for (int kw = 0; kw < 3; ++kw)
            {
                int iw = w + kw - 1;
                if (iw < 0 || iw >= W)
                    continue;
                float inp_val = input[((n * C + c) * H + ih) * W + iw];
                float w_val = weight[(((o * C + c) * 3 + kh) * 3 + kw)];
                acc += inp_val * w_val;
            }
        }
    }

    output[((n * O + o) * H + h) * W + w] = acc;
}

torch::Tensor conv3x3_pad1_forward(
    torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias)
{
    TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "Tensors must be CUDA");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int O = weight.size(0);

    auto output = torch::empty({N, O, H, W}, input.options());

    int threads = 256;
    int blocks = (H * W + threads - 1) / threads;
    dim3 grid(blocks, O, N);

    conv3x3_pad1_kernel<<<grid, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C, H, W, O);

    return output;
}
