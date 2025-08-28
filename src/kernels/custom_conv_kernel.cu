#include <cuda_runtime.h>

// The CUDA kernel for 3x3 convolution
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
    if (hw >= H * W) return;

    int h = hw / W;
    int w = hw % W;

    float acc = bias ? bias[o] : 0.0f;

    for (int c = 0; c < C; ++c)
    {
        for (int kh = 0; kh < 3; ++kh)
        {
            for (int kw = 0; kw < 3; ++kw)
            {
                int ih = h + kh - 1;
                int iw = w + kw - 1;

                if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                {
                    float inp_val = input[((n * C + c) * H + ih) * W + iw];
                    float w_val = weight[(((o * C + c) * 3 + kh) * 3 + kw)];
                    acc += inp_val * w_val;
                }
            }
        }
    }

    output[((n * O + o) * H + h) * W + w] = acc;
}

// Launcher function that will be called from the C++ file
extern "C" void launch_conv3x3_pad1_kernel(
    const float *input,
    const float *weight,
    const float *bias,
    float *output,
    int N, int C, int H, int W, int O)
{
    int threads = 256;
    int blocks = (H * W + threads - 1) / threads;
    dim3 grid(blocks, O, N);

    conv3x3_pad1_kernel<<<grid, threads>>>(
        input, weight, bias, output, N, C, H, W, O);
}