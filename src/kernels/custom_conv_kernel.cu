#include <cuda_runtime.h>

// Correctly implements a standard 2D convolution for a 3x3 kernel with padding=1
__global__ void conv3x3_pad1_kernel(
    const float *__restrict__ input,
    const float *__restrict__ weight,
    const float *__restrict__ bias,
    float *__restrict__ output,
    int N, int C, int H, int W, int O)
{
    // Calculate output pixel coordinates
    int n = blockIdx.z;
    int o = blockIdx.y;
    int hw = blockIdx.x * blockDim.x + threadIdx.x;
    if (hw >= H * W) return;

    int h = hw / W;
    int w = hw % W;

    // Accumulator for the output pixel
    float acc = bias ? bias[o] : 0.0f;

    // Loop over all input channels
    for (int c = 0; c < C; ++c)
    {
        // Loop over the 3x3 kernel
        for (int kh = 0; kh < 3; ++kh)
        {
            for (int kw = 0; kw < 3; ++kw)
            {
                int ih = h + kh - 1; // Input height with padding
                int iw = w + kw - 1; // Input width with padding

                // Boundary check (for padding)
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