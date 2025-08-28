#include <cuda_runtime.h>

#define TILE_WIDTH 16
#define KERNEL_WIDTH 3
#define BLOCK_WIDTH (TILE_WIDTH + KERNEL_WIDTH - 1)

// The tiled convolution kernel
__global__ void conv3x3_tiled_kernel(
    const float *__restrict__ input,
    const float *__restrict__ weight,
    const float *__restrict__ bias,
    float *__restrict__ output,
    int N, int C, int H, int W, int O)
{
    __shared__ float s_data[BLOCK_WIDTH][BLOCK_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int n = blockIdx.z / O;
    int o = blockIdx.z % O;

    int out_tile_x_start = blockIdx.x * TILE_WIDTH;
    int out_tile_y_start = blockIdx.y * TILE_WIDTH;

    int in_tile_x_start = out_tile_x_start - KERNEL_WIDTH / 2;
    int in_tile_y_start = out_tile_y_start - KERNEL_WIDTH / 2;

    float acc = bias ? bias[o] : 0.0f;

    for (int c = 0; c < C; ++c) {
        int in_x = in_tile_x_start + tx;
        int in_y = in_tile_y_start + ty;

        if (in_x >= 0 && in_x < W && in_y >= 0 && in_y < H) {
            s_data[ty][tx] = input[((n * C + c) * H + in_y) * W + in_x];
        } else {
            s_data[ty][tx] = 0.0f;
        }
        __syncthreads();

        if (ty < TILE_WIDTH && tx < TILE_WIDTH) {
            for (int kh = 0; kh < KERNEL_WIDTH; kh++) {
                for (int kw = 0; kw < KERNEL_WIDTH; kw++) {
                    acc += s_data[ty + kh][tx + kw] * 
                           weight[(((o * C + c) * KERNEL_WIDTH + kh) * KERNEL_WIDTH + kw)];
                }
            }
        }
        __syncthreads();
    }

    if (ty < TILE_WIDTH && tx < TILE_WIDTH) {
        int out_x = out_tile_x_start + tx;
        int out_y = out_tile_y_start + ty;
        if (out_x < W && out_y < H) {
            output[((n * O + o) * H + out_y) * W + out_x] = acc;
        }
    }
}

// Launcher function that will be called from the C++ file
extern "C" void launch_conv3x3_tiled_kernel(
    const float *input,
    const float *weight,
    const float *bias,
    float *output,
    int N, int C, int H, int W, int O)
{
    dim3 block(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 grid((W + TILE_WIDTH - 1) / TILE_WIDTH, (H + TILE_WIDTH - 1) / TILE_WIDTH, N * O);

    conv3x3_tiled_kernel<<<grid, block>>>(
        input, weight, bias, output, N, C, H, W, O);
}