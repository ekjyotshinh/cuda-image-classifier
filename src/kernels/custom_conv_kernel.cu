#include <cuda_runtime.h>

#define TILE_WIDTH 16
#define KERNEL_WIDTH 3

// Final, optimized tiled 3x3 convolution kernel
__global__ void conv3x3_tiled_kernel(
    const float *__restrict__ input,
    const float *__restrict__ weight,
    const float *__restrict__ bias,
    float *__restrict__ output,
    int N, int C, int H, int W, int O)
{
    // Shared memory for a tile of the input data
    __shared__ float s_data[TILE_WIDTH][TILE_WIDTH + KERNEL_WIDTH - 1];

    // Identify the specific output pixel this thread will compute
    int n = blockIdx.z / O;
    int o = blockIdx.z % O;
    int out_tile_x = blockIdx.x * TILE_WIDTH;
    int out_tile_y = blockIdx.y * TILE_WIDTH;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // This thread's output pixel coordinates
    int out_x = out_tile_x + tx;
    int out_y = out_tile_y + ty;

    // Accumulator for the output pixel
    float acc = bias ? bias[o] : 0.0f;

    // Loop over all input channels
    for (int c = 0; c < C; c++) {
        // Load a tile of the input data into shared memory
        int in_x = out_tile_x + tx - KERNEL_WIDTH / 2;
        int in_y = out_tile_y + ty - KERNEL_WIDTH / 2;

        if (in_x >= 0 && in_x < W && in_y >= 0 && in_y < H) {
            s_data[ty][tx] = input[((n * C + c) * H + in_y) * W + in_x];
        } else {
            s_data[ty][tx] = 0.0f;
        }
        __syncthreads();

        // Perform the convolution using the data in shared memory
        if (out_x < W && out_y < H) {
            for (int kh = 0; kh < KERNEL_WIDTH; kh++) {
                for (int kw = 0; kw < KERNEL_WIDTH; kw++) {
                    acc += s_data[ty][tx - KERNEL_WIDTH / 2 + kw] * 
                           weight[(((o * C + c) * KERNEL_WIDTH + kh) * KERNEL_WIDTH + kw)];
                }
            }
        }
        __syncthreads();
    }

    // Write the final result to global memory
    if (out_x < W && out_y < H) {
        output[((n * O + o) * H + out_y) * W + out_x] = acc;
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
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((W + TILE_WIDTH - 1) / TILE_WIDTH, (H + TILE_WIDTH - 1) / TILE_WIDTH, N * O);

    conv3x3_tiled_kernel<<<grid, block>>>(
        input, weight, bias, output, N, C, H, W, O);
}