#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath> // for std::min / std::max

#define MASK_WIDTH 5
#define MASK_RADIUS (MASK_WIDTH / 2)
#define TILE_WIDTH 16
#define SHARED_MEMORY_WIDTH (TILE_WIDTH + MASK_WIDTH - 1)
#define CLAMP(x) (fminf(fmaxf((x), 0.0f), 1.0f))

__global__ void convolution_kernel(
    const float *input, const float *mask, float *output,
    int channels, int width, int height)
{
    __shared__ float smem[SHARED_MEMORY_WIDTH][SHARED_MEMORY_WIDTH];

    int c = blockIdx.z; // channel
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * TILE_WIDTH;
    int by = blockIdx.y * TILE_WIDTH;

    int x = bx + tx;
    int y = by + ty;

    // load into shared memory
    int smem_x = tx;
    int smem_y = ty;
    int ix = x - MASK_RADIUS;
    int iy = y - MASK_RADIUS;

    if (ix >= 0 && ix < width && iy >= 0 && iy < height)
        smem[smem_y][smem_x] = input[(iy * width + ix) * channels + c];
    else
        smem[smem_y][smem_x] = 0.0f;

    __syncthreads();

    float acc = 0.0f;
    for (int i = 0; i < MASK_WIDTH; i++)
        for (int j = 0; j < MASK_WIDTH; j++)
            acc += smem[ty + i][tx + j] * mask[i * MASK_WIDTH + j];

    if (x < width && y < height)
        output[(y * width + x) * channels + c] = CLAMP(acc);
}

void launch_convolution(const float *input, const float *mask, float *output,
                        int channels, int width, int height)
{
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_WIDTH - 1) / TILE_WIDTH, channels);

    convolution_kernel<<<grid, block>>>(input, mask, output, channels, width, height);
}
