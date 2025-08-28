#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define MASK_WIDTH 5
#define MASK_RADIUS MASK_WIDTH / 2
#define TILE_WIDTH 16
#define SHARED_MEMORY_WIDTH (TILE_WIDTH + MASK_WIDTH - 1)
#define CLAMP(x) (fmin(fmax((x), 0.0f), 1.0f))

__global__ void convolution(float *inputImage, const float *__restrict__ convolutionMask, float *outputImage,
                            int numChannels, int imageWidth, int imageHeight)
{
    __shared__ float sharedMemory[SHARED_MEMORY_WIDTH][SHARED_MEMORY_WIDTH];
    int channel;

    for (channel = 0; channel < numChannels; channel++)
    {
        // First batch loading
        int destIndex = threadIdx.y * TILE_WIDTH + threadIdx.x;
        int destY = destIndex / SHARED_MEMORY_WIDTH, destX = destIndex % SHARED_MEMORY_WIDTH;
        int srcY = blockIdx.y * TILE_WIDTH + destY - MASK_RADIUS;
        int srcX = blockIdx.x * TILE_WIDTH + destX - MASK_RADIUS;
        int srcIndex = (srcY * imageWidth + srcX) * numChannels + channel;

        if (srcY >= 0 && srcY < imageHeight && srcX >= 0 && srcX < imageWidth)
        {
            sharedMemory[destY][destX] = inputImage[srcIndex];
        }
        else
        {
            sharedMemory[destY][destX] = 0;
        }

        // Second batch loading
        destIndex = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
        destY = destIndex / SHARED_MEMORY_WIDTH;
        destX = destIndex % SHARED_MEMORY_WIDTH;
        srcY = blockIdx.y * TILE_WIDTH + destY - MASK_RADIUS;
        srcX = blockIdx.x * TILE_WIDTH + destX - MASK_RADIUS;
        srcIndex = (srcY * imageWidth + srcX) * numChannels + channel;

        if (destY < SHARED_MEMORY_WIDTH)
        {
            if (srcY >= 0 && srcY < imageHeight && srcX >= 0 && srcX < imageWidth)
            {
                sharedMemory[destY][destX] = inputImage[srcIndex];
            }
            else
            {
                sharedMemory[destY][destX] = 0;
            }
        }
        __syncthreads(); // Wait for all threads to finish loading

        float accumulatedValue = 0;
        int y, x;
        // Perform convolution
        for (y = 0; y < MASK_WIDTH; y++)
        {
            for (x = 0; x < MASK_WIDTH; x++)
            {
                accumulatedValue += sharedMemory[threadIdx.y + y][threadIdx.x + x] * convolutionMask[y * MASK_WIDTH + x];
            }
        }

        // Compute output pixel location
        int outputY = blockIdx.y * TILE_WIDTH + threadIdx.y;
        int outputX = blockIdx.x * TILE_WIDTH + threadIdx.x;

        if (outputY < imageHeight && outputX < imageWidth)
        {
            outputImage[(outputY * imageWidth + outputX) * numChannels + channel] = CLAMP(accumulatedValue);
        }
        __syncthreads(); // Synchronize threads
    }
}
