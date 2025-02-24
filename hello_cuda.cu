#include <iostream>

__global__ void helloCUDA()
{
    int threadIdX = threadIdx.x;
    int threadIdY = threadIdx.y;
    int threadIdZ = threadIdx.z;
    int threadId = threadIdX + threadIdY * blockDim.x + threadIdZ * blockDim.x * blockDim.y;

    int blockIdX = blockIdx.x;
    int blockIdY = blockIdx.y;
    int blockIdZ = blockIdx.z;
    int blockDimX = blockDim.x;
    int blockDimY = blockDim.y;
    int blockDimZ = blockDim.z;
    int blockId = blockIdX + blockIdY * gridDim.x + blockIdZ * gridDim.x * gridDim.y;
    printf("Hello from CUDA kernel! with threadIdX: %d, threadIdY: %d, threadIdZ: %d, threadId: %d, with blockIdX: %d, blockIdY: %d, blockIdZ: %d, blockId: %d\n", threadIdX, threadIdY, threadIdZ, threadId, blockIdX, blockIdY, blockIdZ, blockId);
}

int main()
{
    // simulating a 2D matrix of NxN
    int N = 8;
    dim3 threadPerBlock(4, 4);
    dim3 numBlocks((N + threadPerBlock.x - 1) / threadPerBlock.x, (N + threadPerBlock.y - 1) / threadPerBlock.y);
    helloCUDA<<<numBlocks, threadPerBlock>>>(); // Launch kernel with 1 block, 1 thread
    cudaDeviceSynchronize();                    // Ensure kernel execution completes
    return 0;
}
