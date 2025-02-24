#include <iostream>

__global__ void kernel3D()
{
    int x = threadIdx.x;
    int y = threadIdx.y;
    int z = threadIdx.z;

    printf("Hello from thread (%d, %d, %d)\n", x, y, z);
}

int main()
{
    dim3 threadsPerBlock(4, 4, 4); // 4x4x4 = 64 threads per block

    // Launch kernel with 1 block containing a 3D thread layout
    kernel3D<<<1, threadsPerBlock>>>();

    cudaDeviceSynchronize(); // Wait for GPU execution to complete
    return 0;
}
