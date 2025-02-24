#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(int *A, int *B, int *C, int N)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x; // Unique thread index in grid
    if (index < N)
    {                                   // Prevent out-of-bounds memory access
        C[index] = A[index] + B[index]; // Element-wise addition
    }
}

int main()
{
    // reset device
    cudaDeviceReset();
    const int N = 4096;       // Number of elements
    int size = N * sizeof(int); // Size in bytes

    // Allocate memory on host (CPU)
    int h_A[N], h_B[N], h_C[N];

    // Initialize vectors with values
    for (int i = 0; i < N; i++)
    {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // Allocate memory on device (GPU)
    int *d_A, *d_B, *d_C;
    // testing: malloc these memory at CPU
    // d_A = (int *)malloc(size);
    // d_B = (int *)malloc(size);
    // d_C = (int *)malloc(size);

    if (cudaMalloc((void **)&d_A, size) != cudaSuccess)
    {
        std::cerr << "Error allocating memory for d_A" << std::endl;
        return -1;
    }
    if (cudaMalloc((void **)&d_B, size) != cudaSuccess)
    {
        std::cerr << "Error allocating memory for d_B" << std::endl;
        return -1;
    }
    if (cudaMalloc((void **)&d_C, size) != cudaSuccess)
    {
        std::cerr << "Error allocating memory for d_C" << std::endl;
        return -1;
    }
    // Copy input data from host to device
    if (cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::cerr << "Error copying data from host to device for d_A" << std::endl;
        return -1;
    }
    if (cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::cerr << "Error copying data from host to device for d_B" << std::endl;
        return -1;
    }

    // Define execution configuration
    int threadsPerBlock = 64;                                    // Number of threads per block
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock; // Compute number of blocks

    std::cout << "Number of blocks: " << numBlocks << ", Threads per block: " << threadsPerBlock << std::endl;

    // Launch kernel
    clock_t start, end;
    start = clock();
    vectorAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    if (cudaGetLastError() != cudaSuccess)
    {
        std::cerr << "Error launching kernel" << std::endl;
        return -1;
    }
    end = clock();
    double time_taken = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "CUDA Time taken: " << time_taken << " seconds" << std::endl;

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print results
    // std::cout << "Vector Addition Results: " << std::endl;
    // for (int i = 0; i < N; i++)
    // {
    //     std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;
    // }

    // Free device memory
    if (cudaFree(d_A) != cudaSuccess)
    {
        std::cerr << "Error freeing memory for d_A" << std::endl;
        return -1;
    }
    if (cudaFree(d_B) != cudaSuccess)
    {
        std::cerr << "Error freeing memory for d_B" << std::endl;
        return -1;
    }
    if (cudaFree(d_C) != cudaSuccess)
    {
        std::cerr << "Error freeing memory for d_C" << std::endl;
        return -1;
    }
    // Free host memory
    // free(d_A);
    // free(d_B);
    // free(d_C);

    return 0;
}
