#include <stdio.h>
#include <stdlib.h>

// Kernel definition
__global__ void VecAdd(float *A, float *B, float *C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
    // allocate memory on the host
    const int N = 10;
    float A[N], B[N], C[N];
    // Initialize A and B, randomly
    for (int i = 0; i < N; i++)
    {
        A[i] = rand() / (float)RAND_MAX;
        B[i] = rand() / (float)RAND_MAX;
    }

    // print A and B
    printf("A: ");
    for (int i = 0; i < N; i++)
    {
        printf("%f ", A[i]);
    }
    printf("\nB: ");
    for (int i = 0; i < N; i++)
    {
        printf("%f ", B[i]);
    }
    printf("\n");

    // time the execution
    clock_t start, end;
    start = clock();
    VecAdd<<<1, N>>>(A, B, C);
    cudaDeviceSynchronize();
    end = clock();
    double time_taken = double(end - start) / CLOCKS_PER_SEC;
    printf("CUDA Time taken: %f seconds\n", time_taken);

    // use cpu to calculate the result
    start = clock();
    for (int i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];
    }
    end = clock();
    time_taken = double(end - start) / CLOCKS_PER_SEC;
    printf("CPU Time taken: %f seconds\n", time_taken);

    return 0;
}