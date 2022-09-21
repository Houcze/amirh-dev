#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include "func.h"

int main(void)
{
    double* host_a;
    double* host_b;
    double* result;
    double* a;
    double* b;
    double* deviceResult;
    int N1{100};
    int N2{100};
    host_a = (double*) malloc(N1 * N2 * sizeof(double));
    host_b = (double*) malloc(N1 * N2 * sizeof(double));
    result = (double*) malloc(N1 * N2 * sizeof(double));

    cudaMalloc(&a, sizeof(double) * N1 * N2);
    cudaMalloc(&b, sizeof(double) * N1 * N2);
    cudaMalloc(&deviceResult, sizeof(double) * N1 * N2);

    for(int i=0; i<N1; i++)
    {
        for(int j=0; j<N2; j++)
        {
            host_a[N1 * i + j] = 4.5;
            host_b[N1 * i + j] = 1.;
        }
    }

    cudaMemcpy(a, host_a, sizeof(double) * N1 * N2, cudaMemcpyHostToDevice);
    cudaMemcpy(b, host_b, sizeof(double) * N1 * N2, cudaMemcpyHostToDevice);
    Add(a, b, deviceResult, N1, N2);
    cudaMemcpy(result, deviceResult, sizeof(double) * N1 * N2, cudaMemcpyDeviceToHost);

    for(int i=0; i<N1; i++)
    {
        for(int j=0; j<N2; j++)
        {
            std::cout << result[i * N1 + j] << '\t';
        }
        std::cout << '\n';
    }

}