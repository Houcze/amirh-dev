#include <iostream>
#include <cmath>
#include <cuda_runtime.h>


__global__ void bias_i(double *input, double *output, int width, int height, int i)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int y_index = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y_index * width + x_index;

	if(((index / height) >= i) && ((index / height) < (width + i)))
		output[index - i * height] = input[index];
	
}


__global__ void bias_j(double *input, double *output, int width, int height, int j)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int y_index = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y_index * width + x_index;

    if(((index % height) >= j) && ((index % height) < (height + j)))
        output[index - j] = input[index];
	
}

double* bias(double* base, int size, int d1, int d2, int i, int j)
{
	/*
	* 该函数只针对2d数组进行定义，但是不检查数组形状
	*/

	double *result_i;
	double *result_j;

	cudaMalloc(&result_i, size * sizeof(double));
	cudaMalloc(&result_j, size * sizeof(double));
    cudaMemset(result_i, 0, sizeof(double) * d1 * d2);
    cudaMemset(result_j, 0, sizeof(double) * d1 * d2);

	bias_i<<<ceil(d1 * d2 / 32), 32>>>(base, result_i, d1, d2, i);
	bias_j<<<ceil(d1 * d2 / 32), 32>>>(result_i, result_j, d1, d2, j);
	
	return result_j;
}


int main(void)
{
    double* a_host;
    double* a_dev;
    int N{3};
    a_host = (double*) malloc(N * N * sizeof(double));
    cudaMalloc(&a_dev, N * N * sizeof(double));
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<N; j++)
        {
            a_host[i* N + j] = 1.;
        }
    }
    std::cout << "Before:\n";
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<N; j++)
        {
            std::cout << a_host[i * N + j] << '\t';
        }
        std::cout << '\n';
    }
    cudaMemcpy(a_dev, a_host, sizeof(double) * N * N, cudaMemcpyHostToDevice);
    a_dev = bias(a_dev, N * N, N, N, 1, 1);
    cudaMemcpy(a_host, a_dev, sizeof(double) * N * N, cudaMemcpyDeviceToHost);
    std::cout << "After:\n";
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<N; j++)
        {
            std::cout << a_host[i * N + j] << '\t';
        }
        std::cout << '\n';
    }
}