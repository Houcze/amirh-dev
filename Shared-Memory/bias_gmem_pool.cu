#include <iostream>
#include <cmath>
#include <cuda_runtime.h>


__global__ void bias_i(double *input, double *output, int d1, int d2, int i)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int y_index = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y_index * d1 + x_index;

    if((((index / d2) + i) < d1) && (((index / d2) + i) >= 0))
    {
        output[index + i * d2] = input[index];
    }	
}


__global__ void bias_j(double *input, double *output, int d1, int d2, int j)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int y_index = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y_index * d1 + x_index;

    if((((index % d2) + j) < d2) && (((index % d2) + j) >= 0))
    {
        output[index + j] = input[index];
    }

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
	bias_i<<<ceil(double(d1 * d2) / 128), 128>>>(base, result_i, d1, d2, i);
	bias_j<<<ceil(double(d1 * d2) / 128), 128>>>(result_i, result_j, d1, d2, j);
	
	return result_j;
}


int main(void)
{
    double* a_host;
    double* a_dev;
    int N1{100};
    int N2{100};
    a_host = (double*) malloc(N1 * N2 * sizeof(double));
    cudaMalloc(&a_dev, N1 * N2 * sizeof(double));
    for(int i=0; i<N1; i++)
    {
        for(int j=0; j<N2; j++)
        {
            a_host[i* N1 + j] = 1.;
        }
    }
    /*
    std::cout << "Before:\n";
    for(int i=0; i<N1; i++)
    {
        for(int j=0; j<N2; j++)
        {
            std::cout << a_host[i * N1 + j] << '\t';
        }
        std::cout << '\n';
    }
    */
    cudaMemcpy(a_dev, a_host, sizeof(double) * N1 * N2, cudaMemcpyHostToDevice);
    a_dev = bias(a_dev, N1 * N2, N1, N2, -1, -1);
    cudaMemcpy(a_host, a_dev, sizeof(double) * N1 * N2, cudaMemcpyDeviceToHost);
    
    std::cout << "After:\n";
    for(int i=0; i<N1; i++)
    {
        for(int j=0; j<N2; j++)
        {
            std::cout << a_host[i * N1 + j] << '\t';
        }
        std::cout << '\n';
    }
    
}