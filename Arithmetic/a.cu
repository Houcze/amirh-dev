#include <cuda_runtime.h>
#include <iostream>


__global__ void add_kernel(double* x, double* y, double* result, int width, int height)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int y_index = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y_index * width + x_index;

    if(index < width * height)
        result[index] = x[index] + y[index];
}


int add_host(double* x, double* y, double* result, int d1, int d2)
{
    add_kernel<<<ceil(double(d1 * d2) / 128), 128, d1 * d2 * sizeof(double)>>>(x, y, result, d1, d2);
    // std::cout << __func__ << std::endl;
	return EXIT_SUCCESS;
}


__global__ void sub_kernel(double* x, double* y, double* result, int width, int height)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int y_index = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y_index * width + x_index;

    if(index < width * height)
        result[index] = x[index] - y[index];
}


int sub_host(double* x, double* y, double* result, int d1, int d2)
{
    sub_kernel<<<ceil(double(d1 * d2) / 128), 128, d1 * d2 * sizeof(double)>>>(x, y, result, d1, d2);
    // std::cout << __func__ << std::endl;
	return EXIT_SUCCESS;
}


__global__ void mul_kernel(double* x, double* y, double* result, int width, int height)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int y_index = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y_index * width + x_index;

    if(index < width * height)
        result[index] = x[index] * y[index];
}


int mul_host(double* x, double* y, double* result, int d1, int d2)
{
    mul_kernel<<<ceil(double(d1 * d2) / 128), 128, d1 * d2 * sizeof(double)>>>(x, y, result, d1, d2);
    // std::cout << __func__ << std::endl;
	return EXIT_SUCCESS;
}


__global__ void div_kernel(double* x, double* y, double* result, int width, int height)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int y_index = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y_index * width + x_index;

    if(index < width * height)
        result[index] = x[index] / y[index];
}


int div_host(double* x, double* y, double* result, int d1, int d2)
{
    div_kernel<<<ceil(double(d1 * d2) / 128), 128, d1 * d2 * sizeof(double)>>>(x, y, result, d1, d2);
    // std::cout << __func__ << std::endl;
	return EXIT_SUCCESS;
}


int main(void)
{
    /*------------------------------------- INIT -----------------------------------------------------------*/
    double* result_host;
    double* result_dev;
    int N1{10};
    int N2{10};
    result_host = (double*) malloc(N1 * N2 * sizeof(double));
    cudaMalloc(&result_dev, N1 * N2 * sizeof(double));
    /*------------------------------------------------------------------------------------------------------*/
    
    double* x_host;
    double* x_dev;
    x_host = (double*) malloc(N1 * N2 * sizeof(double));
    cudaMalloc(&x_dev, N1 * N2 * sizeof(double));
    for(int i=0; i<N1; i++)
    {
        for(int j=0; j<N2; j++)
        {
            x_host[i* N1 + j] = 2.;
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
    cudaMemcpy(x_dev, x_host, sizeof(double) * N1 * N2, cudaMemcpyHostToDevice);
    
    add_host(x_dev, x_dev, result_dev, N1, N2);
    x_dev = result_dev;
    mul_host(x_dev, x_dev, result_dev, N1, N2);
    x_dev = result_dev;
    
    
    cudaMemcpy(result_host, result_dev, sizeof(double) * N1 * N2, cudaMemcpyDeviceToHost);
    
    std::cout << "After:\n";
    for(int i=0; i<N1; i++)
    {
        for(int j=0; j<N2; j++)
        {
            std::cout << result_host[i * N1 + j] << '\t';
        }
        std::cout << '\n';
    }
      
}