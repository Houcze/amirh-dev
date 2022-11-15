#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

/*
    设计范式: 函数统一写为两类，逻辑层的函数不可以直接访问
    a_dev: a在gpu上
    a_host: a在cpu上
    a可以是任何变量
    但是考虑到计算的开销，我认为实际上计算过程中并不需要step-step同步数据的方式，此编程范式的意义在于为调试器留下接口
    同时这个组件的启发也包括，牵扯到逻辑层面的代码，一定不可以直接面向gpu，函数也应当采用
    func_host
    func_dev
    的范式区分，前者是func的cpu部分，后者是gpu部分
*/

__global__ void bias2d_dev(double* input, double* output, int d1, int d2)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int y_index = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y_index * d1 + x_index;
    int i;
    int j;
   
    i = 0;
    j = 1;
    if((((index % d2) + j) < d2) && (((index % d2) + j) >= 0) && (((index / d2) + i) < d1) && (((index / d2) + i) >= 0))
    {

        output[index + i * d2 + j] += input[index];
    }

    i = 0;
    j = -1;
    if((((index % d2) + j) < d2) && (((index % d2) + j) >= 0) && (((index / d2) + i) < d1) && (((index / d2) + i) >= 0))
    {

        output[index + i * d2 + j] += input[index];
    }

    i = 1;
    j = 0;
    if((((index % d2) + j) < d2) && (((index % d2) + j) >= 0) && (((index / d2) + i) < d1) && (((index / d2) + i) >= 0))
    {

        output[index + i * d2 + j] += input[index];
    }

    i = -1;
    j = 0;
    if((((index % d2) + j) < d2) && (((index % d2) + j) >= 0) && (((index / d2) + i) < d1) && (((index / d2) + i) >= 0))
    {

        output[index + i * d2 + j] += input[index];
    }

    if(index < d1 * d2)
    {
        output[index] -= 4 * input[index];
    }
       
}


int bias2d_host(double* base, double* result, int d1, int d2)
{
	/*
	* 该函数只针对2d数组进行定义，但是不检查数组形状
	*/
    cudaMemset(result, 0, sizeof(double) * d1 * d2);
	bias2d_dev<<<ceil(double(d1 * d2) / 128), 128>>>(base, result, d1, d2);
    // std::cout << __func__ << std::endl;
	return EXIT_SUCCESS;
}


int main(void)
{
    /*------------------------------------- INIT -----------------------------------------------------------*/
    double* result_host;
    double* result_dev;
    int N1{3};
    int N2{4};
    result_host = (double*) malloc(N1 * N2 * sizeof(double));
    cudaMalloc(&result_dev, N1 * N2 * sizeof(double));
    /*------------------------------------------------------------------------------------------------------*/
    
    double* p_host;
    double* p_dev;
    p_host = (double*) malloc(N1 * N2 * sizeof(double));
    cudaMalloc(&p_dev, N1 * N2 * sizeof(double));
    for(int i=0; i<N1; i++)
    {
        for(int j=0; j<N2; j++)
        {
            p_host[i* N2 + j] = 1.;
        }
    }
    /*
    std::cout << "Before:\n";
    for(int i=0; i<N1; i++)
    {
        for(int j=0; j<N2; j++)
        {
            std::cout << p_host[i * N2 + j] << '\t';
        }
        std::cout << '\n';
    }
    */
    
    cudaMemcpy(p_dev, p_host, sizeof(double) * N1 * N2, cudaMemcpyHostToDevice);
    /*
        向右和向下移动为正
        反之为负
    */

    bias2d_host(p_dev, result_dev, N1, N2);
    
    cudaMemcpy(result_host, result_dev, sizeof(double) * N1 * N2, cudaMemcpyDeviceToHost);
    
    std::cout << "After:\n";
    for(int i=0; i<N1; i++)
    {
        for(int j=0; j<N2; j++)
        {
            std::cout << result_host[i * N2 + j] << '\t';
        }
        std::cout << '\n';
    }

    
}