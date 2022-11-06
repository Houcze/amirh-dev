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

__global__ void bias2d_dev(double* input, double* output, int2 d, int2 ij)
{
    extern __shared__ int s[];
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int y_index = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y_index * d.x + x_index;

   
    if((((index % d.y) + ij.y) < d.y) && (((index % d.y) + ij.y) >= 0))
    {
        s[index + ij.y] = input[index];
    }
    
    if((((index / d.y) + ij.x) < d.x) && (((index / d.y) + ij.x) >= 0))
    {
        output[index + ij.x * d.y] = s[index];
    }
   
}


int bias2d_host(double* base, double* result, int2 d, int2 ij)
{
	/*
	* 该函数只针对2d数组进行定义，但是不检查数组形状
	*/
    cudaMemset(result, 0, sizeof(double) * d.x * d.y);
	bias2d_dev<<<ceil(double(d.x * d.y) / 128), 128, d.x * d.y * sizeof(double)>>>(base, result, d, ij);
    // std::cout << __func__ << std::endl;
	return EXIT_SUCCESS;
}


int main(void)
{
    /*------------------------------------- INIT -----------------------------------------------------------*/
    double* result_host;
    double* result_dev;
    int N1{5};
    int N2{6};
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
    
    std::cout << "Before:\n";
    for(int i=0; i<N1; i++)
    {
        for(int j=0; j<N2; j++)
        {
            std::cout << p_host[i * N2 + j] << '\t';
        }
        std::cout << '\n';
    }
    
    cudaMemcpy(p_dev, p_host, sizeof(double) * N1 * N2, cudaMemcpyHostToDevice);
    /*
        向右和向下移动为正
        反之为负
    */

    bias2d_host(p_dev, result_dev, make_int2(N1, N2), make_int2(-1, -1));
    
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