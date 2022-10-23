#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include "Dtypes.h"

class Sphere
{
    private:
        double* data;
        int wCells, lCells;
    public:
        Sphere(int w, int l);
        double* value(int, int);
};

Sphere::Sphere(int w, int l)
{
    cudaMalloc(&data, sizeof(double) * w * l);
    //double* hostValue;
    //hostValue = (double*) malloc(w * l * sizeof(double));
    std::vector<double> hostInit;
    // std::cout << "w, l are " << w << " " << l << std::endl;
    /*
    for(int i=0; i<w; i++)
    {
        for(int j=0; j<l; j++)
        {
            hostValue[i * w + j] = 4.;
        }
    }*/
    for(int i=0; i< w * l; i++)
    {
        hostInit.push_back(0.);
    }
    cudaMemcpy(data, hostInit.data(), sizeof(double) * w * l, cudaMemcpyHostToDevice);
    wCells = w;
    lCells = l;
}

double* Sphere::value(int i, int j)
{
    return data + i * wCells + j;
}


class Func
{
    private:
        int InputNum;
        int wid;
        int len;
        double *x;
        double *y;
        double *result;
        F1 f;
        F2 f;
    public:
        Func(int m, int n, F1 f);
        Func(int m, int n, F2 f);
        int Input(double*, double*);
        int Input(double*);
        int rst(double*);
        int run();
};


// 完成中央差分的基础函数
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


double* move(double* base, double* result, int i, int j)
{
	/*
	* 该函数只针对2d数组进行定义，但是不检查数组形状
	*/

	double *result_i;

	cudaMalloc(&result_i, w * l * sizeof(double));

	
	bias_i<<<ceil(w * l / double(1024)), 1024>>>(base, result_i, w, l, i);
	bias_j<<<ceil(w * l / double(1024)), 1024>>>(result_i, result, w, l, j);
	
	cudaFree(result_i);
	return result;
}

int main(void)
{
    Sphere Earth{3, 3};
    double demo;
    
    cudaMemcpy(&demo, Earth.value(1, 1), sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "demo is " << demo << std::endl;
}