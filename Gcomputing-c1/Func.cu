#include <cuda_runtime.h>
#include "Func.h"
#define NodeSuccess 1

// Add
Func::Func(int m, int n, double (*f)(double, double))
{
    wid = m;
    len = n;
    f2 = f;
    InputNum = 2;
}

Func::Func(int m, int n, double (*f)(double))
{
    wid = m;
    len = n;
    f1 = f;
    InputNum = 1;
}

__global__ void Ops(double* x, double* result, F1 f1, int N1, int N2)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int y_index = blockIdx.y * blockDim.y + threadIdx.y;	

	int index = x_index + y_index * N2;
	if(index < N1 * N2)
		result[index] = (*f1)(x[index]);
}


__global__ void Ops(double* x, double* y, double* result, F2 f2, int N1, int N2)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int y_index = blockIdx.y * blockDim.y + threadIdx.y;	

	int index = x_index + y_index * N2;
	if(index < N1 * N2)
		result[index] = (*f2)(x[index], y[index]);
}


int Func::run()
{
    switch (InputNum)
    {
    case 1:
        Ops<<<ceil(wid * len / double(1024)), 1024>>>(x, result, *f1, wid, len);
        break;
    case 2:
        Ops<<<ceil(wid * len / double(1024)), 1024>>>(x, y, result, *f2, wid, len);
        break;    
    
    default:
        break;
    }
    return NodeSuccess;
}

int Func::Input(double* x1, double* x2)
{
    x = x1;
    y = x2;
    return EXIT_SUCCESS;
}

int Func::Input(double* x1)
{
    x = x1;
    return EXIT_SUCCESS;
}

int Func::rst(double* rst)
{
    result = rst;
    return EXIT_SUCCESS;
}