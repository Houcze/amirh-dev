#include "Ops.h"
#include "Dtype.h"

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