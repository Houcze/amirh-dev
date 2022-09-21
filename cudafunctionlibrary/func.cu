#include <cuda_runtime.h>
typedef double (*F1)(double);
typedef double (*F2)(double, double);

__global__ void Ops(double* x, double* y, double* result, F2 f2, int N1, int N2)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int y_index = blockIdx.y * blockDim.y + threadIdx.y;	

	int index = x_index + y_index * N2;
	if(index < N1 * N2)
		result[index] = (*f2)(x[index], y[index]);
}

__device__ double add(double x1, double x2)
{
    return x1 + x2;
}

__device__ F2 fp_add = add;

void Add(double* x, double* y, double* result, int N1, int N2)
{
    F2 Add;
    cudaMemcpyFromSymbol(&Add, fp_add, sizeof(F2));
    Ops<<<ceil(N1 * N2 / double(1024)), 1024>>>(x, y, result, Add, N1, N2);
}