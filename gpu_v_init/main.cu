#include <cuda_runtime.h>
#include <iostream>

__device__ double ad[5] = {1, 2, 3, 4, 5};
__device__ double *adh = ad;


int main(void)
{
	double *ah;
    ah = (double *) malloc(5 * sizeof(double));
	cudaMemcpyFromSymbol(ah, adh, sizeof(double) * 5);
    double *b;
    b = (double *) malloc(5 * sizeof(double));
    cudaMemcpy(b, ah, 5 * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << b[0] << std::endl;
}