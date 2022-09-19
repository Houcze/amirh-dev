#include "Mem.h"
#include <cuda_runtime.h>

double* devMem(int w, int l)
{
    // 模拟内存池
    double* MemBlock;
    cudaMalloc(&MemBlock, w * l * sizeof(double));
    return MemBlock;
}

double* hostMem(int w, int l)
{
    return (double*) malloc(w * l * sizeof(double));
}