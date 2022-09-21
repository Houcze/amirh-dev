#include <cuda_runtime.h>
#include "Dtype.h"

__global__ void Ops(double* x, double* result, F1 f1, int N1, int N2);
__global__ void Ops(double* x, double* y, double* result, F2 f2, int N1, int N2);