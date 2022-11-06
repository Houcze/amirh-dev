#include <cuda_runtime.h>
#include <iostream>

int main(int argc, char* argv[])
{
    cudaError_t cudaStatus;
    int gpuc;
    cudaStatus = cudaGetDeviceCount(&gpuc);
    std::cout << gpuc << std::endl;
    return EXIT_SUCCESS;   
}