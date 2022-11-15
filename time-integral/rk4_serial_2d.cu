#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <fstream>
#include <string>
#include <io/netcdf>



__global__ void add_bias_dev(double* phi, double* result, int2 d, int2 ij)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int index = x_index;
    
    int i = ij.x;
    int j = ij.y;

    if(((index / d.y + i) < d.x) && ((index / d.y + i) >= 0) && ((index % d.y + j) < d.y) && ((index % d.y + j) >= 0))
    {
        result[index + i * d.y + j] += phi[index];
    }
}


__global__ void weighted_sub_dev(double* x, double* y, double* result, int2 d, int2 c)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = x_index;
    
    if(index < d.x * d.y)
    {
        result[index] = (c.x) * x[index] - (c.y) * y[index];
    }
}


int laplace_host(double* phi, double* result, int N1, int N2)
{
    cudaMemset(result, 0, N1 * N2 * sizeof(double));
    add_bias_dev<<<dim3(std::ceil(double(N1 * N2) / 1024), 1, 1), dim3(1024, 1, 1)>>>(phi, result, make_int2(N1, N2), make_int2(1, 0));
    add_bias_dev<<<dim3(std::ceil(double(N1 * N2) / 1024), 1, 1), dim3(1024, 1, 1)>>>(phi, result, make_int2(N1, N2), make_int2(-1, 0));
    add_bias_dev<<<dim3(std::ceil(double(N1 * N2) / 1024), 1, 1), dim3(1024, 1, 1)>>>(phi, result, make_int2(N1, N2), make_int2(0, 1));
    add_bias_dev<<<dim3(std::ceil(double(N1 * N2) / 1024), 1, 1), dim3(1024, 1, 1)>>>(phi, result, make_int2(N1, N2), make_int2(0, -1));
    weighted_sub_dev<<<dim3(std::ceil(double(N1 * N2) / 1024), 1, 1), dim3(1024, 1, 1)>>>(result, phi, result, make_int2(N1, N2), make_int2(1, 4));
    return EXIT_SUCCESS;
}

/**************************************** 乘法函数 **********************************************/
__global__ void mul_dev(double* x, double* y, double* result, int2 d)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = x_index;
    if(index < d.x * d.y)
    {
        result[index] = x[index] * y[index];
    }    
       
}


int mul_host(double* x, double* y, double* result, int N1, int N2)
{
    mul_dev<<<ceil(double(N1 * N2) / 128), 128>>>(x, y, result, make_int2(N1, N2));
    return EXIT_SUCCESS;
}
/***********************************************************************************************/
/**************************************** 加法函数 **********************************************/
__global__ void add_dev(double* x, double* y, double* result, int2 d)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = x_index;
    if(index < d.x * d.y)
    {
        result[index] = x[index] + y[index];
    }    
       
}


int add_host(double* x, double* y, double* result, int N1, int N2)
{
    add_dev<<<ceil(double(N1 * N2) / 128), 128>>>(x, y, result, make_int2(N1, N2));
    return EXIT_SUCCESS;
}
/***********************************************************************************************/
int constant_matrix_dev(double* mat_dev, double value, int N1, int N2)
{
    double* mat_host;
    mat_host = (double*) malloc(N1 * N2 * sizeof(double)); 
    for(int i=0; i < N1; i++)
    {
        for(int j=0; j < N2; j++)
        {
            mat_host[i * N2 + j] = value;
        }
    }
    cudaMemcpy(mat_dev, mat_host, sizeof(double) * N1 * N2, cudaMemcpyHostToDevice);
    free(mat_host);
    return EXIT_SUCCESS;
}


int R(double* phi, double* result, int N1, int N2)
{
    laplace_host(phi, result, N1, N2);
    return EXIT_SUCCESS;
}


int phi1(double* phi, double* result, double h, int N1, int N2)
{
    double* t;
    cudaMalloc(&t, sizeof(double) * N1 * N2);
    constant_matrix_dev(t, h / 3, N1, N2);
    R(phi, result, N1, N2);
    mul_host(t, result, result, N1, N2);
    add_host(phi, result, result, N1, N2);
    return EXIT_SUCCESS;
}


int phi2(double* phi, double* result, double h, int N1, int N2)
{
    double* t;
    cudaMalloc(&t, sizeof(double) * N1 * N2);
    constant_matrix_dev(t, h / 2, N1, N2);

    double* phi1_result;
    cudaMalloc(&phi1_result, N1 * N2 * sizeof(double));

    phi1(phi, phi1_result, h, N1, N2);
    R(phi1_result, result, N1, N2);
    mul_host(t, result, result, N1, N2);
    add_host(phi, result, result, N1, N2);
    cudaFree(phi1_result);
    return EXIT_SUCCESS;
}


int next(double* phi, double* result, double h, int N1, int N2)
{
    double* t;
    cudaMalloc(&t, sizeof(double) * N1 * N2);
    constant_matrix_dev(t, h, N1, N2);

    double* phi2_result;
    cudaMalloc(&phi2_result, N1 * N2 * sizeof(double));
    
    phi2(phi, phi2_result, h, N1, N2);
    R(phi2_result, result, N1, N2);
    mul_host(t, result, result, N1, N2);
    add_host(phi, result, result, N1, N2);
    
    cudaFree(phi2_result);
    return EXIT_SUCCESS;        
}

int main(void)
{
    int N1{64};
    int N2{48};
    double h{0.1};
    double* init_host;
    double* result_host;
    init_host = (double*) malloc(N1 * N2 * sizeof(double));   
    result_host = (double*) malloc(N1 * N2 * sizeof(double)); 
    double* init_dev;
    double* result_dev;
    cudaMalloc(&init_dev, N1 * N2 * sizeof(double));
    cudaMalloc(&result_dev, N1 * N2 * sizeof(double));

    char filepath[] = "./input.nc";
    char varname[] = "temperature";
    netcdf::ds(init_host, filepath, varname);

    cudaMemcpy(init_dev, init_host, sizeof(double) * N1 * N2, cudaMemcpyHostToDevice);
    cudaMemcpy(result_dev, result_host, sizeof(double) * N1 * N2, cudaMemcpyHostToDevice);

    for(int i=0; i<36000; i++)
    {
        std::cout << "Round " << i + 1 << std::endl;
        next(init_dev, result_dev, h, N1, N2);
        cudaMemcpy(init_dev, result_dev, sizeof(double) * N1 * N2, cudaMemcpyDeviceToDevice);
        
        if(i % 600 == 0)
        {
            cudaMemcpy(result_host, result_dev, sizeof(double) * N1 * N2, cudaMemcpyDeviceToHost);

            std::ofstream outfile;
            outfile.open("./result/" + std::to_string(i + 1) + ".txt");
            for(int j=0; j < N1; j++)
            {
                for(int k=0; k <N2; k++)
                {
                    outfile << result_host[j * N2 + k] << '\t';
                }
                outfile << '\n';
            }
            outfile.close();
        }
    }
}