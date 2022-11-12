#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <fstream>
#include <string>

__global__ void laplace_dev(double* phi, double* result, int2 d)
{
    /*
    extern __shared__ double s[];
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int y_index = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y_index * d.x + x_index;

    int i;
    int j;


    i = 1;
    j = 0;
    if((((index % d.y) + j) < d.y) && (((index % d.y) + j) >= 0))
    {
        s[index + j] = phi[index];
    }
    
    if((((index / d.y) + i) < d.x) && (((index / d.y) + i) >= 0))
    {
        result[index + i * d.y] += s[index];
    }

    i = -1;
    j = 0;
    if((((index % d.y) + j) < d.y) && (((index % d.y) + j) >= 0))
    {
        s[index + j] = phi[index];
    }
    
    if((((index / d.y) + i) < d.x) && (((index / d.y) + i) >= 0))
    {
        result[index + i * d.y] += s[index];
    }       

    i = 0;
    j = 1;
    if((((index % d.y) + j) < d.y) && (((index % d.y) + j) >= 0))
    {
        s[index + j] = phi[index];
    }
    
    if((((index / d.y) + i) < d.x) && (((index / d.y) + i) >= 0))
    {
        result[index + i * d.y] += s[index];
    }

    i = 0;
    j = -1;
    if((((index % d.y) + j) < d.y) && (((index % d.y) + j) >= 0))
    {
        s[index + j] = phi[index];
    }
    
    if((((index / d.y) + i) < d.x) && (((index / d.y) + i) >= 0))
    {
        result[index + i * d.y] += s[index];
    }     

    if(index < d.x * d.y)
    {
        result[index] -= 4 * phi[index];
    }
    */

}


int laplace_host(double* phi, double* result, int N1, int N2)
{

    laplace_dev<<<ceil(double(N1 * N2) / 128), 128, N1 * N2 * sizeof(double)>>>(phi, result, make_int2(N1, N2));
    return EXIT_SUCCESS;
}


__global__ void mul_dev(double* x, double* y, double* result, int2 d)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int y_index = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y_index * d.x + x_index;
    if(index < d.x * d.y)
    {
        result[index] = x[index] * y[index];
    }    
       
}

int mul_host(double* x, double* y, double* result, int N1, int N2)
{
    mul_dev<<<ceil(double(N1 * N2) / 128), 128, N1 * N2 * sizeof(double)>>>(x, y, result, make_int2(N1, N2));
    return EXIT_SUCCESS;
}

__global__ void add_dev(double* x, double* y, double* result, int2 d)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int y_index = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y_index * d.x + x_index;
    if(index < d.x * d.y)
    {
        result[index] = x[index] + y[index];
    }    
       
}

int add_host(double* x, double* y, double* result, int N1, int N2)
{
    add_dev<<<ceil(double(N1 * N2) / 128), 128, N1 * N2 * sizeof(double)>>>(x, y, result, make_int2(N1, N2));
    return EXIT_SUCCESS;
}


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
    //mul_host(t, result, result, N1, N2);
    //add_host(phi, result, result, N1, N2);
    return EXIT_SUCCESS;
}


int phi2(double* phi, double* result, double h, int N1, int N2)
{
    double* t;
    cudaMalloc(&t, sizeof(double) * N1 * N2);
    constant_matrix_dev(t, h / 2, N1, N2);
    phi1(phi, result, h, N1, N2);
    //R(result, result, N1, N2);
    //mul_host(t, result, result, N1, N2);
    //add_host(phi, result, result, N1, N2);
    return EXIT_SUCCESS;
}

int next(double* phi, double* result, double h, int N1, int N2)
{
    double* t;
    cudaMalloc(&t, sizeof(double) * N1 * N2);
    constant_matrix_dev(t, h, N1, N2);
    phi2(phi, result, h, N1, N2);
    //R(result, result, N1, N2);
    //mul_host(t, result, result, N1, N2);
    //add_host(phi, result, result, N1, N2);
    return EXIT_SUCCESS;        
}

int main(void)
{
    int N1{100};
    int N2{100};
    double h{0.1};
    double* init_host;
    double* result_host;
    init_host = (double*) malloc(N1 * N2 * sizeof(double));   
    result_host = (double*) malloc(N1 * N2 * sizeof(double)); 
    double* init_dev;
    double* result_dev;
    cudaMalloc(&init_dev, N1 * N2 * sizeof(double));
    cudaMalloc(&result_dev, N1 * N2 * sizeof(double));

    for(int i=0; i<N1; i++)
    {
        for(int j=0; j<N2; j++)
        {
            init_host[i * N2 + j] = 1.;
            result_host[i * N2 + j] = 0.;
        }
    }
    init_host[50 * 100 + 50] = 3.;
    cudaMemcpy(init_dev, init_host, sizeof(double) * N1 * N2, cudaMemcpyHostToDevice);
    cudaMemcpy(result_dev, result_host, sizeof(double) * N1 * N2, cudaMemcpyHostToDevice);
    for(int i=0; i<100; i++)
    {
        std::cout << "Round " << i + 1 << std::endl;
        next(init_dev, result_dev, h, N1, N2);
        cudaMemcpy(init_dev, result_dev, sizeof(double) * N1 * N2, cudaMemcpyDeviceToDevice);
        cudaMemcpy(result_host, result_dev, sizeof(double) * N1 * N2, cudaMemcpyDeviceToHost);
        std::ofstream outfile;
        outfile.open("./result-cpp/" + std::to_string(i + 1) + ".txt");
        for(int j=0; j < N1; j++)
        {
            for(int k=0; k <N2; k++)
            {
                outfile << result_host[i * N2 + j] << '\t';
            }
            outfile << '\n';
        }
        outfile.close();

    }


}