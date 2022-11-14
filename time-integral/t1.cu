#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <fstream>
#include <string>
#include <cmath>
#include <io/netcdf>

__global__ void laplace_dev(double* phi, double* result, double* s1, double* s2, int2 d)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    //int y_index = blockIdx.y * blockDim.y + threadIdx.y;

    // int index = y_index * d.x + x_index;
    // int index = threadIdx.x;
    int index = x_index;
    int i;
    int j;

    if(index < d.x * d.y)
    {
        result[index] = 0.;
        s1[index] = 0.;
        s2[index] = 0.;

    }
    /*
    i = 1;
    j = 0;
    if((((index % d.y) + j) < d.y) && (((index % d.y) + j) >= 0) && (((index / d.y) + i) < d.x) && (((index / d.y) + i) >= 0) && (index < d.x * d.y))
    {

        s1[index + i * d.y + j] = phi[index];
    }
    */
    i = 1;
    j = 0;
    
    if(((index / d.y + i) < d.x) && ((index / d.y + i) >= 0) && ((index % d.y + j) < d.y) && ((index % d.y + j) >= 0))
    {
        s1[index + i * d.y + j] = phi[index];
    }

    
    /*
    i = -1;
    j = 0;
    if((((index % d.y) + j) < d.y) && (((index % d.y) + j) >= 0) && (((index / d.y) + i) < d.x) && (((index / d.y) + i) >= 0))
    {

        s2[index + i * d.y + j] = phi[index];
    }     
    */
    
    if(index < d.x * d.y)
    {
        result[index] = s1[index]; // + s2[index];
    }

    
    /*
    i = 0;
    j = 1;
    if((((index % d.y) + j) < d.y) && (((index % d.y) + j) >= 0) && (((index / d.y) + i) < d.x) && (((index / d.y) + i) >= 0))
    {

        result[index + i * d.y + j] += phi[index];
    }
    
    
    i = 0;
    j = -1;
    if((((index % d.y) + j) < d.y) && (((index % d.y) + j) >= 0) && (((index / d.y) + i) < d.x) && (((index / d.y) + i) >= 0))
    {

        result[index + i * d.y + j] += phi[index];
    }     
    */
    /*


  
    if(index < d.x * d.y)
    {
        result[index] -= 4 * phi[index];
    }
    */
    
}



/*
* 共享内存限制太多
*/
int laplace_host(double* phi, double* result, int N1, int N2)
{
    double* s1;
    double* s2;
    cudaMalloc(&s1, N1 * N2 * sizeof(double));
    cudaMalloc(&s2, N1 * N2 * sizeof(double));
    laplace_dev<<<std::ceil(double(N1 * N2) / 1024), 1024>>>(phi, result, s1, s2, make_int2(N1, N2));
    cudaFree(s1);
    cudaFree(s2);
    return EXIT_SUCCESS;
}

int main(void)
{
    int N1{10};
    int N2{10};
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

    for(int i=0; i<N1; i++)
    {
        for(int j=0; j<N2; j++)
        {
            std::cout << init_host[i * N2 + j] << '\t';
        }
        std::cout << '\n';
    }

    char filepath[] = "./input.nc";
    char varname[] = "temperature";
    // netcdf::ds(init_host, filepath, varname);
    /********************************************************************************************************/
    std::ofstream check_init;
    check_init.open("./init.txt");
    for(int j=0; j < N1; j++)
    {
        for(int k=0; k <N2; k++)
        {
            check_init << init_host[j * N2 + k] << '\t';
        }
        check_init << '\n';
    }
    check_init.close();
    /********************************************************************************************************/
    cudaMemcpy(init_dev, init_host, sizeof(double) * N1 * N2, cudaMemcpyHostToDevice);
    cudaMemcpy(result_dev, result_host, sizeof(double) * N1 * N2, cudaMemcpyHostToDevice);
    for(int i=0; i<1; i++)
    {
        std::cout << "Round " << i + 1 << std::endl;
        laplace_host(init_dev, result_dev, N1, N2);
        // cudaMemcpy(init_dev, result_dev, sizeof(double) * N1 * N2, cudaMemcpyDeviceToDevice);
        cudaMemcpy(result_host, result_dev, sizeof(double) * N1 * N2, cudaMemcpyDeviceToHost);
        std::ofstream outfile;
        outfile.open("./1r.txt");
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