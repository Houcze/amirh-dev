#include <cuda_runtime.h>
#include <iostream>
#include <io/netcdf>
#include <Prop.h>


__global__ void central_diff_dev(float* input, float* output, int2 d)
{
   extern __shared__ int s[];
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int y_index = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y_index * d.x + x_index;
    /*
    * output需要初始化为0
    */
    int i, j;
    i = 1;
    j = 1;
    if((((index % d.y) + j) < d.y) && (((index % d.y) + j) >= 0))
    {
        s[index + j] = input[index];
    }
    
    if((((index / d.y) + i) < d.x) && (((index / d.y) + i) >= 0))
    {
        output[index + i * d.y] += s[index];
    }
    i = 1;
    j = -1;
    if((((index % d.y) + j) < d.y) && (((index % d.y) + j) >= 0))
    {
        s[index + j] = input[index];
    }
    
    if((((index / d.y) + i) < d.x) && (((index / d.y) + i) >= 0))
    {
        output[index + i * d.y] += s[index];
    }
    i = -1;
    j = 1;
    if((((index % d.y) + j) < d.y) && (((index % d.y) + j) >= 0))
    {
        s[index + j] = input[index];
    }
    
    if((((index / d.y) + i) < d.x) && (((index / d.y) + i) >= 0))
    {
        output[index + i * d.y] += s[index];
    }
    i = -1;
    j = -1;
    if((((index % d.y) + j) < d.y) && (((index % d.y) + j) >= 0))
    {
        s[index + j] = input[index];
    }
    
    if((((index / d.y) + i) < d.x) && (((index / d.y) + i) >= 0))
    {
        output[index + i * d.y] += s[index];
    }
    if((((index / d.y) + i) < d.x) && (((index / d.y) + i) >= 0))
    {
        output[index + i * d.y] += s[index];
    }
    i = 0;
    j = 0;
    if((((index % d.y) + j) < d.y) && (((index % d.y) + j) >= 0))
    {
        s[index + j] = input[index];
    }
    
    if((((index / d.y) + i) < d.x) && (((index / d.y) + i) >= 0))
    {
        output[index + i * d.y] -= 4 * s[index];
    }

}


int central_diff_host(float* base, float* result, int2 d)
{
	/*
	* 该函数只针对2d数组进行定义，但是不检查数组形状
	*/
    cudaMemset(result, 0, sizeof(float) * d.x * d.y);
	central_diff_dev<<<ceil(float(d.x * d.y) / 128), 128, d.x * d.y * sizeof(float)>>>(base, result, d);
    // std::cout << __func__ << std::endl;
	return EXIT_SUCCESS;
}

int main(void)
{
    char path[] = {"./clim_01_me.nc"};
    char varn[] = {"c_o2"};


    Prop::shape p;

    netcdf::ds_prop(&p, path, varn);
    
    int size = Prop::size(p);
    int dims = Prop::dims(p);  
    
    std::cout << "Dims is " << dims << std::endl;
    std::cout << "Size is " << size << std::endl;

    float* temperature;
    temperature = (float *) malloc(size * sizeof(float));
    netcdf::ds(temperature, path, varn);

    float* t1_host;
    float* t2_host;
    t1_host = (float*) malloc(p["latitude"] * p["longitude"] * sizeof(float));
    t2_host = (float*) malloc(p["latitude"] * p["longitude"] * sizeof(float));

    float* t1_dev;
    float* t2_dev;
    cudaMalloc(&t1_dev, p["latitude"] * p["longitude"] * sizeof(float));
    cudaMalloc(&t2_dev, p["latitude"] * p["longitude"] * sizeof(float));

    memcpy(t1_host, temperature, p["latitude"] * p["longitude"] * sizeof(float));
    memcpy(t2_host, &temperature[p["latitude"] * p["longitude"]], p["latitude"] * p["longitude"] * sizeof(float));  

    cudaMemcpy(t1_dev, t1_host, p["latitude"] * p["longitude"] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(t2_dev, t2_host, p["latitude"] * p["longitude"] * sizeof(float), cudaMemcpyHostToDevice);

    float* result_host;
    result_host = (float*) malloc(p["latitude"] * p["longitude"] * sizeof(float));
    float* result_dev;
    cudaMalloc(&result_dev, p["latitude"] * p["longitude"] * sizeof(float));


    central_diff_host(t1_dev, result_dev, make_int2(p["latitude"], p["longitude"]));

    cudaMemcpy(result_host, result_dev, sizeof(double) * p["latitude"] * p["longitude"], cudaMemcpyDeviceToHost);
    std::cout << '\n';
    for(int i=0; i<p["latitude"]; i++)
    {
        for(int j=0; j<p["longitude"]; j++)
        {
            std::cout << result_host[i * p["longitude"] + j] << '\t';
        }
        std::cout << '\n';
    }

}