#include <cuda_runtime.h>
#include <vector>
#include <iostream>

int prod(std::vector<int> p)
{
    int size{1};
    for(const auto& it : p)
    {
        size *= it;
    }
    return size;
}



/*
__global__ void add_dev(float* vlist, float* result, int size)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index < size)
    {
        result[index] += vlist[index];
    }
}
*/

/*
* 目前来说这个index是1维还是2维的配置在运行时间上几乎没有任何差异
*/

__global__ void add_dev(float* vlist, float* result, int2 d)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int y_index = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y_index * d.x + x_index;
    // int index = x_index;

    if(index < d.x * d.y)
    {
        result[index] += vlist[index];
    }
}


/*
int add_host(std::vector<float*> vlist, float* result, std::vector<int> dlist)
{
    int size = prod(dlist);
    cudaMemset(result, 0, sizeof(float) * size);
    for(int i=0; i < vlist.size(); i++)
    {
	    add_dev<<<ceil(float(size) / 128), 128, size * sizeof(float)>>>(vlist[i], result, size);
    }
    return EXIT_SUCCESS;
}
*/
int add_host(std::vector<float*> vlist, float* result, std::vector<int> dlist)
{
    int size = prod(dlist);
    cudaMemset(result, 0, sizeof(float) * size);
    
    for(int i=0; i < vlist.size(); i++)
    {
	    add_dev<<<ceil(float(size) / 128), 128, size * sizeof(float)>>>(vlist[i], result, make_int2(dlist[0], dlist[1]));
    }
    return EXIT_SUCCESS;
}

int main(void)
{
    std::vector<float*> vlist;
    int N1{5760};
    int N2{11520};
    float* a_host;
    float* b_host;
    float* c_host;
    a_host = (float*) malloc(N1 * N2 * sizeof(float));
    b_host = (float*) malloc(N1 * N2 * sizeof(float));
    c_host = (float*) malloc(N1 * N2 * sizeof(float));
    float* a_dev;
    float* b_dev;
    float* c_dev;
    cudaMalloc(&a_dev, N1 * N2 * sizeof(float));
    cudaMalloc(&b_dev, N1 * N2 * sizeof(float));
    cudaMalloc(&c_dev, N1 * N2 * sizeof(float));

    for(int i=0; i < N1; i++)
    {
        for(int j=0; j < N2; j++)
        {
            a_host[i * N2 + j] = 1.;
            b_host[i * N2 + j] = 2.;
            c_host[i * N2 + j] = 3.;
        }
    }

    cudaMemcpy(a_dev, a_host, N1 * N2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b_host, N1 * N2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c_dev, c_host, N1 * N2 * sizeof(float), cudaMemcpyHostToDevice);

    vlist.push_back(a_dev);
    vlist.push_back(b_dev);
    vlist.push_back(c_dev);

    float* result_host;
    float* result_dev;

    result_host = (float*) malloc(N1 * N2 * sizeof(float));
    cudaMalloc(&result_dev, N1 * N2 * sizeof(float));   

    std::vector<int> dlist;
    dlist.push_back(N1);
    dlist.push_back(N2);

    add_host(vlist, result_dev, dlist);

    cudaMemcpy(result_host, result_dev, N1 * N2 * sizeof(float), cudaMemcpyDeviceToHost);  
    
    std::cout << "Input data\n";
    for(int i=0; i < N1; i++)
    {
        for(int j=0; j < N2; j++)
        {
            std::cout << a_host[i * N2 + j] << '\t';
        }
        std::cout << '\n';
    }
    std::cout << "Input data\n";
    for(int i=0; i < N1; i++)
    {
        for(int j=0; j < N2; j++)
        {
            std::cout << b_host[i * N2 + j] << '\t';
        }
        std::cout << '\n';
    }
    std::cout << "Input data\n";
    for(int i=0; i < N1; i++)
    {
        for(int j=0; j < N2; j++)
        {
            std::cout << c_host[i * N2 + j] << '\t';
        }
        std::cout << '\n';
    }
    std::cout << "Output data\n";
    for(int i=0; i < N1; i++)
    {
        for(int j=0; j < N2; j++)
        {
            std::cout << result_host[i * N2 + j] << '\t';
        }
        std::cout << '\n';
    }
    
    

}