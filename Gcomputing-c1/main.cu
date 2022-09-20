#include <cstdlib>
#include <iostream>
#include <cmath>
#include <list>
#include <cuda_runtime.h>
#include "Mem.h"
#include "Func.h"

__device__ double dsin(double x) {return sin(x);}
__device__ double dcos(double x) {return cos(x);}
__device__ double dtan(double x) {return sin(x) / cos(x);}
__device__ double dcot(double x) {return cos(x) / sin(x);}
__device__ F1 fp_sin = dsin;
__device__ F1 fp_cos = dcos;
__device__ F1 fp_tan = dtan;
__device__ F1 fp_cot = dcot;

int main(void)
{
    int wid{6000};
    int len{12000};
    double* host_x1;
    double* host_x2;
    host_x1 = hostMem(wid, len);
    host_x2 = hostMem(wid, len);

    double* x1;
    double* x2;

    x1 = devMem(wid, len);
    x2 = devMem(wid, len);

    for(int i=0; i<wid; i++)
    {
        for(int j=0; j<len; j++)
        {
            host_x1[i * wid + j] = 0.;
            host_x2[i * wid + j] = 2.;
        }
    }
   cudaMemcpy(x1, host_x1, wid * len * sizeof(double), cudaMemcpyHostToDevice);
   cudaMemcpy(x2, host_x2, wid * len * sizeof(double), cudaMemcpyHostToDevice);



    double *rst;
    rst = devMem(wid, len);

    // Func Add{wid, len, add};
    // Func Sub{wid, len, sub};
    F1 fsin;
    F1 fcos;
    F1 ftan;
    F1 fcot;
    cudaMemcpyFromSymbol(&fsin, fp_sin, sizeof(F1));
    cudaMemcpyFromSymbol(&fcos, fp_cos, sizeof(F1));
    cudaMemcpyFromSymbol(&ftan, fp_tan, sizeof(F1));
    cudaMemcpyFromSymbol(&fcot, fp_cot, sizeof(F1));
    Func Sin{wid, len, fsin};
    Func Cos{wid, len, fcos};
    Func Tan{wid, len, ftan};
    Func Cot{wid, len, fcot};

    std::list<Func> fl
    {
        Sin,
        Cos,
        Tan
    };
    Seq b{fl};

    b.IN(1);
    b.Input(x1);
    b.rst(rst);
    b.run();

    double* hostRst;
    hostRst = hostMem(wid, len);

    cudaMemcpy(hostRst, rst, wid * len * sizeof(double), cudaMemcpyDeviceToHost);
    
    for(int i=0; i<wid; i++)
    {
        for(int j=0; j<len; j++)
        {
            std::cout << hostRst[i * wid + j] << '\t';
        }
        std::cout << '\n';
    }
       
}