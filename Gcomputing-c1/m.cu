#include <cstdlib>
#include <iostream>
#include <cmath>
#include <list>
#include <cuda_runtime.h>
#define NodeSuccess 1

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

typedef double (*F1)(double);
typedef double (*F2)(double, double);

class Func
{
    private:
        int InputNum;
        int wid;
        int len;
        double *x;
        double *y;
        double *result;
        F1 f1;
        F2 f2;
    public:
        Func(int m, int n, double (*f)(double));
        Func(int m, int n, double (*f)(double, double));
        int Input(double *, double *);
        int Input(double *);
        int rst(double *);
        int run();
};

class Seq
{
    private:
        int InputNum;
        double *x;
        double *y;
        double *result;
    public:
        Seq(std::list<Func> fl);
        int IN(int);
        std::list<Func> l;
        int Input(double *);
        int Input(double *, double *);
        int rst(double *);
        int run();
};


// Add
Func::Func(int m, int n, double (*f)(double, double))
{
    wid = m;
    len = n;
    f2 = f;
    InputNum = 2;
}

Func::Func(int m, int n, double (*f)(double))
{
    wid = m;
    len = n;
    f1 = f;
    InputNum = 1;
}

__global__ void Ops(double* x, double* result, F1 f1, int N1, int N2)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int y_index = blockIdx.y * blockDim.y + threadIdx.y;	

	int index = x_index + y_index * N2;
	if(index < N1 * N2)
		result[index] = (*f1)(x[index]);
}


__global__ void Ops(double* x, double* y, double* result, F2 f2, int N1, int N2)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int y_index = blockIdx.y * blockDim.y + threadIdx.y;	

	int index = x_index + y_index * N2;
	if(index < N1 * N2)
		result[index] = (*f2)(x[index], y[index]);
}

int Func::run()
{
    switch (InputNum)
    {
    case 1:
        Ops<<<ceil(wid * len / double(1024)), 1024>>>(x, result, *f1, wid, len);
        break;
    case 2:
        Ops<<<ceil(wid * len / double(1024)), 1024>>>(x, y, result, *f2, wid, len);
        break;    
    
    default:
        break;
    }
    return NodeSuccess;
}

int Func::Input(double* x1, double* x2)
{
    x = x1;
    y = x2;
    return EXIT_SUCCESS;
}

int Func::Input(double* x1)
{
    x = x1;
    return EXIT_SUCCESS;
}

int Func::rst(double* rst)
{
    result = rst;
    return EXIT_SUCCESS;
}

double add(double x1, double x2)
{
    return x1 + x2;
}

double sub(double x1, double x2)
{
    return x1 - x2;
}


Seq::Seq(std::list<Func> fl)
{
    l = fl;
}

int Seq::Input(double* x1, double* x2)
{
    x = x1;
    y = x2;
    return EXIT_SUCCESS;
}

int Seq::Input(double* x1)
{
    x = x1;
    return EXIT_SUCCESS;
}

int Seq::rst(double* rst)
{
    result = rst;
    return EXIT_SUCCESS;
}

int Seq::run()
{
    switch (InputNum)
    {
    case 1:
        {
            int Flag{0};
            for(Func n : l)
            {
                switch (Flag)
                {
                case 0:
                    n.Input(x);
                    Flag = 1;
                    break;
                case 1:
                    n.Input(result);
                    break;
                default:
                    break;
                }  
                n.rst(result);
                n.run();
            }
            break;
        }
    case 2:
        for(Func n : l)
        {
            n.Input(x, y);
            n.rst(result);
            n.run();
        }
        break;
    default:
        break;
    }

    return EXIT_SUCCESS;
}

int Seq::IN(int INum)
{
    InputNum = INum;
    return EXIT_SUCCESS;
}


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