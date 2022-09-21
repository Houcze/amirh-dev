#include <cuda_runtime.h>
#include "Func.h"
#include "Dtype.h"
#define NodeSuccess 1

uFunc::uFunc(double (*f)(double, double))
{
    cudaMemcpyFromSymbol(&f2, *f, sizeof(F2));
    // f2 = f;
    InputNum = 2;
}

uFunc::uFunc(double (*f)(double))
{
    cudaMemcpyFromSymbol(&f1, *f, sizeof(F1));
    // f1 = f;
    InputNum = 1;
}

F1 uFunc::get_f1()
{
    return f1;
}

F2 uFunc::get_f2()
{
    return f2;
}

int uFunc::getInputNum()
{
    return InputNum;
}

Func::Func(int m, int n, uFunc f)
{
    wid = m;
    len = n;
    InputNum = f.getInputNum();
    switch (InputNum)
    {
    case 1:
        f1 = f.get_f1();
        break;
    case 2:
        f2 = f.get_f2();
        break;
    default:
        break;
    }
}

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