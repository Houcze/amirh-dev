#include <list>
#include <cuda_runtime.h>
#include "Dtype.h"

class uFunc
{
    private:
        int InputNum;
        F1 f1;
        F2 f2;
    public:
        uFunc(double (*f)(double));
        uFunc(double (*f)(double, double));
        F1 get_f1();
        F2 get_f2();
        int getInputNum();
};


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
        Func(int m, int n, uFunc f);
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

/*
extern __device__ double dsin(double x);
extern __device__ double dcos(double x); 
extern __device__ double dtan(double x); 
extern __device__ double dcot(double x); 
*/
extern __device__ double dsin(double x) {return sin(x);}
extern __device__ double dcos(double x) {return cos(x);}
extern __device__ double dtan(double x) {return sin(x) / cos(x);}
extern __device__ double dcot(double x) {return cos(x) / sin(x);}

extern __device__ F1 fp_sin = dsin;
extern __device__ F1 fp_cos = dcos;
extern __device__ F1 fp_tan = dtan;
extern __device__ F1 fp_cot = dcot;

uFunc uSin{dsin};
uFunc uCos{dcos};
uFunc uTan{dtan};
uFunc uCot{dcot};

