#include <list>
#include <cuda_runtime.h>
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

__device__ void Ops(double* x, double* result, F1 f1, int N1, int N2);
__device__ void Ops(double* x, double* y, double* result, F2 f2, int N1, int N2);