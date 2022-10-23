#include <list>

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
        Func(int m, int n, F1 f);
        Func(int m, int n, F2 f);
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