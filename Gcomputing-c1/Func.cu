#include <cuda_runtime.h>
#include "Func.h"
#define NodeSuccess 1

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