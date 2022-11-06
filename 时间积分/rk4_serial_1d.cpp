#include <iostream>
#include <vector>

typedef double (*F)(double, double);

double f(double x, double t)
{
    return x;
}

double k1(F func, double t_n, double y_n, double h)
{
    return func(t_n, y_n);
}

double k2(F func, double t_n, double y_n, double h)
{
    return func(t_n + h / 2, y_n + k1(func, t_n, y_n, h) * h / 2);
}

double k3(F func, double t_n, double y_n, double h)
{
    return func(t_n + h / 2, y_n + k2(func, t_n, y_n, h) * h / 2);
}

double k4(F func, double t_n, double y_n, double h)
{
    return func(t_n + h, y_n + k3(func, t_n, y_n, h) * h);
}

double next(F func, double t_n, double y_n, double h)
{
    return y_n + (1 / 6) * (k1(func, t_n, y_n, h) + k2(func, t_n, y_n, h) + k3(func, t_n, y_n, h) + k4(func, t_n, y_n, h)) * h;
}

int main(void)
{
    double t0{0};
    double y0{0};
    double y{y0};
    double h{0.1};

    std::vector<double> index;
    std::vector<double> result;

    for(int i=0; i < 100; i++)
    {
        y = next(f, y, t0 + i * h, h);
        index.push_back(t0 + i * h);
        result.push_back(y);
    }

    for(int i=0; i < 100; i++)
    {
        std::cout << "( " << index[i] << ", " << result[i] << ")\n";
    }

}