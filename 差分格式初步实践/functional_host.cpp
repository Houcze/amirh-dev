#include <iostream>
#include <vector>

int main(void)
{
    float* a;
    float* b;
    float* c;
    float* result;
    int N1{5760};
    int N2{11520};
    a = (float *) malloc(N1 * N2 * sizeof(float));
    b = (float *) malloc(N1 * N2 * sizeof(float));
    c = (float *) malloc(N1 * N2 * sizeof(float));
    result = (float *) malloc(N1 * N2 * sizeof(float));

    for(int i=0; i < N1; i++)
    {
        for(int j=0; j < N2; j++)
        {
            a[i * N2 + j] = 1.;
            b[i * N2 + j] = 2.;
            c[i * N2 + j] = 3.;
        }
    }
    std::vector<float*> vlist;
    vlist.push_back(a);
    vlist.push_back(b);
    vlist.push_back(c);

    for(int i=0; i < vlist.size(); i++)
    {
        for(int j=0; j < N1; j++)
        {
            for(int k=0; k < N2; k++)
            {
                result[i * N2 + j] += (vlist[i])[j * N2 + k];
            }
        }        
    }

    /*   
    std::cout << "Input data\n";
    for(int i=0; i < N1; i++)
    {
        for(int j=0; j < N2; j++)
        {
            std::cout << a[i * N2 + j] << '\t';
        }
        std::cout << '\n';
    }
    std::cout << "Input data\n";
    for(int i=0; i < N1; i++)
    {
        for(int j=0; j < N2; j++)
        {
            std::cout << b[i * N2 + j] << '\t';
        }
        std::cout << '\n';
    }
    std::cout << "Input data\n";
    for(int i=0; i < N1; i++)
    {
        for(int j=0; j < N2; j++)
        {
            std::cout << c[i * N2 + j] << '\t';
        }
        std::cout << '\n';
    }
    std::cout << "Output data\n";
    for(int i=0; i < N1; i++)
    {
        for(int j=0; j < N2; j++)
        {
            std::cout << result[i * N2 + j] << '\t';
        }
        std::cout << '\n';
    }
    */
}