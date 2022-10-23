#include <iostream>

int main(void)
{
    int a[5] = {1, 2, 3, 4, 5};
    int *b = a;
    for(int i=0; i < 5; i++)
    {
        std::cout << b[i] << std::endl;
    }
}