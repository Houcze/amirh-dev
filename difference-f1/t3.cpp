#include <iostream>
#include <vector>

int a(float* b)
{
    std::cout << "Hello World!" << std::endl;
    std::cout << (b)[2] <<  std::endl;
}


int main(void)
{
    std::vector<float> c{1, 2, 3, 4, 5};
    std::vector<float>* d = &c;
    a((*d).data());
}