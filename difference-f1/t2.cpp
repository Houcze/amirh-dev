#include <iostream>
#include <cstring>

int main(void)
{
    int a[5] = {1, 2, 3, 4, 5};
    int b[3];
    memcpy(b, &a[2], 3 * sizeof(int));
    for(int i=0; i<3; i++)
    {
        std::cout << b[i] << '\n';
    }
}