#include <map>
#include <iostream>

int prod(std::map<char*, int>* p)
{
    int size{1};
    for(auto it = (*p).begin(); it != (*p).end(); ++it)
    {
        size *= (it->second);
        std::cout << (it->second) << '\t';
    }
    std::cout << '\n';
    return size;
}

int main(void)
{
    std::map<char*, int> a;
    a["HOU"] = 1;
    a["H"] = 2;
    for (auto it = a.begin(); it != a.end(); ++it) {
        std::cout << it->first << ", " << it->second << '\n';
    }
    std::cout << prod(&a) << std::endl;
}