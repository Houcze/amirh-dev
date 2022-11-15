#ifndef PROP_H
#define PROP_H
#include <map>
#include <string>

namespace Prop{
    using shape = std::map<std::string, int>;
    int size(shape);
    int dims(shape);
}

int Prop::size(Prop::shape p)
{
    int size{1};
    for(const auto& it : p)
    {
        size *= (it.second);
    }
    return size;
}

int Prop::dims(Prop::shape p)
{
    return p.size();
}

#endif