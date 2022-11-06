// #include <cuda_runtime.h>
#include <io/netcdf>
#include <iostream>
#include <cstdlib>
#include <unordered_map>
/*
    球面
    问题，我们在计算过程中不能出现任何意义的经纬度坐标
*/


int prod(std::unordered_map<char*, int> p)
{
    int size{1};
    
    for(auto it = p.begin(); it != p.end(); ++it)
    {
        size *= (it->second);
        std::cout << "p Size is " << size << '\n';
    }
    
    char k1[] = "latitude";
    char k2[] = "longitude";
    std::cout << p[k1] << "\t" << p[k2] << std::endl;
    return size;
}


int main(void)
{
    size_t size{1};
    size_t dims;
    char path[] = {"./mola32.nc"};
    char varn[] = {"alt"};

    // netcdf::get_size(&size, &dims, path, varn);
    // std::cout << "Size is " << size << std::endl;
    std::unordered_map<char*, int> p;

    netcdf::ds_prop(&p, path, varn);
    
    // size = p["size"];
    size = prod(p);
    dims = p.size();
    char k1[] = "latitude";
    char k2[] = "longitude";
    std::cout << p[k1] << "\t" << p[k2] << std::endl;
    std::cout << "Dims is " << dims << std::endl;
    std::cout << "Size is " << size << std::endl;
    /*
    double* topo;
    topo = (double *) malloc(size * sizeof(double));
    size_t* shape;
    shape = (size_t*) malloc(sizeof(size_t) * (dims));
    netcdf::ds(topo, shape, path, varn);
    */
    
    /*
    for(int i=0; i < size; i++)
    {
        std::cout << topo[i] << '\t';
    }
    std::cout << '\n';
    */

}