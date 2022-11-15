#include <cuda_runtime.h>
#include <io/netcdf>
#include <iostream>
#include <Prop.h>

/*
    球面
    问题，我们在计算过程中不能出现任何意义的经纬度坐标
*/


int main(void)
{
    char path[] = {"./MY32_all_var_eo.nc"};
    char varn[] = {"temp"};


    Prop::shape p;

    netcdf::ds_prop(&p, path, varn);
    
    int size = Prop::size(p);
    int dims = Prop::dims(p);

    std::cout << "Dims is " << dims << std::endl;
    std::cout << "Size is " << size << std::endl;

    float* topo;
    topo = (float *) malloc(size * sizeof(float));
    netcdf::ds(topo, path, varn);
    
    

}