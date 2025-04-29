#include <iostream>
#include <math.h>
#include "gas.h"
#include <vector>

int main (int argc, char **argv)
{
    int min_l=1;
    int max_l=800;
    int step_l=2;
    
    std::vector<double> machnumbers={0.5,0.9,2.0,4.0,6.0};
    
    gas gasC;
    
    for (int i=0; i<machnumbers.size(); i++)
    {
        for (int l=min_l; l<max_l; l+=step_l)
        {
            std::complex<long double> result=gasC.Im(l,machnumbers[i]);
            
            std::cout<<machnumbers[i]<<"  "<<l<<"  "<<result.real()<<"  "<<result.imag()<<std::endl;
        }
        
        std::cout<<std::endl;
    }

}
