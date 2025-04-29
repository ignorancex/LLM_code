#include "gas.h"

long double  gas::ReS(int l, int m, long double Ma)
{
    if (m!=0)
    {
        long double result=-M_PI_2*boost::math::sph_bessel(l,abs(m*Ma))*boost::math::sph_neumann(l-1,abs(m*Ma));
        return result;
    }
    else
    {
        long double result=M_PI_2/(4*l*l-1);
        return result;
    }   
}

long double gas::ImS(int l,int m,long double Ma)
{
    if (m!=0)
    {
        long double result=(m<0?-1.0:1.0)*M_PI_2*boost::math::sph_bessel(l,abs(m*Ma))*boost::math::sph_bessel(l-1,abs(m*Ma));
        return  result;
    }
    else
    {
        return 0;
    }
}



std::complex<long double> gas::Im(int maxL, long double Ma)
{
    std::complex<long double> result=sum(std::bind( &gas::ReS,this, std::placeholders::_1,std::placeholders::_2,Ma ),std::bind(&gas::ImS,this, std::placeholders::_1,std::placeholders::_2,Ma),maxL  );
    return  std::complex<long double> (Ma*Ma*result.real(),Ma*Ma*result.imag());   
}


long double gas::ReI(int maxL,long double Ma )
{
    return Ma*Ma*sumSingle(std::bind( &gas::ReS,this, std::placeholders::_1,std::placeholders::_2,Ma ),maxL,true);
}


long double gas::ImI(int maxL,long double Ma )
{
     return Ma*Ma*sumSingle(std::bind( &gas::ImS,this, std::placeholders::_1,std::placeholders::_2,Ma ),maxL,false);
}


