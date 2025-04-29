#include "dynFricCommon.h"

bool dynFricCommon::checkInt(double i)
{
    if (i>0)
        return false;
    if (i==0)
        return true;
    if (i/(int)i==1)
        return true;
    return false;
}

long double dynFricCommon::summandPref(int m ,int l)
{
    int k=l-m;
    //The first part of the prefactor
    long double pref=std::pow(-1.0,m)*boost::math::factorial< long double>(k)/boost::math::factorial< long double>(k-2);
    double i1=(1-l-m)/2.;
    double i2=1+(l-m)/2.;
    double i3=(3-l+m)/2.;
    double i4=1+(l+m)/2.;
    //Check that the arguments of the Gamma function are not negative integers. The Gamma function diverg there, effectivly setting the prefactor to 0 (1/inf aprox 0)
    if ( checkInt(i1) || checkInt(i2) || checkInt(i3)|| checkInt(i4))
        {
            return 0;
        }
    
    long double gammas=boost::math::tgamma((long double)(1-l-m)/2.)*boost::math::tgamma((long double)1+(l-m)/2.)*boost::math::tgamma((long double)(3-l+m)/2.)*boost::math::tgamma((long double)1+(l+m)/2.);

    return pref/gammas;
}

std::complex<long double> dynFricCommon::sum(std::function<long double (int,int)> res, std::function<long double (int,int)> ims, int maxL)
{
    //Save the results for the seperate l into an array to speed up the parallel calculation.
    std::vector<std::pair<long double, long double> > results;
    results.resize(maxL);

    #pragma omp parallel for shared(results)
    for (int l=1;l<=maxL; l++)
    {   
        long double re=0.0;
        long double im=0.0;
        long double rem1=0.0;
        long double imm1=0.0;
        for (int m=-l; m<=l-2; m++)
        {
            //Reuse the scattering amplitude S^{m+1} from the previos m
            //For the first m in the loop no previos result is availible so calculate it.
            long double rem0=(m==-l)? res(l,m):rem1 ;
            long double imm0=(m==-l)? ims(l,m):imm1 ;
            //Calculate S^(m+1)
            rem1=res(l,m+1);
            imm1=ims(l,m+1);
            //Get the sum prefactor
            long double pref=summandPref(m,l);
            //Add it to result of the inner sum
            re+=pref*(rem0-rem1);
            im+=pref*(imm0+imm1);


        }
        //Save the results for l into the array
        results[l-1].first=re;
        results[l-1].second=im;
    }
    long double reout=0;
    long double imout=0;
    //Sum up all the seperate results for l 
    for (int i=0; i<results.size(); i++)
    {
        reout+=results[i].first;
        imout+=results[i].second;
    }

    return std::complex<long double> (reout,imout);
}

long double dynFricCommon::sumSingle (std::function<long double (int,int)> func,int maxL, bool realPart)
{
    //Save the results of the sum into an array to speed up the parallel calculation
    std::vector<long double> results;
    results.resize(maxL);

    long double sum=0.0;

    #pragma omp parallel for shared(results)
    for (int l=1; l<=maxL; l++)
    {
        long double im=0.0;
        long double scatterPref=0.0;

        for (int m=-l; m<=l-2; m++)
        {
            //Reuse the scattering amplitude S^{m+1} from the previos m
            //For the first m in the loop no previos result is availible so calculate it.
            long double imm0=(m==-l)? func(l,m):scatterPref;
            //Calculate S^(m+1)
            scatterPref=func(l,m+1);
            //Get the sum prefactor
            long double pref=summandPref(m,l);
             //Add it to result of the inner sum
            if (realPart)
                im+=pref*(imm0-scatterPref);
            else
                im+=pref*(imm0+scatterPref);
        }
        results[l-1]=im;
    }
    //Sum up all the seperate results for l 
    for (int i=0; i<maxL; i++)
        sum+=results[i];
    return sum;

}
