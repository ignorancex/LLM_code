#include <iostream>
#include <iomanip>
#include <chrono>

using namespace std;

/*Prints the time lapse between t0 and t1 divided by iter (default iter=1)*/
void print_t(chrono::high_resolution_clock::time_point t1, chrono::high_resolution_clock::time_point t0, int iter){
    auto t_nano = chrono::duration_cast<chrono::nanoseconds>(t1-t0).count();
    t_nano /= iter;
    if(t_nano<1000){
        cout<<t_nano<<"ns";
        return;
    }
    auto t_micro = chrono::duration_cast<chrono::microseconds>(t1-t0).count();
    t_micro /= iter;
    t_nano -= t_micro*1000;
    if(t_micro<1000){
        cout<<t_micro<<"µs "<<t_nano<<"ns";
        return;
    }
    auto t_milli = chrono::duration_cast<chrono::milliseconds>(t1-t0).count();
    t_milli /= iter;
    t_micro -= t_milli*1000;
    if(t_milli<1000){
        cout<<t_milli<<"ms "<<t_micro<<"µs";
        return;
    }
    auto t_sec = chrono::duration_cast<chrono::seconds>(t1-t0).count();
    t_sec /= iter;
    t_milli -= t_sec*1000;
    if(t_sec<60){
        cout<<t_sec<<"s "<<t_milli<<"ms";
        return;
    }
    auto t_min = chrono::duration_cast<chrono::minutes>(t1-t0).count();
    t_min /= iter;
    t_sec -= t_min*60;
    if(t_min<60){
        cout<<t_min<<"'"<<t_sec<<"s";
        return;
    }
    auto t_hour = chrono::duration_cast<chrono::hours>(t1-t0).count();
    t_hour /= iter;
    t_min -= t_hour*60;
    cout<<t_hour<<"h"<<t_min<<"'";
}
// Sign function 
double sign(const double x){
    return (x>=0) - (x<0);
}

//Smooth function returning 0 if x<< x0 and 1 if x>> x0 in log scale
//s determines how sharp the function is around x0.
double Activation_log(const double x,const double x0, const double s){
    double y = s*log10(x/x0);
    return (erf(y)+1.)/2.;
}
