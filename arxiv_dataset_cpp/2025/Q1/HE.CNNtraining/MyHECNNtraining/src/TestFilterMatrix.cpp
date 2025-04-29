#include <iostream>
#include <cmath>
#include <set>
#include <iomanip>
#include <string>

using namespace std;

int main()
{
    int fdimNums = 512, slots = 512*64, rowidx = 0, batch = 64;
    double* filtermatrix = new double[slots]();
    for (long idx=0; idx < batch; ++idx)
            filtermatrix[idx * fdimNums + (idx + rowidx+17) % batch] = 1;


    for (long r = 0; r < batch; ++r) {
        for (long c = 10; c < fdimNums; ++c)
            filtermatrix[r * fdimNums + c] = 0;
    }


    for (long r = 0; r < batch; ++r) {
        for (long c = 0; c < fdimNums/5; ++c)
            cout << filtermatrix[r * fdimNums + c] ;
        cout << endl;
    }


    int lglabel = int( floor( log(512 / 10.0) / log(2) ) ); 
    cout << endl << "lglabel: " << lglabel <<  "\t" << 10 * int( pow(2, lglabel) ) << endl;

    int* rolist = new int[1+lglabel]();
    for (int i = 0; i < 1+lglabel; ++i) {
        cout << 10 * int( pow(2, i) ) << endl;
        rolist[i] = 10 * int( pow(2, i) );
    }

    int featureNum = 501;
    cout << "A" << (ceil(featureNum / 10.0) +1) * 10 << endl;
    auto newcolnum =   (ceil(featureNum / 10.0) +1) * 10;
    for(int i = lglabel; i>=0; --i) {
        while(newcolnum>=rolist[i]) {
            cout << rolist[i] << "\t";
            newcolnum -= rolist[i];
        }

    }

/*
    if( 441 + 10 > 10 * int( pow(2, lglabel) ) ) {
        cout << endl << 441 + 10 - 10 * int( pow(2, lglabel) ) << endl;
        auto left = 441 + 10 - 10 * int( pow(2, lglabel) );
        cout << "A" << int( ceil( log(left / 10.0) / log(2) ) )  << endl;
        auto lgleft = int( ceil( log(left / 10.0) / log(2) ) );
        cout << "B" << 10 * int( pow(2, lgleft) ) << endl;

        cout << "C" << 10 * int( pow(2, lgleft) )  + 10 * int( pow(2, lglabel) ) - 512 << endl;
    }
*/




cout << endl << endl << endl;
    for (long i=0; i < 5; ++i) 
         cout << 10 * int( pow(2, i) )  << "\t\t";



cout << endl << endl;
    for (long i=0; i < 64; ++i) 
        cout << 512 * i << ",\t";






    int pos[] = { +0, +1, +2, +4, +8, +16, +32, +64, +128, +256, +512, +1024,
        +1536, +2048, +2560, +3072, +3584, +4096, +4608, +5120, +5632,
        +6144, +6656, +7168, +7680, +8192, +8704, +9216, +9728, +10240,
        +10752, +11264, +11776, +12288, +12800, +13312, +13824, +14336,
        +14848, +15360, +15872, +16384, +16896, +17408, +17920, +18432,
        +18944, +19456, +19968, +20480, +20992, +21504, +22016, +22528,
        +23040, +23552, +24064, +24576, +25088, +25600, +26112, +26624,
        +27136, +27648, +28160, +28672, +29184, +29696, +30208, +30720,
        +31232, +31744, +32256,      0,    512,   1024,   1536,   2048,   
          2560,   3072,   3584,   4096,   4608,   5120,   5632,   6144,  
          6656,   7168,   7680,   8192,   8704,   9216,   9728,  10240,  
         10752,  11264,  11776,  12288,  12800,  13312,  13824,  14336,  
         14848,  15360,  15872,  16384,  16896,  17408,  17920,  18432,  
         18944,  19456,  19968,  20480,  20992,  21504,  22016,  22528,  
         23040,  23552,  24064,  24576,  25088,  25600,  26112,  26624,  
         27136,  27648,  28160,  28672,  29184,  29696,  30208,  30720,  
         31232,  31744,  32256, 12312341 };
    set<int> leftset;
    for (long i = 0; i < 73+64+12+111111; ++i) {

        if(pos[i] != 12312341) leftset.insert(pos[i] );
        else break;
    }
    cout << endl << endl;
    cout << "SETSIZE: " << leftset.size() << endl;

    cout << endl << endl;
    for( auto it = leftset.begin(); it != leftset.end(); ++it)
        cout  << *it << ",  ";
    cout << endl;





    filtermatrix = new double[64*512]();
    for (long idx = 0; idx < 10; ++idx) {
        for (long col = 0; col < 512; ++col)
            if( (col-idx+1)%10 == 0 )
                filtermatrix[idx * 512 + col] = 1;
    }
    
    for (long idx = 0; idx < 10; ++idx) {
        for (long col = 0; col < 20; ++col)
            cout << setw(2) << filtermatrix[idx * 512 + col] ;
        cout << endl;
    }


    int pppos[] = { +0, +1, +2, 3, +4, 5, 6, 7, +8, 9, 10, +16, +32, +64, +128, +256, +512, +1024,
            +1536, +2048, +2560, +3072, +3584, +4096, +4608, +5120, +5632,
            +6144, +6656, +7168, +7680, +8192, +8704, +9216, +9728, +10240,
            +10752, +11264, +11776, +12288, +12800, +13312, +13824, +14336,
            +14848, +15360, +15872, +16384, +16896, +17408, +17920, +18432,
            +18944, +19456, +19968, +20480, +20992, +21504, +22016, +22528,
            +23040, +23552, +24064, +24576, +25088, +25600, +26112, +26624,
            +27136, +27648, +28160, +28672, +29184, +29696, +30208, +30720,
            +31232, +31744, +32256  };
    
    cout << endl << endl << endl;   
    set<int> rightset(pppos, pppos+79);
    set<int> resutset;
    for (auto it = rightset.begin(); it != rightset.end(); ++it){
        auto item = log(*it)/log(2);
        if( item - int(item) > 1e-8 ) {
            resutset.insert( *it );
            cout << *it << "\t";
        }
    }
    cout << endl;
    cout << endl;
    cout << endl;

    for (auto it = resutset.begin(); it != resutset.end(); ++it)
        cout << *it << ",\t";
    cout << endl << endl << resutset.size() << endl;




    cout << endl;
    cout << endl;
    cout << endl;
    cout << endl;
    cout << endl;
    cout << string("The ") + to_string( 4 ) << string("-th iteration begins ... ")  << endl;

}



                