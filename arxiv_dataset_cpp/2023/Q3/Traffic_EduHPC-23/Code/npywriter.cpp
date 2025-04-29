/// @file npywriter.cpp
/// @brief details in npywriter.h

#include "npywriter.h"
#include <iostream>
#include <sstream>

namespace npy {

/// @brief internal routine to create an npy header for a 2d array. Assumes little endian integers.
std::string npy_header(int length, int maxsequence=1000000000, int fixlen=0)
{
    // properties are a python dictionary
    std::stringstream dict;
    dict << "{'descr': '<i" << sizeof(int) << "', 'fortran_order': False, "
         << "'shape': (" << maxsequence << ',' << length << "), }\n";
    // must pad to have header (incl. newline) be a multiple of 16 characters long
    int preheadlen = 10;
    int headlen = preheadlen + dict.str().length() + 1; // one for the newline below
    int add2len = (fixlen?fixlen:int((headlen+15)/16)*16) - headlen;
    if (add2len < 0) {
        std::cerr << "NPY HEADER CREATION ERROR\n";
        return ""; // signals an error
    }
    dict << std::string(add2len, ' ') << '\n'; // the newline is part of the format!
    // build the header
    std::string prehead("\x93NUMPY\x01\0\0\0", preheadlen);
    int dictlen = dict.tellp();
    prehead[8] = char(dictlen%256);
    prehead[9] = char(dictlen/256);
    return prehead + dict.str();
}

/// @brief Open npy files for writing positions and velocities on a linear grid.
Writer writer_open(std::string filename, int length)
{
    Writer result;
    result.npydens  = std::make_unique<std::ofstream>(filename+"-dens.npy");
    result.npyvelo  = std::make_unique<std::ofstream>(filename+"-velo.npy");
    result.npytime  = std::make_unique<std::ofstream>(filename+"-time.npy");
    result.length = length;
    result.sequencenumber = 0;
    std::string npy2dheader = npy_header(length);
    std::string npy1dheader = npy_header(1);
    result.npydensheadersize = result.npyveloheadersize = npy2dheader.length(); 
    result.npytimeheadersize = npy1dheader.length();
    result.npydens->write((char*)npy2dheader.data(),result.npydensheadersize);
    result.npyvelo->write((char*)npy2dheader.data(),result.npyveloheadersize);
    result.npytime->write((char*)npy1dheader.data(),result.npytimeheadersize);
    return result;
}
    
/// @brief Write time snapshot t of the positions and velocities to files.
void writer_add(Writer& wrtr, int t, const std::vector<int>& x, const std::vector<int>& v)
{
    int L = wrtr.length;
    int N = x.size();

    // create the density field of the grid
    std::vector<int> density(L,0);
    for (int y: x) 
        density[y%L]++;
    wrtr.npydens->write((char*)(density.data()), L*sizeof(int));

    // create the velocity on the grid
    const int UNSETMARK = -1;
    std::vector<int> velocity(L,UNSETMARK);
    for (int i=0; i<N; i++)  
        velocity[x[i]%L] = v[i];
    // on points without cars, take the velocity of the car ahead
    for (int i=0; i<L; i++)  
        if (velocity[i] == UNSETMARK) 
            for (int j = 1; j<L; j++) 
                if (velocity[(i+j)%L] != UNSETMARK) {
                    velocity[i] = velocity[(i+j)%L];
                    break;
                }
    wrtr.npyvelo->write((char*)(velocity.data()), L*sizeof(int));

    // write time value
    wrtr.npytime->write((char*)(&t), sizeof(t));

    wrtr.sequencenumber++;
}

/// @brief Close the npy files and update headers
void writer_close(Writer& wrtr)
{
    // correct headers and close files
    std::string npy2dheader = npy_header(wrtr.length, wrtr.sequencenumber, wrtr.npydensheadersize);
    wrtr.npydens->seekp(0, std::ios::beg);
    wrtr.npydens->write((char*)npy2dheader.data(), wrtr.npydensheadersize);
    wrtr.npydens->seekp(0, std::ios::end);   
    wrtr.npydens.reset(nullptr);

    wrtr.npyvelo->seekp(0, std::ios::beg);
    wrtr.npyvelo->write((char*)npy2dheader.data(), wrtr.npyveloheadersize);
    wrtr.npyvelo->seekp(0, std::ios::end);
    wrtr.npyvelo.reset(nullptr);
    
    std::string npy1dheader = npy_header(1, wrtr.sequencenumber, wrtr.npytimeheadersize);
    wrtr.npytime->seekp(0, std::ios::beg);
    wrtr.npytime->write((char*)npy1dheader.data(), wrtr.npytimeheadersize);
    wrtr.npytime->seekp(0, std::ios::end);
    wrtr.npytime.reset(nullptr);
}

}
