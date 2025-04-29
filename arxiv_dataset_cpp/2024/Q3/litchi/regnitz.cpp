/*
 * This file is part of litchi, a lightweight C++ library
 * for Minkowski analysis
 * 
 * Copyright (C) 2021-2023 Caroline Collischon <caroline.collischon@fau.de>
 * 
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */


#include "litchi_eat.hpp"

#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <cmath>
#include <type_traits>
#include <chrono>



const double pi = 3.14159265358979;


using namespace std;

int main(int argc,char **argv)
{
    string inname = "../litchi/COM_CMB_IQU-smica_2048_R3.00_hm1.fits", outname = "./testmap.fits";
    
    paramStruct params{};
    
    
    vector<string> arguments(argv + 1, argv + argc);    
    for(uint i=0; i<arguments.size(); i++)
    {
        string thisArg = arguments.at(i);
        if(thisArg=="--infile" || thisArg=="-i")
        {
            inname = arguments.at(++i);
        }
        else if(thisArg=="--outfile" || thisArg=="-o")
        {
            outname = arguments.at(++i);
        }
        else if(thisArg=="--rankA" || thisArg=="-A")
        {
            int number = stoi(arguments.at(++i));
            if(number>=0) params.rankA = uint(number);
            else
            {
                std::cerr << "Illegal value after rankA: " << number << " , must be positive integer or zero \n";
                return 1;
            }
        }
        else if (thisArg=="--rankB" || thisArg=="-B")
        {
            int number = stoi(arguments.at(++i));
            if(number>=0) params.rankB = uint(number);
            else
            {
                std::cerr << "Illegal value after rankB: " << number << " , must be positive integer or zero \n";
                return 1;
            }
        }
        else if (thisArg=="-l") //treat l as rankB for use in irreducible case
        {
            int number = stoi(arguments.at(++i));
            if(number>=0) {
                params.rankB = uint(number);
                params.curvIndex = 1;
            }
            else
            {
                std::cerr << "Illegal value after l: " << number << " , must be positive integer or zero \n";
                return 1;
            }
        }
        else if (thisArg=="--curvI" || thisArg=="-C" || thisArg=="-c")
        {
            int number = stoi(arguments.at(++i));
            if(number>=0 && number<=2) params.curvIndex = uint(number);
            else
            {
                std::cerr << "Illegal value after curvI: " << number << " , must be 0, 1, or 2 \n";
                return 1;
            }
        }
        else if (thisArg=="--smooth")
        {
            int number = stoi(arguments.at(++i));
            if((number&(number-1))==0) params.smooth = uint(number);
            else
            {
                std::cerr << "Illegal value after smooth: " << number << " , must be power of two or zero \n";
                return 1;
            }
        }
        else if (thisArg=="--smoothRad" || thisArg=="-s")
        {
            double number = stod(arguments.at(++i));
            params.smoothRad = number;
        }
        else if (thisArg=="--nside_out" || thisArg=="--NsideOut" || thisArg=="--Nside_out" || thisArg=="--no")
        {
            int number = stoi(arguments.at(++i));
            if((number&(number-1))==0) params.NsideOut = uint(number);
            else
            {
                std::cerr << "Illegal value after NsideOut: " << number << " , must be power of two or zero \n";
                return 1;
            }
        }
        else if (thisArg=="--mint")
        {
            params.mint = stod(arguments.at(++i));
        }
        else if (thisArg=="--maxt")
        {
            params.maxt = stod(arguments.at(++i));
        }
        else if (thisArg=="--numt")
        {
            int number = stoi(arguments.at(++i));
            if(number>=1) params.numt = uint(number);
            else
            {
                std::cerr << "Illegal value after numt: " << number << " , must be positive integer \n";
                return 1;
            }
        }
        else if (thisArg=="--nside" || thisArg=="--Nside")
        {
            int number = stoi(arguments.at(++i));
            if( (number&(number-1))==0 ) params.Nside = uint(number);
            else
            {
                std::cerr << "Illegal value after nside: " << number << " , must be power of two \n";
                return 1;
            }
        }
        else if (thisArg=="--linThresh" || thisArg=="--linthresh")
        {
            params.linThresh = true;
        }
        else if (thisArg=="--logThresh" || thisArg=="--logthresh")
        {
            params.linThresh = false;
        }
        else if (thisArg=="--forceOutname" || thisArg=="--forceoutname")
        {
            params.forceOutname = true;
        }
        else if (thisArg=="--trace")
        {
            params.function = "tr";
        }
        else if (thisArg=="--EVDir" || thisArg=="--evdir" || thisArg=="--evd" || thisArg=="--cartDir" || thisArg=="--cd")
        {
            params.function = "evd";
        }
        else if (thisArg=="--EVquotient" || thisArg=="--evquotient" || thisArg=="--evq" || thisArg=="--EVQuo" || thisArg=="--cartAniso" || thisArg=="--ca")
        {
            params.function = "evq";
        }
        else if (thisArg=="--irrAniso" || thisArg=="--anisoIrr" || thisArg=="--ia")
        {
            params.function = "irrAniso";
        }
        else if (thisArg=="--irrDir" || thisArg=="--dirIrr" || thisArg=="--id")
        {
            params.function = "irrDir";
        }
        else if (thisArg=="--sequence")
        {
            params.sequence = true;
        }
        else if (thisArg=="--mask" || thisArg=="-m")
        {
            params.maskname = arguments.at(++i);
        }
        else if (thisArg=="--maskThresh")
        {
            params.maskThresh = stod(arguments.at(++i));
        }
        else
        {
            std::cerr << "Illegal argument: " << thisArg << "\n";
            return 1;
        }
    }
    
    
    auto start = chrono::high_resolution_clock::now();
    
    makeMinkmap(inname, params, outname);
    
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout << "Execution took " << duration.count()/1000. << " seconds" << endl;
    
    return 0;
}
