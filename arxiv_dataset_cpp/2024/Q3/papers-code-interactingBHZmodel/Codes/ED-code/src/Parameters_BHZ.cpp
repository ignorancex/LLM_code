#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "Parameters_BHZ.hpp"

using namespace std;

void Parameters_BHZ::Initialize(string input_file){

        string PBC_X_string,PBC_Y_string,SBC_X_string;
        string pbc_x_out, pbc_y_out, sbc_x_out;

        cout<<"-------------------------------------------"<<endl;
        cout<<"Reading the input file = "<<input_file<<endl;
        cout<<"-------------------------------------------"<<endl;

        PBC_X_string=matchstring2(input_file,"PBC_X");
        if(PBC_X_string=="True"){
                PBC_X=true;
                pbc_x_out="PBC";
        }
        else{
                PBC_X=false;
                pbc_x_out="OBC";
        }

        PBC_Y_string=matchstring2(input_file,"PBC_Y");
        if(PBC_Y_string=="True"){
                PBC_Y=true;
                pbc_y_out="PBC";
        }
        else{
                PBC_Y=false;
                pbc_y_out="OBC";
        }

        SBC_X_string=matchstring2(input_file,"SBC_X");
        if(SBC_X_string=="True"){
                SBC_X=true;
                sbc_x_out="ON";
        }
        else{
                SBC_X=false;
                sbc_x_out="OFF";
        }

        Lx = int(matchstring(input_file, "Cells_X"));
        Ly = int(matchstring(input_file, "Cells_Y"));
        N_Orbs = int(matchstring(input_file, "Total_Orbs"));
        Wx = int(matchstring(input_file, "Smoothness_Width"));

        Fill = matchstring(input_file, "Filling");

        Total_Cells = Lx*Ly;
        Total_Sites = N_Orbs*Total_Cells;
        Ham_Size = 2*Total_Sites;

        Total_Particles=Fill*Ham_Size;

        cout<<"Total size of the Hamiltonian = "<<Ham_Size<<"x"<<Ham_Size<<endl;
        cout<<"Total number of unit cells = "<<Total_Cells<<endl;
        cout<<"Total number of particles = "<<Total_Particles<<endl;
        cout<<"Boundary conditions along-x = "<<pbc_x_out<<endl;
        cout<<"Boundary conditions along-y = "<<pbc_y_out<<endl;
        cout<<"Smooth boundary along-x = "<<sbc_x_out<<endl;

        A_val = matchstring(input_file, "Orbital_Mixing_A");
        B_val = matchstring(input_file, "Intra_Orbital_Parameter_B");
        M_val = matchstring(input_file, "Gap_Parameter_M");
        Vo_val = matchstring(input_file, "Confining_Potential_Vo");

        Temp = matchstring(input_file, "Temperature");
        beta = 1.0/(1.0*Temp);

        cout<<"(A, B, M) = ("<<A_val<<" , "<<B_val<<" , "<<M_val<<")"<<endl;
        cout<<"Smooth Edge Parameters: (Wx , Vo) = ("<<Wx<<" , "<<Vo_val<<")"<<endl;

        w_min = matchstring(input_file, "Min_w");
        w_max = matchstring(input_file, "Max_w");

        dw_dos = matchstring(input_file, "Width_dw");
        eta_dos = matchstring(input_file, "Broadening_eta");


        cout<<"-------------------------------------------"<<endl;
        cout<<"Finish Reading the input file"<<endl;
        cout<<"-------------------------------------------"<<endl;

}


double Parameters_BHZ::matchstring(string file, string match){

        string test;
        string line;
        ifstream inputfile(file);
        double amount;

        bool pass=false;

        while(getline(inputfile,line)){
                istringstream iss(line);

                if(getline(iss,test,'=') && pass == false){
                        if(iss >> amount && test == match){
                                pass = true;
                        }
                        else{pass == false;}
                        if(pass)
                                break;
                }
        }

        if(pass == false){
                string errorout = match;
                errorout += "= the argument is missing in the input file!";
                throw std::invalid_argument(errorout);
        }
   //     cout<< match <<" = "<< amount <<endl;
        return amount;
}


string Parameters_BHZ::matchstring2(string file, string match){
        string line;
        ifstream inputfile(file);
        string amount;
        int offset;

        if(inputfile.is_open()){
                while(!inputfile.eof()){

                        getline(inputfile, line);

                        /* Referenced from https://cplusplus.com/forum/beginner/121556/ 
                         * https://stackoverflow.com/questions/12463750/ */

                        if((offset = line.find(match,0)) != string::npos){
                                amount = line.substr(offset + match.length() + 1);
                        }

                }
                inputfile.close();
        }
        else{cout << "Unable to open input file while in the Parameter class." << endl;}

        cout << match << " = " << amount << endl;
        return amount;
}
