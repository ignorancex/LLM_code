#include <iostream>
#include <math.h>
#include <time.h>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <assert.h>
#include "tensor.h"
#include <algorithm>
using namespace std;

int main(int argc, char** argv){
	int Lx,Ly;
	Lx=atoi(argv[1]);
	Ly=atoi(argv[2]);

	string head("lattice_geometry_for_");
	string mid("x");
	string base(".tex");

	string x;

	stringstream ss1,ss2;
	ss1 << Lx;
	ss2 << Ly;

	string x_str=ss1.str();
	string y_str=ss2.str();

	string file_out_tex(head + x_str + mid + y_str +base); 
	ofstream file_out(file_out_tex.c_str());

	file_out<<"\\documentclass[tikz,border=0.1pt]{standalone}"<<endl;
	file_out<<"\\usepackage{tikz}"<<endl;
	file_out<<"\\usetikzlibrary{shapes}"<<endl;
	file_out<<"\\usetikzlibrary{positioning,arrows.meta}"<<endl;
	file_out<<"\\usepackage[utf8x]{inputenc}"<<endl;
	file_out<<"\\usepackage{graphicx}"<<endl;
	file_out<<""<<endl;
	file_out<<""<<endl;
	file_out<<"\\definecolor{light green}{RGB}{204,255,153}"<<endl;
	file_out<<"\\definecolor{light red}{RGB}{255,102,102}"<<endl;
	//file_out<<"\\newcommand*\\scalenodetext[2][.5]{\\pgfmathsetmacro\\revscale{1/.5}\\scalebox{#1}{\\parbox{\\revscale\\linewidth}{#2}}}"<<endl;

	file_out<<""<<endl;
	file_out<<"\\begin{document}"<<endl;
	file_out<<"\\begin{tikzpicture}"<<endl;
	file_out<<""<<endl;

	double x1,y1,x2,y2,a,ap;
	double xa,xb,ya,yb;
	x1=0.0;y1=0.0;x2=0.0;y2=0.0;
	xa=0.0;xb=0.0;ya=0.0;yb=0.0;
	a=1.0;ap=0.95*a;

	for(int x=0; x<Lx; x++){
		for(int y=0; y<Ly;  y++){
			x1=a*x;		y1=a*y;

			if(y==0){
				file_out<<"\\draw[-, densely dotted, black]"<<"("<<x1-0.05<<","<<0.05<<") to [bend right,looseness=1.3,out=10,in=-190]("<<x1-0.1<<","<<1.0*(Ly-1)-0.1<<");"<<endl;
			}

			if(x!=Lx-1){
				file_out<<"\\draw[line width=0.15 mm, double, black] "<<"("<<x1+0.085<<","<<y1<<")"<<" -- "<<"("<<x1+a-0.085<<","<<y1<<")"<<";"<<endl;
			}
			if(y!=Ly-1){
				file_out<<"\\draw[line width=0.15 mm, double, black] "<<"("<<x1<<","<<y1+0.085<<")"<<" -- "<<"("<<x1<<","<<y1+a-0.085<<")"<<";"<<endl;
			}

			file_out<<"\\filldraw[fill=red, draw=red]" <<"("<<x1-0.05<<","<<y1-0.05<<")"<<" circle ("<<0.2<<"mm);"<<endl;
			file_out<<"\\filldraw[fill=blue, draw=blue]" <<"("<<x1+0.05-0.02<<","<<y1+0.05-0.02<<")"<<" rectangle ("<<x1+0.05+0.02<<","<<y1+0.05+0.02<<");"<<endl;
			file_out<<"\\draw[rotate around={-45:("<<1.0*x1<<","<<1.0*y1<<")},magenta] ("<<1.0*x1<<","<<1.0*y1<<") ellipse (0.7mm and 1.4mm);"<<endl;

			if(x==Lx-1 && y==0){
				file_out<<"\\node [below] at"<<" ("<<x1+a<<","<<y1+0.25*a<<") "<<"{\\scalebox{0.7}{$x$}};"<<endl;
				file_out<<"\\node [below] at"<<" ("<<x1+0.6*a<<","<<y1+0.75*a<<") "<<"{\\scalebox{0.7}{$y$}};"<<endl;
				file_out<<"\\draw [arrows={-Triangle[width=0.9mm]}, black,line width=0.1mm] "<<"("<<x1+0.5*a<<","<<y1<<")"<<" -- "<<"("<<x1+1.5*a<<","<<y1<<");"<<endl;
				file_out<<"\\draw [arrows={-Triangle[width=0.9mm]}, black,line width=0.1mm] "<<"("<<x1+0.5*a<<","<<y1<<")"<<" -- "<<"("<<x1+0.5*a<<","<<y1+a<<");"<<endl;

				file_out<<"\\filldraw[fill=red, draw=red]" <<"("<<x1+0.5*a<<","<<y1+1.5*a<<")"<<" circle ("<<0.5<<"mm);"<<endl;
				file_out<<"\\filldraw[fill=blue, draw=blue]" <<"("<<x1+0.5*a-0.04<<","<<y1+1.7*a-0.04<<")"<<" rectangle ("<<x1+0.5*a+0.04<<","<<y1+1.7*a+0.04<<");"<<endl;

				file_out<<"\\node [right,red] at"<<" ("<<x1+0.5*a<<","<<y1+1.49*a<<") "<<"{\\scalebox{0.7}{$\\rightarrow s$}};"<<endl;
				file_out<<"\\node [right,blue] at"<<" ("<<x1+0.5*a<<","<<y1+1.69*a<<") "<<"{\\scalebox{0.7}{$\\rightarrow p$}};"<<endl;

			}


			if(x==1){
				if(y==1){

					file_out<<"\\draw [arrows={-Triangle[width=0.9mm]}, cyan] "<<"("<<x1+0.05<<","<<y1+0.15<<")"<<" -- "<<"("<<x1+ap<<","<<y1+0.15<<");"<<endl;
					file_out<<"\\draw [arrows={-Triangle[width=0.9mm]}, cyan] "<<"("<<x1+0.15<<","<<y1+0.05<<")"<<" -- "<<"("<<x1+0.15<<","<<y1+ap<<");"<<endl;

					file_out<<"\\node[cyan] at"<<" ("<<x1+ap-0.09<<","<<y1+0.35<<") "<<"{\\scalebox{0.7}{$\\hat{a}_x$}};"<<endl;
					file_out<<"\\node[cyan] at"<<" ("<<x1+0.35<<","<<y1+ap-0.1<<") "<<"{\\scalebox{0.7}{$\\hat{a}_y$}};"<<endl;
				}
			}
		}
	}




	file_out<<"\\end{tikzpicture}"<<endl;
	file_out<<"\\end{document}"<<endl;


	complex<double> comp,comp_inv;
	comp.real(2.0);
	comp.imag(1.0);

	comp_inv=(1.0)/comp;
	cout<<"comp_inverse="<<comp_inv<<endl;

	return 0;
}

