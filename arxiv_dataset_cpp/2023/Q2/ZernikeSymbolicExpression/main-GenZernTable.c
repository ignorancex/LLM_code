/*
 * main-GenZernTable.c
 * 
 * Copyright 2022 Hong-Yan ZHANG <hongyan@hainnu.edu.cn>
 * 
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


#include "ZernikeSymExpr.h"
#include "TerminalOperation.h"


void GenLaTeXCompileCommand(char* command, char* pdfcreator, char* filename);

void ClearAuxiliaryLaTeXFile();

/***********************************************************************
 * Compile at termninal: 
 *       gcc ZernikeSymExpr.c main*.c Terminal*.c -o GenTable -lm 
 * Run at terminal: 
      with Unix-like: ./GenTable 13 18
      with Windows:     GenTable.exe 13 18
 * Run at Unix-like terminal: ./GenTable 1 120
 * Run at Unix-like terminal: ./GenTable 1 210
 * ********************************************************************/
 
int main(int argc, char* argv[])
{
	SingleIndex jmin = atoi(argv[1]);
	SingleIndex jmax = atoi(argv[2]);
	
	char* prefix = "TableZernikePolynom-";	
	char* str4jmin = argv[1];
	char* str4jmax = argv[2];
	
	if( jmax > 465 ){
		printf("jmax is large, it is cut off by replacing with 465.\n");
		jmax = 465; // use the min(465, jmax)
		str4jmax = "465";
	    //Note that it is useless for large jmax in applications
	}
	
	int   filenamelen = CalcFilenameLength(prefix, str4jmin, str4jmax);
	char* filename = malloc((filenamelen + 1)*sizeof(char)); 
    // The memory returned by malloc() is usually not zeroed.
    *filename = '\0'; // initialization
	GenLaTeXFileName(filename, prefix, str4jmin, str4jmax);
	
	
	LayoutInfor* ptr2layout = (LayoutInfor*) malloc(5*sizeof(char*));
	if(jmax > 120){
		ptr2layout->orient = "landscape";
	}else{
		ptr2layout->orient = "portrait";
	}
	ptr2layout->left   = "2cm";
	ptr2layout->right  = "2cm";
	ptr2layout->top    = "2cm";
	ptr2layout->bottom = "2cm";	

	
	int ncols = 5;
	TableInfor* ptr2tab = (TableInfor*) malloc(sizeof(TableInfor) + sizeof(int) + (ncols+2)*sizeof(char*));
	ptr2tab->ncols = ncols;
	ptr2tab->alignctrl = "ccrcp{0.75\\textwidth}";
	ptr2tab->caption = "Zernike Circular Polynomials $\\Zern_j(\\rho,\\theta)$";
	ptr2tab->attribname[0] = "$j$";
	ptr2tab->attribname[1] = "$n$",
	ptr2tab->attribname[2] = "$m$",
	ptr2tab->attribname[3] = "$N^m_n$",
	ptr2tab->attribname[4] = "ZEMAX $\\Zern_j(\\rho,\\theta)=\\Zernpoly{n}{m}(\\rho,\\theta)=\\Radipoly{n}{m}(\\rho)\\Theta_m(\\theta)$";
	
	
	/*******************************************************************
	 * Create the latex file for generating Table of Zernike Polynomials
	 *   Note that if jmax > 120, the output pdf file will be set in the
	 * landscape mode instead of the portrait mode.
	 * ****************************************************************/
    GenLaTeXFile(filename, ptr2tab, ptr2layout, jmin, jmax);
    
    
    /*******************************************************************
     * If the LaTeX is installed in your OS, then run the terminal to
     * generate the *.pdf file for the table of the Zernike circular 
     * polynomials with the range of order jmin <= j <= jmax.
     * Otherwise, please remove them by commenting them.  
     *********************BEGIN****************************************/
    char* pdfcreator = "xelatex"; // "pdflatex" is OK
    int   len_command = strlen(pdfcreator) + strlen(" ") + filenamelen; 
    char* command = malloc((len_command)*sizeof(char));
    // The memory returned by malloc() is usually not zeroed.
    *command = '\0';  //initialization
    GenLaTeXCompileCommand(command, pdfcreator, filename);
    // If you comment the line `*command = '\0';`, you can just print 
    // the command, and you will find there are some strange charactors
    // at the begining of command. Another way to avoid this is use `strcpy`
    // instead of `strcat` at line 10 in `TerminalOperation.c`.
    printf("%s\n", command);
    system(command);  // first  compiling: xelatex  *.tex
    system(command);  // second compiling: xelatex  *.tex
    //system("evince *.pdf"); // if "evince" is available, just try it!    
    free(command);    
    ClearAuxiliaryLaTeXFile();
    /********************* END ****************************************/
 
    free(filename);
    free(ptr2tab);
    free(ptr2layout);
    	
	return 0;	
}


