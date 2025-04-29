/*
 * ZernikeSymExpr.c
 * 
 * Copyright 2022 Hong-Yan ZHANG <hongyan@hainnu.edu.cn>
 * 
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "ZernikeSymExpr.h"

Type Abs(Type x){ return x > 0 ? x:(-x); }

int  CvtNM2K(DoubleIndex pair)
{
	int k = (pair.n- Abs(pair.m))/2;	 
	return k;
}

SingleIndex  CvtNM2J(DoubleIndex pair)
{
	SingleIndex j;
	if( pair.m >= 0){
		j = (pair.n*(pair.n+1))/2 + pair.m + 1;
    }else{
        j =  (pair.n*(pair.n+1))/2 - pair.m;
    }
   return j;	
}

DoubleIndex CvtJ2NM(SingleIndex j)
{
	DoubleIndex pair;
	pair.n = (unsigned int) ceil((-3+ sqrt(8*j+1))/2);
	int r = j - (pair.n*(pair.n + 1))/2;
	if( (pair.n % 2) == 0){
		if( (r % 2) == 0 ){
		    pair.m = -r;
	    }else{
			pair.m = r - 1;
		}		
	}else{
		if( (r % 2) == 0 ){
		    pair.m = r - 1;
	    }else{
			pair.m = -r;
		}			
	}
	
	return pair;	
}

double CalcBinomCoef(double x, unsigned int i)
{
	if( i == 0 ){ return 1;}
	double prod = 1;
	for(int t = 0; t <= i-1; t++){
		prod *= (x-t)/(i-t);
	}
	return prod; 
}


void CalcRadiPolynomPowerCoef(int* ptr2power, int* ptr2coef, DoubleIndex pair)
{    
	int k = CvtNM2K(pair);

	int sign = 1;
	for(int s = 0; s <= k; s++){
		ptr2coef[s] = sign*CalcBinomCoef(pair.n - s, k)*CalcBinomCoef(k,s);
		ptr2power[s] = pair.n - 2*s;	
		sign = -sign;	
	}
}	

void CalcRadiPolynomPower(int* ptr2power, DoubleIndex pair)
{
	unsigned k = CvtNM2K(pair);

	for(int s = 0; s <= k; s++){
		ptr2power[s] = pair.n - 2*s;	
	}
}

void CalcRadiPolynomCoef(int* ptr2coef, DoubleIndex pair)
{
	unsigned k = CvtNM2K(pair);

	int sign = 1;
	for(int s = 0; s <= k; s++){
		ptr2coef[s] = sign*CalcBinomCoef(pair.n - s, k)*CalcBinomCoef(k,s);
		sign = -sign;	
	}
}


void GenSexprCatSign(FILE* fp, Type coef, int i)
{
    if(coef < 0){
        fprintf(fp, "-");
    }else{
        if( i != 0) fprintf(fp, "+"); // not the first term
    }
}

void GenSexprUsgnGeneralTerm(FILE*fp, char* var, char* op, Type coef, int power)
{
	int ucoef = Abs(coef);
	switch(power){
        case 1:  
			if(ucoef != 1){
				fprintf(fp, "%d%s", ucoef, var);
			}else{
                fprintf(fp, "%s", var);
			} 
			break;
        case 0:  fprintf(fp, "%d", ucoef); break;
        default: 
			if(ucoef != 1){
				fprintf(fp, "%d%s%s{%d}", ucoef, var, op, power);
			}else{
				fprintf(fp, "%s%s{%d}", var, op, power);
			}
    }
}

void GenSexprGeneralTerm(FILE* fp, int i, char* var, char* op, Type coef, int power)
{
    GenSexprCatSign(fp, coef, i);
    GenSexprUsgnGeneralTerm(fp, var, op, coef, power); 
}


void GenSexprPolynom(FILE* fp, char* var, char* op, Type* coef, int* power, int size)
{
    for(int i = 0; i < size; i++){
       if(coef[i] == 0) continue; // just ignore the term since it is zero
       GenSexprGeneralTerm(fp, i, var, op, coef[i], power[i]);
       fprintf(fp, " ");
    }
}

void GenSexprRadiPolynom(FILE* fp, DoubleIndex pair, char* var, char* op)
{
    int size = 1 + CvtNM2K(pair);
    int*  ptr2power = (int*)malloc(sizeof(int)*size);
    Type* ptr2coef  = (Type*)malloc(sizeof(Type)*size);
    CalcRadiPolynomPowerCoef(ptr2power, ptr2coef, pair);
	GenSexprPolynom(fp, var, op, ptr2coef, ptr2power, size);
    free(ptr2power);
    free(ptr2coef);
}

void GenSexprRadiNormCoef(FILE* fp, DoubleIndex pair)
{
	if( pair.m == 0 )
		fprintf(fp, "\\sqrt{%d}", pair.n+1);
    else
		fprintf(fp, "\\sqrt{%d}", 2*(pair.n+1));
}

void GenSexprAnguFun(FILE* fp, DoubleIndex pair)
{
	SingleIndex j = CvtNM2J(pair);
	pair.m = Abs(pair.m);
	if( j%2 == 0){ // even Zernike function, angular function is "cos"
	    switch(pair.m){
			case 0:  
				if (j == 0) fprintf(fp, "1"); 	
				break;
			case 1:  
				fprintf(fp, "\\cos(\\theta)");  
				break;
			default: 
				fprintf(fp, "\\cos(%d\\theta)", pair.m); 
				break;
		}
	}else{         // odd Zernike function, angular function is "sin"
	    switch(pair.m){
			case 0:  // nothing to be done
				break;
			case 1:  
				fprintf(fp, "\\sin(\\theta)"); 
				break;
			default: 
				fprintf(fp, "\\sin(%d\\theta)", pair.m); 
				break;
		}
	}
}


void GenSexprZernikePolynomNM(FILE* fp, DoubleIndex pair)
{	
	unsigned int size = 1 + CvtNM2K(pair);	
	char* var = "\\rho"; // for LaTeX
	char* op  = "^";     // for LaTeX
	if( (pair.m == 0) || (size == 1)){
		GenSexprRadiPolynom(fp, pair, var, op);
	}else{
		fprintf(fp,"(");
		GenSexprRadiPolynom(fp, pair, var, op);
	    fprintf(fp, ")");
	}
	GenSexprAnguFun(fp, pair);
}

void GenSexprZernikePolynomJ(FILE* fp, SingleIndex j)
{
	DoubleIndex pair = CvtJ2NM(j);
    GenSexprZernikePolynomNM(fp, pair);
}

/***********************************************************************
 * Procedures for LaTeX long table environment        
 * ************** BEGIN ***********************************************/
 
void SetLtabNewline(FILE* fp){ 
	fprintf(fp, "\\\\"); fprintf(fp, "\n"); 
}

void SetLtabHline(FILE* fp){ 
	fprintf(fp, "\\hline\n"); 
}

void SetLtabBeginCenter(FILE* fp){ 
	fprintf(fp, "\\begin{center}\n"); 
}

void SetLtabBeginLtable(FILE* fp, TableInfor* ptr2tab){ 	
	fprintf(fp, "\\begin{longtable}{%s}\n", ptr2tab->alignctrl); 
}

void SetLtabCaption(FILE* fp, TableInfor* ptr2tab){ 
	fprintf(fp, "\\caption{%s}", ptr2tab->caption); 
	SetLtabNewline(fp); 
}

void SetLtabCaptionContinue(FILE* fp, TableInfor* ptr2tab){
	fprintf(fp, "\\caption{%s%s}", ptr2tab->caption, " (continued)"); 
	SetLtabNewline(fp);
}

void SetLtabTableHead(FILE* fp, TableInfor* ptr2tab)
{
    fprintf(fp, "%s", ptr2tab->attribname[0]);
    for(int i = 1; i < ptr2tab->ncols; i++){
		fprintf(fp," & %s", ptr2tab->attribname[i]);
	}
    SetLtabNewline(fp);
}

void SetLtabEndFirstHead(FILE* fp){ 
	fprintf(fp, "\\endfirsthead\n"); 
}

void SetLtabEndHead(FILE* fp){ 
	fprintf(fp, "\\endhead\n"); 
}

void SetLtabEndFoot(FILE* fp){ 
	fprintf(fp, "\\endfoot\n"); 
}

void SetLtabEndLastFoot(FILE* fp){ 
	fprintf(fp, "\\endlastfoot\n"); 
}

void SetLtabEndLtable(FILE* fp){ 
	fprintf(fp, "\\end{longtable}\n"); 
}    

void SetLtabEndCenter(FILE* fp){ 
	fprintf(fp, "\\end{center}\n"); 
}

/* This function depends on the concrete problem to be solved 
 * it can be implemented with polymorphism
 *    dynamic polymorphism: virtual function with different implementations
 *    static  polymorphism: overiding a function with different arguments 
 * */
void GenLaTeXTableLine(FILE*fp, SingleIndex j)
{
	unsigned n;
	int m;
	DoubleIndex pair = CvtJ2NM(j);
	
	// print the single and double indices j, n, and m one by one  
	fprintf(fp, " $%d$  & $%d$  & $%d$", j, pair.n, pair.m);
	
	// print the normalizaiton coefficient
	fprintf(fp, "  &$");
	GenSexprRadiNormCoef(fp, pair); 
	fprintf(fp, "$");
	
	// print the Zernike Circular Polynomial
	fprintf(fp, "  &$"); 
	GenSexprZernikePolynomNM(fp, pair); 
	fprintf(fp, "$");

	SetLtabNewline(fp);
}






/* The framework of this function is generic for for many problems*/  
void GenLaTeXLtable(FILE* fp, TableInfor* ptr2tab, int rowbegin, int rowend)
{
    SetLtabBeginCenter(fp);                               
    SetLtabBeginLtable(fp, ptr2tab);          
    SetLtabCaption(fp, ptr2tab);                        
    SetLtabHline(fp);                                     
    SetLtabTableHead(fp, ptr2tab);                 
    SetLtabHline(fp);                                     
    SetLtabEndFirstHead(fp);                              
    SetLtabCaptionContinue(fp, ptr2tab); 
    SetLtabHline(fp);                                     
    SetLtabTableHead(fp, ptr2tab);              
    SetLtabHline(fp);                                     
    SetLtabEndHead(fp);                                   
    SetLtabHline(fp);                                     
    SetLtabEndFoot(fp);                                   
    SetLtabHline(fp);                                     
    SetLtabEndLastFoot(fp);                               
    
    for(int idx = rowbegin; idx <= rowend; ++idx){
        GenLaTeXTableLine(fp, idx);   // depnds on concrete problem
    }

    SetLtabHline(fp);                              
    SetLtabEndLtable(fp);                       
    SetLtabEndCenter(fp); 
}



void SetLaTeXDocCalss(FILE* fp)
{
    fprintf(fp, "\\documentclass[11pt,a4paper]{article}\n\n");
}

/* The preamble depends on the concrete problem to tbe solved */
void SetLaTeXDocPreamble(FILE* fp, LayoutInfor *ptr2layout)
{
    fprintf(fp, "\\usepackage{amsmath,amsfonts,amssymb}\n");
    fprintf(fp, "\\usepackage[%s,left=%s,right=%s,top=%s,bottom=%s]{geometry}\n",
	            ptr2layout->orient,
	            ptr2layout->left,
	            ptr2layout->right,
	            ptr2layout->top,
	            ptr2layout->bottom);	              
	
	// The following operations can be modified according to the very problem
    fprintf(fp, "\\usepackage{longtable}\n\n");    
    fprintf(fp, "\\DeclareMathOperator{\\Zern}{Z}\n");
    fprintf(fp, "\\DeclareMathOperator{\\Radi}{R}\n");
    fprintf(fp, "\\newcommand{\\Zernpoly}[2]{\\Zern_{#1}^{#2}}\n");
    fprintf(fp, "\\newcommand{\\Radipoly}[2]{\\Radi_{#1}^{#2}}\n\n");
    
    
    // The following operations can be modified according to the very problem and the programmer
    fprintf(fp, "\\title{Table of Zernike Circular Polynomials}\n");
    fprintf(fp, "\\author{Hong-Yan Zhang, hongyan@hainnu.edu.cn, \\\\");
    fprintf(fp, "Hainan Normal University, Haikou 571158, China}\n");
}


void GenKeyContentsInTeXFile(FILE* fp, TableInfor* ptr2tab, int rowbegin, int rowend)
{
	fprintf(fp, "\\begin{document}\n\n");
	fprintf(fp, "\\maketitle\n\n");
	fprintf(fp, "\n");
	
	GenLaTeXLtable(fp, ptr2tab, rowbegin, rowend);	
	
	fprintf(fp, "\n");
	fprintf(fp, "\\end{document}\n");
}

/* If you code with C++/Python/Octave/MATLAB, it will be much simpler */
int CalcFilenameLength(char* prefix, char* str4jmin, char* str4jmax)
{
	return strlen(prefix) + strlen(str4jmin) + strlen("-") 
	       + strlen(str4jmax) + strlen(".tex");
}

/* If you code with C++/Python/Octave/MATLAB, it will be much simpler */
void GenLaTeXFileName(char* filename, char* prefix, char* str4jmin, char* str4jmax)
{
	char* symboldash = "-";
	char* suffix = ".tex";
	strcat(filename,prefix);
	strcat(filename,str4jmin);
	strcat(filename,symboldash);
	strcat(filename,str4jmax); 
	strcat(filename,suffix);
}

void GenLaTeXFile(char* filename, TableInfor* ptr2tab, LayoutInfor* ptr2layout, int rowbegin, int rowiend)
{
    FILE* fp = fopen(filename, "w");
    SetLaTeXDocCalss(fp);
    SetLaTeXDocPreamble(fp, ptr2layout);
    GenKeyContentsInTeXFile(fp, ptr2tab, rowbegin, rowiend);
    fclose(fp);
}

