#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "TerminalOperation.h"

void GenLaTeXCompileCommand(char* command, char* pdfcreator, char* filename)
{   
	char* emptyspaces = " ";
	strcat(command, pdfcreator);
	strcat(command, emptyspaces);
	strcat(command, filename);
}

void ClearAuxiliaryLaTeXFile()
{
	#ifdef WINVER
	   printf("Windows OS, delete *.aux and *.log files.\n");
	   system("del *.aux");
	   system("del *.log");
	#elif  __unix  // Unix, Linux, Apple, Sun SPARC, Sun x86, AIX, ... 
	   printf("Unix-like OS, remove *.aux and *.log files.\n");
	   system("rm *.aux"); 
	   system("rm *.log"); 
	#else
	   printf("Unknown OS. Do nothing!\n");
	#endif
}
