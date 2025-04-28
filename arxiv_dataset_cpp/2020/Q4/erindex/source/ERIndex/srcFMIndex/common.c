#include <stdio.h>
#include "common.h"

int Verbose;             /* produce verbose output */

//Questi erano tutti commentati: provo a decommentarli per compilare compr_main.c
int Type_compression;    /* indicates type of compression adopted */
int Is_dictionary;       /* a dictionary has to be processed */
int Is_URL;              /* a dictionary of URLS has to be processed */
int Is_huffword;         /* compression output of 7-bit huffword */
int Bucket_size_lev1;    /* size of superbucket (multiple of 1K) */
int Bucket_size_lev2;    /* size of bucket (multiple of 1K & divides _lev1) */
int Mtf_save;            /* # of copied MTF element */
int Num_bucs_lev1;       /* overall number of superbuckets */
int Num_bucs_lev2;       /* overall number of buckets */
int Start_prologue_info_sb; // starting byte of info superbuckets
int Start_prologue_info_b;  // starting byte of info buckets
int Start_prologue_occ;     // starting byte of char occurrences
//// int Retrieved_occ;          // Number of retrieved explicit pos
double Marked_char_freq;    // maximum frequency of the marked char


/* ======= global variables for I/O ====== */
FILE *Infile;            /* input file */
FILE *Outfile;           /* output file */
int Infile_size;         /* size of input file */
int  Outfile_size;       /* size of the output file */

FILE *Infile_head;            /* header file of a dictionary */
FILE *Infile_dict;            /* dictionary file */


/* ======= global variables for file management ====== */
uchar *File_start;  // byte where mapped file starts in memory
uchar *File_end;    // byte where mapped file starts in memory
uchar *File_pos;    // byte where read/write are currently positioned


int Type_mem_ops;   // It may assume one of the three values below

int my_getc_macro_tmp;
uint32 t_macro;
