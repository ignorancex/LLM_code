/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton implementation for Bison's Yacc-like parsers in C

   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301, USA.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.3"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Using locations.  */
#define YYLSP_NEEDED 0



/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     NUM = 258,
     SFNCT = 259,
     BOGUS = 260,
     LVAR = 261,
     VAR = 262,
     FNCT = 263,
     IDN = 264,
     CROSS = 265,
     FLUXP = 266,
     FLUXM = 267,
     NUFLUX = 268,
     EFT_FLUX_COEFF_FILE = 269,
     EFT_XSEC_COEFF_FILE = 270,
     SYS_ON_FUNCTION = 271,
     SYS_OFF_FUNCTION = 272,
     SYS_MULTIEX_ERRORS = 273,
     GRP = 274,
     GID = 275,
     FNAME = 276,
     VERS = 277,
     SIGNAL = 278,
     BG = 279,
     ENERGY = 280,
     CHANNEL = 281,
     NDEF = 282,
     GRPOPEN = 283,
     GRPCLOSE = 284,
     PM = 285,
     FLAVOR = 286,
     NOGLOBES = 287,
     RULESEP = 288,
     RULEMULT = 289,
     NAME = 290,
     RDF = 291,
     ENDEXP = 292,
     ENDDET = 293,
     NEG = 294
   };
#endif
/* Tokens.  */
#define NUM 258
#define SFNCT 259
#define BOGUS 260
#define LVAR 261
#define VAR 262
#define FNCT 263
#define IDN 264
#define CROSS 265
#define FLUXP 266
#define FLUXM 267
#define NUFLUX 268
#define EFT_FLUX_COEFF_FILE 269
#define EFT_XSEC_COEFF_FILE 270
#define SYS_ON_FUNCTION 271
#define SYS_OFF_FUNCTION 272
#define SYS_MULTIEX_ERRORS 273
#define GRP 274
#define GID 275
#define FNAME 276
#define VERS 277
#define SIGNAL 278
#define BG 279
#define ENERGY 280
#define CHANNEL 281
#define NDEF 282
#define GRPOPEN 283
#define GRPCLOSE 284
#define PM 285
#define FLAVOR 286
#define NOGLOBES 287
#define RULESEP 288
#define RULEMULT 289
#define NAME 290
#define RDF 291
#define ENDEXP 292
#define ENDDET 293
#define NEG 294




/* Copy the first part of user declarations.  */
#line 22 "glb_parser.y"

#if HAVE_CONFIG_H   /* config.h should come before any other includes */
#  include "config.h"
#endif



#define YYDEBUG 1
//int yydebug = 1; /* Uncomment this line to get debug output from the parser */
#include <math.h>
#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <globes/globes.h>
#include <gsl/gsl_spline.h>
#include "glb_smear.h"
#include "glb_multiex.h"
#include "glb_types.h"
#include "glb_error.h"
#include "glb_lexer.h"
#include "glb_parser_type.h"
#include "glb_fluxes.h"
#include "glb_sys.h"
#include "glb_minimize.h"

#define TWICE 100

#define UNTYPE -1
#define INT 0
#define DOUBLE 1
#define INT_LIST 2
#define DOUBLE_LIST 3
#define DOUBLE_LIST_INDEXED 4
#define INT_LIST_INDEXED 5
#define ENERGY_MATRIX 6
#define INT_INDEXED 7
//#define INT_INDEXED_PAIR 8
#define DOUBLE_INDEXED 9
#define DOUBLE_INDEXED_PAIR 10
#define DOUBLE_INDEXED_PAIR_INV 11
#define CHAR 12
#define FUN 13
#define DOUBLE_LIST_INDEXED_SL 14
#define DOUBLE_LIST_INDEXED_BL 15
#define COUNTER 16
#define DOUBLE_PAIR 17
#define INT_PAIR 18
#define GMAX 1E100

  static int exp_count=1;
  static int energy_len;
  static int loc_count;
  static int energy_count=-1;
  static int cross_count=-1;
  static int flux_count=-1;
  static struct glb_experiment buff;
  static struct glb_experiment *buff_list[GLB_MAX_EXP];
  static glb_smear ibf;
  static glb_option_type opt;
  static glb_flux flt;
  static glb_xsec xsc;
  static glb_nuisance nuis;
  static int errordim_sys_on=-1;         /* Temp. storage for the old numerical errordims */
  static int errordim_sys_off=-1;
  static char *context;

  int yyerror (const char *s, ...);           /* Forward declaration to suppress compiler warning */


  /* The name and symbol tables: a chain of `struct glb_symrec'.  */
  static glb_namerec *name_table = (glb_namerec *) NULL;
  glb_symrec *sym_table = (glb_symrec *) NULL;
  glb_symrec *pre_sym_table = (glb_symrec *) NULL;
          /* cannot use static here, since its declared earlier as extern */

 typedef struct
  {
    char *token; // the string which is used as lhs in the ini-file
    int scalar; // data type flag
    double rl;
    double ru; // allowed range
    void *ptr; // this is a pointer the corresponding part of the exp structure
    void *len; /* this is a pointer which points to the length of the vector
                *  in a struct glb_experiment. Example: if we parse densitytab,
                *  this things points
                *  to psteps
                */

    char *ctx; /* here the type of the environment is given, e.g. rule
                * thus the corresponding token is matched onyl within
                * a rule type environment
                */
  } glb_parser_decl;


  /* IMPORTANT NOTE: Token names that are substrings of other token names
     (e.g. @error and @error_list) should appear *after* that other token */
  static glb_parser_decl token_list[]={
    {"$version",CHAR,0,1E8,&buff.version,NULL,"global"},
    {"$citation",CHAR,0,1E8,&buff.citation,NULL,"global"},
    {"$parent_energy",DOUBLE,0,GMAX,&buff.emax,NULL,"global"},
    {"$oscillation_engine",CHAR,0,1E8,&buff.osc_engine.name,NULL,"global"},
    {"$target_mass",DOUBLE,0,GMAX,&buff.targetmass,NULL,"global"},
#ifdef GLB_OLD_AEDL
    {"$simbins" ,COUNTER,0,500,&buff.simbins,NULL,"global"},
    {"$simtresh" ,DOUBLE,0,GMAX,&buff.simtresh,NULL,"global"},
    {"$simbeam" ,DOUBLE,0,GMAX,&buff.simbeam,NULL,"global"},
    {"$numofbins",COUNTER,0,500,&buff.numofbins,NULL,"global"},
    {"$simbinsize",DOUBLE_LIST,0,GMAX,&buff.simbinsize,
     &buff.simbins,"global"},
/* JK - Has been removed
    {"$errorfunction",INT,0,1,&buff.errorfunction,NULL,"global"},
    {"@treshold_setttings" ,DOUBLE_INDEXED_PAIR,
     0,100,&buff.bgtcenter[0],&loc_count,"rule"},
    {"@treshold_error" ,DOUBLE_INDEXED_PAIR,
     0,100,&buff.bgterror[0],&loc_count,"rule"}, */
#endif /* GLB_OLD_AEDL */
    {"$sampling_points" ,COUNTER,0,500,&buff.simbins,NULL,"global"},
    {"$sampling_min" ,DOUBLE,0,GMAX,&buff.simtresh,NULL,"global"},
    {"$sampling_max" ,DOUBLE,0,GMAX,&buff.simbeam,NULL,"global"},
    /* FIXME -- this was not properly recognized due to the fact the first match is performed
     * by matching only the letters up to the length of the word ins this list, ie.
     * if 'bins' is before 'binsize' in this list, 'binsize' will be matched as
     * 'bins'
     */
    {"$binsize",DOUBLE_LIST,0,GMAX,&buff.binsize,
     &buff.numofbins,"global"},

    {"$bins",COUNTER,0,500,&buff.numofbins,NULL,"global"},
    {"$emin",DOUBLE,0,GMAX,&buff.emin,NULL,"global"},
    {"$emax",DOUBLE,0,GMAX,&buff.emax,NULL,"global"},

    {"$baseline",DOUBLE,0,2*GLB_EARTH_RADIUS,&buff.baseline,NULL,"global"},
    {"$profiletype",INT,1,3,&buff.density_profile_type,NULL,"global"},

    {"$sampling_stepsize",DOUBLE_LIST,0,GMAX,&buff.simbinsize,
     &buff.simbins,"global"},

   {"$densitytab",DOUBLE_LIST,0,GMAX,&buff.densitytab,&buff.psteps,"global"},
   {"$lengthtab",DOUBLE_LIST,0,2*GLB_EARTH_RADIUS,
    &buff.lengthtab,&buff.psteps,"global"},
   {"rulechannellist",INT_LIST_INDEXED,0,GMAX,&buff.rulechannellist[0],
    &buff.lengthofrules[0],"rule"},
   {"rulescoeff",DOUBLE_LIST_INDEXED,0,GMAX,&buff.rulescoeff[0],
    &buff.lengthofrules[0],"rule"},

   {"bgrulechannellist",INT_LIST_INDEXED,0,GMAX,&buff.bgrulechannellist[0],
    &buff.lengthofbgrules[0],"rule"},
   {"bgrulescoeff",DOUBLE_LIST_INDEXED,0,GMAX,&buff.bgrulescoeff[0],
    &buff.lengthofbgrules[0],"rule"},
   {"rule",UNTYPE,-1,1,NULL,&buff.numofrules,"global"},
   {"@signal",UNTYPE,-1,1,NULL,&loc_count,"rule"},
   {"@background",UNTYPE,-1,1,NULL,&loc_count,"rule"},

    {"@pre_smearing_efficiencies",DOUBLE_LIST_INDEXED,0,GMAX,
     &buff.user_pre_smearing_channel[0],&loc_count,"channel"},

    {"@pre_smearing_background",DOUBLE_LIST_INDEXED,0,GMAX,
     &buff.user_pre_smearing_background[0],&loc_count,"channel"},

    {"@post_smearing_efficiencies",DOUBLE_LIST_INDEXED,0,GMAX,
     &buff.user_post_smearing_channel[0],&loc_count,"channel"},

    {"@post_smearing_background",DOUBLE_LIST_INDEXED,0,GMAX,
     &buff.user_post_smearing_background[0],&loc_count,"channel"},


   {"@errordim_sys_on",INT,0,20,&errordim_sys_on,NULL,"rule"},
   {"@errordim_sys_off",INT,0,20,&errordim_sys_off,NULL,"rule"},
   {"@data_flag",INT_INDEXED,0,1,&buff.data_on_off[0],&loc_count,"rule"},
   {"@data",DOUBLE_LIST_INDEXED,-GMAX,GMAX,&buff.data[0],&loc_count,"rule"},

   {"@sys_on_function",CHAR,0,20,&buff.sys_on_strings[0],&loc_count,"rule"},
   {"@sys_off_function",CHAR,0,20,&buff.sys_off_strings[0],&loc_count,"rule"},

   {"channel",UNTYPE,-1,1,NULL,&buff.numofchannels,"global"},
   {"@channel",INT_LIST_INDEXED,-1,1,&buff.listofchannels[0],
    &buff.numofchannels,"channel"},
   {"energy",UNTYPE,-1,1,NULL,&buff.num_of_sm,"global"},
   {"@energy",ENERGY_MATRIX,-1,GMAX,&buff.smear[0],&loc_count,"energy"},

   {"@signalerror",DOUBLE_INDEXED_PAIR,
    0,100,&buff.signal_errors[0],&loc_count,"rule"},
   {"@backgrounderror",DOUBLE_INDEXED_PAIR,
    0,100,&buff.bg_errors[0],&loc_count,"rule"},
   {"@backgroundcenter",DOUBLE_INDEXED_PAIR,
    0,100,&buff.bg_centers[0],&loc_count,"rule"},

   {"@sys_on_errors",DOUBLE_LIST_INDEXED,0,GMAX,
    &buff.sys_on_errors[0],&loc_count,"rule"},
   {"@sys_off_errors",DOUBLE_LIST_INDEXED,0,GMAX,
    &buff.sys_off_errors[0],&loc_count,"rule"},
   {"@sys_on_multiex_errors_sig",  ENERGY_MATRIX, 1, GLB_MAX_NUISANCE,
     &buff.sys_on_multiex_errors_sig[0], &loc_count, "rule"},
   {"@sys_on_multiex_errors_bg",   ENERGY_MATRIX, 1, GLB_MAX_NUISANCE,
     &buff.sys_on_multiex_errors_bg[0], &loc_count, "rule"},
   {"@sys_off_multiex_errors_sig", ENERGY_MATRIX, 1, GLB_MAX_NUISANCE,
     &buff.sys_off_multiex_errors_sig[0], &loc_count, "rule"},
   {"@sys_off_multiex_errors_bg",  ENERGY_MATRIX, 1, GLB_MAX_NUISANCE,
     &buff.sys_off_multiex_errors_bg[0], &loc_count, "rule"},


   {"sys", UNTYPE, 0, 20, NULL, &buff.n_nuisance, "global"},
   {"@energy_list", DOUBLE_LIST, 0, GMAX, &nuis.energy_list, &nuis.n_energies, "sys"},
   {"@error_list",  DOUBLE_LIST, 0, GMAX, &nuis.error_list,  &nuis.n_energies, "sys"},
   {"@error",       DOUBLE,      0, GMAX, &nuis.error,       NULL,             "sys"}, 
   {"@systype",     INT,         0, 20, &nuis.systype,       NULL,             "sys"},

   {"@energy_window" ,DOUBLE_INDEXED_PAIR_INV,0,GMAX,&buff.energy_window[0],
    &loc_count,"rule"},
   {"$densitysteps",COUNTER,1,GMAX,&buff.psteps,NULL,"global"},

   {"$filter_state",INT,GLB_OFF,GLB_ON,&buff.filter_state,NULL,"global"},
   {"$filter_value",DOUBLE,0,GMAX,&buff.filter_value,NULL,"global"},

   {"cross",UNTYPE,0,20,NULL,&buff.num_of_xsecs,"global"},
   {"@cross_file",CHAR,0,20,&xsc.file_name,NULL,"cross"},
#ifdef GLB_EFT
   {"@eft_coeff_file",CHAR,0,20,&xsc.eft_coeff_file,NULL,"cross"},
   {"@quark_flavors",INT_PAIR,0,2,&xsc.q,NULL,"cross"},
#endif

   {"flux",UNTYPE,0,20,NULL,&buff.num_of_fluxes,"global"},
   {"@flux_file",CHAR,0,20,&flt.file_name,NULL,"flux"},

   {"@builtin",INT,1,4,&flt.builtin,NULL,"flux"},
   {"@time",DOUBLE,0,GMAX,&flt.time,NULL,"flux"},
   {"@power",DOUBLE,0,GMAX,&flt.target_power,NULL,"flux"},
   {"@stored_muons",DOUBLE,0,GMAX,&flt.stored_muons,NULL,"flux"},
   {"@parent_energy",DOUBLE,0,GMAX,&flt.parent_energy,NULL,"flux"},
   {"@norm",DOUBLE,0,GMAX,&flt.norm,NULL,"flux"},
   {"@gamma",DOUBLE,0,GMAX,&flt.gamma,NULL,"flux"},
   {"@end_point",DOUBLE,0,GMAX,&flt.end_point,NULL,"flux"},
   {"@stored_ions",DOUBLE,0,GMAX,&flt.stored_muons,NULL,"flux"},

   {"nuflux",UNTYPE,0,20,NULL,&buff.num_of_fluxes,"global"},
   {"@flux_file",CHAR,0,20,&flt.file_name,NULL,"nuflux"},

   {"@builtin",INT,1,4,&flt.builtin,NULL,"nuflux"},
   {"@time",DOUBLE,0,GMAX,&flt.time,NULL,"nuflux"},
   {"@power",DOUBLE,0,GMAX,&flt.target_power,NULL,"nuflux"},
   {"@stored_muons",DOUBLE,0,GMAX,&flt.stored_muons,NULL,"nuflux"},
   {"@parent_energy",DOUBLE,0,GMAX,&flt.parent_energy,NULL,"nuflux"},
   {"@norm",DOUBLE,0,GMAX,&flt.norm,NULL,"nuflux"},
   {"@gamma",DOUBLE,0,GMAX,&flt.gamma,NULL,"nuflux"},
   {"@end_point",DOUBLE,0,GMAX,&flt.end_point,NULL,"nuflux"},
   {"@stored_ions",DOUBLE,0,GMAX,&flt.stored_muons,NULL,"nuflux"},
#ifdef GLB_EFT
   {"@eft_coeff_file",CHAR,0,20,&flt.eft_coeff_file,NULL,"nuflux"},
   {"@quark_flavors",INT_PAIR,0,2,&flt.q,NULL,"nuflux"},
#endif

   {"@type" ,INT,1,2,&ibf.type,&loc_count,"energy"},
   {"@sigma_e" ,DOUBLE_LIST,0,GMAX,&ibf.sigma,&ibf.num_of_params,"energy"},
   {"@sigma_function" ,FUN,0,1000,&ibf.sig_f,NULL,"energy"},



   {NULL,UNTYPE,0,0,NULL,NULL,"global"}
};


static void grp_start(char* name, int id) // id = index of the group among groups of same type
   {
     if(strncmp(name,"rule",4)==0 )
       {
         /* Reset variables to default values */
         errordim_sys_on  = -1.0;
         errordim_sys_off = -1.0;
       }
   }


static void grp_end(char* name, int id) // id = index of the group among groups of same type
   {
     if (strncmp(name,"energy",6)==0 )
       {
         if(id-1 >= 0)
           {
             ibf.options=glb_option_type_alloc();
             ibf.options=(glb_option_type *) memmove(ibf.options,&opt,
                                             sizeof(glb_option_type));

             if (buff.smear_data[id-1] != NULL)
             {
               glb_smear_free(buff.smear_data[id-1]);
               buff.smear_data[id-1] = NULL;
             }
             buff.smear_data[id-1] = glb_smear_alloc();
             buff.smear_data[id-1] = glb_copy_smear(buff.smear_data[id-1],&ibf);
             glb_option_type_free(ibf.options);
             glb_option_type_reset(&opt);
             if(ibf.sigma!=NULL) glb_free(ibf.sigma);
             glb_smear_reset(&ibf);
           }
       }

     if (strncmp(name,"flux",4)==0 )
       {
         if(id > 0)
           {
             glb_error("The 'flux' directive is deprecated (consider using 'nuflux').\n"
                       "The flux normalization may not be what you expect.\n"
                       "Please, consult the manual!");
             if(flt.builtin<=0) flt.builtin=GLB_OLD_NORM;

             if (buff.fluxes[id-1] != NULL)
             { 
               glb_free_flux(buff.fluxes[id-1]);
               buff.fluxes[id-1] = NULL;
             }
             buff.fluxes[id-1] = glb_malloc(sizeof(glb_flux));
             memset(buff.fluxes[id-1], 0, sizeof(*buff.fluxes[0]));
             glb_reset_flux(buff.fluxes[id-1]);
             if (glb_copy_flux(buff.fluxes[id-1], &flt) != GLB_SUCCESS)
               glb_error("grp_end: Error copying flux data");
             glb_reset_flux(&flt);
           }
       }


     if (strncmp(name,"nuflux",6)==0 )
       {
         if(id > 0)
           {
             if (buff.fluxes[id-1] != NULL)
             { 
               glb_free_flux(buff.fluxes[id-1]);
               buff.fluxes[id-1] = NULL;
             }
             buff.fluxes[id-1] = glb_malloc(sizeof(glb_flux));
             memset(buff.fluxes[id-1], 0, sizeof(*buff.fluxes[0]));
             glb_reset_flux(buff.fluxes[id-1]);
             if (glb_copy_flux(buff.fluxes[id-1], &flt) != GLB_SUCCESS)
               glb_error("grp_end: Error copying flux data");
             glb_reset_flux(&flt);
           }
       }

     if (strncmp(name,"cross",5)==0 )
       {
         if(id > 0)
           {
             if (buff.xsecs[id-1] != NULL)
             {
               glb_free_xsec(buff.xsecs[id-1]);
               buff.xsecs[id-1] = NULL;
             }
             buff.xsecs[id-1] = glb_malloc(sizeof(glb_xsec));
             memset(buff.xsecs[id-1], 0, sizeof(*buff.xsecs[0]));
             glb_reset_xsec(buff.xsecs[id-1]);
             if (glb_copy_xsec(buff.xsecs[id-1], &xsc) != GLB_SUCCESS)
               glb_error("grp_end: Error copying cross section data");
             glb_reset_xsec(&xsc);
           }
       }



     if (strncmp(name,"rule",4)==0 )
       {
         int nr = id - 1;

         /* Parse old (numerical) errordims */
         if (buff.sys_on_strings[nr] == NULL  &&  errordim_sys_on >= 0)
           buff.sys_on_strings[nr] = glbConvertErrorDim(errordim_sys_on);
         if (buff.sys_off_strings[nr] == NULL  &&  errordim_sys_off >= 0)
           buff.sys_off_strings[nr] = glbConvertErrorDim(errordim_sys_off);
       }

     if (strncmp(name,"sys",3) == 0)
     {
       if (nuis.name)
       {
         glb_free(nuis.name);
         nuis.name = NULL;
       }
       nuis.name = strdup(name_table->name);
       if (!buff.nuisance_params[id-1])
         buff.nuisance_params[id-1] = glb_alloc_nuisance();
       glb_nuisance *n = buff.nuisance_params[id-1];
       if (n)  memcpy(n, &nuis, sizeof(glb_nuisance));
       glbResetNuisance();
     }

     glb_free(context);
     context = (char *) strdup("global");

   }

static int set_channel_data(int x[6],int loc_count)
   {
     // I am sorry for this -- it's a kludge
     int i;

     for(i=0;i<6;i++) {
       if (loc_count >= buff.numofchannels) /* Don't realloc when parsing a re-definition */
         buff.listofchannels[i]=
                        (int*) glb_realloc(buff.listofchannels[i]
                                        ,sizeof(int)*loc_count);

       buff.listofchannels[i][loc_count-1]=x[i];
     }


     return 0;
   }


static int step_counter(char *name)
{
  int i=0;
  int *ibf;
  ibf=NULL;
  for(i=0;token_list[i].token!=NULL;i++)
    {
      if(strncmp(name,token_list[i].token,
                 strlen(token_list[i].token))==0 )
        {

                     ibf=(int*) token_list[i].len;
                     if(*ibf==-1) *ibf=1;// first time encounter
                     else (*ibf)++;

        }
    }
  if(ibf!=NULL) return *ibf; // otherwise a SEGFAULT occurs when there
  // is no matching name
  return 0;
}

static int set_fnct(char *name,void *in)
{
  int i;
  sigfun *dbf;


  for(i=0;token_list[i].token!=NULL;i++)
    {
      if(strncmp(name,token_list[i].token,
                 strlen(token_list[i].token))==0&&
         strncmp(context,token_list[i].ctx,
                 strlen(token_list[i].ctx))==0 )
        {
             if(token_list[i].scalar==FUN) //double
               {
                 dbf=(sigfun *) token_list[i].ptr;
                 *dbf=(sigfun) in;
                 return 0;
               }
             else
               {
                 fprintf(stderr,"Error: Value for %s out of range\n",
                         token_list[i].token);
                 return 2;
               }
        }


    }


  return 1;
}


static int set_exp(char *name,double value,int scalar)
{
  int i;
  double *dbf;
  int *ibf;
  int *xibf;
  for(i=0;token_list[i].token!=NULL;i++)
    {
      if(strncmp(name,token_list[i].token,
                 strlen(token_list[i].token))==0 &&
         strncmp(context,token_list[i].ctx,
                 strlen(token_list[i].ctx))==0
         )
        {
             if(token_list[i].scalar==DOUBLE) //double
               {
                 if(value >= token_list[i].rl && value <= token_list[i].ru)
                   {
                     dbf=(double*) token_list[i].ptr;
                     *dbf=value;
                     return 0;
                   }
                 else
                   {
                     fprintf(stderr,"Error: Value for %s out of range\n",
                             token_list[i].token);
                     return 2;
                   }
               }
             if(token_list[i].scalar==INT) //int
               {
                 if(value >= token_list[i].rl && value <= token_list[i].ru)
                   {
                     ibf=(int*) token_list[i].ptr;
                     *ibf=value;
                     return 0;
                   }
                 else
                   {
                     fprintf(stderr,"Error: Value for %s out of range\n",
                             token_list[i].token);
                     return 2;
                   }
               }

             if(token_list[i].scalar==COUNTER) //counter
               {
                 if(value >= token_list[i].rl && value <= token_list[i].ru)
                   {

                     ibf=(int*) token_list[i].ptr;
/* JK 2012-06-25 Remove this restriction */
/*                     if(!((*ibf == -1) || (*ibf == (int) value))) {
                       glb_warning("Given length does not"
                                   " match actual length");
                       return 2;

                     }*/
                     *ibf=value;
                     return 0;
                   }
                 else
                   {
                     fprintf(stderr,"Error: Value for %s out of range\n",
                             token_list[i].token);
                     return 2;
                   }
               }

             if(token_list[i].scalar==INT_INDEXED) //int
               {
                 if(value >= token_list[i].rl && value <= token_list[i].ru)
                   {
                     xibf=(int*) token_list[i].ptr;
                     xibf[loc_count-1]=(int) value;
                     return 0;
                   }
                 else
                   {
                     fprintf(stderr,"Error: Value for %s out of range\n",
                             token_list[i].token);
                     return 2;
                   }
               }

             if(token_list[i].scalar==DOUBLE_INDEXED) //int
               {
                 if(value >= token_list[i].rl && value <= token_list[i].ru)
                   {
                     dbf=(double*) token_list[i].ptr;
                     dbf[loc_count-1]=(double) value;
                     return 0;
                   }
                 else
                   {
                     fprintf(stderr,"Error: Value for %s out of range\n",
                             token_list[i].token);
                     return 2;
                   }
               }


           }
       }

   return 1;
}

static int set_pair(char *name,double value,double value2,int scalar)
{
  int i;
  double *dbf;
  int *ibf;
  for(i=0;token_list[i].token!=NULL;i++)
    {
      if(strncmp(name,token_list[i].token,
                 strlen(token_list[i].token))==0&&
         strncmp(context,token_list[i].ctx,
                 strlen(token_list[i].ctx))==0 )
        {
             if(token_list[i].scalar==DOUBLE_PAIR)
               {
                 if(value >= token_list[i].rl && value <= token_list[i].ru)
                   {

                     dbf=(double*) token_list[i].ptr;
                     dbf[0]=(double) value;
                     dbf[1]=(double) value2;
                     return 0;
                   }
                 else
                   {
                     fprintf(stderr,"Error: Value for %s out of range\n",
                             token_list[i].token);
                     return 2;
                   }
               }

             if(token_list[i].scalar==INT_PAIR)
               {
                 if(value >= token_list[i].rl && value <= token_list[i].ru)
                   {
                     ibf=(int*) token_list[i].ptr;
                     ibf[0]=(int) value;
                     ibf[1]=(int) value2;
                     return 0;
                   }
                 else
                   {
                     fprintf(stderr,"Error: Value for %s out of range\n",
                             token_list[i].token);
                     return 2;
                   }
               }

             // JK - 2012-05-17 I think the following wouldn't work. It's not used,
             // so I comment it out for the moment
//             if(token_list[i].scalar==INT_INDEXED_PAIR) //int
//               {
//                 if(value >= token_list[i].rl && value <= token_list[i].ru)
//                   {
//
//                     ibf=(int*) token_list[i].ptr;
//                     ibf[(loc_count-1)+0*32]=(int) value;
//                     return 0;
//                   }
//                 else
//                   {
//                     fprintf(stderr,"Error: Value for %s out of range\n",
//                             token_list[i].token);
//                     return 2;
//                   }
//               }

             if(token_list[i].scalar==DOUBLE_INDEXED_PAIR) //int
               {
                 if(value >= token_list[i].rl && value <= token_list[i].ru)
                   {
                     if (strcmp(token_list[i].ctx, "rule") != 0)
                       fprintf(stderr, "Error: DOUBLE_INDEXED_PAIR works only inside "
                                       "a rule environment! Ignoring it.\n");
                     else
                     {
                       int delta = GLB_MAX_RULES;
                       dbf=(double*) token_list[i].ptr;
                       dbf[(loc_count-1)+0*delta]=(double) value;
                       dbf[(loc_count-1)+1*delta]=(double) value2;
                     }
                     return 0;
                   }
                 else
                   {
                     fprintf(stderr,"Error: Value for %s out of range\n",
                             token_list[i].token);
                     return 2;
                   }
               }

             if(token_list[i].scalar==DOUBLE_INDEXED_PAIR_INV) //int
               {
                 if(value >= token_list[i].rl && value <= token_list[i].ru)
                   {

                     dbf=(double*) token_list[i].ptr;
                     dbf[(loc_count-1)*2+0]=(double) value;
                     dbf[(loc_count-1)*2+1]=(double) value2;
                     return 0;
                   }
                 else
                   {
                     fprintf(stderr,"Error: Value for %s out of range\n",
                             token_list[i].token);
                     return 2;
                   }
               }
           }
       }

   return 1;
}


static int set_string(char *name, char *str)
{
  int i;
  for(i=0; token_list[i].token != NULL; i++)
  {
    if (strncmp(name, token_list[i].token, strlen(token_list[i].token)) == 0)
    {
      char **p = (char **)(token_list[i].ptr);
      if (*p)  glb_free(*p);
      *p = strdup(str);
      return 0;
    }
  }
  return 1;
}


static size_t list_length (glb_List *head)
{
  size_t n;
  for (n = 0; head; ++n)
    head = head->next;
  return n;
}


static void list_free(glb_List *stale)
{
 glb_List *ptr;
 glb_List *dummy;
 ptr=stale;
 while(ptr != (glb_List *) NULL)
   {
     dummy=ptr->next;
     glb_free(ptr);
     ptr=dummy;
   }

}


static double glb_reverse(double x)
{
  return x;
}


static glb_List *list_cons (glb_List *tail, double newdata)
{
  glb_List *res = (glb_List*) glb_malloc(sizeof(glb_List));
  res->next=tail;
  res->entry=newdata;
  return res;
}

/* this functions threads a funct_t function (double f (double x))
   over a list */

static double glb_list_copy(double x)
{
  return x;
}

static glb_List *thread_list(func_t f, int reverse, int destroy ,glb_List *tail)
{
  glb_List *res, *head;
  size_t n,l;
  double nv;
  res=NULL;
  /* the problem is that threading reverses the list, hence we
     re-reverse it by default, unless someone wants to reverse himself
  */
  if(reverse==1)
    {
      head=tail;
      for (n = 0; head; ++n)
        {
          nv=f(head->entry);
          res=list_cons(res,nv);
          head = head->next;
        }
    }
  else
    {
      double *rlist,x;
      l=list_length(tail);
      rlist=(double *) malloc(sizeof(double)*l);
      head=tail;
      for (n = 0; head; ++n)
        {
          rlist[n]=head->entry;
          head = head->next;
        }

      for(n=l;n>0; n--)
        {
          x=f(rlist[n-1]);
          res=list_cons(res,x);
        }
      free(rlist);
    }
  if(destroy==1) list_free(tail);
  if(destroy==-1) {list_free(res);res=tail;}
  return res;
}


static void showlist(glb_List *lp)
{

  if (lp){
    showlist(lp->next);             // show the tail
    printf("%.10g " , lp->entry);     // show the head
  }

}

static double list_take(glb_List *li,int k)
{
  int i;
  double erg;
  glb_List *bf;
  bf=li;
  erg=bf->entry;
  for(i=0;i<k;i++)
    {
      bf=bf->next;
      erg=bf->entry;
    }
  return erg;
}


static glb_List *glb_interpolation(glb_List *xval,glb_List *yval,int flag,glb_List *where)
{
  glb_List *head,*res=NULL;
  size_t xl,yl,rl,i;
  double *xlist,*ylist,*rlist;
  gsl_interp_type type;

  if(flag==1) type=*gsl_interp_linear;
  else if (flag==2) type=*gsl_interp_cspline;
  else {glb_error("Invalid interpolation scheme flag");return NULL;}

  xl=list_length(xval);
  yl=list_length(yval);
  rl=list_length(where);

  if(yl!=xl) {glb_error("Xval and Yval in glb_interpolation are not of the same length"); return NULL;}

  xlist=(double*) malloc(sizeof(double)*xl);
  ylist=(double*) malloc(sizeof(double)*yl);
  rlist=(double*) malloc(sizeof(double)*rl);

  gsl_interp_accel *acc = gsl_interp_accel_alloc();
  gsl_spline *spline = gsl_spline_alloc (&type,yl);

  head=xval;
  for (i = yl-1; head; i--){
    xlist[i]=head->entry;
    head = head->next;
  }

  head=yval;
  for (i = yl-1; head; i--){
    ylist[i]=head->entry;
    head = head->next;
  }

  gsl_spline_init(spline,xlist,ylist,yl);

  head=where;
  for(i=0; head; i++){

    /* PH 06/09/16 in going from GSL 1.14 to 1.15 the behavior
     changed: if head->entry is outside the range of xlist a fatal
     error is generated, i.e. no more implicit extrapolation, so we
     have to catch that by hand */

    if(head->entry<xlist[0])
      {
	glb_warning("Linear extrapolation used because x value %f is out of range %f - %f",head->entry,xlist[0],xlist[yl-1]);
	rlist[i]=ylist[0]-(xlist[0]-head->entry)*(ylist[1]-ylist[0])/(xlist[1]-xlist[0]);
      }
    else if(head->entry>xlist[yl-1])
      {
	glb_warning("Linear extrapolation used because x value %f is out of range %f - %f",head->entry,xlist[0],xlist[yl-1]);
	rlist[i]=ylist[yl-1]-(xlist[yl-1]-head->entry)*(ylist[yl-2]-ylist[yl-1])/(xlist[yl-2]-xlist[yl-1]);
      }
    else
      {   
	rlist[i]=gsl_spline_eval(spline,head->entry,acc);
      }
     head=head->next; }

  for(i=rl;i>0;i--) res=list_cons(res,rlist[i-1]);

  free(xlist);
  free(ylist);
  free(rlist);
  gsl_spline_free(spline);
  gsl_interp_accel_free(acc);

  return res;
}


static int set_exp_list(char *name,glb_List *value,int scalar)
{
  int i,k;
  double **dbf;
  int **ibf;
  int *lbf;
  int len;
  double val;
  double *list;
  int *ilist;
  for(i=0;token_list[i].token!=NULL;i++)
    {
      if(strncmp(name,token_list[i].token,strlen(token_list[i].token))==0 &&
         strncmp(context,token_list[i].ctx,
                 strlen(token_list[i].ctx))==0)
        {
          switch((int) token_list[i].scalar) {
          case DOUBLE_LIST:

            //here we will have to do a lot asking asf.
            len=list_length(value); // how long is the list
            lbf=(int*) token_list[i].len;
            if(*lbf==-1) *lbf=len;  // setting the length correctly in exp
/* JK 2012-06-26 Removed to allow for redefinitions of binning
            else if(*lbf!=len) glb_fatal("Line %d: Length mismatch or list"
                                         " length changed", glb_line_num);*/


            dbf = (double**) token_list[i].ptr;
            if(*dbf!=NULL){glb_free(*dbf);*dbf=NULL;}
            list=(double*) glb_malloc(sizeof(double)*len);
            *dbf=list;


            for(k=0;k<len;k++)
               {
                  val=list_take(value,len-k-1);

                  if(val >= token_list[i].rl && val <= token_list[i].ru)
                    {
                      list[k]=val;
                    }
                  else
                    {
                      fprintf(stderr,"Error: Value for %s out of range\n",
                              token_list[i].token);
                      glb_free(list);
                      *dbf=NULL;
                      return 2;
                    }

               }
            if(scalar!=TWICE)  list_free(value);
            return 0;
            break;

          case DOUBLE_LIST_INDEXED:
            len=list_length(value); // how long is the list
            lbf=(int*) token_list[i].len;


            //  lbf[loc_count-1]=len;  // setting the length correctly in exp

            dbf= (double**) token_list[i].ptr;
            if(dbf[loc_count-1]!=NULL){glb_free(dbf[loc_count-1]);dbf[loc_count-1]=NULL;}
            list=(double*) glb_malloc(sizeof(double)*(len+1));

            dbf[loc_count-1]=list;
            list[len]=-1;

            for(k=0;k<len;k++)
              {
                val=list_take(value,len-k-1);

                if(val >= token_list[i].rl && val <= token_list[i].ru)
                  {
                    list[k]=val;

                  }
                else
                  {
                    fprintf(stderr,"Error: In line %d: "
                            "Value for %s out of range\n",
                            glb_line_num,token_list[i].token);
                    glb_free(list);
                    dbf[loc_count-1]=NULL;

                   return 2;
                  }
              }
            if(scalar!=TWICE) list_free(value);
            return 0;
            break;


          case INT_LIST_INDEXED: //integer list indexed
            //with loc_counter

            //here we will have to do a lot asking asf.
            len=list_length(value); // how long is the list
            lbf=(int*) token_list[i].len; //FIXME danger !!!!
            lbf[loc_count-1]=len;  // setting the length correctly in exp
            ibf= (int**) token_list[i].ptr;
            if(ibf[loc_count-1]!=NULL)glb_free(ibf[loc_count-1]);
            ilist=(int*) glb_malloc(sizeof(int)*len);

            ibf[loc_count-1]=ilist;


            for(k=0;k<len;k++)
              {
                val=list_take(value,len-k-1);

                if(val >= token_list[i].rl && val <= token_list[i].ru)
                  {
                    ilist[k]=(int) val;

                  }
                else
                  {
                    fprintf(stderr,"Error: Value for %s out of range\n",
                            token_list[i].token);
                   glb_free(ilist);
                    return 2;
                  }
              }
            if(scalar!=TWICE) list_free(value);
            return 0;
            break;
          default:
            return 1;
            break;
          }
        }
    }

  return 1;
}

static int set_exp_energy(char *name, glb_List **value)
{
  int i,k,l;
  double ***dbf;
  int len;
  double val;
  int v1,v2;
  double **list;

  for(i=0; token_list[i].token != NULL; i++)
  {
    if(strncmp(name,token_list[i].token, strlen(token_list[i].token)) == 0 &&
       strncmp(context,token_list[i].ctx, strlen(token_list[i].ctx)) == 0)
    {
      switch((int) token_list[i].scalar)
      {
        case ENERGY_MATRIX:
          list = (double**) glb_malloc(sizeof(double* ) * (energy_len+1));
          buff.lowrange[loc_count-1] = (int*) glb_malloc((energy_len+1)*sizeof(int));
          buff.uprange[loc_count-1]  = (int*) glb_malloc((energy_len+1)*sizeof(int));

          /* Loop over all analysis bins */
          for(l=0; l < energy_len; l++)
          {
            len = (int) list_length(value[l]); /* how long is the list provided by the user? */
            if (len < 2) {
              fprintf(stderr, "Error: in line %d: in @smear number %d: sublist %d too short!\n",
                              glb_line_num, loc_count, l); return 2;
            }

            dbf = (double***) token_list[i].ptr;
            list[l] = (double*) glb_malloc(sizeof(double)*(len-2+1));
            dbf[loc_count-1] = list;

            v1 = (int) list_take(value[l], len-0-1); /* Sampling point range for this bin */
            v2 = (int) list_take(value[l], len-1-1);

            if(v1 >= 0  && v2 <= buff.simbins  &&
               v2 >= v1 && v2-v1 == len-3)
            {
              buff.lowrange[loc_count-1][l] = v1;
              buff.uprange[loc_count-1][l]  = v2;
            }
            else
            {
              fprintf(stderr,"Error: In line %d: Value for ranges in smear out of range\n",
                      glb_line_num);
              glb_free(list[l]);
              glb_free(buff.lowrange[loc_count-1]);
              glb_free(buff.uprange[loc_count-1]);
              return 2;
            }

            /* Loop over all sampling points contributing to the current analysis bin */
            for(k=0; k < len-2; k++)
            {
              val = list_take(value[l], len-(k+2)-1);
              if (val >= token_list[i].rl && val <= token_list[i].ru)
                list[l][k]=val;
              else
              {
                fprintf(stderr,"Error: In line %d: Value for %s out of range\n",
                        glb_line_num, token_list[i].token);
                free(list[l]);
                return 2;
              }
            } /* for (k) */
            list_free(value[l]);
            value[l] = NULL;

            list[l][len-2] = -1; /* This signals the end of the list */
          } /* for (l) */
          glb_free(value);
          value = NULL;

          list[energy_len] = NULL; /* This signals the end of the list */
          buff.lowrange[loc_count-1][energy_len] = -1;
          buff.uprange[loc_count-1][energy_len] = -1;
          return 0;
          break;
      } /* switch */
    }
  } /* for (i) */
  return 1;
}


static int set_multiex_errors(char *name, glb_List **value)
{
  int i, j, k;
  for(i=0; token_list[i].token != NULL; i++)
  {
      if(strncmp(name, token_list[i].token, strlen(token_list[i].token)) == 0  &&
         strncmp(context,token_list[i].ctx, strlen(token_list[i].ctx)) == 0)
      {
        switch((int) token_list[i].scalar)
        {
          case ENERGY_MATRIX:
            if (energy_len > GLB_MAX_RULES)
            {
              fprintf(stderr, "Error in line %d: @sys_XX_multiex_errors_YY definition too long.\n",
                      glb_line_num);
              return 2;
            }
            for (j=0; j < energy_len; j++)
            {
              int len = (int) list_length(value[j]);
              if (len > GLB_MAX_CHANNELS-1)
              {
                fprintf(stderr, "Error in line %d: Entry %d in @sys_XX_multiex_errors_YY too long.\n",
                        glb_line_num, j+1);
                return 3;
              }
              int *(*x)[GLB_MAX_CHANNELS] = (int *(*)[GLB_MAX_CHANNELS]) token_list[i].ptr;
              x[loc_count-1][j] = glb_malloc(sizeof(int) * (len+1));
              if (len > 0)
              {
                for (k=0; k < len; k++)
                {
                  double val = list_take(value[j], len-k-1);
                  if(val >= token_list[i].rl && val <= token_list[i].ru)
                    x[loc_count-1][j][k] = (int)(val+0.5 - 1);
                      /* -1 because first sys<> name has index 1 in parser */
                  else
                  {
                    fprintf(stderr, "Error in line %d: Value for %s out of range\n",
                            glb_line_num, token_list[i].token);
                    glb_free(x[loc_count-1][j]);
                    x[loc_count-1][j] = NULL;
                    return 4;
                  }
                } /* for (k) */
                x[loc_count-1][j][k] = -1; /* This signals the end of the list */
              }
              else
                x[loc_count-1][j][0] = -1;
 
              list_free(value[j]);
              value[j] = NULL;
            } /* for (j) */

            free(value);
            return 0;
            break;
        }
      }
  } /* for (i) */

  return 1;
}


 

/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* Enabling the token table.  */
#ifndef YYTOKEN_TABLE
# define YYTOKEN_TABLE 0
#endif

#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
#line 1210 "glb_parser.y"
{
  double  val;  /* For returning numbers.                   */
  double *dpt;  /* for rules */
  glb_List *ptr;
  glb_List **ptrq;
  glb_symrec  *tptr;  /* For returning symbol-table pointers      */
  char *name;
  char *iname;
  int in;
  glb_namerec *nameptr;
}
/* Line 193 of yacc.c.  */
#line 1375 "glb_parser.c"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 1388 "glb_parser.c"

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char yytype_int8;
#else
typedef short int yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(e) ((void) (e))
#else
# define YYUSE(e) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
# define YYID(n) (n)
#else
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static int
YYID (int i)
#else
static int
YYID (i)
    int i;
#endif
{
  return i;
}
#endif

#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     ifndef _STDLIB_H
#      define _STDLIB_H 1
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (YYID (0))
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined _STDLIB_H \
       && ! ((defined YYMALLOC || defined malloc) \
	     && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef _STDLIB_H
#    define _STDLIB_H 1
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss;
  YYSTYPE yyvs;
  };

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  YYSIZE_T yyi;				\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (YYID (0))
#  endif
# endif

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack)					\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack, Stack, yysize);				\
	Stack = &yyptr->Stack;						\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (YYID (0))

#endif

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  38
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   330

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  54
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  26
/* YYNRULES -- Number of rules.  */
#define YYNRULES  82
/* YYNRULES -- Number of states.  */
#define YYNSTATES  175

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   294

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
      51,    52,    46,    45,    39,    44,     2,    47,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,    53,
       2,    40,    41,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,    49,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    43,     2,    42,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,    50,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    48
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     5,     8,    10,    12,    14,    16,    18,
      20,    22,    24,    26,    28,    32,    36,    40,    46,    51,
      55,    59,    63,    67,    70,    74,    78,    80,    82,    87,
      91,    95,    98,   102,   106,   110,   115,   120,   124,   126,
     130,   141,   143,   147,   148,   157,   165,   166,   169,   171,
     173,   175,   177,   179,   181,   183,   185,   187,   189,   193,
     197,   201,   205,   209,   213,   227,   229,   231,   233,   235,
     237,   239,   243,   247,   252,   256,   260,   264,   268,   270,
     272,   276,   280
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int8 yyrhs[] =
{
      55,     0,    -1,    56,    -1,    55,    56,    -1,    32,    -1,
      50,    -1,    62,    -1,    57,    -1,    60,    -1,    37,    -1,
      38,    -1,     3,    -1,    35,    -1,     7,    -1,     7,    40,
      57,    -1,     9,    40,    57,    -1,     9,    40,     4,    -1,
       9,    40,    57,    33,    57,    -1,     8,    51,    57,    52,
      -1,    57,    45,    57,    -1,    57,    44,    57,    -1,    57,
      46,    57,    -1,    57,    47,    57,    -1,    44,    57,    -1,
      57,    49,    57,    -1,    51,    57,    52,    -1,    66,    -1,
      27,    -1,     9,    33,    40,     6,    -1,    57,    39,    57,
      -1,    59,    39,    57,    -1,    43,    42,    -1,    43,    59,
      42,    -1,    43,    57,    42,    -1,     9,    40,    60,    -1,
       8,    51,    59,    52,    -1,     8,    51,    60,    52,    -1,
       8,    51,    52,    -1,     6,    -1,     6,    40,    60,    -1,
       8,    51,    60,    39,    60,    39,    57,    39,    60,    52,
      -1,    58,    -1,    57,    34,    57,    -1,    -1,    20,    51,
      35,    52,    63,    28,    64,    29,    -1,    20,    51,    36,
      52,    28,    64,    29,    -1,    -1,    64,    65,    -1,    57,
      -1,    60,    -1,    79,    -1,    76,    -1,    72,    -1,    67,
      -1,    68,    -1,    69,    -1,    70,    -1,    71,    -1,    22,
      40,    21,    -1,    10,    40,    21,    -1,    11,    40,    21,
      -1,    13,    40,    21,    -1,    14,    40,    21,    -1,    15,
      40,    21,    -1,    26,    40,    73,    33,    74,    33,    31,
      33,    31,    33,    73,    33,    73,    -1,    35,    -1,    27,
      -1,    30,    -1,    45,    -1,    44,    -1,    60,    -1,    75,
      33,    60,    -1,    25,    40,    75,    -1,    25,    40,    75,
      53,    -1,    24,    40,    61,    -1,    77,    33,    61,    -1,
      23,    40,    61,    -1,    78,    33,    61,    -1,    77,    -1,
      78,    -1,    16,    40,    21,    -1,    17,    40,    21,    -1,
      18,    40,    75,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,  1271,  1271,  1272,  1273,  1274,  1278,  1279,  1280,  1281,
    1282,  1286,  1287,  1288,  1289,  1290,  1295,  1299,  1304,  1310,
    1311,  1312,  1313,  1314,  1315,  1316,  1317,  1318,  1322,  1331,
    1336,  1340,  1341,  1342,  1343,  1348,  1349,  1350,  1351,  1352,
    1353,  1354,  1358,  1369,  1368,  1379,  1386,  1387,  1391,  1392,
    1393,  1394,  1395,  1396,  1397,  1398,  1399,  1400,  1404,  1414,
    1424,  1436,  1448,  1457,  1466,  1485,  1486,  1490,  1491,  1492,
    1497,  1505,  1519,  1523,  1530,  1539,  1550,  1560,  1571,  1580,
    1589,  1595,  1601
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "NUM", "SFNCT", "BOGUS", "LVAR", "VAR",
  "FNCT", "IDN", "CROSS", "FLUXP", "FLUXM", "NUFLUX",
  "EFT_FLUX_COEFF_FILE", "EFT_XSEC_COEFF_FILE", "SYS_ON_FUNCTION",
  "SYS_OFF_FUNCTION", "SYS_MULTIEX_ERRORS", "GRP", "GID", "FNAME", "VERS",
  "SIGNAL", "BG", "ENERGY", "CHANNEL", "NDEF", "GRPOPEN", "GRPCLOSE", "PM",
  "FLAVOR", "NOGLOBES", "RULESEP", "RULEMULT", "NAME", "RDF", "ENDEXP",
  "ENDDET", "','", "'='", "'>'", "'}'", "'{'", "'-'", "'+'", "'*'", "'/'",
  "NEG", "'^'", "'\\247'", "'('", "')'", "';'", "$accept", "input",
  "topleveldirective", "exp", "listcopy", "seq", "list", "rulepart",
  "group", "@1", "ingroup", "ingroup_statement", "version", "cross",
  "flux", "nuflux", "eft_flux_coeff", "eft_xsec_coeff", "channel", "name",
  "pm", "ene", "energy", "brule", "srule", "rule", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,    44,
      61,    62,   125,   123,    45,    43,    42,    47,   294,    94,
     167,    40,    41,    59
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    54,    55,    55,    55,    55,    56,    56,    56,    56,
      56,    57,    57,    57,    57,    57,    57,    57,    57,    57,
      57,    57,    57,    57,    57,    57,    57,    57,    58,    59,
      59,    60,    60,    60,    60,    60,    60,    60,    60,    60,
      60,    60,    61,    63,    62,    62,    64,    64,    65,    65,
      65,    65,    65,    65,    65,    65,    65,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    73,    74,    74,    74,
      75,    75,    76,    76,    77,    77,    78,    78,    79,    79,
      79,    79,    79
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     2,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     3,     3,     3,     5,     4,     3,
       3,     3,     3,     2,     3,     3,     1,     1,     4,     3,
       3,     2,     3,     3,     3,     4,     4,     3,     1,     3,
      10,     1,     3,     0,     8,     7,     0,     2,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     3,     3,
       3,     3,     3,     3,    13,     1,     1,     1,     1,     1,
       1,     3,     3,     4,     3,     3,     3,     3,     1,     1,
       3,     3,     3
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       0,    11,    38,    13,     0,     0,     0,     0,    27,     4,
      12,     9,    10,     0,     0,     5,     0,     0,     2,     7,
      41,     8,     6,    26,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    31,     0,     0,    23,     0,     1,     3,
       0,     0,     0,     0,     0,     0,     0,    39,    14,    37,
       0,     0,     0,     0,    16,    15,    34,     0,     0,    58,
       0,     0,     0,    33,     0,    32,    25,    20,    19,    21,
      22,    24,     0,     0,    18,    35,     0,    36,    28,     0,
      43,     0,     0,    29,    30,     0,     0,    17,     0,    46,
       0,    46,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    45,    48,    49,
      47,    53,    54,    55,    56,    57,    52,    51,    78,    79,
      50,     0,    44,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    59,    60,
      61,    62,    63,    80,    81,    70,    82,     0,    76,    74,
      72,    66,    65,     0,    75,    77,    40,     0,     0,    73,
       0,    71,    42,    67,    69,    68,     0,     0,     0,     0,
       0,     0,     0,     0,    64
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,    17,    18,   147,    20,    51,    21,   148,    22,    88,
      92,   110,    23,   111,   112,   113,   114,   115,   116,   153,
     166,   146,   117,   118,   119,   120
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -136
static const yytype_int16 yypact[] =
{
     194,  -136,   -34,   -28,   -33,    41,   -22,    -3,  -136,  -136,
    -136,  -136,  -136,    12,   253,  -136,   253,    77,  -136,   161,
    -136,  -136,  -136,  -136,     2,   253,    24,    15,   243,    83,
      36,    19,    33,  -136,   254,   -35,    29,   269,  -136,  -136,
     253,   253,   253,   253,   253,    40,    55,  -136,   161,  -136,
      78,    -4,    13,    76,  -136,    56,  -136,    44,    54,  -136,
     253,   232,   253,  -136,   253,  -136,  -136,    62,    62,    29,
      29,    29,    24,     2,  -136,  -136,     2,  -136,  -136,   253,
    -136,    59,   278,   161,   161,   178,    74,   161,    65,  -136,
     253,  -136,   130,   263,   169,    86,    89,    91,    92,    95,
     102,   109,   110,   120,   128,   129,   148,  -136,   161,  -136,
    -136,  -136,  -136,  -136,  -136,  -136,  -136,  -136,   156,   157,
    -136,     2,  -136,   188,   190,   197,   198,   207,   209,   212,
       2,   253,   253,     2,    31,   253,   253,    64,  -136,  -136,
    -136,  -136,  -136,  -136,  -136,  -136,   166,   117,  -136,  -136,
     -31,  -136,  -136,   182,  -136,  -136,  -136,     2,   253,  -136,
     -21,  -136,   161,  -136,  -136,  -136,   201,   211,   210,   217,
     220,    31,   222,    31,  -136
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -136,  -136,   240,     0,  -136,   245,   -23,    35,  -136,  -136,
     172,  -136,  -136,  -136,  -136,  -136,  -136,  -136,  -136,  -135,
    -136,   131,  -136,  -136,  -136,  -136
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -1
static const yytype_uint8 yytable[] =
{
      19,    47,   157,    52,    64,    56,    24,    65,     2,   163,
      45,    46,    25,    34,    36,     1,    37,    19,    26,     3,
      31,    32,   159,   164,   165,    48,    50,     1,    55,    29,
       2,     3,     4,     5,     7,    64,   172,    30,   174,     8,
      67,    68,    69,    70,    71,    13,     7,    10,    75,    52,
      56,     8,    76,    86,    33,    53,    14,    59,   151,    10,
      82,    55,    83,    16,    84,    77,   152,    13,    14,   109,
      60,   109,    85,    61,    27,    16,    49,    38,    44,    87,
       1,    28,    78,     2,     3,     4,     5,    89,    27,    79,
      93,    72,   108,    91,   108,    73,    80,     6,   137,     7,
      40,    41,    42,    43,     8,    44,    81,   145,    42,    43,
     145,    44,    10,    90,    11,    12,   156,    62,    57,    58,
      13,    14,    40,    41,    42,    43,   123,    44,    16,   124,
      74,   125,   126,     1,   161,   127,     2,     3,     4,     5,
      95,    96,   128,    97,    98,    99,   100,   101,   102,   129,
     130,   158,     7,   103,   104,   105,   106,     8,   162,   107,
     131,    40,    41,    42,    43,    10,    44,   149,   132,   133,
     154,   155,     1,    13,    14,     2,     3,     4,     5,    95,
      96,    16,    97,    98,    99,   100,   101,   102,   134,   135,
     136,     7,   103,   104,   105,   106,     8,     1,   122,   157,
       2,     3,     4,     5,    10,    40,    41,    42,    43,   138,
      44,   139,    13,    14,     6,   160,     7,    62,   140,   141,
      16,     8,    40,    41,    42,    43,     9,    44,   142,    10,
     143,    11,    12,   144,   167,     1,    54,    13,    14,     3,
      31,    32,   168,   169,    15,    16,     1,    54,   170,     2,
       3,     4,     5,   171,     7,   173,     1,    39,    35,     8,
       3,    31,    32,    94,   150,     7,     0,    10,     0,     0,
       8,     0,     0,     0,     0,     7,    14,     0,    10,     0,
       8,     0,     0,    16,     0,     0,    13,    14,    10,     0,
       0,     0,     0,    62,    16,     0,    63,    14,    40,    41,
      42,    43,   121,    44,    16,     0,     0,    40,    41,    42,
      43,     0,    44,    40,    41,    42,    43,     0,    44,     0,
       0,    66,    40,    41,    42,    43,     0,    44,     0,     0,
      74
};

static const yytype_int16 yycheck[] =
{
       0,    24,    33,    26,    39,    28,    40,    42,     6,    30,
       8,     9,    40,    13,    14,     3,    16,    17,    51,     7,
       8,     9,    53,    44,    45,    25,    26,     3,    28,    51,
       6,     7,     8,     9,    22,    39,   171,    40,   173,    27,
      40,    41,    42,    43,    44,    43,    22,    35,    52,    72,
      73,    27,    39,    76,    42,    40,    44,    21,    27,    35,
      60,    61,    62,    51,    64,    52,    35,    43,    44,    92,
      51,    94,    72,    40,    33,    51,    52,     0,    49,    79,
       3,    40,     6,     6,     7,     8,     9,    28,    33,    33,
      90,    51,    92,    28,    94,    40,    52,    20,   121,    22,
      44,    45,    46,    47,    27,    49,    52,   130,    46,    47,
     133,    49,    35,    39,    37,    38,    52,    39,    35,    36,
      43,    44,    44,    45,    46,    47,    40,    49,    51,    40,
      52,    40,    40,     3,   157,    40,     6,     7,     8,     9,
      10,    11,    40,    13,    14,    15,    16,    17,    18,    40,
      40,    34,    22,    23,    24,    25,    26,    27,   158,    29,
      40,    44,    45,    46,    47,    35,    49,   132,    40,    40,
     135,   136,     3,    43,    44,     6,     7,     8,     9,    10,
      11,    51,    13,    14,    15,    16,    17,    18,    40,    33,
      33,    22,    23,    24,    25,    26,    27,     3,    29,    33,
       6,     7,     8,     9,    35,    44,    45,    46,    47,    21,
      49,    21,    43,    44,    20,    33,    22,    39,    21,    21,
      51,    27,    44,    45,    46,    47,    32,    49,    21,    35,
      21,    37,    38,    21,    33,     3,     4,    43,    44,     7,
       8,     9,    31,    33,    50,    51,     3,     4,    31,     6,
       7,     8,     9,    33,    22,    33,     3,    17,    13,    27,
       7,     8,     9,    91,   133,    22,    -1,    35,    -1,    -1,
      27,    -1,    -1,    -1,    -1,    22,    44,    -1,    35,    -1,
      27,    -1,    -1,    51,    -1,    -1,    43,    44,    35,    -1,
      -1,    -1,    -1,    39,    51,    -1,    42,    44,    44,    45,
      46,    47,    39,    49,    51,    -1,    -1,    44,    45,    46,
      47,    -1,    49,    44,    45,    46,    47,    -1,    49,    -1,
      -1,    52,    44,    45,    46,    47,    -1,    49,    -1,    -1,
      52
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     6,     7,     8,     9,    20,    22,    27,    32,
      35,    37,    38,    43,    44,    50,    51,    55,    56,    57,
      58,    60,    62,    66,    40,    40,    51,    33,    40,    51,
      40,     8,     9,    42,    57,    59,    57,    57,     0,    56,
      44,    45,    46,    47,    49,     8,     9,    60,    57,    52,
      57,    59,    60,    40,     4,    57,    60,    35,    36,    21,
      51,    40,    39,    42,    39,    42,    52,    57,    57,    57,
      57,    57,    51,    40,    52,    52,    39,    52,     6,    33,
      52,    52,    57,    57,    57,    57,    60,    57,    63,    28,
      39,    28,    64,    57,    64,    10,    11,    13,    14,    15,
      16,    17,    18,    23,    24,    25,    26,    29,    57,    60,
      65,    67,    68,    69,    70,    71,    72,    76,    77,    78,
      79,    39,    29,    40,    40,    40,    40,    40,    40,    40,
      40,    40,    40,    40,    40,    33,    33,    60,    21,    21,
      21,    21,    21,    21,    21,    60,    75,    57,    61,    61,
      75,    27,    35,    73,    61,    61,    52,    33,    34,    53,
      33,    60,    57,    30,    44,    45,    74,    33,    31,    33,
      31,    33,    73,    33,    73
};

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto yyerrlab

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      yytoken = YYTRANSLATE (yychar);				\
      YYPOPSTACK (1);						\
      goto yybackup;						\
    }								\
  else								\
    {								\
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (YYID (0))


#define YYTERROR	1
#define YYERRCODE	256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)				\
    do									\
      if (YYID (N))                                                    \
	{								\
	  (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;	\
	  (Current).first_column = YYRHSLOC (Rhs, 1).first_column;	\
	  (Current).last_line    = YYRHSLOC (Rhs, N).last_line;		\
	  (Current).last_column  = YYRHSLOC (Rhs, N).last_column;	\
	}								\
      else								\
	{								\
	  (Current).first_line   = (Current).last_line   =		\
	    YYRHSLOC (Rhs, 0).last_line;				\
	  (Current).first_column = (Current).last_column =		\
	    YYRHSLOC (Rhs, 0).last_column;				\
	}								\
    while (YYID (0))
#endif


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL
#  define YY_LOCATION_PRINT(File, Loc)			\
     fprintf (File, "%d.%d-%d.%d",			\
	      (Loc).first_line, (Loc).first_column,	\
	      (Loc).last_line,  (Loc).last_column)
# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
#endif


/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (YYLEX_PARAM)
#else
# define YYLEX yylex ()
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      yy_symbol_print (stderr,						  \
		  Type, Value); \
      YYFPRINTF (stderr, "\n");						  \
    }									  \
} while (YYID (0))


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# else
  YYUSE (yyoutput);
# endif
  switch (yytype)
    {
      default:
	break;
    }
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_stack_print (yytype_int16 *bottom, yytype_int16 *top)
#else
static void
yy_stack_print (bottom, top)
    yytype_int16 *bottom;
    yytype_int16 *top;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; bottom <= top; ++bottom)
    YYFPRINTF (stderr, " %d", *bottom);
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_reduce_print (YYSTYPE *yyvsp, int yyrule)
#else
static void
yy_reduce_print (yyvsp, yyrule)
    YYSTYPE *yyvsp;
    int yyrule;
#endif
{
  int yynrhs = yyr2[yyrule];
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      fprintf (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       		       );
      fprintf (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, Rule); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif



#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
yystrlen (const char *yystr)
#else
static YYSIZE_T
yystrlen (yystr)
    const char *yystr;
#endif
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
yystpcpy (char *yydest, const char *yysrc)
#else
static char *
yystpcpy (yydest, yysrc)
    char *yydest;
    const char *yysrc;
#endif
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
	switch (*++yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (yyres)
	      yyres[yyn] = *yyp;
	    yyn++;
	    break;

	  case '"':
	    if (yyres)
	      yyres[yyn] = '\0';
	    return yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into YYRESULT an error message about the unexpected token
   YYCHAR while in state YYSTATE.  Return the number of bytes copied,
   including the terminating null byte.  If YYRESULT is null, do not
   copy anything; just return the number of bytes that would be
   copied.  As a special case, return 0 if an ordinary "syntax error"
   message will do.  Return YYSIZE_MAXIMUM if overflow occurs during
   size calculation.  */
static YYSIZE_T
yysyntax_error (char *yyresult, int yystate, int yychar)
{
  int yyn = yypact[yystate];

  if (! (YYPACT_NINF < yyn && yyn <= YYLAST))
    return 0;
  else
    {
      int yytype = YYTRANSLATE (yychar);
      YYSIZE_T yysize0 = yytnamerr (0, yytname[yytype]);
      YYSIZE_T yysize = yysize0;
      YYSIZE_T yysize1;
      int yysize_overflow = 0;
      enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
      char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
      int yyx;

# if 0
      /* This is so xgettext sees the translatable formats that are
	 constructed on the fly.  */
      YY_("syntax error, unexpected %s");
      YY_("syntax error, unexpected %s, expecting %s");
      YY_("syntax error, unexpected %s, expecting %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s");
# endif
      char *yyfmt;
      char const *yyf;
      static char const yyunexpected[] = "syntax error, unexpected %s";
      static char const yyexpecting[] = ", expecting %s";
      static char const yyor[] = " or %s";
      char yyformat[sizeof yyunexpected
		    + sizeof yyexpecting - 1
		    + ((YYERROR_VERBOSE_ARGS_MAXIMUM - 2)
		       * (sizeof yyor - 1))];
      char const *yyprefix = yyexpecting;

      /* Start YYX at -YYN if negative to avoid negative indexes in
	 YYCHECK.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;

      /* Stay within bounds of both yycheck and yytname.  */
      int yychecklim = YYLAST - yyn + 1;
      int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
      int yycount = 1;

      yyarg[0] = yytname[yytype];
      yyfmt = yystpcpy (yyformat, yyunexpected);

      for (yyx = yyxbegin; yyx < yyxend; ++yyx)
	if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
	  {
	    if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
	      {
		yycount = 1;
		yysize = yysize0;
		yyformat[sizeof yyunexpected - 1] = '\0';
		break;
	      }
	    yyarg[yycount++] = yytname[yyx];
	    yysize1 = yysize + yytnamerr (0, yytname[yyx]);
	    yysize_overflow |= (yysize1 < yysize);
	    yysize = yysize1;
	    yyfmt = yystpcpy (yyfmt, yyprefix);
	    yyprefix = yyor;
	  }

      yyf = YY_(yyformat);
      yysize1 = yysize + yystrlen (yyf);
      yysize_overflow |= (yysize1 < yysize);
      yysize = yysize1;

      if (yysize_overflow)
	return YYSIZE_MAXIMUM;

      if (yyresult)
	{
	  /* Avoid sprintf, as that infringes on the user's name space.
	     Don't have undefined behavior even if the translation
	     produced a string with the wrong number of "%s"s.  */
	  char *yyp = yyresult;
	  int yyi = 0;
	  while ((*yyp = *yyf) != '\0')
	    {
	      if (*yyp == '%' && yyf[1] == 's' && yyi < yycount)
		{
		  yyp += yytnamerr (yyp, yyarg[yyi++]);
		  yyf += 2;
		}
	      else
		{
		  yyp++;
		  yyf++;
		}
	    }
	}
      return yysize;
    }
}
#endif /* YYERROR_VERBOSE */


/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yymsg, yytype, yyvaluep)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  YYUSE (yyvaluep);

  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  switch (yytype)
    {

      default:
	break;
    }
}


/* Prevent warnings from -Wmissing-prototypes.  */

#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int yyparse (void *YYPARSE_PARAM);
#else
int yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */



/* The look-ahead symbol.  */
int yychar;

/* The semantic value of the look-ahead symbol.  */
YYSTYPE yylval;

/* Number of syntax errors so far.  */
int yynerrs;



/*----------.
| yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void *YYPARSE_PARAM)
#else
int
yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{
  
  int yystate;
  int yyn;
  int yyresult;
  /* Number of tokens to shift before error messages enabled.  */
  int yyerrstatus;
  /* Look-ahead token as an internal (translated) token number.  */
  int yytoken = 0;
#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

  /* Three stacks and their tools:
     `yyss': related to states,
     `yyvs': related to semantic values,
     `yyls': related to locations.

     Refer to the stacks thru separate pointers, to allow yyoverflow
     to reallocate them elsewhere.  */

  /* The state stack.  */
  yytype_int16 yyssa[YYINITDEPTH];
  yytype_int16 *yyss = yyssa;
  yytype_int16 *yyssp;

  /* The semantic value stack.  */
  YYSTYPE yyvsa[YYINITDEPTH];
  YYSTYPE *yyvs = yyvsa;
  YYSTYPE *yyvsp;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  YYSIZE_T yystacksize = YYINITDEPTH;

  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;


  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY;		/* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

  yyssp = yyss;
  yyvsp = yyvs;

  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	yytype_int16 *yyss1 = yyss;


	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow (YY_("memory exhausted"),
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),

		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	yytype_int16 *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyexhaustedlab;
	YYSTACK_RELOCATE (yyss);
	YYSTACK_RELOCATE (yyvs);

#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;


      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     look-ahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to look-ahead token.  */
  yyn = yypact[yystate];
  if (yyn == YYPACT_NINF)
    goto yydefault;

  /* Not known => get a look-ahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid look-ahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yyn == 0 || yyn == YYTABLE_NINF)
	goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the look-ahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token unless it is eof.  */
  if (yychar != YYEOF)
    yychar = YYEMPTY;

  yystate = yyn;
  *++yyvsp = yylval;

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 2:
#line 1271 "glb_parser.y"
    {}
    break;

  case 3:
#line 1272 "glb_parser.y"
    {}
    break;

  case 4:
#line 1273 "glb_parser.y"
    {YYABORT;}
    break;

  case 5:
#line 1274 "glb_parser.y"
    {YYABORT;}
    break;

  case 6:
#line 1278 "glb_parser.y"
    {}
    break;

  case 7:
#line 1279 "glb_parser.y"
    {}
    break;

  case 8:
#line 1280 "glb_parser.y"
    {}
    break;

  case 9:
#line 1281 "glb_parser.y"
    { glb_copy_buff();  glbReset(); }
    break;

  case 10:
#line 1282 "glb_parser.y"
    { glbNewDetector(); }
    break;

  case 11:
#line 1286 "glb_parser.y"
    { (yyval.val) = (yyvsp[(1) - (1)].val);                     }
    break;

  case 12:
#line 1287 "glb_parser.y"
    { (yyval.val) = (yyvsp[(1) - (1)].nameptr)->value;              }
    break;

  case 13:
#line 1288 "glb_parser.y"
    { (yyval.val) = (yyvsp[(1) - (1)].tptr)->value.var;          }
    break;

  case 14:
#line 1289 "glb_parser.y"
    { (yyval.val) = (yyvsp[(3) - (3)].val); (yyvsp[(1) - (3)].tptr)->value.var = (yyvsp[(3) - (3)].val); }
    break;

  case 15:
#line 1290 "glb_parser.y"
    {
  if(set_exp((yyvsp[(1) - (3)].name),(yyvsp[(3) - (3)].val),0)==1) yyerror("Unknown identifier: %s", (yyvsp[(1) - (3)].name));
  (yyval.val) = (yyvsp[(3) - (3)].val);
  if ((yyvsp[(1) - (3)].name))  { glb_free((yyvsp[(1) - (3)].name));  (yyvsp[(1) - (3)].name)=NULL; }
}
    break;

  case 16:
#line 1295 "glb_parser.y"
    {
  if(set_fnct((yyvsp[(1) - (3)].name),(yyvsp[(3) - (3)].nameptr)->sf)==1) yyerror("Unknown identifier: %s", (yyvsp[(1) - (3)].name));
  if ((yyvsp[(1) - (3)].name))  { glb_free((yyvsp[(1) - (3)].name));  (yyvsp[(1) - (3)].name)=NULL; }
}
    break;

  case 17:
#line 1299 "glb_parser.y"
    {
  if(set_pair((yyvsp[(1) - (5)].name),(yyvsp[(3) - (5)].val),(yyvsp[(5) - (5)].val),0)==1) yyerror("Unknown identifier: %s", (yyvsp[(1) - (5)].name));
  (yyval.val) = (yyvsp[(3) - (5)].val);
  if ((yyvsp[(1) - (5)].name))  { glb_free((yyvsp[(1) - (5)].name));  (yyvsp[(1) - (5)].name)=NULL; }
}
    break;

  case 18:
#line 1304 "glb_parser.y"
    {
  /* added safety in case the function pointer is NULL, which is
     sometimes useful for special functions */
  if((yyvsp[(1) - (4)].tptr)->value.fnctptr==NULL) yyerror("Improper use of special function %s", (yyvsp[(1) - (4)].tptr)->name);
  else (yyval.val) = (*((yyvsp[(1) - (4)].tptr)->value.fnctptr))((yyvsp[(3) - (4)].val)); }
    break;

  case 19:
#line 1310 "glb_parser.y"
    { (yyval.val) = (yyvsp[(1) - (3)].val) + (yyvsp[(3) - (3)].val);      }
    break;

  case 20:
#line 1311 "glb_parser.y"
    { (yyval.val) = (yyvsp[(1) - (3)].val) - (yyvsp[(3) - (3)].val);      }
    break;

  case 21:
#line 1312 "glb_parser.y"
    { (yyval.val) = (yyvsp[(1) - (3)].val) * (yyvsp[(3) - (3)].val);      }
    break;

  case 22:
#line 1313 "glb_parser.y"
    { (yyval.val) = (yyvsp[(1) - (3)].val) / (yyvsp[(3) - (3)].val);      }
    break;

  case 23:
#line 1314 "glb_parser.y"
    { (yyval.val) = -(yyvsp[(2) - (2)].val);          }
    break;

  case 24:
#line 1315 "glb_parser.y"
    { (yyval.val) = pow ((yyvsp[(1) - (3)].val), (yyvsp[(3) - (3)].val)); }
    break;

  case 25:
#line 1316 "glb_parser.y"
    { (yyval.val) = (yyvsp[(2) - (3)].val);           }
    break;

  case 26:
#line 1317 "glb_parser.y"
    { (yyval.val) = 0;            }
    break;

  case 27:
#line 1318 "glb_parser.y"
    { yyerror("Unknown name: %s", (yyvsp[(1) - (1)].name)); YYERROR; }
    break;

  case 28:
#line 1322 "glb_parser.y"
    {
  glb_List *ltemp;
  ltemp=thread_list(&glb_list_copy,0,0,(yyvsp[(4) - (4)].tptr)->list);
  if(set_exp_list((yyvsp[(1) - (4)].name),ltemp,3)==1) yyerror("Unknown identifier");
  (yyval.ptr) = ltemp;
  if ((yyvsp[(1) - (4)].name))  { glb_free((yyvsp[(1) - (4)].name));  (yyvsp[(1) - (4)].name)=NULL; }
}
    break;

  case 29:
#line 1331 "glb_parser.y"
    {
   glb_List *buf = list_cons(NULL, (yyvsp[(1) - (3)].val));
   buf = list_cons(buf, (yyvsp[(3) - (3)].val));
   (yyval.ptr)  = buf;
}
    break;

  case 30:
#line 1336 "glb_parser.y"
    { (yyval.ptr) = list_cons((yyvsp[(1) - (3)].ptr), (yyvsp[(3) - (3)].val)); }
    break;

  case 31:
#line 1340 "glb_parser.y"
    {(yyval.ptr)=NULL;}
    break;

  case 32:
#line 1341 "glb_parser.y"
    {(yyval.ptr)=(yyvsp[(2) - (3)].ptr); }
    break;

  case 33:
#line 1342 "glb_parser.y"
    {(yyval.ptr)=list_cons(NULL,(yyvsp[(2) - (3)].val)); }
    break;

  case 34:
#line 1343 "glb_parser.y"
    {
  if(set_exp_list((yyvsp[(1) - (3)].name),(yyvsp[(3) - (3)].ptr),3)==1)  yyerror("Unknown identifier");
  (yyval.ptr) = (yyvsp[(3) - (3)].ptr);
  if ((yyvsp[(1) - (3)].name))  { glb_free((yyvsp[(1) - (3)].name));  (yyvsp[(1) - (3)].name)=NULL; }
}
    break;

  case 35:
#line 1348 "glb_parser.y"
    {(yyval.ptr) = thread_list((yyvsp[(1) - (4)].tptr)->value.fnctptr,(yyvsp[(1) - (4)].tptr)->reverse,(yyvsp[(1) - (4)].tptr)->destroy,(yyvsp[(3) - (4)].ptr));}
    break;

  case 36:
#line 1349 "glb_parser.y"
    {(yyval.ptr) = thread_list((yyvsp[(1) - (4)].tptr)->value.fnctptr,(yyvsp[(1) - (4)].tptr)->reverse,(yyvsp[(1) - (4)].tptr)->destroy,(yyvsp[(3) - (4)].ptr));}
    break;

  case 37:
#line 1350 "glb_parser.y"
    {(yyval.ptr) = (*((yyvsp[(1) - (3)].tptr)->value.lfnctptr))();}
    break;

  case 38:
#line 1351 "glb_parser.y"
    { (yyval.ptr) = (yyvsp[(1) - (1)].tptr)->list;              }
    break;

  case 39:
#line 1352 "glb_parser.y"
    { (yyval.ptr) = (yyvsp[(3) - (3)].ptr); (yyvsp[(1) - (3)].tptr)->list = (yyvsp[(3) - (3)].ptr); }
    break;

  case 40:
#line 1353 "glb_parser.y"
    {(yyval.ptr)=glb_interpolation((yyvsp[(3) - (10)].ptr),(yyvsp[(5) - (10)].ptr),floor((yyvsp[(7) - (10)].val)),(yyvsp[(9) - (10)].ptr));}
    break;

  case 41:
#line 1354 "glb_parser.y"
    {}
    break;

  case 42:
#line 1358 "glb_parser.y"
    {
  double *buf;
  buf=(double*) glb_malloc(sizeof(double)*2);
  buf[0]=(yyvsp[(1) - (3)].val);
  buf[1]=(yyvsp[(3) - (3)].val)-1;
  (yyval.dpt)=buf;
}
    break;

  case 43:
#line 1369 "glb_parser.y"
    { if((yyvsp[(3) - (4)].nameptr)->value==-1) {(yyvsp[(3) - (4)].nameptr)->value=step_counter((yyvsp[(1) - (4)].name)); }
  loc_count=(yyvsp[(3) - (4)].nameptr)->value;
  glb_free(context);
  context =(char *) strdup((yyvsp[(1) - (4)].name));
  grp_start(context, loc_count);
  if ((yyvsp[(1) - (4)].name))  { glb_free((yyvsp[(1) - (4)].name));  (yyvsp[(1) - (4)].name)=NULL; }
}
    break;

  case 44:
#line 1376 "glb_parser.y"
    {
  grp_end(context, loc_count);
}
    break;

  case 45:
#line 1379 "glb_parser.y"
    {
    yyerror("Redefinition of an automatic variable %s", (yyvsp[(3) - (7)].nameptr)->name); YYERROR;
    if ((yyvsp[(1) - (7)].name))  { glb_free((yyvsp[(1) - (7)].name));  (yyvsp[(1) - (7)].name)=NULL; }
}
    break;

  case 48:
#line 1391 "glb_parser.y"
    {}
    break;

  case 49:
#line 1392 "glb_parser.y"
    {}
    break;

  case 50:
#line 1393 "glb_parser.y"
    {}
    break;

  case 51:
#line 1394 "glb_parser.y"
    {}
    break;

  case 52:
#line 1395 "glb_parser.y"
    {}
    break;

  case 53:
#line 1396 "glb_parser.y"
    {}
    break;

  case 54:
#line 1397 "glb_parser.y"
    {}
    break;

  case 55:
#line 1398 "glb_parser.y"
    {}
    break;

  case 56:
#line 1399 "glb_parser.y"
    {}
    break;

  case 57:
#line 1400 "glb_parser.y"
    {}
    break;

  case 58:
#line 1404 "glb_parser.y"
    {
//  buff.version=strdup($3);
  if (set_string((yyvsp[(1) - (3)].name), (yyvsp[(3) - (3)].name)) != 0)
    yyerror("Unknown identifier: %s", (yyvsp[(1) - (3)].name));
  if ((yyvsp[(1) - (3)].name))  { glb_free((yyvsp[(1) - (3)].name));  (yyvsp[(1) - (3)].name)=NULL; }
  if ((yyvsp[(3) - (3)].name))  { glb_free((yyvsp[(3) - (3)].name));  (yyvsp[(3) - (3)].name)=NULL; }
}
    break;

  case 59:
#line 1414 "glb_parser.y"
    {
  //load_cross($3,loc_count-1);
  xsc.file_name=strdup((yyvsp[(3) - (3)].name));
  (yyval.name)=(yyvsp[(3) - (3)].name);
  if ((yyvsp[(1) - (3)].name))  { glb_free((yyvsp[(1) - (3)].name));  (yyvsp[(1) - (3)].name)=NULL; }
  if ((yyvsp[(3) - (3)].name))  { glb_free((yyvsp[(3) - (3)].name));  (yyvsp[(3) - (3)].name)=NULL; }
}
    break;

  case 60:
#line 1424 "glb_parser.y"
    {
  //load_flux($3,loc_count-1,1);
  flt.file_name=strdup((yyvsp[(3) - (3)].name));

  //if(set_exp($1,$3,0)==1) yyerror("Unknown identifier");
  (yyval.name)=(yyvsp[(3) - (3)].name);
  if ((yyvsp[(1) - (3)].name))  { glb_free((yyvsp[(1) - (3)].name));  (yyvsp[(1) - (3)].name)=NULL; }
  if ((yyvsp[(3) - (3)].name))  { glb_free((yyvsp[(3) - (3)].name));  (yyvsp[(3) - (3)].name)=NULL; }
}
    break;

  case 61:
#line 1436 "glb_parser.y"
    {
  //load_flux($3,loc_count-1,1);
  flt.file_name=strdup((yyvsp[(3) - (3)].name));

  //if(set_exp($1,$3,0)==1) yyerror("Unknown identifier");
  (yyval.name)=(yyvsp[(3) - (3)].name);
  if ((yyvsp[(1) - (3)].name))  { glb_free((yyvsp[(1) - (3)].name));  (yyvsp[(1) - (3)].name)=NULL; }
  if ((yyvsp[(3) - (3)].name))  { glb_free((yyvsp[(3) - (3)].name));  (yyvsp[(3) - (3)].name)=NULL; }
}
    break;

  case 62:
#line 1448 "glb_parser.y"
    {
  flt.eft_coeff_file=strdup((yyvsp[(3) - (3)].name));

  (yyval.name)=(yyvsp[(3) - (3)].name);
  if ((yyvsp[(1) - (3)].name))  { glb_free((yyvsp[(1) - (3)].name));  (yyvsp[(1) - (3)].name)=NULL; }
  if ((yyvsp[(3) - (3)].name))  { glb_free((yyvsp[(3) - (3)].name));  (yyvsp[(3) - (3)].name)=NULL; }
}
    break;

  case 63:
#line 1457 "glb_parser.y"
    {
  xsc.eft_coeff_file=strdup((yyvsp[(3) - (3)].name));
  (yyval.name)=(yyvsp[(3) - (3)].name);
  if ((yyvsp[(1) - (3)].name))  { glb_free((yyvsp[(1) - (3)].name));  (yyvsp[(1) - (3)].name)=NULL; }
  if ((yyvsp[(3) - (3)].name))  { glb_free((yyvsp[(3) - (3)].name));  (yyvsp[(3) - (3)].name)=NULL; }
}
    break;

  case 64:
#line 1467 "glb_parser.y"
    {

  int x[6];
  x[0]=(yyvsp[(3) - (13)].nameptr)->value - 1 ;
  x[1]=(yyvsp[(5) - (13)].in);
  x[2]=(yyvsp[(7) - (13)].in);
  x[3]=(yyvsp[(9) - (13)].in);
  x[4]=(int) (yyvsp[(11) - (13)].nameptr)->value -1;
  x[5]=(int) (yyvsp[(13) - (13)].nameptr)->value -1;

  set_channel_data(x,loc_count);
  if ((yyvsp[(1) - (13)].name))  { glb_free((yyvsp[(1) - (13)].name));  (yyvsp[(1) - (13)].name)=NULL; }
}
    break;

  case 65:
#line 1485 "glb_parser.y"
    {(yyval.nameptr)=(yyvsp[(1) - (1)].nameptr);}
    break;

  case 66:
#line 1486 "glb_parser.y"
    { yyerror("Unknown name: %s", (yyvsp[(1) - (1)].name)); YYERROR; }
    break;

  case 67:
#line 1490 "glb_parser.y"
    {(yyval.in)=(yyvsp[(1) - (1)].in);}
    break;

  case 68:
#line 1491 "glb_parser.y"
    {(yyval.in)=1;}
    break;

  case 69:
#line 1492 "glb_parser.y"
    {(yyval.in)=-1;}
    break;

  case 70:
#line 1497 "glb_parser.y"
    {
  glb_List **buf;
  energy_len=1;

  buf=(glb_List**) glb_malloc(sizeof( glb_List* ) );
  buf[0]=(yyvsp[(1) - (1)].ptr);
  (yyval.ptrq)=buf;
}
    break;

  case 71:
#line 1506 "glb_parser.y"
    {
  glb_List **buf;
  buf=(yyvsp[(1) - (3)].ptrq);
  energy_len++;

  buf=(glb_List**) glb_realloc((void**) buf , sizeof( glb_List* ) * energy_len);

  buf[energy_len-1]=(yyvsp[(3) - (3)].ptr);
  (yyval.ptrq)=buf;
}
    break;

  case 72:
#line 1519 "glb_parser.y"
    {
  set_exp_energy("@energy",(yyvsp[(3) - (3)].ptrq));
  if ((yyvsp[(1) - (3)].name))  { glb_free((yyvsp[(1) - (3)].name));  (yyvsp[(1) - (3)].name)=NULL; }
}
    break;

  case 73:
#line 1523 "glb_parser.y"
    {
  set_exp_energy("@energy",(yyvsp[(3) - (4)].ptrq)); 
  if ((yyvsp[(1) - (4)].name))  { glb_free((yyvsp[(1) - (4)].name));  (yyvsp[(1) - (4)].name)=NULL; }
}
    break;

  case 74:
#line 1530 "glb_parser.y"
    {
  glb_List **buf;
  buf=(glb_List**) glb_malloc(sizeof(glb_List*)*2);
  buf[0]=list_cons(NULL,(yyvsp[(3) - (3)].dpt)[0]);
  buf[1]=list_cons(NULL,(yyvsp[(3) - (3)].dpt)[1]);
  glb_free((yyvsp[(3) - (3)].dpt));
  if ((yyvsp[(1) - (3)].name))  { glb_free((yyvsp[(1) - (3)].name));  (yyvsp[(1) - (3)].name)=NULL; }
  (yyval.ptrq)=buf;
}
    break;

  case 75:
#line 1539 "glb_parser.y"
    {
  glb_List **buf;
  buf=(yyvsp[(1) - (3)].ptrq);
  buf[0]=list_cons(buf[0],(yyvsp[(3) - (3)].dpt)[0]);
  buf[1]=list_cons(buf[1],(yyvsp[(3) - (3)].dpt)[1]);
  glb_free((yyvsp[(3) - (3)].dpt));
  (yyval.ptrq)=buf;
}
    break;

  case 76:
#line 1550 "glb_parser.y"
    {
  glb_List **buf;

  buf=(glb_List**) glb_malloc(sizeof(glb_List*)*2);
  buf[0]=list_cons(NULL,(yyvsp[(3) - (3)].dpt)[0]);
  buf[1]=list_cons(NULL,(yyvsp[(3) - (3)].dpt)[1]);
  glb_free((yyvsp[(3) - (3)].dpt));
  if ((yyvsp[(1) - (3)].name))  { glb_free((yyvsp[(1) - (3)].name));  (yyvsp[(1) - (3)].name)=NULL; }
  (yyval.ptrq)=buf;
}
    break;

  case 77:
#line 1560 "glb_parser.y"
    {
  glb_List **buf;
  buf=(yyvsp[(1) - (3)].ptrq);
  buf[0]=list_cons(buf[0],(yyvsp[(3) - (3)].dpt)[0]);
  buf[1]=list_cons(buf[1],(yyvsp[(3) - (3)].dpt)[1]);
  glb_free((yyvsp[(3) - (3)].dpt));
  (yyval.ptrq)=buf;
}
    break;

  case 78:
#line 1571 "glb_parser.y"
    {
  int flag;
  (yyval.ptrq)=(yyvsp[(1) - (1)].ptrq);
  flag=set_exp_list("bgrulescoeff",(yyvsp[(1) - (1)].ptrq)[0],0);
  if(flag==1) yyerror("Invalid coefficient in @background");
  flag=set_exp_list("bgrulechannellist",(yyvsp[(1) - (1)].ptrq)[1],0);
  if(flag==1) yyerror("Invalid channel in @background");
  glb_free((yyvsp[(1) - (1)].ptrq));
}
    break;

  case 79:
#line 1580 "glb_parser.y"
    {
  int flag;
  (yyval.ptrq)=(yyvsp[(1) - (1)].ptrq);
  flag=set_exp_list("rulescoeff",(yyvsp[(1) - (1)].ptrq)[0],0);
  if(flag==1) yyerror("Invalid coefficient in @signal");
  flag=set_exp_list("rulechannellist",(yyvsp[(1) - (1)].ptrq)[1],0);
  if(flag==1) yyerror("Invalid channel in @signal");
  glb_free((yyvsp[(1) - (1)].ptrq));
}
    break;

  case 80:
#line 1589 "glb_parser.y"
    {
//JK, 2012-05-17  buff.sys_on_strings[buff.numofrules-1] = strdup($3);
  buff.sys_on_strings[loc_count-1] = strdup((yyvsp[(3) - (3)].name));
  if ((yyvsp[(1) - (3)].name))  { glb_free((yyvsp[(1) - (3)].name));  (yyvsp[(1) - (3)].name)=NULL; }
  if ((yyvsp[(3) - (3)].name))  { glb_free((yyvsp[(3) - (3)].name));  (yyvsp[(3) - (3)].name)=NULL; }
}
    break;

  case 81:
#line 1595 "glb_parser.y"
    {
//JK, 2012-05-17  buff.sys_off_strings[buff.numofrules-1] = strdup($3);
  buff.sys_off_strings[loc_count-1] = strdup((yyvsp[(3) - (3)].name));
  if ((yyvsp[(1) - (3)].name))  { glb_free((yyvsp[(1) - (3)].name));  (yyvsp[(1) - (3)].name)=NULL; }
  if ((yyvsp[(3) - (3)].name))  { glb_free((yyvsp[(3) - (3)].name));  (yyvsp[(3) - (3)].name)=NULL; }
}
    break;

  case 82:
#line 1601 "glb_parser.y"
    {
  set_multiex_errors((yyvsp[(1) - (3)].name), (yyvsp[(3) - (3)].ptrq));
  if ((yyvsp[(1) - (3)].name))  { glb_free((yyvsp[(1) - (3)].name));  (yyvsp[(1) - (3)].name)=NULL; }
}
    break;


/* Line 1267 of yacc.c.  */
#line 3341 "glb_parser.c"
      default: break;
    }
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;


  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
      {
	YYSIZE_T yysize = yysyntax_error (0, yystate, yychar);
	if (yymsg_alloc < yysize && yymsg_alloc < YYSTACK_ALLOC_MAXIMUM)
	  {
	    YYSIZE_T yyalloc = 2 * yysize;
	    if (! (yysize <= yyalloc && yyalloc <= YYSTACK_ALLOC_MAXIMUM))
	      yyalloc = YYSTACK_ALLOC_MAXIMUM;
	    if (yymsg != yymsgbuf)
	      YYSTACK_FREE (yymsg);
	    yymsg = (char *) YYSTACK_ALLOC (yyalloc);
	    if (yymsg)
	      yymsg_alloc = yyalloc;
	    else
	      {
		yymsg = yymsgbuf;
		yymsg_alloc = sizeof yymsgbuf;
	      }
	  }

	if (0 < yysize && yysize <= yymsg_alloc)
	  {
	    (void) yysyntax_error (yymsg, yystate, yychar);
	    yyerror (yymsg);
	  }
	else
	  {
	    yyerror (YY_("syntax error"));
	    if (yysize != 0)
	      goto yyexhaustedlab;
	  }
      }
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse look-ahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  yydestruct ("Error: discarding",
		      yytoken, &yylval);
	  yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse look-ahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (yyn != YYPACT_NINF)
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;


      yydestruct ("Error: popping",
		  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  *++yyvsp = yylval;


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#ifndef yyoverflow
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEOF && yychar != YYEMPTY)
     yydestruct ("Cleanup: discarding lookahead",
		 yytoken, &yylval);
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (yyresult);
}


#line 1607 "glb_parser.y"


extern glb_symrec *sym_table;



/***************************************************************************
 * Function yyerror                                                        *
 ***************************************************************************
 * Print parser errors including the line number where the error occured   *
 ***************************************************************************/
int yyerror (const char *s, ...)  /* Called by yyparse on error */
{
  va_list args;
  va_start(args, s);

  if(yydebug > 0) fprintf(stderr,"*****************************************\n");
  fprintf (stderr,"%s:%d: error: ",
           glb_file_id, glb_line_num+1);
  vfprintf(stderr, s, args);
  fprintf(stderr, "\n");
  if(yydebug > 0) fprintf(stderr,"*****************************************\n");
  va_end(args);
  return 0;
}

int
yywarn (const char *s)  /* Called by yyparse on warning */
{
  fprintf (stderr,"%s:%d: warning: %s\n",
           glb_file_id, glb_line_num+1, s);
  return 0;
}




struct glb_init
{
  char *fname;
  double (*fnct)(double);
  /* these two flags determine the behaviour under threading
     reverse 0 leaves the list as is (i.e reverse)
     reverse 1 reverse the list (right order)

     destroy -1 leaves the list (even the memory location) unscathed
     destroy 0  procuces a copy
     destroy 1 destroys the input list and returns a new list
  */
  int reverse;
  int destroy;
};

struct glb_init_list
{
  char *fname;
  glb_List *(*fnct)(void);
};


struct glb_init_sig
{
  char *fname;
  sigfun sf;
};

static double echo(double x)
{
  fprintf(stdout,"%f ",x);
  return x;
}

static double echon(double x)
{
  fprintf(stdout,"%f\n",x);
  return x;
}


static double line(double x)
{
  size_t n,i;
  n=floor(x);
  for(i=0;i<n;i++) fprintf(stdout,"\n");
  return x;
}

static struct glb_init arith_fncts[] =
  {
    {"sin",  sin, 0,1},
    {"cos",  cos,0,1},
    {"tan",  tan,0,1},
    {"asin", asin,0,1},
    {"acos", acos,0,1},
    {"atan", atan,0,1},
    {"log",   log,0,1},
    {"exp",  exp,0,1},
    {"log10", log10,0,1},
    {"sqrt", sqrt,0,1},
    {"reverse",glb_reverse,1,1},
    {"copy",glb_list_copy,0,0},
    {"echo",echo,0,-1},
    {"echon",echon,0,-1},
    {"line",line,0,-1},
    {"interpolation",NULL,0,0},
    {NULL, NULL,0,0}
     };

/***************************************************************************
 * Function glb_free_names                                                 *
 ***************************************************************************
 * Destroy a list of name record                                           *
 ***************************************************************************/
void glb_free_names(glb_naming *stale)
{
  glb_naming *ptr;
  glb_naming *dummy;
  ptr = stale;
  while(ptr != (glb_naming *) NULL)
  {
    glb_free(ptr->name);
    glb_free(ptr->context);
    dummy=ptr->next;
    glb_free(ptr);
    ptr=dummy;
  }
  stale=NULL;
}


/***************************************************************************
 * Function glb_copy_names_from_parser                                     *
 ***************************************************************************
 * Prepends the current parser namespace the GLoBES namesapce head and     *
 * returns the new head                                                    *
 ***************************************************************************/
glb_naming *glb_copy_names_from_parser(glb_naming *head)
{
  glb_namerec *pp; /* Parser name record */
  glb_naming *pg;  /* GLoBES name record */
  for (pp=name_table; pp != NULL; pp=pp->next)
  {
    pg = (glb_naming *) glb_malloc(sizeof(glb_naming));
    pg->name    = strdup(pp->name);
    pg->context = strdup(pp->context);
    pg->value   = pp->value;
    pg->next    = head;
    head        = pg;
  }
  return head;
}

/***************************************************************************
 * Function glb_copy_names                                                 *
 ***************************************************************************
 * Duplicates a list of glb_naming records and prepends them to head       *
 ***************************************************************************/
glb_naming *glb_copy_names(glb_naming *in, glb_naming *head)
{
  glb_naming *p;
  glb_naming *pnew;
  for (p=in; p != NULL; p=p->next)
  {
    pnew = (glb_naming *) glb_malloc(sizeof(glb_naming));
    pnew->name    = strdup(p->name);
    pnew->context = strdup(p->context);
    pnew->value   = p->value;
    pnew->next    = head;
    head          = pnew;
  }
  return head;
}



//void glb_set_up_smear_data(glb_smear *test,const struct glb_experiment *head)
static glb_List *glb_bincenter(void)
{
  int i;
  glb_List *res=NULL;
  glb_smear *test;
  test=glb_smear_alloc();

  if(buff.numofbins<0) {glb_error("Cannot compute bincenter. Binning not set up properly."); return NULL;}

  glb_set_up_smear_data(test,&buff);


  for(i=0;i<test->numofbins;i++)
    {
      res=list_cons(res,test->bincenter[i]);
    }

  glb_smear_free(test);

  return res;
}


static glb_List *glb_samplingbincenter(void)
{
  int i;
  glb_List *res=NULL;
  glb_smear *test;
  test=glb_smear_alloc();

  if(buff.simbins<0) {glb_error("Cannot compute samplingbincenter. Sampling-binning not set up properly."); return NULL;}


  glb_set_up_smear_data(test,&buff);

  for(i=0;i<test->simbins;i++)
    {

      res=list_cons(res,test->simbincenter[i]);
    }

  glb_smear_free(test);


  return res;
}




static struct glb_init_list list_fncts[] =
  {
    {"bincenter",glb_bincenter},
    {"samplingbincenter",glb_samplingbincenter},
    {NULL, NULL}
  };

static double standard_sig(double x ,double *in)
{

  return in[0]*x + in[1]*sqrt(x) + in[2];
}


static double chooz_sig(double x, double *in)
{
  if(x<=0.0018) return in[0]/1000.0;
  else return in[0]*sqrt((x-0.0018+0.001)*1000.0)/1000.0;
}

/* Number of parameters ? */
struct glb_init_sig sig_fncts[] =
  {
    {"#standard",  standard_sig},
    {"#inverse_beta", chooz_sig},
    {NULL, NULL}
  };


#define BIN_LIST 1
#define SAMPLING_LIST 2
#define DENSITY_LIST 3


/* Put arithmetic functions in table.
 * And all user-defined stuff.
 */
static void
init_table (void)
{
  int i;
  glb_symrec *ptr,*p;

  glb_namerec *sptr;
  for (i = 0; arith_fncts[i].fname != 0; i++)
    {
      ptr = glb_putsym (arith_fncts[i].fname, FNCT);
      ptr->value.fnctptr = arith_fncts[i].fnct;
      ptr->destroy=arith_fncts[i].destroy;
      ptr->reverse=arith_fncts[i].reverse;

    }
  for (i = 0; list_fncts[i].fname != 0; i++)
    {
      ptr = glb_putsym (list_fncts[i].fname, FNCT);
      ptr->value.lfnctptr = list_fncts[i].fnct;
    }

  ptr=pre_sym_table;
  /* Copying the contents of pre_sym_table to sym_table */
  for (i=0; ptr !=0; i++)
    {
      p=glb_putsym(ptr->name,ptr->type);
      p->value.var=ptr->value.var;
      if(ptr->list!=NULL) p->list=thread_list(&glb_list_copy,0,0,ptr->list);
    ptr=ptr->next;
    }

  for (i = 0; sig_fncts[i].fname != 0; i++)
    {
      sptr = glb_putname (sig_fncts[i].fname,"energy",SFNCT);
      sptr->sf = sig_fncts[i].sf;
    }
}

static void
free_symtable()
{
    glb_symrec *ptr;
    glb_symrec *dummy;
    ptr=sym_table;
    while(ptr != (glb_symrec *) NULL)
      {
        glb_free(ptr->name);
        if(ptr->list!=NULL){ list_free(ptr->list);}
        dummy=ptr->next;
        glb_free(ptr);
        ptr=dummy;
      }
    sym_table=NULL;
}

static void
free_presymtable()
{
    glb_symrec *ptr;
    glb_symrec *dummy;
    ptr=pre_sym_table;
    while(ptr != (glb_symrec *) NULL)
      {
        glb_free(ptr->name);
        if(ptr->list!=NULL){ list_free(ptr->list);}
        dummy=ptr->next;
        glb_free(ptr);
        ptr=dummy;
      }
    pre_sym_table=NULL;
}




static void
free_nametable()
{
    glb_namerec *ptr;
    glb_namerec *dummy;
    ptr=name_table;
    while(ptr != (glb_namerec *) NULL)
      {
        glb_free(ptr->name);
        glb_free(ptr->context);
        dummy=ptr->next;
        glb_free(ptr);
        ptr=dummy;
      }
    name_table=NULL;
}

glb_symrec *
glb_putsym (char *sym_name, int sym_type)
{
  glb_symrec *ptr;
  ptr = (glb_symrec *) glb_malloc (sizeof (glb_symrec));
  ptr->name = (char *) glb_malloc (strlen (sym_name) + 1);
  strcpy (ptr->name,sym_name);
  ptr->type = sym_type;
  ptr->value.var = GLB_NAN; /* set value to GLB_NAN, which ensures an
                               error if an undefined variable is used  */
  ptr->list= NULL;
  ptr->next = (struct glb_symrec *) sym_table;
  sym_table = ptr;
  return ptr;
}





/* Name handling */

static glb_naming *glb_putnames (char *sym_name, char *context, int value,
                                 glb_naming *in)
{
  glb_naming *ptr;
  ptr = (glb_naming *) glb_malloc (sizeof (glb_naming));
  ptr->name = (char *) glb_malloc (strlen (sym_name) + 1);
  strcpy (ptr->name,sym_name);
  ptr->context = (char *) glb_malloc (strlen (context) + 1);
  strcpy (ptr->context,context);
  ptr->value = value; /* set value to -1 for new ones  */

  ptr->next = (struct glb_naming *) in;
  //in = ptr;
  return ptr;
}

static glb_naming *copy_names (glb_naming *in)
{

  glb_namerec *ptr;
  for (ptr = name_table; ptr != (glb_namerec *) NULL;
       ptr = (glb_namerec *)ptr->next)
    {
      in=glb_putnames(ptr->name,ptr->context,ptr->value,in);
    }
  return in;
}



glb_namerec *glb_getname (const char *sym_name, char* context)
{
  glb_namerec *ptr;
  for (ptr = name_table; ptr != (glb_namerec *) NULL;
       ptr = (glb_namerec *)ptr->next)
    if (strcmp (ptr->name,sym_name) == 0 )
        return ptr;
  return 0;
}

glb_namerec *glb_putname (char *sym_name, char *context, int sym_type)
{
  glb_namerec *ptr;
  ptr = (glb_namerec *) glb_malloc (sizeof (glb_namerec));
  ptr->name = (char *) glb_malloc (strlen (sym_name) + 1);
  strcpy (ptr->name,sym_name);
  ptr->context = (char *) glb_malloc (strlen (context) + 1);
  strcpy (ptr->context,context);
  ptr->type = sym_type;
  ptr->value = -1; /* set value to -1 for new ones  */

  ptr->next = (struct glb_namerec *)name_table;
  name_table = ptr;
  return ptr;
}

glb_symrec *
glb_getsym (const char *sym_name)
{
  glb_symrec *ptr;
  for (ptr = sym_table; ptr != (glb_symrec *) NULL;
       ptr = (glb_symrec *)ptr->next)
    if (strcmp (ptr->name,sym_name) == 0)
      return ptr;
  return 0;
}
/* User pre-def variables */
glb_symrec *
glb_putpresym (const char *sym_name, int sym_type)
{
  glb_symrec *ptr;
  ptr = (glb_symrec *) glb_malloc (sizeof (glb_symrec));
  ptr->name = (char *) glb_malloc (strlen (sym_name) + 1);
  strcpy (ptr->name,sym_name);
  ptr->type = sym_type;
  ptr->value.var = 0; /* set value to 0 even if fctn.  */
  ptr->list=NULL;
  ptr->next = (struct glb_symrec *) pre_sym_table;
  pre_sym_table = ptr;
  return ptr;
}

glb_symrec *
glb_getpresym (const char *sym_name)
{
  glb_symrec *ptr;
  for (ptr = pre_sym_table; ptr != (glb_symrec *) NULL;
       ptr = (glb_symrec *)ptr->next)
    if (strcmp (ptr->name,sym_name) == 0)
      return ptr;
  return 0;
}


/* A new an powerful function which allows the user to define variables
 * for substitution in the AEDL files
 */


void glbDefineAEDLVariable(const char *name, double value)
{
  glb_symrec *ptr;
  ptr=glb_getpresym(name);
  if(ptr==0) ptr = glb_putpresym (name, VAR);
  ptr->value.var = value;
  return;
}

double glbGetAEDLVariable(const char *name)
{
  glb_symrec *ptr;
  ptr=glb_getpresym(name);
  if (!ptr)
    return GLB_NAN;
  return ptr->value.var;
}

void glbClearAEDLVariables()
{
   if(pre_sym_table!=NULL) free_presymtable();
}


/* same for lists */

void glbDefineAEDLList(const char *name, double *list, size_t length)
{
  size_t i;
  glb_symrec *ptr;
  if(name==NULL) return;
  if(name[0]!='%'){ fprintf(stderr,"ERROR: Names of AEDL lists have to start with %%\n");return;}
  ptr=glb_getpresym(name);
  if(ptr==0) ptr = glb_putpresym (name, LVAR);
  for(i=0;i<length;i++) {ptr->list=list_cons(ptr->list,list[i]);
  }

  return;
}




void glb_copy_buff()
{
  /* I am not sure how well this assigment really works */
  buff.names=copy_names(buff.names);
  if (buff.filename)  glb_free(buff.filename);
  buff.filename=strdup(glb_file_id);
  buff_list[exp_count]  = glbAllocExp();
  *buff_list[exp_count] = buff;
  if (buff.parent)
  {
    struct glb_experiment *p = buff.parent;
    glbExpRemoveChild(p, &buff);
    glbExpAddChild(p, buff_list[exp_count]);
  }
  exp_count++;
}


/***************************************************************************
 * Function glbReset                                                       *
 ***************************************************************************
 * Resets the parser's internal data structures to prepare for the parsing *
 * of a new experiment.                                                    *
 ***************************************************************************/
void glbReset()
{
  glb_line_num =  0;
  energy_len   =  1;
  energy_count = -1;
  loc_count    = -1;
  flux_count   = -1;
  glbResetNuisance();
  glbInitExp(&buff);

  if(name_table!=NULL) free_nametable();
  if(sym_table!=NULL) free_symtable();
  name_table =(glb_namerec *) NULL;
  sym_table =(glb_symrec *) NULL;
  init_table ();
}


/***************************************************************************
 * Function glbNewDetector                                                 *
 ***************************************************************************
 * Starts a new detector that inherits everything from the previously      *
 * defined one, including the namespace.                                   *
 ***************************************************************************/
void glbNewDetector()
{
  glb_copy_buff();
  energy_len = 1;
  glbResetNuisance();
  glbInitExpFromParent(&buff, buff_list[exp_count-1]);
     //FIXME FIXME FIXME What if parent is not/incorrectly defined?

  // Remove rule names from namespace (rules are not copied by glbInitExpFromParent)
  glb_namerec **ptr = &name_table;
  while (*ptr)
  {
    if (strcmp((*ptr)->context, "rule") == 0)
    {
      glb_namerec *ptr_old = *ptr;
      *ptr = ptr_old->next;
      glb_free(ptr_old->name);        ptr_old->name    = NULL;
      glb_free(ptr_old->context);     ptr_old->context = NULL;
      ptr_old->next   = NULL;
      ptr_old->sf     = NULL;
      ptr_old->type   = ptr_old->value = -1;
      glb_free(ptr_old);
    }
    else
      ptr = &((*ptr)->next);
  }

}

void glbResetNuisance()
{
  nuis.name        = NULL;
  nuis.error       = GLB_NAN;
  nuis.a           = GLB_NAN;
  nuis.systype = 0;
  nuis.n_energies  = -1;
  nuis.energy_list = NULL;
  nuis.error_list  = NULL;
  nuis.a_list      = NULL;
  nuis.ref_count   = 1;
}

void glbResetCounters()
{
  flux_count=-1;
  cross_count=-1;
}

void glbResetEOF()
{
  int i;
  exp_count=0;
  glb_line_num=0;
  energy_len=1;
  energy_count=-1;
  loc_count=-1;
  flux_count=-1;
  glbResetNuisance();
  glbInitExp(&buff);
  for(i=0;i < GLB_MAX_EXP; i++)
    if (buff_list[i])
    {
      glbFreeExp(buff_list[i]);
      buff_list[i] = NULL;
    }
  /* this here would be the place to check for unuses variables, but
     for that we need an access counter */
  if(name_table!=NULL) free_nametable();
  if(sym_table!=NULL) free_symtable();

  name_table =(glb_namerec *) NULL;
  sym_table =(glb_symrec *) NULL;
  init_table ();
  glb_reset_flux(&flt);
  glb_reset_xsec(&xsc);

}

void glb_clean_parser()
{
  if(name_table!=NULL) free_nametable();
  if(sym_table!=NULL) free_symtable();
  if(pre_sym_table!=NULL) free_presymtable();

}

int glbInitExperiment(char *inf,glb_exp *in, int *counter)
{
  FILE *input;
  int k,i;
  const char tch[]="%!GLoBES";
  char tct[11];
  struct glb_experiment **ins;
  ins=(struct glb_experiment **) in;

  // yydebug=1;
  context=(char *) strdup("global");
  glb_smear_reset(&ibf);
  glb_option_type_reset(&opt);
  memset(&flt, 0, sizeof(flt));
  glb_reset_flux(&flt);
  memset(&xsc, 0, sizeof(xsc));
  glb_reset_xsec(&xsc);
  input=glb_fopen(inf,"r");
  if(input==NULL) return -2;
  /* This line produces a warning with -Wall:
   * 'warning: char format, different type arg (arg 3)'
   * I think however that understand the problem.
   */
  fscanf(input,"%8c",&tct);
  if(strncmp(tch,tct,8)!=0) {glb_error("Not a GLoBES file!"); return -2;}
  glb_fclose(input);
  yyin=glb_fopen(inf,"r");
  if(yyin==NULL) return -2;
  glb_file_id=(char*) strdup(inf);
  glbResetEOF();
  k=yyparse ();

  /* Copy last experiment and reset data structures */
  glb_copy_buff();
  glbReset();

  glb_fclose(yyin);
  glb_free(context);
  glb_free(glb_file_id);

  if(k!=0) return -2;

  k=0;

  if(*counter+exp_count>GLB_MAX_EXP) glb_fatal("Too many experiments!");
  for(i=0;i<exp_count;i++)
    {
      ins[*counter+i] = buff_list[i];
      buff_list[i]    = NULL; /* Remove our pointer to that exp to prevent destruction */
      k              += glbDefaultExp(ins[*counter+i]);
    }
  (*counter)= (*counter) + exp_count;

  /* Reallocate data structures for the minimizer */
  glb_init_minimizer();

  if(k!=0) return -1;

  return 0;

}

