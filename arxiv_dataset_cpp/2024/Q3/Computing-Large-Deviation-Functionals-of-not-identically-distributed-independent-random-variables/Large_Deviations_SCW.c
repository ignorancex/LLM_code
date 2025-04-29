/* Large_Deviations_SCW.c */
/* Computes the I-functions for the LD approach to the SCW, with w = 0.*/
#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include "newdefs.hd"
#include "file_in1.hd"
#include "linalg.h"
#include "stat2013.h"
#include "smallfun.hd"

#define SVARS struct input_params *VARS
#define AVARS &VARS

struct input_params
{
 int Sim;  
 int Seed;  
 int N;  
 int Extra;
 int trapN;
 int ChooseI;
 int Biastwoone;
 int Biastwotwo;
 int Biasone;
 double x;
 double eps;
 double Gammatwoone;
 double Gammatwotwo;
 double Gammaone;
};

double trapk(double x,double eps,double thetaone,double thetatwo,
double a,double b,int n,SVARS);
double trapinvq(double x,double eps,double thetaone,double thetatwo,
double a,double b,int n,SVARS);
double integraloneoverq(double x,double eps,double thetaone,double thetatwo,SVARS);
int choosethetas(double x,double eps,double *thetaone,double *thetatwo,int chooseI,SVARS);
double h(double x,double eps,double thetaone,double thetatwo,double y,SVARS);
double trapezoidparone(double x,double eps,double thetaone,double thetatwo,
double a,double b,int n,SVARS);
double trapezoidpartwo(double x,double eps,double thetaone,double thetatwo,
double a,double b,int n,SVARS);
double parcparthetaone(double x,double eps,double thetaone,double thetatwo,SVARS);
double parcparthetatwo(double x,double eps,double thetaone,double thetatwo,SVARS);
double k(double x,double eps,double thetaone,double thetatwo,SVARS);
double kalt(double x,double eps,double thetaone,double thetatwo,SVARS);
double trapezoidparktwo(double x,double eps,double thetaone,double thetatwo,
double a,double b,int n,SVARS);
int main()
{
   FILE *info, *in, *out1, *out2, *out3, *out4, *out5, *out6;

  int intnum,floatnum,named_ints,named_floats,Out_namelength;
  struct cvector names,Out1,Out2,Out3,Out4,Out5,Out6;
  struct ivector ints;struct vector doubles;

  char info_file[] = "SCW.inf";
  char rt[] = "main";

  /*long time_st,time_fin;

  time(&time_st);*/

  Out_namelength = 9;

  /* Out_namelength must be one more than the number of
  characters in the names of output files. */

  intnum = 300;
  floatnum = 400;

  named_ints = 50;
  named_floats = 50; 
  
  incvec(&names,1,25*(named_ints+named_floats),"names");
  incvec(&Out1,1,Out_namelength,"Out1");
  incvec(&Out2,1,Out_namelength,"Out2");
  incvec(&Out3,1,Out_namelength,"Out3");
  incvec(&Out4,1,Out_namelength,"Out4");
  incvec(&Out5,1,Out_namelength,"Out5");
  incvec(&Out6,1,Out_namelength,"Out6");

  inivec(&ints,1,intnum,"ints");invec(&doubles,1,floatnum,"doubles");

  if(intnum < 101) nrerror("Error in main: not enough ints."); else;
  /* For passing to named_x routines: */
  /* WARNING: ints[99] and ints[100] are reserved for passing
     intnum and named_ints and named_floats to routines. */
  
  if( (info = fopen(info_file,"wt")) != NULL){
 
  if( (in = fopen("SCW.in","rt")) != NULL){

/* For "scanin", message length is assumed < 50 characters;
and variable namelength is <= 25 characters. Output file namelength
is derived from vector "Outx" and must be one more than
the number of characters in the output file names.
The routine scans in the message, the
output file names, the variable names (stored in cvector "names"),
and the ints and doubles (stored in ivector "ints" and
vector "doubles"). Don't put empty lines between
the information lines, and be sure that the variable names
don't extend past 25 characters. */

/* void scanin(FILE *in,FILE *info,int named_ints,int named_floats,
struct cvector *Out1,struct cvector *Out2,struct cvector *Out3,
struct cvector *Out4,struct cvector *Out5, struct cvector *Out6,
struct cvector *names,struct ivector *ints,struct vector *doubles) */ 

    scanin(in,info,named_ints,named_floats,
     &Out1,&Out2,&Out3,&Out4,&Out5,&Out6,&names,&ints,&doubles);

    fclose(in);

  } else nrerror("error in main: input file didn't open");

 struct input_params VARS;

 printf("Program info in %s\n",info_file);

   /* Input ends here */

 if( (out1 = fopen(&Out1.vec[1],"wt")) != NULL){
 if( (out2 = fopen(&Out2.vec[1],"wt")) != NULL){
 if( (out3 = fopen(&Out3.vec[1],"wt")) != NULL){
 if( (out4 = fopen(&Out4.vec[1],"wt")) != NULL){
 if( (out5 = fopen(&Out5.vec[1],"wt")) != NULL){
 if( (out6 = fopen(&Out6.vec[1],"wt")) != NULL){

/*struct input_params
{
 int Sim;  
 int Seed;  
 int N;  
 int Extra;
 double x;
 double eps;
 double Gammatwoone;
 double Gammatwotwo;
 double Gammaone;
};*/


   VARS.Sim = named_int("Sim",rt,ANID);
   VARS.Seed = named_int("Seed",rt,ANID);
   VARS.N = named_int("N",rt,ANID);
   VARS.Extra = named_int("Extra",rt,ANID);
   VARS.trapN = named_int("trapN",rt,ANID);
   VARS.ChooseI = named_int("ChooseI",rt,ANID);
   VARS.Biasone = named_int("Biasone",rt,ANID);
   VARS.Biastwoone = named_int("Biastwoone",rt,ANID);
   VARS.Biastwotwo = named_int("Biastwotwo",rt,ANID);
 
   VARS.x = named_float("x",rt,ANID);
   VARS.eps = named_float("eps",rt,ANID);
   VARS.Gammatwoone = named_float("Gammatwoone",rt,ANID);
   VARS.Gammatwotwo = named_float("Gammatwotwo",rt,ANID);
   VARS.Gammaone = named_float("Gammaone",rt,ANID);

   srandom(VARS.Seed);

 if(VARS.Sim == 1){
   int k,n,ok,success,num;
   double dum,thetaone,thetatwo,x,eps,rr,y;
   x = VARS.x;
   eps = VARS.eps;
  
   success=0; 
   for(n=1;n<=VARS.N;n++){
     ok = choosethetas(x,eps,&thetaone,&thetatwo,2,AVARS);
     if(ok == 1){
     for(k=1;k<=VARS.N;k++){
      rr = ran0();
      y = -rr +(1.0-rr);
/*double h(double x,double eps,double thetaone,double thetatwo,double y,SVARS);*/
      dum = h(x,eps,thetaone,thetatwo,y,AVARS);
       if(dum < 0.5){
        printf("Good;h = %f.\n",dum);
        success++;
       }
       else printf("Oops; h = %f\n.",dum);
      }
     }

     /*else nrerror("Oops; not OK.");*/
     else;
   }
   num = VARS.N*VARS.N;

   printf("Success fraction = %f.\n",(1.0*success)/num);
  } else;

  if(VARS.Sim == 2){
   int n,ok,success;
   double sum,dum,dum1,thetaone,thetatwo,x,eps;
   x = VARS.x;
   eps = VARS.eps;

   success=0;sum = 0.0;
   for(n=1;n<=VARS.N;n++){

     ok = choosethetas(x,eps,&thetaone,&thetatwo,2,AVARS);
     if(ok==1){
      success++;
      dum = trapinvq(x,eps,thetaone,thetatwo,-1.0,1.0,VARS.trapN,AVARS);
      dum1 = integraloneoverq(x,eps,thetaone,thetatwo,AVARS);
      /* printf("the discrepancy was %f.\n",dum1 - dum);*/
      sum = sum + fabs(dum1 - dum);
     } else;
    }

       printf("Success rate was %f.\n",1.0*success/VARS.N);
       printf("Average discrepancy was %f.\n",sum/VARS.N);
 } else; /* End Sim==2*/

  if(VARS.Sim == 3){
   int n,ok,success;
   double sum,dum,dum1,thetaone,thetatwo,x,eps;
   x = VARS.x;
   eps = VARS.eps;

   success=0;sum = 0.0;
   for(n=1;n<=VARS.N;n++){

     ok = choosethetas(x,eps,&thetaone,&thetatwo,2,AVARS);
     if(ok==1){
      success++;
      dum = trapezoidparone(x,eps,thetaone,thetatwo,-1.0,1.0,VARS.trapN,AVARS);
      dum1 = parcparthetaone(x,eps,thetaone,thetatwo,AVARS);
       printf("trapezoidparone gave %f.\n",dum);
       printf("parcparthetaone gave %f.\n",dum1);
      sum = sum + fabs(dum1 - dum);
     } else;
    }

       printf("Success rate was %f.\n",1.0*success/VARS.N);
       printf("Average discrepancy was %f.\n",sum/VARS.N);
 } else; /* End Sim==3*/
     
  if(VARS.Sim == 4){
   int n,ok,success;
   double sum,dum,dum1,thetaone,thetatwo,x,eps;
   x = VARS.x;
   eps = VARS.eps;

   success=0;sum = 0.0;
   for(n=1;n<=VARS.N;n++){

     ok = choosethetas(x,eps,&thetaone,&thetatwo,2,AVARS);
     if(ok==1){
      success++;
      dum = trapezoidpartwo(x,eps,thetaone,thetatwo,-1.0,1.0,VARS.trapN,AVARS);
      dum1 = parcparthetatwo(x,eps,thetaone,thetatwo,AVARS);
      /* printf("the discrepancy was %f.\n",dum1 - dum);*/
      sum = sum + fabs(dum1 - dum);
     } else;
    }

       printf("Success rate was %f.\n",1.0*success/VARS.N);
       printf("Average discrepancy was %f.\n",sum/VARS.N);
 } else; /* End Sim==4*/
     

  if(VARS.Sim == 5){ /*Plots to find the constraint set.*/
   int n,ok,hits;
   double dumone,dumtwo,thetaone,thetatwo,x,eps;
   x = VARS.x;
   eps = VARS.eps;

   hits=0;
   for(n=1;n<=VARS.N;n++){
     ok = choosethetas(x,eps,&thetaone,&thetatwo,2,AVARS);
     if(ok==1){
      dumone = parcparthetaone(x,eps,thetaone,thetatwo,AVARS);
      dumtwo = parcparthetatwo(x,eps,thetaone,thetatwo,AVARS);
      if((dumone < 0.0)&&(dumtwo > 0.0)){
       hits++;
       fprintf(out1,"%f\n",thetaone);
       fprintf(out2,"%f\n",thetatwo);
     } 
     else{
       fprintf(out3,"%f\n",thetaone);
       fprintf(out4,"%f\n",thetatwo); 
     } 

    } else;
   }
     
       printf("Number of hits was %i.\n",hits);
 } else; /* End Sim==5*/

  if(VARS.Sim == 6){ /*Approximates infimums by sampling */
   int j,m,n,ok,success,successok;
   double storethetaone,storethetatwo,storeone,storetwo,dum,dumone,dumtwo,I,thetaone,thetatwo,x,eps;
   x = VARS.x;
   eps = VARS.eps;

  for(j=1;j<=2;j++){
   VARS.ChooseI = j;
   success=0;I = 1.0e+100;successok=0;
   for(m=1;m<=VARS.Extra;m++){
   for(n=1;n<=VARS.N;n++){

    if(VARS.ChooseI == 1) ok = choosethetas(x,eps,&thetaone,&thetatwo,1,AVARS);
    else ok = choosethetas(x,eps,&thetaone,&thetatwo,2,AVARS);
    if(ok==1){
      successok++;
      dumone = parcparthetaone(x,eps,thetaone,thetatwo,AVARS);
      dumtwo = parcparthetatwo(x,eps,thetaone,thetatwo,AVARS);
     if(VARS.ChooseI == 2){
      if((dumone < - 1.0e-5)&&(dumtwo > 1.0e-5)){
      success++;
       dum = k(x,eps,thetaone,thetatwo,AVARS);
     /* dum = kalt(x,eps,thetaone,thetatwo,AVARS);
      dum = trapk(x,eps,thetaone,thetatwo,-1.0,1.0,VARS.trapN,AVARS); */
      /*printf("k was %f.\n",dum);*/
      if(dum < I){
        I = dum; 
        storeone = dumone;
        storetwo = dumtwo;
        storethetaone = thetaone;
        storethetatwo = thetatwo;
      } else;
      } else;
      }
     else{ 
      if(dumone < - 1.0e-2){
      success++;
       dum = k(x,eps,thetaone,thetatwo,AVARS);
      /* dum = kalt(x,eps,thetaone,thetatwo,AVARS);*/
      /*printf("k was %f.\n",dum);*/
     /* printf("Thetaone was %f.\n",thetaone);*/
      if(dum < I){
       I = dum;
       storeone = dumone;
        storethetaone = thetaone;
       } else;
      } else;
     }

    } else;
   }
  }
     printf("OK rate was %g.\n",1.0*successok/(VARS.Extra*VARS.N));
     printf("Success rate was %g.\n",1.0*success/(VARS.Extra*VARS.N));
     if(VARS.ChooseI == 1){
        printf("I_1 was %g.\n",I);
        printf("partial c/partial thetaone was %g.\n",storeone);
       printf("thetaone was %g.\n",storethetaone);
     }
     else{
       printf("I__2 was %g.\n",I);
       printf("partial c/partial thetaone was %g.\n",storeone);
       printf("partial c/partial thetatwo was %g.\n",storetwo);
       printf("thetaone was %g.\n",storethetaone);
       printf("thetatwo was %g.\n",storethetatwo);
   }
  } 
 } else; /* End Sim==6*/
 
  if(VARS.Sim == 7){
   int n,ok,success;
   double min,dum,thetaone,thetatwo,x,eps;
   x = VARS.x;
   eps = VARS.eps;

   min = 1.0e+10;
   success=0;
   for(n=1;n<=VARS.N;n++){

     ok = choosethetas(x,eps,&thetaone,&thetatwo,2,AVARS);
     if(ok==1){
      if(thetatwo > 0.0){
      success++;
      dum = trapezoidparktwo(x,eps,thetaone,thetatwo,-1.0,1.0,VARS.trapN,AVARS);
       printf("%f = ;",dum);
       if(dum < min) min = dum;else; 
     } else;
     } else;
    }

       printf("Success rate was %f.\n",1.0*success/VARS.N);
       printf("Minimum integral was %f.\n",min);
 } else; /* End Sim==7*/

  if(VARS.Sim == 8){
     int n;
     double A,B,C,x,disc,eps,dthetatwo,thetatwo,thetaone,thetatwomax,thetatwomin;
     x = VARS.x;
     eps = VARS.eps;
     thetatwomax = 4.0;
     thetatwomin = -4.0;
     dthetatwo = (thetatwomax - thetatwomin)/(VARS.N) + 0.000026534;

   for(n=1;n<=VARS.N;n++){
     thetatwo = thetatwomin + dthetatwo*n;
     A = 4.0*(1-x);B= -2.0*(2*thetatwo*eps + 1);C = thetatwo*thetatwo; 
     disc = B*B - 4*A*C;
     if(disc>0.0){
      thetaone = solve_quad(A,B,C,1);
      if(fabs(thetaone) > 0.0){
       fprintf(out1,"%f\n",thetaone);
       fprintf(out2,"%f\n",thetatwo);
      } else;
      thetaone = solve_quad(A,B,C,-1);
      if(fabs(thetaone) > 0.0){
       fprintf(out1,"%f\n",thetaone);
       fprintf(out2,"%f\n",thetatwo);
      } else;
     } else;

   }
    

 } else; /* End Sim==8*/
    
  if(VARS.Sim == 9){
     int n;
     double x,eps,dthetatwo,thetatwo,thetatwomax,thetatwomin;
     x = VARS.x;
     eps = VARS.eps;
     thetatwomax = 4.0;
     thetatwomin = -4.0;
     dthetatwo = (thetatwomax - thetatwomin)/(VARS.N) + 0.000026534;
   for(n=1;n<=VARS.N;n++){
     thetatwo = thetatwomin + dthetatwo*n;
       fprintf(out2,"%f\n",thetatwo);
       fprintf(out3,"%f\n",-(0.5+thetatwo*(1.0+eps))/x);
       fprintf(out4,"%f\n",-(0.5-thetatwo*(1.0-eps))/x);
       fprintf(out5,"%f\n",0.5*thetatwo);
       fprintf(out6,"%f\n",-0.5*thetatwo);
    }
     
 } else; /* End Sim==9*/

  if(VARS.Sim == 10){

       fprintf(out1,"%f\n",0.1);
       fprintf(out2,"%f\n",1.688);
       fprintf(out3,"%f\n",1.692);

       fprintf(out1,"%f\n",0.2);
       fprintf(out2,"%f\n",0.9958);
       fprintf(out3,"%f\n",0.9967);

       fprintf(out1,"%f\n",0.3);
       fprintf(out2,"%f\n",0.592);
       fprintf(out3,"%f\n",0.601);

       fprintf(out1,"%f\n",0.4);
       fprintf(out2,"%f\n",0.3185);
       fprintf(out3,"%f\n",0.3456);

       fprintf(out1,"%f\n",0.5);
       fprintf(out2,"%f\n",0.131);
       fprintf(out3,"%f\n",0.192);

       fprintf(out1,"%f\n",0.6);
       fprintf(out2,"%f\n",0.0229);
       fprintf(out3,"%f\n",0.1394);

       fprintf(out1,"%f\n",0.7);
       fprintf(out2,"%f\n",0.0);
       fprintf(out3,"%f\n",0.139);

 } else; /* End Sim==10*/

  if(VARS.Sim == 11){
   int n,ok,success;
   double sum,dum,dum1,thetaone,thetatwo,x,eps;
   x = VARS.x;
   eps = VARS.eps;

   success=0;sum = 0.0;
   for(n=1;n<=VARS.N;n++){

     ok = choosethetas(x,eps,&thetaone,&thetatwo,2,AVARS);
     if(ok==1){
      success++;
      dum = k(x,eps,thetaone,thetatwo,AVARS);
      dum1 = trapk(x,eps,thetaone,thetatwo,-1.0,1.0,VARS.trapN,AVARS);
      /* printf("k = %f; trapk =%f.\n",dum,dum1);*/
      sum = sum + fabs(dum1 - dum);
     } else;
    }

       printf("Success rate was %f.\n",1.0*success/VARS.N);
       printf("Average discrepancy was %f.\n",sum/VARS.N);
 } else; /* End Sim==11*/
  
  if(VARS.Sim == 12){
   int n,N,stop;
   double p,x,eps,G,dth,left,thetaone;
   N = VARS.N;
   x = VARS.x;
   eps = VARS.eps;
   left = -0.5/x;
   dth = left/N;
 
   stop=0;thetaone = dth;p=1.0e+10;
/*double integraloneoverq(double x,double eps,double thetaone,double thetatwo,SVARS);*/
   G= -1.0 + 0.5*integraloneoverq(x,eps,thetaone,0.0,AVARS);
   if(G < 0.0);  
   else nrerror("Oops, G >=0 on first step!");

   while((stop==0)&&(n < N -2)){
    thetaone = thetaone + dth; 
    G= -1.0 + 0.5*integraloneoverq(x,eps,thetaone,0.0,AVARS);
    if(G < 0.0) n++;  
    else{
     stop++; 
     p = thetaone;
    }
   }

   if(stop ==0) p = left;else;

   printf("Final thetaone = %f while left = %f.\n",thetaone,left);
   printf("x = %f, left = %f; p = %f.\n",x,left,p);
 } else; /* End Sim==12*/

  
     
    } else fnrerror(info,"error in main: out6 didn't open");
    fclose(out6);
   } else fnrerror(info,"error in main: out5 didn't open");
    fclose(out5);
  } else fnrerror(info,"error in main: out4 didn't open");
    fclose(out4);
 } else fnrerror(info,"error in main: out3 didn't open");
    fclose(out3);
} else fnrerror(info,"error in main: out2 didn't open");
    fclose(out2);
} else fnrerror(info,"error in main: out1 didn't open");
   fclose(out1);

/*  time(&time_fin);
  printf("Elapsed time = %i seconds\n",time_fin - time_st);
  fprintf(info,"Elapsed time = %i seconds\n",time_fin - time_st);
*/
    fclose(info);
  } else nrerror("error in main: info didn't open");

    freeivec(&ints);freevec(&doubles);
    freecvec(&Out1);freecvec(&Out2);freecvec(&Out3);
    freecvec(&Out4);freecvec(&Out5);freecvec(&Out6);

  printf("Done!\n");
  return 1;
}

double h(double x,double eps,double thetaone,double thetatwo,double y,SVARS)
{

 double ret;

  ret = thetaone*(1.0 - x - y*y) + thetatwo*(y - eps);

   return ret;
}

double b(double x,double eps,double thetaone,double thetatwo,SVARS)
{

 double ret;

  ret = 1.0 - 2*thetaone*(1.0 - x) + 2*thetatwo*eps;

   return ret;
}


double q(double x,double eps,double thetaone,double thetatwo,double y,SVARS)
{

   double ret;

   ret = 1.0 - 2*h(x,eps,thetaone,thetatwo,y,VARS);

   return ret;
}

double invq(double x,double eps,double thetaone,double thetatwo,double y,SVARS)
{

   double ret,dum;

   dum = q(x,eps,thetaone,thetatwo,y,VARS);
   if(dum > 0.0) ret = 1.0/dum;
   else nrerror("Error in invq: q not positive?.");

   return ret;
}

double logq(double x,double eps,double thetaone,double thetatwo,double y,SVARS)
{

   double ret,dum;

   dum = q(x,eps,thetaone,thetatwo,y,VARS);
   if(dum > 0.0) ret = log(dum);
   else nrerror("Error in invq: q not positive?.");

   return ret;
}

double trapinvq(double x,double eps,double thetaone,double thetatwo,
double a,double b,int n,SVARS)
{
 int i;double dx,sum;
 if(n <=1) nrerror("Error in trapezoid: n<=1.");else;
 if(b > a)  dx = (b - a)/n;
 else nrerror("Error in trapezoid: b <= a.");
 sum =0.0;
 for(i=1;i<=(n-1);i++) sum = sum + invq(x,eps,thetaone,thetatwo,a + i*dx,VARS);
 sum = sum + 0.5*( invq(x,eps,thetaone,thetatwo,a,VARS) + invq(x,eps,thetaone,thetatwo,b,VARS) );
 sum = sum*dx;
 return sum;
}

double trapk(double x,double eps,double thetaone,double thetatwo,
double a,double b,int n,SVARS)
{
 int i;double dx,sum;
 if(n <=1) nrerror("Error in trapezoid: n<=1.");else;
 if(b > a)  dx = (b - a)/n;
 else nrerror("Error in trapezoid: b <= a.");
 sum =0.0;
 for(i=1;i<=(n-1);i++) sum = sum + invq(x,eps,thetaone,thetatwo,a + i*dx,VARS)
    + logq(x,eps,thetaone,thetatwo,a + i*dx,VARS);
 sum = sum + 0.5*( invq(x,eps,thetaone,thetatwo,a,VARS) + invq(x,eps,thetaone,thetatwo,b,VARS) )
 + 0.5*( logq(x,eps,thetaone,thetatwo,a,VARS) + logq(x,eps,thetaone,thetatwo,b,VARS) );
 sum = -1.0 + 0.5*sum*dx;
 return sum;
}

double trapezoidparone(double x,double eps,double thetaone,double thetatwo,
double a,double b,int n,SVARS)
{
 int i;double y,dum,dx,sum;
 if(n <=1) nrerror("Error in trapezoid: n<=1.");else;
 if(b > a)  dx = (b - a)/n;
 else nrerror("Error in trapezoid: b <= a.");
 sum =0.0;
 for(i=1;i<=(n-1);i++){
  y = a + i*dx;
  dum = 1.0 - x - y*y;
  sum = sum + dum*invq(x,eps,thetaone,thetatwo,y,VARS);
  }
 sum = sum + 0.5*( (1.0 - x - a*a)*invq(x,eps,thetaone,thetatwo,a,VARS) + 
(1.0 - x -b*b)*invq(x,eps,thetaone,thetatwo,b,VARS) );
 sum = sum*dx;
 return sum;
}

double trapezoidpartwo(double x,double eps,double thetaone,double thetatwo,
double a,double b,int n,SVARS)
{
 int i;double y,dum,dx,sum;
 if(n <=1) nrerror("Error in trapezoid: n<=1.");else;
 if(b > a)  dx = (b - a)/n;
 else nrerror("Error in trapezoid: b <= a.");
 sum =0.0;
 for(i=1;i<=(n-1);i++){
  y = a + i*dx;
  dum = y - eps;
  sum = sum + dum*invq(x,eps,thetaone,thetatwo,y,VARS);
  }
 sum = sum + 0.5*( (a - eps)*invq(x,eps,thetaone,thetatwo,a,VARS) + 
(b - eps)*invq(x,eps,thetaone,thetatwo,b,VARS) );
 sum = sum*dx;
 return sum;
}

double kintegrand(double x,double eps,double thetaone,double thetatwo,double y,SVARS)
{

   double ret,dum;

   dum = q(x,eps,thetaone,thetatwo,y,VARS);
   if(fabs(dum) > 0.0);
   else nrerror("Error in invq: q = 0?.");

   ret =( (y - eps)*(1.0 - dum) )/(dum*dum);
   return ret;
}


double trapezoidparktwo(double x,double eps,double thetaone,double thetatwo,
double a,double b,int n,SVARS)
{
 int i;double dx,sum;
 if(n <=1) nrerror("Error in trapezoid: n<=1.");else;
 if(b > a)  dx = (b - a)/n;
 else nrerror("Error in trapezoid: b <= a.");
 sum =0.0;
 for(i=1;i<=(n-1);i++) sum = sum + kintegrand(x,eps,thetaone,thetatwo,a + i*dx,VARS);
 sum = sum + 0.5*( kintegrand(x,eps,thetaone,thetatwo,a,VARS) 
   + kintegrand(x,eps,thetaone,thetatwo,b,VARS) );
 sum = sum*dx;
 return sum;
}

double integraloneoverq(double x,double eps,double thetaone,double thetatwo,SVARS)
{

  double ret,denom,rplus,rminus,A,B,C,E,disc,sqrdisc,fac,dac,dumone,dumtwo;

   disc = 4.0*thetatwo*thetatwo - 8.0*thetaone*b(x,eps,thetaone,thetatwo,VARS);
  
  if(disc > 0.0){

    sqrdisc = sqrt(disc);

   rplus = (2.0*thetatwo + sqrdisc)/(4.0*thetaone);  
   rminus = (2.0*thetatwo - sqrdisc)/(4.0*thetaone);  

   if((fabs(rplus) > 1.0)&&(fabs(rminus) > 1.0));
    else nrerror("Oops, roots inside the unit interval? In routine integraloneoverq.");
   if((fabs(rplus  - rminus) > 0.0));
    else nrerror("Oops, roots are equal? In routine integraloneoverq.");

     E = 1.0/(rminus - rplus);
     
     if(fabs(thetaone) > 0.0)
          fac = E/(2.0*thetaone);
    else nrerror("Oops, thetaone = 0? In routine integraloneoverq.");

     denom = -1.0 - rminus;
     if(fabs(denom) > 0.0) dumone = (1.0 - rminus)/denom;
    else nrerror("Oops, denom = 0? In routine integraloneoverq.");
     denom = - 1.0 - rplus;
     if(fabs(denom) > 0.0) dumtwo = (1.0 - rplus)/denom;
    else nrerror("Oops, denom = 0? In routine integraloneoverq.");

     if((dumone > 0.0)&&(dumtwo > 0.0))
    ret = fac*(log(dumone) - log(dumtwo));
    else nrerror("Oops, args of log negative? In routine integraloneoverq.");

    }

    else{

     if(fabs(disc) > 0.0){

     if(fabs(thetaone) > 0.0){
        A = 2.0*thetaone; 
        B = thetatwo/A;  
        C = b(x,eps,thetaone,thetatwo,VARS) - A*B*B;
     }
    else nrerror("Oops, thetaone = 0? In routine integraloneoverq.");
    if(A*C > 0.0){ 
     fac = 1.0/sqrt(A*C);
     dac = sqrt(A/C);
     ret= fac*( atan(dac*(1.0-B)) - atan(dac*(-1.0-B)) );
     } 
    else nrerror("Oops, A.C <= 0? In routine integraloneoverq.");

    } else nrerror("Oops, disc = 0? In routine integraloneoverq.");

  }
    return ret;
 }


double parcparthetaone(double x,double eps,double thetaone,double thetatwo,SVARS)
{
     double A,B,C,qone,qmone,dum,dummy,ret;
    if(fabs(thetaone) > 0.0); else nrerror("Oops, thetaone = 0 in routine parctwo.");


      A = -1.0/(2*thetaone);
      B = 1.0 - x -A*b(x,eps,thetaone,thetatwo,VARS);
      C = 2*thetatwo*A;
      dum = integraloneoverq(x,eps,thetaone,thetatwo,VARS);
      qone = q(x,eps,thetaone,thetatwo,1.0,VARS);
      qmone = q(x,eps,thetaone,thetatwo,-1.0,VARS);
     if((qmone > 0.0)&&(qmone > 0.0)) dummy = (C/(4*thetaone))*log( qone/qmone); 
    else nrerror("Oops, qmone or qone not positive in routine parcparthetaone.");
   
    ret = 2*A + ( C*thetatwo/(2*thetaone) + B )*dum + dummy;
    return ret;
}



double parcparthetatwo(double x,double eps,double thetaone,double thetatwo,SVARS)
{
     double E,F,qone,qmone,dum,ret;
    if(fabs(thetaone) > 0.0); else nrerror("Oops, thetaone = 0 in routine parctwo.");

      dum = integraloneoverq(x,eps,thetaone,thetatwo,VARS);
      E = (thetatwo/(2*thetaone) - eps)*dum; 
      qone = q(x,eps,thetaone,thetatwo,1.0,VARS);
      qmone = q(x,eps,thetaone,thetatwo,-1.0,VARS);

     if((qmone > 0.0)&&(qmone > 0.0)) F = (1.0/(4*thetaone))*log( qone/qmone); 
    else nrerror("Oops, qmone or qone not positive in routine parcparthetaone.");

    ret = E + F;
    return ret;
}

double k(double x,double eps,double thetaone,double thetatwo,SVARS)
{
   double ret,qone,qmone;
   
    if(fabs(thetaone) > 0.0); else nrerror("Oops, thetaone = 0 in routine parctwo.");

      qone = q(x,eps,thetaone,thetatwo,1.0,VARS);
      qmone = q(x,eps,thetaone,thetatwo,-1.0,VARS);

     if((qmone > 0.0)&&(qmone > 0.0));
    else nrerror("Oops, qmone or qone not positive in routine k.");

  ret = -3.0 + 0.5*log(qone*qmone) - (0.25*thetatwo/thetaone)*log(qone/qmone) + 
       0.5*( 1.0 - thetatwo*thetatwo/thetaone + 
      2*b(x,eps,thetaone,thetatwo,VARS) )*integraloneoverq(x,eps,thetaone,thetatwo,VARS);
  
  return ret;
}

double kalt(double x,double eps,double thetaone,double thetatwo,SVARS)
{
   double ret,qone,qmone,F,c,pone,ptwo;
   
    if(fabs(thetaone) > 0.0); else nrerror("Oops, thetaone = 0 in routine parctwo.");

      qone = q(x,eps,thetaone,thetatwo,1.0,VARS);
      qmone = q(x,eps,thetaone,thetatwo,-1.0,VARS);

     if((qmone > 0.0)&&(qmone > 0.0)) F = log( qone/qmone); 
    else nrerror("Oops, qmone or qone not positive in routine k.");

     c = 2.0 - 0.5*log(qone*qmone) + (0.25*thetatwo/thetaone)*F + 
      ( 0.5*thetatwo*thetatwo/thetaone - 
      b(x,eps,thetaone,thetatwo,VARS) )*integraloneoverq(x,eps,thetaone,thetatwo,VARS);
 
   pone = parcparthetaone(x,eps,thetaone,thetatwo,VARS);
   ptwo = parcparthetatwo(x,eps,thetaone,thetatwo,VARS);

   ret = thetaone*pone + thetatwo*ptwo - c;

  return ret;
}

/*The next routine delivers a random sample between a and b, generated from  u, uniform
on [0,1]; if which = -1, biased towards a;
if which = 1, biased towards b. */
double biasedsampling(double a,double b,double gamma,double u,int which)
{
 double dum,ret,z;

   if(gamma <= 0.0)
    nrerror("Oops, gamma not positive in routine biasedsampling?");
   else;
   if(b <= a)
    nrerror("Oops, b <= a in routine biasedsampling?");
   else;
   if((u <0.0)||(u > 1.0))
    nrerror("Oops, u not in [-1,1] in routine biasedsampling?");
   else;

  z = (exp(gamma*(b-a)) - 1.0)/gamma;
  if(which > 0){
  ret = a + log(z*gamma*u + 1.0)/gamma;
  }
  else{
   dum = 1.0 - z*gamma*u*exp(-gamma*(b-a));
   if( dum > 0.0) ret = a - log(dum)/gamma;
    else nrerror("Oops, arg of log negative in routine biasedsampling?");
   }

   if((ret < a)||(ret > b))
    nrerror("Oops, x out of range in routine biasedsampling?");
   else;

  return ret;
}

int choosethetas(double x,double eps,double *thetaone,double *thetatwo,int chooseI,SVARS)
{
  int n,yes,ok;
  double A,B,C,dum,upper,rr,alpha,thone,thtwo,rat;
  if((eps > 0.0)&&(eps < 0.5));
    else nrerror("Oops, bad eps in routine choosethetas.");
  if((x > 0.0)&&(1.0 - x > 0.0));
    else nrerror("Oops, bad x in routine choosethetas.");

    if(chooseI == 2){
    rr = ran0();
   /*  alpha = - rr*x/(1.0+eps) + (1.0 - rr)*x/(1.0-eps);*/ 
       alpha = biasedsampling(-x/(1.0+eps)+1.0e-6,x/(1.0 - eps)-1.0e-6,VARS->Gammatwotwo,rr,
       VARS->Biastwotwo);

     A = alpha*alpha + 4.0*(1.0 - x - alpha*eps);
     B = (alpha*alpha - 2.0*alpha*eps)/x - 2.0;
     C = alpha*alpha/(4.0*x*x);


     upper = solve_quad(A,B,C,-1);
     dum = solve_quad(A,B,C,+1);
     if(dum > upper) upper = dum;
     if(upper > 0.0);
     else nrerror("Trouble in choosethetas:upper not positive");
 
    rr = ran0();
   /* thone =  - 0.5*rr/x + (1.0-rr)*upper; */
    thone = biasedsampling( -0.5/x+1.0e-6, upper-1.0e-6,VARS->Gammatwoone,rr,VARS->Biastwoone);
    }
    else{
    alpha = 0.0;
    rr = ran0();
   /* thone =  - 0.5*rr/x + (1.0-rr)*0.5/(1.0-x);*/
    thone = biasedsampling( -0.5/x+1.0e-6, 0.5/(1.0 - x)-1.0e-6,VARS->Gammaone,rr,VARS->Biasone);
    }

     
    if(fabs(thone) > 0.0);
    else nrerror("Oops, theta1 = 0 in routine choosethetas?");

    thtwo = alpha*(thone + 0.5/x);
  
     yes = 0;
   
   if(h(x,eps,thone,thtwo,1.0,VARS) < 0.5) yes++;else;
   if(h(x,eps,thone,thtwo,-1.0,VARS) < 0.5) yes++;else;

   rat = thtwo/(2*thone);
   if(fabs(rat) < 1.0){ 
    if(h(x,eps,thone,thtwo,rat,VARS) < 0.5) yes++;else;
   } else yes++;
   if(yes == 3){
    ok = 1;
    (*thetaone) = thone; (*thetatwo) = thtwo;
    } else ok = 0;
   return ok;
}











