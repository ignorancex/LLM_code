#include "common.h"

static double relbeg,relend,absbeg,absend;

static double bin2th(int ith,int logbin,int nth,double thmin,double thmax)
{
  if(logbin)
    return pow(10.,log10(thmin)+(ith+0.5)*log10(thmax/thmin)/nth);
  else
    return thmin+(ith+0.5)*(thmax-thmin)/nth;
}

static inline int th2bin(double cth,int logbin,
			 double n_logint,double log_th_max,
			 int nb_theta,double i_theta_max)
{
  int ith;
  cth=(MIN((1.),(cth)));
  
  if(logbin) {
    if(cth!=1) {
#ifdef _TRUE_ACOS
      cth=log10(acos((MIN(1,cth))));
#else //_TRUE_ACOS
      cth=1-MIN(1,cth);
      cth=0.5*log10(2*cth+0.3333333*cth*cth+
		     0.0888888889*cth*cth*cth);
#endif //_TRUE_ACOS
      ith=(int)(n_logint*(cth-log_th_max)+nb_theta);
    }
    else ith=-1;
  }
  else {
#ifdef _TRUE_ACOS
    cth=acos((MIN(1,cth)));
#else //_TRUE_ACOS
    cth=1-MIN(1,cth);
    cth=sqrt(2*cth+0.333333333*cth*cth+
	      0.08888888889*cth*cth*cth);
#endif //_TRUE_ACOS
    ith=(int)(cth*nb_theta*i_theta_max);
  }
  
  return ith;
}

void report_error(int level,char *fmt,...)
{
  va_list args;
  char msg[256];

  va_start(args,fmt);
  vsprintf(msg,fmt,args);
  va_end(args);
  
  if(level) {
    fprintf(stderr," Fatal error: %s",msg);
    exit(level);
  }
  else
    fprintf(stderr," Warning: %s",msg);
}

void *my_malloc(size_t size)
{
  void *outptr=malloc(size);
  if(outptr==NULL) report_error(1,"Out of memory\n");

  return outptr;
}

void *my_calloc(size_t nmemb,size_t size)
{
  void *outptr=calloc(nmemb,size);
  if(outptr==NULL)
    report_error(1,"Out of memory\n");

  return outptr;
}

FILE *my_fopen(const char *path,const char *mode)
{
  FILE *fout=fopen(path,mode);
  if(fout==NULL)
    report_error(1,"Couldn't open file %s\n",path);

  return fout;
}

double *read_catalog(char *fname_cat,char *weight_name,char *cut_name,long *ngal)
{
  int status=0;
  long ip,np;
  fitsfile *fptr;

  fits_open_table(&fptr,fname_cat,0,&status);
  fits_get_num_rows(fptr,&np,&status);

  int found=0;
  int i_b,i_l,i_w=-1,i_c=-1;
  found=fits_get_colnum(fptr,CASEINSEN,"B",&i_b,&status);
  if(found==COL_NOT_FOUND)
    report_error(1,"No column found for galactic latitude\n");
  found=fits_get_colnum(fptr,CASEINSEN,"L",&i_l,&status);
  if(found==COL_NOT_FOUND)
    report_error(1,"No column found for galactic longitude\n");
  if(strcmp(weight_name,"NO_WEIGHT")) {
    found=fits_get_colnum(fptr,CASEINSEN,weight_name,&i_w,&status);
    if(found==COL_NOT_FOUND)
      report_error(1,"No column found for weights\n");
  }
  if(strcmp(cut_name,"NO_CUT")) {
    found=fits_get_colnum(fptr,CASEINSEN,cut_name,&i_c,&status);
    if(found==COL_NOT_FOUND)
      report_error(1,"No column found for cuts\n");
  }

  double *b_arr=my_malloc(np*sizeof(double));
  double *l_arr=my_malloc(np*sizeof(double));
  double *w_arr=my_malloc(np*sizeof(double));
  int *c_arr=my_malloc(np*sizeof(int));

  fits_read_col(fptr,TDOUBLE,i_b,1,1,np,NULL,b_arr,NULL,&status);
  fits_read_col(fptr,TDOUBLE,i_l,1,1,np,NULL,l_arr,NULL,&status);
  if(i_w!=-1)
    fits_read_col(fptr,TDOUBLE,i_w,1,1,np,NULL,w_arr,NULL,&status);
  else {
    for(ip=0;ip<np;ip++)
      w_arr[ip]=1;
  }
  if(i_c!=-1)
    fits_read_col(fptr,TINT,i_c,1,1,np,NULL,c_arr,NULL,&status);
  else {
    for(ip=0;ip<np;ip++)
      c_arr[ip]=-1;
  }
  fits_close_file(fptr,&status);

  int n_cut=0;
  for(ip=0;ip<np;ip++) {
    if(c_arr[ip]!=0)
      n_cut++;
  }

  *ngal=n_cut;
  double *pos=my_malloc(3*n_cut*sizeof(double));
  n_cut=0;
  for(ip=0;ip<np;ip++) {
    if(c_arr[ip]!=0) {
      double cth=cos(DTOR*(90-b_arr[ip]));
      double phi=DTOR*l_arr[ip];
      pos[3*n_cut+0]=cth;
      pos[3*n_cut+1]=phi;
      pos[3*n_cut+2]=w_arr[ip];
      n_cut++;
    }
  }

  free(b_arr);
  free(l_arr);
  free(w_arr);
  free(c_arr);

  return pos;
}

void compute_correlation(long ngal,double *pos,long nside,double *fld,double *msk,
			 double thmin,double thmax,int nth,int do_log,
			 double *hf_th,double *hm_th)
{
  long npix=he_nside2npix(nside);
  double thmax_rad=thmax*DTOR;
  double *pos_pix=my_malloc(3*npix*sizeof(double));

  long ip;
  for(ip=0;ip<npix;ip++) {
    double *v=&(pos_pix[3*ip]);
    he_pix2vec_ring(nside,ip,v);
  }

  for(ip=0;ip<nth;ip++) {
    hf_th[ip]=0;
    hm_th[ip]=0;
  }

#pragma omp parallel default(none)			\
  shared(ngal,pos,fld,msk,nside,thmin,thmax,nth,do_log)	\
  shared(hf_th,hm_th,npix,thmax_rad,pos_pix)
  {
    int i;
    double thmin_rad=thmin*DTOR;
    double cthmax=cos(thmax_rad);
    double *hf_th_thr=my_calloc(nth,sizeof(double));
    double *hm_th_thr=my_calloc(nth,sizeof(double));
    int lenlist0=(int)(4*npix*(1-cos(2*thmax_rad)));
    int *listpix=my_malloc(lenlist0*sizeof(int));
    
    int logbin=do_log;
    double log_th_max=log10(thmax_rad);
    double i_theta_max=1./thmax_rad;
    double n_logint=-1;
    if(do_log)
      n_logint=nth/log10(thmax_rad/thmin_rad);
    
#pragma omp for
    for(i=0;i<ngal;i++) {
      int j;
      double pos_g[3];
      int lenlist_half=lenlist0/2;
      double cth_g=pos[3*i+0];
      double phi_g=pos[3*i+1];
      double wei_g=pos[3*i+2];
      pos_g[0]=sqrt(1-cth_g*cth_g)*cos(phi_g);
      pos_g[1]=sqrt(1-cth_g*cth_g)*sin(phi_g);
      pos_g[2]=cth_g;
      he_query_disc(nside,cth_g,phi_g,1.2*thmax_rad,listpix,&lenlist_half,1);
      for(j=0;j<lenlist_half;j++) {
	int ipx=listpix[j];
	double *pos_p=&(pos_pix[3*ipx]);
	double prod=pos_g[0]*pos_p[0]+pos_g[1]*pos_p[1]+pos_g[2]*pos_p[2];
	if(prod>cthmax) {
	  int ith=th2bin(prod,logbin,n_logint,log_th_max,nth,i_theta_max);
	  if((ith<nth)&&(ith>=0)) {
	    hf_th_thr[ith]+=wei_g*fld[ipx];
	    hm_th_thr[ith]+=wei_g*msk[ipx];
	  }
	}
      }
    } //end omp for

#pragma omp critical
    {
      for(i=0;i<nth;i++) {
	hf_th[i]+=hf_th_thr[i];
	hm_th[i]+=hm_th_thr[i];
      }
    } //end omp critical
      
    free(hf_th_thr);
    free(hm_th_thr);
  } //end omp parallel

  free(pos_pix);
}

void write_correlation(char *fname_out,double *hf_th,double *hm_th,double thmin,double thmax,int nth,int do_log)
{
  int ith;
  FILE *fo=my_fopen(fname_out,"w");
  for(ith=0;ith<nth;ith++) {
    double th=bin2th(ith,do_log,nth,thmin,thmax);
    double w=0;
    if(hm_th[ith]>0)
      w=hf_th[ith]/hm_th[ith];
    fprintf(fo,"%lE %lE %lE %lE\n",th,w,hf_th[ith],hm_th[ith]);
  }
  fclose(fo);
}

void timer(int i)
{
  /////
  // Timing routine
  // timer(0) -> initialize relative clock
  // timer(1) -> read relative clock
  // timer(2) -> read relative clock and initialize it afterwards
  // timer(4) -> initialize absolute clock
  // timer(5) -> read absolute clock
  if(i==0)
    relbeg=omp_get_wtime();
  else if(i==1) {
    relend=omp_get_wtime();
    printf("    Relative time ellapsed %.1lf ms\n",1000*(relend-relbeg));
  }
  else if(i==2) {
    relend=omp_get_wtime();
    printf("    Relative time ellapsed %.1lf ms\n",1000*(relend-relbeg));
    relbeg=omp_get_wtime();
  }
  else if(i==4)
    absbeg=omp_get_wtime();
  else if(i==5) {
    absend=omp_get_wtime();
    printf("    Total time ellapsed %.1lf ms \n",1000*(absend-absbeg));
  }
}
