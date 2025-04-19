#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <numpy/arrayobject.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_sf_bessel.h>

typedef struct {
  int nl;
  double th;
  double l0;
  double lf;
  double clf;
  double tilt;
  gsl_spline *spl_cl;
  gsl_interp_accel *ia_cl;
} IntWthParam;

static void int_wth_param_free(IntWthParam *p)
{
  gsl_interp_accel_free(p->ia_cl);
  gsl_spline_free(p->spl_cl);
  free(p);
}

IntWthParam *int_wth_param_new(int nl,double *llist,double *cllist)
{
  IntWthParam *p=malloc(sizeof(IntWthParam));
  p->nl=nl;
  p->l0=llist[0];
  p->lf=llist[nl-1];
  p->clf=cllist[nl-1];
  if((cllist[nl-1]<=0) || (cllist[nl-2]<=0))
    p->tilt=0;
  else 
    p->tilt=log(cllist[nl-1]/cllist[nl-2])/log(llist[nl-1]/llist[nl-2]);

  p->ia_cl=gsl_interp_accel_alloc();
  p->spl_cl=gsl_spline_alloc(gsl_interp_cspline,nl);
  gsl_spline_init(p->spl_cl,llist,cllist,nl);

  return p;
}

static double eval_cl(double l,IntWthParam *p)
{
  if(l<=p->l0)
    return 0;
  else if(l>=p->lf)
    return 0;//p->clf*pow(l/p->lf,p->tilt);
  else
    return gsl_spline_eval(p->spl_cl,l,p->ia_cl);
}

static double wth_integrand(double l,void *params)
{
  IntWthParam *p=(IntWthParam *)params;
  double x=l*p->th;
  double cl=eval_cl(l,p);
  double jbes=gsl_sf_bessel_J0(x);
  return l*jbes*cl;
}

static void wth_all(int nth,double *tharr,double *wtharr,
		    int nl,double *larr,double *clarr)
{
  gsl_set_error_handler_off();
#pragma omp parallel default(none)		\
  shared(nth,tharr,wtharr,nl,larr,clarr)
  {
    int ith;
    double result,eresult;
    gsl_function F;
    IntWthParam *p=int_wth_param_new(nl,larr,clarr);
    gsl_integration_workspace *w=gsl_integration_workspace_alloc(1000);

#pragma omp for
    for(ith=0;ith<nth;ith++) {
      p->th=tharr[ith];
      F.function=&wth_integrand;
      F.params=p;
      //      gsl_integration_qag(&F,larr[0],larr[nl-1],0,1E-4,1000,GSL_INTEG_GAUSS41,w,&result,&eresult);
      gsl_integration_qag(&F,0.,1E5,0,1E-4,1000,GSL_INTEG_GAUSS41,w,&result,&eresult);
      wtharr[ith]=result/(2*M_PI);
    } //end omp for

    gsl_integration_workspace_free(w);
    int_wth_param_free(p);
  } //end omp parallel
}

static int  not_doublevector(PyArrayObject *vec)  {
  if (vec->descr->type_num != NPY_DOUBLE || vec->nd != 1)  {
    PyErr_SetString(PyExc_ValueError,
		    "In not_doublevector: array must be of type Float and 1 dimensional (n).");
    return 1;  }
  return 0;
}

static PyObject *cl2wth_compute_wth(PyObject *self,PyObject *args)
{
  PyArrayObject *olarr,*oclarr,*otharr,*wout=NULL;
  double *larr,*clarr,*tharr,*wtharr;

  if(!PyArg_ParseTuple(args,"O!O!O!",&PyArray_Type,&olarr,&PyArray_Type,&oclarr,&PyArray_Type,&otharr))
    return NULL;

  if(NULL==olarr) return NULL;
  if(NULL==oclarr) return NULL;
  if(NULL==otharr) return NULL;
  if(not_doublevector(olarr)) return NULL;
  if(not_doublevector(oclarr)) return NULL;
  if(not_doublevector(otharr)) return NULL;
  
  larr=malloc(olarr->dimensions[0]*sizeof(double));
  clarr=malloc(olarr->dimensions[0]*sizeof(double));
  tharr=malloc(otharr->dimensions[0]*sizeof(double));
  wtharr=malloc(otharr->dimensions[0]*sizeof(double));
  int ii;
  for(ii=0;ii<olarr->dimensions[0];ii++) {
    larr[ii]=*((double *)(olarr->data+ii*olarr->strides[0]));
    clarr[ii]=*((double *)(oclarr->data+ii*oclarr->strides[0]));
  }
  for(ii=0;ii<otharr->dimensions[0];ii++)
    tharr[ii]=(*((double *)(otharr->data+ii*otharr->strides[0])))*0.01745329251;

  wth_all(otharr->dimensions[0],tharr,wtharr,olarr->dimensions[0],larr,clarr);

  int dims=otharr->dimensions[0];
  wout=(PyArrayObject *)PyArray_FromDims(1,&dims,PyArray_DOUBLE);
  for(ii=0;ii<otharr->dimensions[0];ii++) {
    double *w=(double *)(wout->data+ii*wout->strides[0]);
    *w=wtharr[ii];
  }
  
  return PyArray_Return(wout);
}

static PyMethodDef Cl2wthMethods[] = {
  {"compute_wth",cl2wth_compute_wth,METH_VARARGS,"2PCF from power spectrum"},
  {NULL,NULL,0,NULL}
};

PyMODINIT_FUNC initcl2wth(void)
{
  (void)Py_InitModule("cl2wth",Cl2wthMethods);
  import_array();
}
