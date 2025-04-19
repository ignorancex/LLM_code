#include "spn.h"

void SCN_construct(SCN *self, int p, int d) {

  self->p = p;
  self->d = d;

  self->Pa_h_A = (DCOMPLEX*)malloc(sizeof(DCOMPLEX)*2*d);

  if( self->Pa_h_A == NULL){
    printf("MEM_ERROR\n");
  }
  self->Pa_omega = (DCOMPLEX*)malloc(sizeof(DCOMPLEX)*2*d);

  if( self->Pa_omega == NULL){
    printf("MEM_ERROR\n");
}

  SCN_init(self);
}

void SCN_init(SCN* self) {
  SCN_init_forward(self);
  SCN_init_backward(self);
}

void SCN_init_forward(SCN* self){
}


void SCN_init_backward(SCN* self) {
  memset(self->Pa_h_A,          0,2*self->d);
  memset(self->Pa_omega,        0,2*self->d);

  set_zval(self->F_A,           0,2);
  set_zval(self->h_A,           0,2);
  set_zval(self->TG_Ge_sc,      0,4);
  set_zval(self->DG_sc,         0,4);
  set_zval(self->temp_T_eta,    0,4);
  set_zval(self->Dh_sc,         0,4);
  set_zval(self->Psigma_G_sc,   0,2);
  set_zval(self->Psigma_h_sc,   0,2);
  set_zval(self->DG_A,          0,4);
  set_zval(self->Dh_A,          0,4);
  set_zval(self->S,             0,4);
  set_zval(self->temp_mat,      0,4);
  set_zval(self->Psigma_omega,  0,2);
}


void SCN_destroy(SCN *self) {
  free(self->Pa_h_A);
  free(self->Pa_omega);
  self->Pa_h_A   = NULL;
  self->Pa_omega = NULL;
}




int
SCN_cauchy(SCN* self){
  return 0;
}

void
SCN_grad(SCN* self,  int p, int d, double  *a,  double  sigma, \
  DCOMPLEX z,DCOMPLEX *G, DCOMPLEX *omega, DCOMPLEX *omega_sc,\
  DCOMPLEX *o_grad_a, DCOMPLEX *o_grad_sigma){

    self->F_A[0] = omega[0]+omega_sc[0]-z;
    self->F_A[1] = omega[1]+omega_sc[1]-z;

    // TODO separate check list:
    assert ( cabs(self->F_A[0]*G[0] - 1) < 1e-8 );
    assert ( cabs(self->F_A[1]*G[1] - 1) < 1e-8 );


    if (!(  cabs(self->F_A[0]*G[0] - 1) < 1e-8 || ( cabs(self->F_A[1]*G[1] - 1) < 1e-8 ) ) ){
      printf("omega error\n");
    }

    self->h_A[0] = omega[0] -z;
    self->h_A[1] = omega[1] -z;

    //printf("F_A=%e\n", cabs(F_A[0]));
    //printf("F_A=%e\n", cabs(F_A[1]));

    //printf("G=%e\n", cabs(G[0]));
    //printf("G=%e\n", cabs(G[1]));
    TG_Ge(p,d,sigma,G, self->TG_Ge_sc);
    /*
    printf("abs TG_Ge_sc=%e\n", cabs(TG_Ge_sc[0]));
    printf("abs TG_Ge_sc=%e\n", cabs(TG_Ge_sc[1]));
    printf("abs TG_Ge_sc=%e\n", cabs(TG_Ge_sc[2]));
    printf("abs TG_Ge_sc=%e\n", cabs(TG_Ge_sc[3]));
    //*/
    DG(G, self->TG_Ge_sc, self->DG_sc);
    //printf(" DG_sc= %e \n", cabs(DG_sc[0]));
    //printf(" DG_sc= %e \n", cabs(DG_sc[1]));
    //printf(" DG_sc= %e \n", cabs(DG_sc[2]));
    //printf(" DG_sc= %e \n", cabs(DG_sc[3]));
    T_eta(p,d,self->temp_T_eta);
    //printf("T_eta=%e\n", cabs(temp_T_eta[0]));
    //printf("T_eta=%e\n", cabs(temp_T_eta[1]));
    //printf("T_eta=%e\n", cabs(temp_T_eta[2]));

    Dh(self->DG_sc, self->temp_T_eta, sigma, self->Dh_sc);

    //printf("Dh_sc=%e\n", cabs(Dh_sc[0]));
    //printf("Dh_sc=%e\n", cabs(Dh_sc[1]));
    //printf("Dh_sc=%e\n", cabs(Dh_sc[2]));
    //printf("Dh_sc=%e\n", cabs(Dh_sc[3]));

    Psigma_G(p,d,sigma, G, self->TG_Ge_sc, self->Psigma_G_sc);

    /*
    printf("abs P_sigma_G_sc=%e\n", cabs(Psigma_G_sc[0]));
    printf("abs P_sigma_G_sc=%e\n", cabs(Psigma_G_sc[1]));
    //*/

    Psigma_h(p, d, sigma,G, self->Psigma_G_sc, self->temp_T_eta,\
    self->Psigma_h_sc);
    /*
    printf("abs Psigma_h %e\n", cabs(Psigma_h_sc[0]));
    printf("abs Psigma_h %e\n", cabs(Psigma_h_sc[1]));
    //*/
    des_DG(p, d, a, omega_sc, self->DG_A);

    //printf("abs  DG_A=%e\n", cabs(DG_A[0]));
    //printf("abs  DG_A=%e\n", cabs(DG_A[1]));
    //printf("abs  DG_A=%e\n", cabs(DG_A[2]));
    //printf("abs  DG_A=%e\n", cabs(DG_A[3]));
    des_Dh(self->DG_A, self->F_A,  self->Dh_A);

    /*
    printf("abs  dha=%e\n", cabs(Dh_A[0]));
    printf("abs  dha=%e\n", cabs(Dh_A[1]));
    printf("abs  dha=%e\n", cabs(Dh_A[2]));
    printf("abs  dha=%e\n", cabs(Dh_A[3]));
    //*/
    des_Pa_h(p, d,  a,  omega_sc, self->F_A,  self->Pa_h_A);
    /*
    for (int i = 0; i < d*2; i++) {
    printf("abs  Pa_h_A=%e\n", cabs(Pa_h_A[i]));
    }
    //*/
    ////tpS =  np.linalg.inv(np.eye(2,dtype=np.complex128) -  tpThsc  @ tpThA

    self->S[0] = 1.;
    self->S[1] = 0.;
    self->S[2] = 0.;
    self->S[3] = 1.;
    my_zgemm(2,2,2 , -1.0, self->Dh_sc, self->Dh_A, 1.0, self->S );
    /*
    printf("pre:abs  S=%e\n", cabs(S[0]));
    printf("abs  S=%e\n", cabs(S[1]));
    printf("abs  S=%e\n", cabs(S[2]));
    printf("abs  S=%e\n", cabs(S[3]));
    //*/
    inv2by2_overwrite(self->S);

    /*
    printf("inv:abs  S=%e\n", cabs(S[0]));
    printf("abs  S=%e\n", cabs(S[1]));
    printf("abs  S=%e\n", cabs(S[2]));
    printf("abs  S=%e\n", cabs(S[3]));
    //*/
    my_zgemm(d,2,2 , 1.0, self->Pa_h_A, self->S, 0, self->Pa_omega );
    z_isnan(d*2, self->Pa_omega);
    /*
    printf("abs  Pa_omega=%e\n", cabs(Pa_omega[0]));
    printf("abs  Pa_omega=%e\n", cabs(Pa_omega[1]));
    printf("abs  Pa_omega=%e\n", cabs(Pa_omega[2]));
    printf("abs  Pa_omega=%e\n", cabs(Pa_omega[3]));
    */
    my_zgemm(d,2,2 , 1.0, self->Pa_omega, self->DG_sc, 0,  o_grad_a);

    z_isnan(d*2, o_grad_a);

    my_zgemm(2,2,2, 1.0, self->Dh_A, self->S, 0, self->temp_mat);

    /*
    printf("abs  temp_mat=%e\n", cabs(temp_mat[0]));
    printf("abs  temp_mat=%e\n", cabs(temp_mat[1]));
    //*/

    my_zgemm(1,2,2, 1.0, self->Psigma_h_sc, self->temp_mat, 0, self->Psigma_omega);
    /*
    printf("abs  Psigma_omega=%e\n", cabs(Psigma_omega[0]));
    printf("abs  Psigma_omega=%e\n", cabs(Psigma_omega[1]));
    //*/
    // tpPsigmaG =  tpPsigmaOmega @ tpTGsc + tpPsigmaG
    my_zgemm(1,2,2, 1.0, self->Psigma_omega, self->DG_sc, 1.0, self->Psigma_G_sc);

    /*
    printf("abs  Psigma_G_sc=%e\n", cabs(Psigma_G_sc[0]));
    printf("abs  Psigma_G_sc=%e\n", cabs(Psigma_G_sc[1]));
    //*/
    o_grad_sigma[0] = self->Psigma_G_sc[0];
    o_grad_sigma[1] = self->Psigma_G_sc[1];
    /*
    printf("abs  o_grad_sigma=%e\n", cabs(o_grad_sigma[0]));
    printf("abs  o_grad_sigma=%e\n", cabs(o_grad_sigma[1]));
    //*/
    z_isnan(2, o_grad_sigma);

}



int
cauchy_sc(int p_dim, int dim, double sigma, \
  DCOMPLEX* Z, \
  int max_iter, double thres,\
  DCOMPLEX*  o_G){
    if ( !(thres> 0 )){
      printf("(cauchy_sc)ERROR:Must be thres > 0\n");
      exit(EXIT_FAILURE);
    }
    if (cimag(Z[0]) < 0 || cimag(Z[1]) < 0){
       printf("(cauchy_sc)ERROR:Must be Im Z > 0 \n");
       exit(EXIT_FAILURE);
     }
    int flag = 0;
    int num_iter = 0;
    DCOMPLEX sub_x = 0;
    DCOMPLEX sub_y = 0;
    for(int n = 0; n < max_iter; ++n){
      sub_x = 1./ (Z[0] - (pow(sigma,2)*o_G[1]*p_dim)/dim) - o_G[0];
      sub_y = 1./ (Z[1] - (pow(sigma,2)*o_G[0])) - o_G[1];
      sub_x *= 0.5;
      sub_y *= 0.5;
      if ( pow(cabs(sub_x), 2) + pow(cabs(sub_y),2) < pow(thres,2)){
        flag = 1;
      }
      o_G[0] += sub_x;
      o_G[1] += sub_y;
      if(flag == 1){
        num_iter =n;
        break;
      }
    }
    if (flag == 0){
      num_iter = max_iter;
      printf("(cauchy_sc)reached max_iter.\n");
    }
    return num_iter;
}


int
cauchy_spn(int p_dim, int dim, double* a,double sigma,\
  DCOMPLEX* B,\
  int max_iter,double thres, \
  DCOMPLEX* o_G_sc, DCOMPLEX* o_omega, DCOMPLEX* o_omega_sc){
    int flag = 0;
    int num_total_iter = 0;
    if ( !(thres > 0)){
      printf("(cauchy_spn)ERROR: Must be thres > 0 : %e\n" , thres);
      exit(EXIT_FAILURE);
    }
    for (int n = 0; n< max_iter; ++n){
        num_total_iter += cauchy_sc(p_dim,dim, sigma, o_omega,\
           max_iter, thres,\
          o_G_sc);
        DCOMPLEX W_x = 0;
        DCOMPLEX W_y = 0;
        W_x  = 1./o_G_sc[0] - o_omega[0] + B[0];
        W_y =  1./o_G_sc[1] - o_omega[1] + B[1];

        DCOMPLEX sum_inv_det = 0;
        for (int d = 0; d < dim; d++){
          sum_inv_det += 1./(W_x*W_y  - pow(a[d], 2) );
        }
        DCOMPLEX sub_x = 0;
        DCOMPLEX sub_y = 0;
        // omega_transform - old omega
        sub_x = dim / (W_y*sum_inv_det ) - W_x + B[0] - o_omega[0];
        sub_y = p_dim / (W_x*sum_inv_det + (p_dim - dim)/W_y)  - W_y  +  B[1]  - o_omega[1];
        // If subtraction is small, break the for-loop.
        if ( pow(cabs(sub_x), 2) + pow(cabs(sub_y), 2) < pow(thres, 2) ){
          flag = 1;
        }
        o_omega[0] += sub_x;
        o_omega[1] += sub_y;
        if (flag == 1){
          break;
        }
    }
    num_total_iter += cauchy_sc(p_dim,dim, sigma,o_omega, \
      max_iter, thres, \
      o_G_sc);
    o_omega_sc[0] = 1./o_G_sc[0] - o_omega[0] + B[0];
    o_omega_sc[1] = 1./o_G_sc[1] - o_omega[1] + B[1];
    return num_total_iter;
}



// ***
// Only for debug
// ***
void
grad_cauchy_spn(int p, int d, double  *a,  double  sigma, \
  DCOMPLEX z,DCOMPLEX *G, DCOMPLEX *omega, DCOMPLEX *omega_sc,\
  DCOMPLEX *o_grad_a, DCOMPLEX *o_grad_sigma){

  SCN* net = (SCN*)malloc(sizeof(SCN));
  SCN_construct(net,p,d);
  SCN_grad(net,p,d, a,sigma, \
          z, G, omega,  omega_sc,\
          o_grad_a, o_grad_sigma);

  SCN_destroy(net);
}



void
TG_Ge( const int p, const int d, const double sigma, const DCOMPLEX *G, \
  DCOMPLEX *o_DGe){
  // init diagonal matrix
  o_DGe[0] = 0;
  o_DGe[1] = G[1]*G[1]*pow(sigma,2);
  o_DGe[2] = G[0]*G[0]*pow(sigma,2)*(((double)p )/d + 0*I);
  o_DGe[3] = 0;
}


void
DG(const DCOMPLEX *G,  const  DCOMPLEX *TG_Ge,  DCOMPLEX *o_DG){
  // o_DG = eye -  TG_Ge
  o_DG[0] = 1. - TG_Ge[0];
  o_DG[1] = 0. - TG_Ge[1];
  o_DG[2] = 0. - TG_Ge[2];
  o_DG[3] = 1. - TG_Ge[3];
  // S = ( eye- TG_Ge)^{-1}
  inv2by2_overwrite(o_DG);
  // o_DG = Tz_Ge @ S
  // where Tz_Ge =  - G[0]**2, 0,
  //                  0, -G[1]**2

  o_DG[0] *= -G[0]*G[0];
  o_DG[1] *= -G[0]*G[0];
  o_DG[2] *= -G[1]*G[1];
  o_DG[3] *= -G[1]*G[1];

}

void T_eta(const int p, const int d, DCOMPLEX *o_T_eta){
  o_T_eta[0] = 0;
  o_T_eta[1] = 1;
  o_T_eta[2] = (double)p/d;
  o_T_eta[3] = 0;
}

void Dh(const DCOMPLEX* DG, const DCOMPLEX *T_eta, const double sigma,DCOMPLEX *o_Dh){
  my_zgemm(2,2,2,1.0, DG, T_eta, 0, o_Dh);
  DCOMPLEX temp = -pow(sigma,2) + 0.*I;
  my_zax(4, temp, o_Dh);
}

void Psigma_G(const int p,const int d, const double sigma, const DCOMPLEX *G, const DCOMPLEX *TG_Ge, DCOMPLEX *o_Psigma_G){
  o_Psigma_G[0]=0;
  o_Psigma_G[1]=0;

  DCOMPLEX Psigma_Ge[2];
  Psigma_Ge[0] = 2*sigma*G[0]*G[0]*(G[1]* (double)p/d);
  Psigma_Ge[1] = 2*sigma*G[1]*G[1]*(G[0]);


  DCOMPLEX S[4];
  S[0] = 1 - TG_Ge[0];
  S[1] = 0 - TG_Ge[1];
  S[2] = 0 - TG_Ge[2];
  S[3] = 1 - TG_Ge[3];

  inv2by2_overwrite(S);
  my_zgemm(1,2,2, 1.0, Psigma_Ge,  S,0, o_Psigma_G);

}

// o_Psigma_h ; 1 x 2
void Psigma_h(const int p, const int d, const double sigma, const DCOMPLEX * G, const DCOMPLEX* P_sigma_G, const DCOMPLEX *T_eta,\
DCOMPLEX* o_Psigma_h){
    // -2*sigma*eta_2by2(G)
    o_Psigma_h[0] = -2*sigma*(G[1]* (double)p/d);
    o_Psigma_h[1] = -2*sigma*(G[0]);
    // 1 x 2 mat @ 2 x 2 mat
    my_zgemm(1,2,2, -sigma*sigma, P_sigma_G, T_eta, 1.0, o_Psigma_h);
}




//////////////////////////////
// Total and Partial Derivations of Descrete
///////////////////////////////
// total derivation
//@a: d
//@o_DG; 2 x 2
//@W: 2
void des_DG( int p, int d, const double *a, const DCOMPLEX *W,DCOMPLEX*o_DG){
  DCOMPLEX sum1 = 0;
  DCOMPLEX sum2 = 0;
  for (int i = 0; i < d; i++, a++) {
    DCOMPLEX inv;
    inv = 1./(W[1]*W[0] - *a* (*a));
    inv *= inv;
    sum1 -= inv;
    sum2 -= inv*(*a)*(*a);
  }
  o_DG[0] = (1./d)*W[1]*W[1]*sum1;
  o_DG[1] = (1./p)*sum2;
  o_DG[2] = (1./d)*sum2;
  o_DG[3] = (1./p)*( W[0]*W[0] *sum1 - (double)(p-d)/ (W[1]*W[1]) )  ;
}


//
// 2 x 2
//
//
void des_Dh( const DCOMPLEX *DG, const DCOMPLEX *F,DCOMPLEX*o_Dh){
  for (int m = 0; m < 2; m++) {
    for (int n = 0; n < 2; n++, o_Dh++, DG++) {
      *o_Dh  = - F[n]*F[n]* (*DG);
      if (m == n ){
      *o_Dh -= 1.;
      }
    }
  }
}

/*
partial differencial of Descrete
a: d
o_DG; 2 x 2
W: 2
F: 2
Pa_h : d x 2
*/

void des_Pa_h( int p, int d, const double *a, const DCOMPLEX *W, DCOMPLEX *F, DCOMPLEX *o_Pa_h){
  for (int m = 0; m < d; ++m, ++a, ++o_Pa_h) {
    DCOMPLEX den = W[0]*W[1]- (*a)*(*a) ;
    den *= den;
    if (den == 0)printf("(des_Pa_h)0 division\n");
    assert(den != 0);
    DCOMPLEX inv = 1./den;
    DCOMPLEX temp = 2*(*a)*inv;
    // o_Pa_h[m][0]
    *o_Pa_h = - temp*F[0]*F[0]*W[1]/d;
    // o_Pa_h[m][1]
    o_Pa_h += 1;
    *o_Pa_h = - temp*F[1]*F[1]*W[0]/p;
  }
}



int
grad_loss_cauchy_spn(  int p, int d, \
  double  *a, double  sigma, double scale,\
  int  batch_size, double *batch, \
  double *o_grad_a, double *o_grad_sigma, double *o_loss){
    assert(batch_size > 0);
    SCN *net = (SCN*)malloc(sizeof(SCN));
    SCN_construct(net,p,d);

    DCOMPLEX omega[2] = {I,I};
    DCOMPLEX omega_sc[2] = {I,I};
    DCOMPLEX G[2] ={-I,-I};

    int total_f_iter = 0;
    const int max_iter = 1000;
    const double thres = 1e-8;

    DCOMPLEX B[2] = {0,0};

    DCOMPLEX*t_grad_a;
    t_grad_a = malloc(sizeof(DCOMPLEX)*d*2);
    memset(t_grad_a, 0, d*2);

    DCOMPLEX t_grad_sigma[2] = {0,0};



    for (int n = 0; n < batch_size; n++) {
      DCOMPLEX z = batch[n] + scale*I;
      DCOMPLEX w = csqrt(z);
      B[0] = w;
      B[1] = w;

      total_f_iter += cauchy_spn(p, d,a,sigma,\
        B, \
        max_iter,thres, \
        G,omega, omega_sc);

      double den = -cimag(G[0]*1./w);
      double rho = den/M_PI;


      if (!(rho>0)){
        printf("(grad_loss_cauchy_spn)Must rho > 0: %e\n",rho );
        exit(EXIT_FAILURE);
      }

      assert(rho > 0);
      *o_loss -= log(rho);

      memset(t_grad_a, 0, d*2);
      t_grad_sigma[0] = 0;
      t_grad_sigma[1] = 0;

      SCN_init_backward(net);
      SCN_grad(net,p,d, a,sigma, \
              w, G, omega,  omega_sc,\
              t_grad_a, t_grad_sigma);



      for (int i = 0; i < d; i++) {
      o_grad_a[i] += cimag(t_grad_a[2*i]/w)/den;
      }

      o_grad_sigma[0] += cimag(t_grad_sigma[0]/w)/den;

    }


    my_dax(d, 1./batch_size,  o_grad_a);
    *o_grad_sigma /= batch_size;
    *o_loss /=  batch_size;

    SCN_destroy(net);
    free(net);
    net = NULL;
    free(t_grad_a);
    t_grad_a=NULL;


    return total_f_iter;
}
