#define _FILE_OFFSET_BITS 64 
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<time.h>




#define Nb_proc 6

#define Nb_gi 140
#define Nb_i 200

#define X_min -11.0 // deg
#define X_max 16.0 // deg
#define Nb_X 108

#define Y_min -15.00 // deg
#define Y_max 11.00 // deg
#define Nb_Y 104

#define TAILLEMAX 1000

// Structure de données
struct data {
	int i,j;
	float xi,eta,don,mod,diff;
};

struct Pts{
	double xi,eta;
};


struct Vecteur{
	double xi,eta;
};

struct pts_poly{
	struct Pts A,B,C,D;
};

struct gi_data {
	int ix,jy,gi,magi;
	float val;
};



// Prototypes
void load(struct data data[Nb_X][Nb_Y]);
float gasdev(long *idum);
float ran2(long *idum);
int collision(struct Pts * P, struct Pts M);
int ligne_poly(void);
int calque_int(struct Pts M);
int poly_in(struct pts_poly poly[407][37],struct Pts M);
void load_poly(int nb_ligne,struct pts_poly p_poly[407][37]);
void cmd_load(int x_pix,int y_pix,struct gi_data gi_data[Nb_gi][Nb_i],float *max);


/*********
** Main **
**********/
int main(void){

	int i,j,k,l;
	float x,y;
	int nb_pts_poly;

// Pour le CMD
    int ind_gi,ind_magi,ifield;
    float magg,magi;
	 float Eg,Ei;
    float val,max;
    struct gi_data gi_data[Nb_gi][Nb_i];

	struct Pts M;
	struct pts_poly poly[407][37];
	struct data data[Nb_X][Nb_Y];
	float intervalx,intervaly,densite;
    int dense;
	long idum=-(long)time(NULL); // Graine aléatoire
	FILE * fichier=NULL;

	long Ntot=0;


	// Chargement des données
	load(data);

	// Charge les points des bords des fields
	nb_pts_poly=ligne_poly();
	load_poly(nb_pts_poly,poly);


	// Intervalle
	intervalx=(X_max-X_min)/Nb_X;
	intervaly=(Y_max-Y_min)/Nb_Y;

	//printf("cool\n");

	fichier=fopen("foreground.asc","w");
	if(fichier!=NULL){
	for(i=0;i<Nb_X;i++){
		for(j=0;j<Nb_Y;j++){

			// Savoir densité par pixel carrée
			densite=data[i][j].mod*(intervalx*intervaly)*0.93; //0.93 to take into account that the model substract to many stars (ibata + 14)
			// Densité aléatoire 
			dense=(int)(densite+sqrt(densite)*gasdev(&idum));
            // Charge les donnnées gi du pixel
			cmd_load(i,j,gi_data,&max);
			Ntot+=dense;

  
			for(k=1;k<=dense;k++){ // chaque étoiles
				x=X_min+intervalx*(i+ran2(&idum)); // ran2 entre 0 et 1 donc aléatoire entre i et i+1
				y=Y_min+intervaly*(j+ran2(&idum));
								
				M.xi=x;
				M.eta=y;

				// Calques
                 
			if((x<=-0.5&&y<=-11.0) || (x>=4.0&&x<=7.0&&y<=-11.0) || (x>=11.0&&y>=-5.0) || (x>=0.0&&y>=8.0) || (x>=14.0&&y>=-9.0) || (x>=14&&y<=-13) || (x<=-6.0&&y>=8.0) ||(x<=-9.0&&y>=5.0)) ;else if((ifield=poly_in(poly,M))!=0){ 
				                        l=0;
                        
                        while(l<1){ // boucle pour avoir une bonne valeur de gi et magi

                             ind_gi=(int)Nb_gi*ran2(&idum);
                             ind_magi=(int)Nb_i*ran2(&idum);
                             val=max*ran2(&idum);
//printf("%d,%d,%d,%f,%f\n",k,ind_gi,ind_magi,val,gi_data[ind_gi][ind_magi].val);
                             if(val>gi_data[ind_gi][ind_magi].val){
                                l=0 ;
                             }else{
                                 magi=20.0+ind_magi*0.02+0.02*ran2(&idum);
                                 magg=0.2+0.02*ind_gi+magi+0.02*ran2(&idum);
											Eg=0.032*exp((magg-24.25)/1.1)+0.004;				
											Ei=0.112*exp((magi-24.25)/1.13)+0.003;		
										 // printf("toto\n");
                                 	fprintf(fichier,"%d %f %f %f %f %f %f\n",ifield,x,y,magg,Eg,magi,Ei);
									l=1;
                             }
                        }
                       
                    
				}	
			}
			
		}
	}
	}fclose(fichier);

printf("Ntot :%ld\n",Ntot);

return 1;
}








/*********************************
***     Charge les données     ***
**********************************/
void cmd_load(int x_pix,int y_pix,struct gi_data gi_data[Nb_gi][Nb_i],float *max){

	int i,j,k,l;
	char ligne[TAILLEMAX];
    char name[25];

	FILE * fichier=NULL;
    snprintf(name, sizeof name,"./modele/%d_%d.out",x_pix,y_pix);

	fichier=fopen(name,"r");
	if(fichier!=NULL){

	fscanf(fichier,"%f",&(*max));	
	    for(k=0;k<Nb_gi;k++){

			for(l=0;l<Nb_i;l++){
			    fscanf(fichier,"%d %d, %d %d, %e",&gi_data[k][l].ix,&gi_data[k][l].jy,&gi_data[k][l].gi,&gi_data[k][l].magi,&gi_data[k][l].val);
			}
		}
	}
	fclose(fichier);

return ;
}












/*********************************
***     Charge les données     ***
**********************************/
void load(struct data data[Nb_X][Nb_Y]){

	int i,j;
	char ligne[TAILLEMAX];

	FILE * fichier=NULL;
	fichier=fopen("./N_stars.asc","r");
	if(fichier!=NULL){
		for(i=0;i<Nb_X;i++){
			for(j=0;j<Nb_Y;j++){
				fscanf(fichier," %d %d %f %f %f",&data[i][j].i,&data[i][j].j,&data[i][j].xi,&data[i][j].eta,&data[i][j].mod);
			}
		}
	}
	fclose(fichier);

return ;
}











/************************************
****       Calque interne        ****
************************************/

/* enleve les points dans un 
rayon de 2 deg autour du bord
retourne 1 si dedans 0 dehors */           

int calque_int(struct Pts M){
	if(sqrt(M.xi*M.xi+M.eta*M.eta)<=2){
		return 0;
	}else{
		return 1;
	}
}


/***********************************
***  Nombre de ligne du fichier  ***
***********************************/
int ligne_poly(void){
	
	int nb_ligne=0;
	char ligne[TAILLEMAX];

// Lis le fichien d'entrée pour savoir combien de lignes il possède
	FILE* fichier = NULL;
	fichier=fopen("../../donnees/field_limits_actual.dat","r");
	if(fichier!=NULL){
		while(fgets(ligne,TAILLEMAX,fichier)!=NULL){
			nb_ligne++;
		}
		fclose(fichier);
	}

return nb_ligne;
}

	
/*********************************
***     Charge les données     ***
**********************************/
void load_poly(int nb_ligne,struct pts_poly p_poly[407][37]){
	int i;
	int nb_field,nb_CCD,tmp=0;
	struct Pts A,B,C,D;	

	FILE * fichier=NULL;
	fichier=fopen("../../donnees/field_limits_actual.dat","r");
	if(fichier!=NULL){
		for(i=0;i<nb_ligne;i++){
			fscanf(fichier,"%d %d %lf %lf %lf %lf %lf %lf %lf %lf",&nb_field,&nb_CCD,&A.xi,&A.eta,&B.xi,&B.eta,&C.xi,&C.eta,&D.xi,&D.eta);

		p_poly[nb_field][nb_CCD].A=A;
		p_poly[nb_field][nb_CCD].B=B;
		p_poly[nb_field][nb_CCD].C=C;
		p_poly[nb_field][nb_CCD].D=D;
			if(nb_CCD!=tmp+1){
				p_poly[nb_field][nb_CCD].A.xi=0.0;
				p_poly[nb_field][nb_CCD].A.eta=0.0;
				p_poly[nb_field][nb_CCD].B.xi=0.0;
				p_poly[nb_field][nb_CCD].B.eta=0.0;
				p_poly[nb_field][nb_CCD].C.xi=0.0;
				p_poly[nb_field][nb_CCD].C.eta=0.0;
				p_poly[nb_field][nb_CCD].D.xi=0.0;
				p_poly[nb_field][nb_CCD].D.eta=0.0;
			}
			if(nb_CCD==36){
				tmp=0;
			}else{	
				tmp=nb_CCD;
			}			
		}
	}
	fclose(fichier);
return;
}


/*Pour chaque image : Retourne 1 si pts dedans sinon 0*/
int collision(struct Pts * P, struct Pts M){
	int i;
	double det=0,sin_theta;
	struct Pts A;
	struct Pts B;
	struct Vecteur T,D;	
	for(i=0;i<=3;i++){
		A=P[i];
		if(i==3){
			B=P[0];
		}else{
			B=P[i+1];
		}
		D.xi=B.xi-A.xi;
		D.eta=B.eta-A.eta;
		T.xi=M.xi-A.xi;
		T.eta=M.eta-A.eta;
		det=D.xi*T.eta-D.eta*T.xi;
		if(det<=0){
			return 0;
		}
	}	
return 1;
}

int poly_in(struct pts_poly poly[407][37],struct Pts M){
	int i,j,info=0;
	struct Pts P[4];

	for(i=1;i<=406;i++){
		for(j=1;j<=36;j++){
			P[0]=poly[i][j].A;
			P[1]=poly[i][j].B;
			P[2]=poly[i][j].C;
			P[3]=poly[i][j].D;
			info=collision(P,M);
			if(info==1){
				return i; // si info =1 pts dedans on s'arrete la
			}
		}	
	}
return 0; // si pts pas dedans retourne 0
}
















/********************************
** gasdev Numerical recipises  **
*********************************/
#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0/IM1)
#define IMM1 (IM1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define NDIV (1+IMM1/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)

float ran2(long *idum)
{
	int j;
	long k;
	static long idum2=123456789;
	static long iy=0;
	static long iv[NTAB];
	float temp;

	if (*idum <= 0) {
		if (-(*idum) < 1) *idum=1;
		else *idum = -(*idum);
		idum2=(*idum);
		for (j=NTAB+7;j>=0;j--) {
			k=(*idum)/IQ1;
			*idum=IA1*(*idum-k*IQ1)-k*IR1;
			if (*idum < 0) *idum += IM1;
			if (j < NTAB) iv[j] = *idum;
		}
		iy=iv[0];
	}
	k=(*idum)/IQ1;
	*idum=IA1*(*idum-k*IQ1)-k*IR1;
	if (*idum < 0) *idum += IM1;
	k=idum2/IQ2;
	idum2=IA2*(idum2-k*IQ2)-k*IR2;
	if (idum2 < 0) idum2 += IM2;
	j=iy/NDIV;
	iy=iv[j]-idum2;
	iv[j] = *idum;
	if (iy < 1) iy += IMM1;
	if ((temp=AM*iy) > RNMX) return RNMX;
	else return temp;
}
#undef IM1
#undef IM2
#undef AM
#undef IMM1
#undef IA1
#undef IA2
#undef IQ1
#undef IQ2
#undef IR1
#undef IR2
#undef NTAB
#undef NDIV
#undef EPS
#undef RNMX

float gasdev(long *idum)
{
	float ran2(long *idum);
	static int iset=0;
	static float gset;
	float fac,rsq,v1,v2;

	if  (iset == 0) {
		do {
			v1=2.0*ran2(idum)-1.0;
			v2=2.0*ran2(idum)-1.0;
			rsq=v1*v1+v2*v2;
		} while (rsq >= 1.0 || rsq == 0.0);
		fac=sqrt(-2.0*log(rsq)/rsq);
		gset=v1*fac;
		iset=1;
		return v2*fac;
	} else {
		iset=0;
		return gset;
	}
}
/******************
*** fin gasdev  ***
*******************/
