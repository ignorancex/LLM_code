#include <cmath>
#include "funs.h"
#include <map>
#include <RcppParallel.h>

using Rcpp::Rcout;
using namespace RcppParallel;

//--------------------------------------------------
// normal density N(x, mean, variance)
double pn(double x, double m, double v)
{
	double dif = x-m;
	return exp(-.5*dif*dif/v)/sqrt(2*PI*v);
}
//--------------------------------------------------
// draw from discrete distribution given by p, return index
int rdisc(double *p, RNG& gen)
{
	
	double sum;
	double u = gen.uniform();
	
    int i=0;
    sum=p[0];
    while(sum<u) {
		i += 1;
		sum += p[i];
    }
    return i;
}
//--------------------------------------------------
//evalute tree tr on grid given by xi and write to os
void grm(tree& tr, xinfo& xi, std::ostream& os) 
{
	size_t p = xi.size();
	if(p!=2) {
	  Rcout << "error in grm, p !=2\n";
		return;
	}
	size_t n1 = xi[0].size();
	size_t n2 = xi[1].size();
	tree::tree_cp bp; //pointer to bottom node
	double *x = new double[2];
	for(size_t i=0;i!=n1;i++) {
		for(size_t j=0;j!=n2;j++) {
			x[0] = xi[0][i]; 
			x[1] = xi[1][j]; 
			bp = tr.bn(x,xi);
			os << x[0] << " " << x[1] << " " << bp->getm() << " " << bp->nid() << endl;
		}
	}
	delete[] x;
}
//--------------------------------------------------
//does this bottom node n have any variables it can split on.
bool cansplit(tree::tree_p n, xinfo& xi)
{
	int L,U;
	bool v_found = false; //have you found a variable you can split on
	size_t v=0;
	while(!v_found && (v < xi.size())) { //invar: splitvar not found, vars left
		L=0; U = xi[v].size()-1;
		n->rg(v,&L,&U);
		if(U>=L) v_found=true;
		v++;
	}
	return v_found;
}
//--------------------------------------------------
//compute prob of a birth, goodbots will contain all the good bottom nodes
double getpb(tree& t, xinfo& xi, pinfo& pi, tree::npv& goodbots)
{
	double pb;  //prob of birth to be returned
	tree::npv bnv; //all the bottom nodes
	t.getbots(bnv);
	for(size_t i=0;i!=bnv.size();i++) 
		if(cansplit(bnv[i],xi)) goodbots.push_back(bnv[i]);
	if(goodbots.size()==0) { //are there any bottom nodes you can split on?
		pb=0.0;
	} else { 
		if(t.treesize()==1) pb=1.0; //is there just one node?
		else pb=pi.pb;
	}
	return pb;
}
//--------------------------------------------------
//find variables n can split on, put their indices in goodvars
void getgoodvars(tree::tree_p n, xinfo& xi,  std::vector<size_t>& goodvars)
{
	int L,U;
	for(size_t v=0;v!=xi.size();v++) {//try each variable
		L=0; U = xi[v].size()-1;
		n->rg(v,&L,&U);
		if(U>=L) goodvars.push_back(v);
	}
}

//calibart
//--------------------------------------------------
// number of available cutpoints at node n for variable var
int getnumcuts(tree::tree_p n, xinfo& xi, size_t var)
{
  int L,U;
  
  getpertLU(n,var,xi,&L,&U);
  return std::max(0,U-L+1);
}

//--------------------------------------------------
// Find numbr of variables internal tree node n can split on
void getinternalvars(tree::tree_p n, xinfo& xi,  std::vector<size_t>& goodvars)
{
  int L,U;
  
  for(size_t v=0;v!=xi.size();v++) {//try each variable
    L=0; U = xi[v].size()-1;
    getpertLU(n,v,xi,&L,&U);
    if(U>=L) goodvars.push_back(v);
  }
}



//--------------------------------------------------
//get prob a node grows, 0 if no good vars, else alpha/(1+d)^beta
double pgrow(tree::tree_p n, xinfo& xi, pinfo& pi)
{
	if(cansplit(n,xi)) {
		return pi.alpha/pow(1.0+n->depth(),pi.beta);
	} else {
		return 0.0;
	}
}

//getLU/getpertLU from calibart
//--------------------------------------------------
// Get the L,U values for a node in the tree *given* the tree
// structure both above and below that node.
void getLU(tree::tree_p pertnode, xinfo& xi, int* L, int* U)
{
  tree::tree_p l,r;
  
  *L=0; *U = xi[pertnode->getv()].size()-1;
  l=pertnode->getl();
  r=pertnode->getr();
  
  bool usel,user;
  usel=l->nuse(pertnode->getv());
  user=r->nuse(pertnode->getv());
  if(usel && user)
  {
    l->rl(pertnode->getv(),L);
    r->ru(pertnode->getv(),U);
  }
  else if(usel)
  {
    pertnode->rg(pertnode->getv(),L,U);
    l->rl(pertnode->getv(),L);
  }
  else
  {
    pertnode->rg(pertnode->getv(),L,U);
    r->ru(pertnode->getv(),U);
  }
}

//--------------------------------------------------
// similar except we get it for a prescribed variable pertvar
void getpertLU(tree::tree_p pertnode, size_t pertvar, xinfo& xi, int* L, int* U)
{
  *L=0; *U = xi[pertvar].size()-1;
  
  bool usel,user;
  usel=pertnode->l->nuse(pertvar);
  user=pertnode->r->nuse(pertvar);
  if(usel && user)
  {
    pertnode->l->rl(pertvar,L);
    pertnode->r->ru(pertvar,U);
  }
  else if(usel)
  {
    pertnode->rg(pertvar,L,U);
    pertnode->l->rl(pertvar,L);
  }
  else
  {
    pertnode->rg(pertvar,L,U);
    pertnode->r->ru(pertvar,U);
  }
}
//--------------------------------------------------
//get sufficients stats for all bottom nodes
struct AllSuffWorker: public Worker
{
	// -------------------
	// Inputs
	// -------------------
	tree& x;
	xinfo& xi;
	dinfo& di;
	size_t nb;
	std::map<tree::tree_cp,size_t> bnmap;
	double* weight;

	// -------------------
	// Internal State
	// -------------------
	double n;
	double sy;
	double n0;

	std::vector<sinfo> sv_tmp;
	double *xx;//current x
	double y;  //current y
	size_t ni; //the  index into vector of the current bottom node


	AllSuffWorker(tree& x,
								xinfo& xi,
								dinfo& di,
								std::map<tree::tree_cp,size_t> bnmap,
								size_t nb,
								double* weight):x(x),xi(xi),di(di),nb(nb),bnmap(bnmap),weight(weight) {
		n=0.0;
		sy=0.0;
		n0=0.0;
		sv_tmp.resize(nb);
	}

	AllSuffWorker(const AllSuffWorker& asw, Split):x(asw.x),xi(asw.xi),di(asw.di),nb(asw.nb),bnmap(asw.bnmap),weight(asw.weight) {
		n=0.0;
		sy=0.0;
		n0=0.0;
		sv_tmp.resize(nb);
	}

	void operator()(std::size_t begin, std::size_t end){
		for(size_t i=begin;i<end;i++) {
			xx = di.x + i*di.p;
			y=di.y[i]; // @peter I suspect that di.y is the Rj, this trees residual thing
			
			ni = bnmap[x.bn(xx,xi)];

			sv_tmp[ni].n0 += 1;
			sv_tmp[ni].n += weight[i];
			sv_tmp[ni].sy += weight[i]*y;

		}
	}
	void join(const AllSuffWorker& asw){
		for(size_t i=0; i!=nb; i++){
			sv_tmp[i].n0  += asw.sv_tmp[i].n0;  
			sv_tmp[i].n   += asw.sv_tmp[i].n;
			sv_tmp[i].sy  += asw.sv_tmp[i].sy; 
		}
	}


};





//--------------------------------------------------
//get sufficients stats for all bottom nodes
void allsuff(tree& x,
   		     xinfo& xi, 
			 dinfo& di, 
			 double* weight,
			 tree::npv& bnv, 
			 std::vector<sinfo>& sv)
{
	tree::tree_cp tbn; //the pointer to the bottom node for the current observations
	size_t ni;         //the  index into vector of the current bottom node
	double *xx;        //current x
	double y;          //current y
	
	bnv.clear();

	// This appears to get the bottom nodes from tree x
	// and save them into the bottom node pointer vector bnv
	x.getbots(bnv);
	
	typedef tree::npv::size_type bvsz;
	bvsz nb = bnv.size();
	sv.resize(nb);
	
	std::map<tree::tree_cp,size_t> bnmap;
	for(bvsz i=0;i!=bnv.size();i++) bnmap[bnv[i]]=i;
	
	// It seems like it's using a tree defined by x, with node xi
	// to calculate stuff from data on di
	// 

	AllSuffWorker asw(x,xi,di,bnmap,nb,weight);

	parallelReduce(0, di.n, asw);

	for(size_t i=0; i!=nb; i++){
			sv[i].n0  += asw.sv_tmp[i].n0;  
			sv[i].n   += asw.sv_tmp[i].n;
			sv[i].sy  += asw.sv_tmp[i].sy; 
	}
}

//get counts for all bottom nodes
std::vector<int> counts(tree& x, xinfo& xi, dinfo& di, tree::npv& bnv)
{
  tree::tree_cp tbn; //the pointer to the bottom node for the current observations
	size_t ni;         //the  index into vector of the current bottom node
	double *xx;        //current x
	double y;          //current y
  
	bnv.clear();
	x.getbots(bnv);
	
	typedef tree::npv::size_type bvsz;
	bvsz nb = bnv.size();

  std::vector<int> cts(bnv.size(), 0);
	
	std::map<tree::tree_cp,size_t> bnmap;
	for(bvsz i=0;i!=bnv.size();i++) bnmap[bnv[i]]=i;
	
	for(size_t i=0;i<di.n;i++) {
		xx = di.x + i*di.p;
		y=di.y[i];
		
		tbn = x.bn(xx,xi);
		ni = bnmap[tbn];
		
    cts[ni] += 1;
	}
  return(cts);
}

void update_counts(int i, std::vector<int>& cts, tree& x, xinfo& xi, 
                   dinfo& di, 
                   tree::npv& bnv, //vector of pointers to bottom nodes
                   int sign)
{
  tree::tree_cp tbn; //the pointer to the bottom node for the current observations
  size_t ni;         //the  index into vector of the current bottom node
	double *xx;        //current x
	double y;          //current y

	typedef tree::npv::size_type bvsz;
	bvsz nb = bnv.size();
	
	std::map<tree::tree_cp,size_t> bnmap;
	for(bvsz ii=0;ii!=bnv.size();ii++) bnmap[bnv[ii]]=ii; // bnmap[pointer] gives linear index
	
	xx = di.x + i*di.p;
	y=di.y[i];
	
	tbn = x.bn(xx,xi);
	ni = bnmap[tbn];
	
  cts[ni] += sign;
}

void update_counts(int i, std::vector<int>& cts, tree& x, xinfo& xi, 
                   dinfo& di, 
                   std::map<tree::tree_cp,size_t>& bnmap,
                   int sign)
{
  tree::tree_cp tbn; //the pointer to the bottom node for the current observations
  size_t ni;         //the  index into vector of the current bottom node
  double *xx;        //current x
	double y;          //current y
  /*
	typedef tree::npv::size_type bvsz;
	bvsz nb = bnv.size();
	
	std::map<tree::tree_cp,size_t> bnmap;
	for(bvsz ii=0;ii!=bnv.size();ii++) bnmap[bnv[ii]]=ii; // bnmap[pointer] gives linear index
	*/
	xx = di.x + i*di.p;
	y=di.y[i];
	
	tbn = x.bn(xx,xi);
	ni = bnmap[tbn];
	
  cts[ni] += sign;
}


void update_counts(int i, std::vector<int>& cts, tree& x, xinfo& xi, 
                   dinfo& di, 
                   std::map<tree::tree_cp,size_t>& bnmap,
                   int sign,
                   tree::tree_cp &tbn
                   )
{
  //tree::tree_cp tbn; //the pointer to the bottom node for the current observations
  size_t ni;         //the  index into vector of the current bottom node
  double *xx;        //current x
  double y;          //current y

	xx = di.x + i*di.p;
	y=di.y[i];
	
	tbn = x.bn(xx,xi);
	ni = bnmap[tbn];
	
  cts[ni] += sign;
}

bool min_leaf(int minct, std::vector<tree>& t, xinfo& xi, dinfo& di) {
  bool good = true;
  tree::npv bnv;
  std::vector<int> cts;
  int m = 0;
  for (size_t tt=0; tt<t.size(); ++tt) {
    cts = counts(t[tt], xi, di, bnv);
    m = std::min(m, *std::min_element(cts.begin(), cts.end()));
    if(m<minct) {
      good = false;
      break;
    }
  }
  return good;
}

//--------------------------------------------------
//get sufficient stats for children (v,c) of node nx in tree x
//n = sum_i phi_i, sumy = \sum \phi_iy_i, sumy^2 \sum \phi_iy_i^2

//  Is is probably this get suff that matters
// getsuff(x,nx->getl(),nx->getr(),xi,di,phi,sl,sr)
struct GetSuffBirthWorker: public Worker
{
// -------------------
// Inputs
// -------------------
tree& x;
tree::tree_cp nx;
size_t v;
size_t c;
xinfo& xi;
dinfo& di;
double* phi;

// -------------------
// Internal State
// -------------------
double l_n;
double l_sy;
double l_n0;

double r_n;
double r_sy;
double r_n0;

double *xx;//current x
double y;  //current y

// -------------------
// Constructors
// -------------------
// Standard Constructor
GetSuffBirthWorker(tree& x,
							tree::tree_cp nx,
							size_t v,
							size_t c,
							xinfo& xi,
							dinfo& di,
							double* phi):x(x),nx(nx),v(v),c(c),xi(xi),di(di),phi(phi) {

    l_n=0.0;
	l_sy=0.0;
	l_n0=0.0;

	r_n=0.0;
	r_sy=0.0;
	r_n0=0.0;
} 
// Splitting Constructor
GetSuffBirthWorker(const GetSuffBirthWorker& gsw, Split):x(gsw.x),nx(gsw.nx),v(gsw.v),c(gsw.c),xi(gsw.xi),di(gsw.di),phi(gsw.phi) {

	l_n=0.0;
	l_sy=0.0;
	l_n0=0.0;

	r_n=0.0;
	r_sy=0.0;
	r_n0=0.0;
} 

void operator()(std::size_t begin, std::size_t end){
	for(size_t i=begin;i<end;i++) {
		xx = di.x + i*di.p;
		if(nx==x.bn(xx,xi)) { //does the bottom node = xx's bottom node
			y = di.y[i];
			if(xx[v] < xi[v][c]) {
        		l_n0 += 1;
				l_n += phi[i];
				l_sy += phi[i]*y;
			} else {
       			r_n0 += 1;
				r_n += phi[i];
				r_sy += phi[i]*y;
			}
		}
	}
}

void join(const GetSuffBirthWorker& gsw){
	l_n   += gsw.l_n;
	l_sy  += gsw.l_sy;
	l_n0  += gsw.l_n0;

	r_n   += gsw.r_n;
	r_sy  += gsw.r_sy;
	r_n0  += gsw.r_n0;
}

}; // End Get Suff Birth Worker


// birth get suff 
void getsuffBirth(tree& x, tree::tree_cp nx, size_t v, size_t c, xinfo& xi, dinfo& di, double* phi, sinfo& sl, sinfo& sr)
{
	GetSuffBirthWorker gsw(x,nx,v,c,xi,di,phi);

	parallelReduce(0, di.n, gsw);

	sl.n   = gsw.l_n;
	sl.sy  = gsw.l_sy;
	sl.n0  = gsw.l_n0;

	sr.n   = gsw.r_n;
	sr.sy  = gsw.r_sy;
	sr.n0  = gsw.r_n0;

	
}

// ----------------------------------------------------
// Rcpp Parrell get suff worker
struct GetSuffDeathWorker: public Worker
{
// -------------------
// Inputs
// -------------------
tree& x;
tree::tree_cp nl; 
tree::tree_cp nr; 
xinfo& xi; 
dinfo& di; 
double* phi; 

// -------------------
// Internal State
// -------------------
double l_n;
double l_sy;
double l_n0;

double r_n;
double r_sy;
double r_n0;

double *xx;//current x
double y;  //current y

// -------------------
// Constructors
// -------------------
// Standard Constructor
GetSuffDeathWorker(tree& x,
				   tree::tree_cp nl,
                   tree::tree_cp nr,
				   xinfo& xi,
				   dinfo& di,
				   double* phi):x(x),nl(nl),nr(nr),xi(xi),di(di),phi(phi) {

    l_n=0.0;
	l_sy=0.0;
	l_n0=0.0;

	r_n=0.0;
	r_sy=0.0;
	r_n0=0.0;
} 
// Splitting Constructor
GetSuffDeathWorker(const GetSuffDeathWorker& gsw, Split):x(gsw.x),nl(gsw.nl),nr(gsw.nr),xi(gsw.xi),di(gsw.di),phi(gsw.phi){

	l_n=0.0;
	l_sy=0.0;
	l_n0=0.0;

	r_n=0.0;
	r_sy=0.0;
	r_n0=0.0;
} 

void operator()(std::size_t begin, std::size_t end){
	for(size_t i=begin;i<end;i++) {
		xx = di.x + i*di.p;
		tree::tree_cp bn = x.bn(xx,xi);
        y = di.y[i];
        
		if(bn==nl) {
			l_n0 += 1;
			l_n  += phi[i];
			l_sy += phi[i]*y;
		}
		if(bn==nr) {
			r_n0 += 1;
			r_n  += phi[i];
			r_sy += phi[i]*y;
		}
	}
}

void join(const GetSuffDeathWorker& gsw){
	l_n   += gsw.l_n;
	l_sy  += gsw.l_sy;
	l_n0  += gsw.l_n0;

	r_n   += gsw.r_n;
	r_sy  += gsw.r_sy;
	r_n0  += gsw.r_n0;
}

}; // End Get Suff Death Worker
//--------------------------------------------------
//get sufficient stats for pair of bottom children nl(left) and nr(right) in tree x
// Death get suff
void getsuffDeath(tree& x, tree::tree_cp nl, tree::tree_cp nr, xinfo& xi, dinfo& di, double* phi, sinfo& sl, sinfo& sr)
{
	GetSuffDeathWorker gsw(x,nl,nr,xi,di,phi);
	parallelReduce(0, di.n, gsw);

	sl.n   = gsw.l_n;
	sl.sy  = gsw.l_sy;
	sl.n0  = gsw.l_n0;

	sr.n   = gsw.r_n;
	sr.sy  = gsw.r_sy;
	sr.n0  = gsw.r_n0;

}

//--------------------------------------------------
//log of the integrated likelihood
double lil(double n, double sy, double sigma, double tau)
{
  double d = 1/(tau*tau) + n;// n is \sum phi_i for het
  
  double out = -log(tau) - 0.5*log(d);
  out += 0.5*sy*sy/d;
  return out;
}

struct FitWorker : public Worker
{
   // source arguments
  	tree& t; // tree
	xinfo& xi; // split rules from xinfo
	dinfo& di; // number of observations from dinfo
	// internal args
	double *xx;
	tree::tree_cp bn;
	std::vector<double>& fv; // node means

   
   // constructing the constructor
   FitWorker(tree& t, 
   	   		  xinfo& xi, 
	   		    dinfo& di, 
	   		 		std::vector<double>& fv) 
      		 : t(t), xi(xi), di(di), fv(fv){}
   
   // declaring function
    void operator()(std::size_t begin, std::size_t end) {

		for(size_t i=begin;i<end;i++) {
			xx = di.x + i*di.p;
			bn = t.bn(xx,xi);
			fv[i] = bn->getm();
		}
   }
};

// struct FitWorker2 : public Worker
// {
//    // source arguments
//   double* fv; // node means
// 	dinfo& di; // number of observations from dinfo
// 	xinfo& xi; // split rules from xinfo
// 	tree& t; // tree

// 	// internal args
// 	double *xx;
// 	tree::tree_cp bn;
   
//    // constructing the constructor
//    FitWorker2(tree& t, 
//    	   		 	  xinfo& xi, 
// 	   		 			dinfo& di, 
// 	   		 			double* fv) 
//       		 		: t(t), xi(xi), di(di), fv(fv){}
   
//    // declaring function
//     void operator()(std::size_t begin, std::size_t end) {

// 		for(size_t i=begin;i<end;i++) {
// 			xx = di.x + i*di.p;
// 			bn = t.bn(xx,xi);
// 			fv[i] = bn->getm();
// 		}
//    }
// };

//--------------------------------------------------
//fit
void fit(tree& t, xinfo& xi, dinfo& di, std::vector<double>& fv)
{

  fv.resize(di.n);

  // FitWorker functor (pass inputs)
  FitWorker fir(t, xi, di, fv);
  
  // call parallelFor to do the work
  parallelFor(0, di.n, fir);
}

//--------------------------------------------------
//fit
void fit(tree& t, xinfo& xi, dinfo& di, double* fv)
{
  // // FitWorker functor (pass inputs)
  // FitWorker2 fir(t, xi, di, fv);
  
  // // call parallelFor to do the work
  // parallelFor(0, di.n, fir);

	double *xx;
	tree::tree_cp bn;
	for(size_t i=0;i<di.n;i++) {
		xx = di.x + i*di.p;
		bn = t.bn(xx,xi);
		fv[i] = bn->getm();
	}

}

//--------------------------------------------------
//partition
void partition(tree& t, xinfo& xi, dinfo& di, std::vector<size_t>& pv)
{
	double *xx;
	tree::tree_cp bn;
	pv.resize(di.n);
	for(size_t i=0;i<di.n;i++) {
		xx = di.x + i*di.p;
		bn = t.bn(xx,xi);
		pv[i] = bn->nid();
	}
}
//--------------------------------------------------
// draw all the bottom node mu's

void drmu(tree& t, xinfo& xi, dinfo& di, pinfo& pi, double* weight, RNG& gen)
{


  //  typedef std::vector<tree_p> npv;   //Node Pointer Vector
	tree::npv bnv;

	std::vector<sinfo> sv;


	// get sufficients stats for all bottom nodes
	// sv seems to store the stats of the residuals
	// bnv is the bottom node vector of tree t
	
	allsuff(t,xi,di,weight,bnv,sv);
	
	for(tree::npv::size_type i=0;i!=bnv.size();i++) {
	
		double fcvar = 1.0/(1.0/(pi.tau * pi.tau)+sv[i].n);
		double fcmean = sv[i].sy*fcvar;
		bnv[i]->setm(fcmean + gen.normal()*sqrt(fcvar));

	  if(bnv[i]->getm() != bnv[i]->getm()) { 
		for(int j=0; j<di.n; ++j) Rcout << *(di.x + j*di.p) <<" "; //*(x + p*i+j)
		Rcout << endl <<" fcvar "<< fcvar <<" svi[n] "<< sv[i].n <<" i "<<i;
		Rcout << endl << t;
		Rcpp::stop("drmu failed");
		}
	}
}

//--------------------------------------------------
//write cutpoint information to screen
void prxi(xinfo& xi)
{
  Rcout << "xinfo: \n";
	for(size_t v=0;v!=xi.size();v++) {
	  Rcout << "v: " << v << endl;
		for(size_t j=0;j!=xi[v].size();j++) Rcout << "j,xi[v][j]: " << j << ", " << xi[v][j] << endl;
	}
	Rcout << "\n\n";
}
//--------------------------------------------------
//make xinfo = cutpoints
void makexinfo(size_t p, size_t n, double *x, xinfo& xi, size_t nc)
{
	double xinc;
	
	//compute min and max for each x
	std::vector<double> minx(p,INFINITY);
	std::vector<double> maxx(p,-INFINITY);
	double xx;
	for(size_t i=0;i<p;i++) {
		for(size_t j=0;j<n;j++) {
			xx = *(x+p*j+i);
			if(xx < minx[i]) minx[i]=xx;
			if(xx > maxx[i]) maxx[i]=xx;
		}
	}
	//make grid of nc cutpoints between min and max for each x.
	xi.resize(p);
	for(size_t i=0;i<p;i++) {
		xinc = (maxx[i]-minx[i])/(nc+1.0);
		xi[i].resize(nc);
		for(size_t j=0;j<nc;j++) xi[i][j] = minx[i] + (j+1)*xinc;
	}
}
// get min/max needed to make cutpoints
void makeminmax(size_t p, size_t n, double *x, std::vector<double> &minx, std::vector<double> &maxx)
{
	double xx;
	
	for(size_t i=0;i<p;i++) {
		for(size_t j=0;j<n;j++) {
			xx = *(x+p*j+i);
			if(xx < minx[i]) minx[i]=xx;
			if(xx > maxx[i]) maxx[i]=xx;
		}
	}
}
//make xinfo = cutpoints give the minx and maxx vectors
void makexinfominmax(size_t p, xinfo& xi, size_t nc, std::vector<double> &minx, std::vector<double> &maxx)
{
	double xinc;
	//make grid of nc cutpoints between min and max for each x.
	xi.resize(p);
	for(size_t i=0;i<p;i++) {
		xinc = (maxx[i]-minx[i])/(nc+1.0);
		xi[i].resize(nc);
		for(size_t j=0;j<nc;j++) xi[i][j] = minx[i] + (j+1)*xinc;
	}
}
