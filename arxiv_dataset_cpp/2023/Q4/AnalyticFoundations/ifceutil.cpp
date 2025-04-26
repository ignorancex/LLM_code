#include <torch/extension.h>
#include <iostream>

// lower semi-continuous step function
template <typename scalar_t>
scalar_t heavisize(scalar_t inp) { return inp>0.0 ? 1.0 : 0.0; }

// WENO interpolation at a single point 
template <typename scalar_t>
scalar_t interpolatef(
	scalar_t pts,
	const scalar_t* __restrict__ fun,
	int npx,int fidx)
{	scalar_t PI = 3.141592653589793238462643383279502884197169399375105820974;
	scalar_t dx = 2.0*PI/(npx);
	scalar_t ep = 2.0*std::numeric_limits<scalar_t>::epsilon();

	int ltm,ltk,rtk,rtp;
	scalar_t lmv,ltv,rtv,rpv;
	scalar_t pnt,tht,wgt; 
	scalar_t p2l,p2r,btl,btr;
	
	pnt = pts;											// interpolation point
	pnt = pnt - 2*PI*std::floor( (pnt+PI)/(2.0*PI) ); 	// map from R to R mod 2piZ
	ltk = std::floor( (pnt+PI)/dx );					// point lies in the interval [x_ltk, x_rtk)
	rtk = (ltk == npx-1) ? 0 : ltk+1;							
	ltm = (ltk == 0 ) ? (npx-1) : ltk-1;				// [x_ltm,x_ltk] left-adjacent interval
	rtp = (rtk == npx-1) ? 0 : rtk+1;					// [x_rtk,x_trp] right-adjacent interval

	tht = (pnt - (-PI + ltk*dx))/dx;					// fraction in interval
	wgt = (2.0-tht)/3.0;								// weighting for left polynomial
			
	lmv = fun[ltm+npx*fidx];							// function values at endpoints
	ltv = fun[ltk+npx*fidx];										
	rtv = fun[rtk+npx*fidx];
	rpv = fun[rtp+npx*fidx];

	btl = ep + (rtv-ltv)*(rtv-ltv) + 13.0*(rtv+lmv-2.0*ltv)*(rtv+lmv-2.0*ltv)/12.0;		// smoothness measure for left interpolant
	btr = ep + (rtv-ltv)*(rtv-ltv) + 13.0*(rpv+ltv-2.0*rtv)*(rpv+ltv-2.0*rtv)/12.0;		// smoothness measure for right interpolant

	if( btr > btl )
	{	btl = btl/btr;
		wgt = wgt/( wgt + btl*btl*(1.0-wgt) );	} 
	else
	{	btl = btr/btl;
		wgt = 1.0 - (1.0 - wgt)/( (1.0 - wgt) + btl*btl*wgt );	}

	p2l = ltv + .5*(rtv-lmv)*tht + .5*(rtv+lmv-2.0*ltv)*tht*tht;	// degree 2 Lagrange on left-adjacent interval
	tht = tht - 1.0;
	p2r = rtv + .5*(rpv-ltv)*tht + .5*(rpv+ltv-2.0*rtv)*tht*tht;	// degree 2 Lagrange on right-adjacent interval

	return wgt*p2l + (1.0-wgt)*p2r;	}

template <typename scalar_t>
void chofvar__impl(
	scalar_t* __restrict__ xi,			// inverse maps xi( eta(x) ) = x at grid points
	const scalar_t* __restrict__ spd,	// the local speeds 1/|G'(x)| at grid points
	const scalar_t* __restrict__ len,	// the total lengths of each interface
	int nc,int npx)
{	scalar_t PI = 3.141592653589793238462643383279502884197169399375105820974;
	scalar_t dx = 2.0*PI/(npx);

	scalar_t F1,F2,F3,F4;				// now use RK4 to update inverse map
	scalar_t cur,sig;
	for(int i=0;i<nc;++i)				// process each interface
	{	sig = len[i]/(2.0*PI);			// constant speed for current interface 
		xi[0+npx*i] = -PI;				// initialize xi(-pi) = -pi
		
		for(int j=1;j<npx;++j)
		{	cur = xi[j-1+npx*i];	
			F1  = interpolatef(cur,spd,npx,i);

			cur = xi[j-1+npx*i] + .5*dx*sig*F1;
			F2  = interpolatef(cur,spd,npx,i);

			cur = xi[j-1+npx*i] + .5*dx*sig*F2;
			F3  = interpolatef(cur,spd,npx,i);

			cur = xi[j-1+npx*i] + dx*sig*F3;
			F4  = interpolatef(cur,spd,npx,i);

			cur = xi[j-1+npx*i] + dx*sig*(F1 + 2.0*F2 + 2.0*F3 + F4)/(6.0);
			xi[j+npx*i] = cur;	} } }

void chofvar(torch::Tensor xi,
			 const torch::Tensor spd,
			 const torch::Tensor len)
{	int nc 	= spd.size(0);
	int npx = spd.size(1);

	AT_DISPATCH_FLOATING_TYPES(spd.scalar_type(),"chofvar",([&]{
		chofvar__impl<scalar_t>(
			xi.data_ptr<scalar_t>(),
			spd.data_ptr<scalar_t>(),
			len.data_ptr<scalar_t>(),nc,npx);}));	}

template <typename scalar_t>
void cleanth__impl(
	scalar_t* __restrict__ outtht,
	const scalar_t* __restrict__ inptht,
	int nc,int npx)
{	scalar_t PI = 3.141592653589793238462643383279502884197169399375105820974;

	scalar_t cur,nxt,del;								// process each interface
	for(int i=0;i<nc;++i)
	{	cur             = inptht[0+npx*i];				// remove discontinuities of jump size > pi
		outtht[0+npx*i] = cur;
		for(int j=1;j<npx;++j)
		{	nxt = inptht[j+npx*i];
			del = nxt - cur - 2.0*PI*( heavisize( nxt-cur-PI ) - heavisize( cur-nxt-PI ) );
			cur = nxt;
			outtht[j+npx*i] = outtht[j-1+npx*i]+del; } } }

void cleanth(torch::Tensor outtht,
			 const torch::Tensor inptht)
{	int nc 	= outtht.size(0);
	int npx = outtht.size(1);

	AT_DISPATCH_FLOATING_TYPES(outtht.scalar_type(),"chofvar",([&]{
		cleanth__impl<scalar_t>(
			outtht.data_ptr<scalar_t>(),
			inptht.data_ptr<scalar_t>(),nc,npx);}));	}

template <typename scalar_t>
void getdist__impl(
	scalar_t* __restrict__ dsts,
	const scalar_t* __restrict__ gam,
	int nc,int npx)
{	scalar_t gamx,gamy,psix,psiy,dist;
	int cidx,didx,pidx,qidx;

	int ntot = nc*npx;
	for(int k=0;k<ntot;++k)
	{	cidx = std::floor( k/npx );
		pidx = k - npx*cidx;
		gamx = gam[0 + 2*pidx + 2*npx*cidx];
		gamy = gam[1 + 2*pidx + 2*npx*cidx]; 
		for(int l=k;l<ntot;++l)
		{	didx = std::floor( l/npx );
			qidx = l - npx*didx;
			psix = gam[0 + 2*qidx + 2*npx*didx];
			psiy = gam[1 + 2*qidx + 2*npx*didx]; 
			dist = .5*(gamx-psix)*(gamx-psix) + .5*(gamy-psiy)*(gamy-psiy);

			dsts[k + ntot*l] = dist;
			dsts[l + ntot*k] = dist; } } }

void getdist(torch::Tensor dsts,
			 const torch::Tensor gams)
{	int nc 	= gams.size(0);
	int npx = gams.size(1);

	AT_DISPATCH_FLOATING_TYPES(gams.scalar_type(),"getdist",([&]{
		getdist__impl<scalar_t>(
			dsts.data_ptr<scalar_t>(),
			gams.data_ptr<scalar_t>(),nc,npx);}));	}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m)
{	m.def("chofvar",&chofvar,"chofvar");	
	m.def("cleanth",&cleanth,"cleanth");	
	m.def("getdist",&getdist,"getdist");	}
