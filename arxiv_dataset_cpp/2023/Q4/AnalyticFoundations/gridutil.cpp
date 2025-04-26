#include <torch/extension.h>
#include <iostream>

template <typename scalar_t>
void interpolate__impl(
	scalar_t* __restrict__ val,
	const scalar_t* __restrict__ pts,
	const scalar_t* __restrict__ fun,
	int nfn,int npx,int npt)
{	scalar_t PI = 3.141592653589793238462643383279502884197169399375105820974;
	scalar_t dx = 2.0*PI/(npx);
	scalar_t ep = 2.0*std::numeric_limits<scalar_t>::epsilon();

	int ltm,ltk,rtk,rtp;
	scalar_t lmv,ltv,rtv,rpv;
	scalar_t pnt,tht,wgt; 
	scalar_t p2l,p2r,btl,btr;
	for(int i=0;i<nfn;++i)			// process each function
	{	
		for(int j=0;j<npt;++j)
		{	pnt = pts[j+npt*i];											// interpolation point
			pnt = pnt - 2*PI*std::floor( (pnt+PI)/(2.0*PI) ); 			// map from R to R mod 2piZ
			ltk = std::floor( (pnt+PI)/dx );							// point lies in the interval [x_ltk, x_rtk)
			rtk = (ltk == npx-1) ? 0 : ltk+1;							
			ltm = (ltk == 0 ) ? (npx-1) : ltk-1;						// [x_ltm,x_ltk] left-adjacent interval
			rtp = (rtk == npx-1) ? 0 : rtk+1;							// [x_rtk,x_trp] right-adjacent interval

			tht = (pnt - (-PI + ltk*dx))/dx;							// fraction in interval
			wgt = (2.0-tht)/3.0;										// weighting for left polynomial
			
			lmv = fun[ltm+npx*i];										// function values at endpoints
			ltv = fun[ltk+npx*i];										
			rtv = fun[rtk+npx*i];
			rpv = fun[rtp+npx*i];

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

			val[j+npt*i] = wgt*p2l + (1.0-wgt)*p2r;	} } }

void interpolate(torch::Tensor val,
				 const torch::Tensor pts,
				 const torch::Tensor fun)
{	int nfn = fun.size(0);
	int npx = fun.size(1);
	int npt = pts.size(1);

	AT_DISPATCH_FLOATING_TYPES(fun.scalar_type(),"interpolate",([&]{
		interpolate__impl<scalar_t>(
			val.data_ptr<scalar_t>(),
			pts.data_ptr<scalar_t>(),
			fun.data_ptr<scalar_t>(),nfn,npx,npt);}));	}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m)
{	m.def("interpolate",&interpolate,"interpolate");	}
