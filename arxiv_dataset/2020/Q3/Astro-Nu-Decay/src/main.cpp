#include <omp.h>

#include "Parameters.h"
#include "Figures.h"
#include "Redshift.h"

#include "IceCube.h"
#include "SM.h"

int main()
{
	setbuf(stdout, NULL);

	omp_set_num_threads(40);

	// Initializing functions
	Recalc_Parameters();
	I2_precalc();

printf("%g\n", Rtc(dp_bm, false));
return 0;

	// Each command calculates the data for one figure, see src/Figures.cpp
	// Some take <1 second, some take O(10) minutes on one core, some take O(10) hours on 40 cores
//	Flavor_Ratio(); // quick
//	Invisible(); // quick
//	Visible_g();
//	Visible_m1();
//	Visible_gamma();
//	Visible_R(); // slow
//	Visible_f();
//	Rtc();
//	Analytic_Validate(); // quick
//	IC_gamma();
//	IC_gamma_2D(); // slow
//	Visible_SPS();
//	Visible_nnb();
//	Chisq(); // slowww

	return 0;
}
