#include "arb.h"
#include "acb.h"
#include "arb_fmpz_poly.h"
#include "arf.h"
#include <math.h>

double power(double base, unsigned int exp) {
    int i = 1;
    double result = 1;
    for (i = 0; i < exp; i++)
        result *= base;
    return result;
 }

int main()
{
	fmpz_poly_t polynomial_to_solve;
	fmpz_poly_init(polynomial_to_solve);	
	// Polynomial is (x^7+x-1)*(x-1000)
	fmpz_poly_set_coeff_si(polynomial_to_solve,8,1);
	fmpz_poly_set_coeff_si(polynomial_to_solve,7,-1000);
	fmpz_poly_set_coeff_si(polynomial_to_solve,2,1);
	fmpz_poly_set_coeff_si(polynomial_to_solve,1,-1001);
	fmpz_poly_set_coeff_si(polynomial_to_solve,0,1000);
	
	acb_ptr roots;
	acb_t result; 
	acb_init(result);
	arf_t error;
	arf_init(error);
	arf_set_d(error,2);
	roots = _acb_vec_init(fmpz_poly_degree(polynomial_to_solve));
	int i = 0;
	arf_t target;
	arf_init(target);
	// Sets target precision to 2^{-1000}
	arf_set_si_2exp_si(target,1,-1000);
	//arf_set_si_2exp_si(target,1,-100);
	flint_printf("Target error is: ");
	arf_printd(target,10);
	flint_printf("\n");

	acb_t base;
	acb_init(base);
	acb_set_si(base,50);
	acb_printd(base,3);
	flint_printf("\n");
	while(arf_cmp(error,target) > 0 && i < 24) { 
		arb_fmpz_poly_complex_roots(roots,polynomial_to_solve,0,power(2,i+10));
		flint_printf("Root: ");
		acb_printd(roots+1,10);
		flint_printf("\n");
		acb_pow(result,base,roots+1,power(10,5));
		acb_get_rad_ubound_arf(error,result,power(10,5));
		flint_printf("Run %d had error: ",i+1);
		arf_printd(error,10);
		flint_printf("\n");
		i += 1;
	}
	// For fair timing purposes: Require evaluations at all roots
	for(int i = 0;i<8;i++) { 
		acb_pow(result,base,roots+i,power(10,5));
	}

	flint_printf("Required number of loops: %d",i);	
	flint_printf("\n");
	flint_printf("Rounded evaluation result at x=1000: ");
	acb_printd(roots+1,10);
	flint_printf("\n");
}