#include "SM.h"
#include "Parameters.h"

// The SM probability correctly rescaled for cosmology
double PSM(flavor a, flavor b, decay_params dp)
{
	double p = 0;

	for (int i = 0; i < 3; i++)
		p += std::norm(U[a][i]) * std::norm(U[b][i]);

	if (dp.x_is_redshift)
		p *= pow(1 + dp.x, -dp.gamma);

	return p;
}

