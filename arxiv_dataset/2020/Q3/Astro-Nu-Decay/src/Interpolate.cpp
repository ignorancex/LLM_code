#include <vector>
#include <stdexcept>
#include <cmath>

#include "Interpolate.h"

// An interpolating code
double Interpolate(double x, std::vector<std::pair<double, double> > table)
{
	if ((x > table.back().first) or (x < table[0].first))
		throw std::domain_error("x outside look up table range: " + std::to_string(table[0].first) + " " + std::to_string(x) + " " + std::to_string(table.back().first));

	std::vector<std::pair<double, double> >::iterator it, it2;
	it = lower_bound(table.begin(), table.end(), std::make_pair(x, -1e100));
	// Corner case
	if (it == table.begin()) return it->second;
	it2 = it;
	--it2;
	return it2->second + (it->second - it2->second) * (x - it2->first) / (it->first - it2->first);
}

