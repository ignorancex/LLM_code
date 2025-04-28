/*
 * KCoreWrapper.cpp
 *
 *  Created on: Nov 12, 2018
 *
 */

#include "KCoreWrapper.h"

void BoostDefKCore() {
	def("k_core",KCoreCalculatorWrapper);
}

boost::python::list KCoreCalculatorWrapper(dict converted_graph) {
	ConvertedGNXReciever reciever(converted_graph);
	KCoreFeatureCalculator calc;
	calc.setGraph(reciever.getCacheGraph());
	std::vector<unsigned short> vecResults = calc.Calculate();
	return vectorToPythonList<unsigned short>(vecResults);
}
