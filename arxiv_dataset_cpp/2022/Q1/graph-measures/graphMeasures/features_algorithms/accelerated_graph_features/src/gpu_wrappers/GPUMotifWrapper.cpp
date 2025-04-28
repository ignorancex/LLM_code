/*
 * ExampleWrapper.cpp
 *
 *  Created on: Nov 11, 2018
 *
 */

#include "GPUMotifWrapper.h"

void BoostDefGPUMotifCalculator() {
	def("motif_gpu", GPUMotifCalculatorWrapper);
}


py::list GPUMotifCalculatorWrapper(dict converted_dict,int level, int cudaDevice) {
	bool directed = extract<bool>(converted_dict["directed"]);
	ConvertedGNXReciever reciever(converted_dict);
	GPUMotifCalculator calc(level, directed, cudaDevice);
	calc.setGraph(reciever.getCacheGraph());
	vector<vector<unsigned int>*>* res = calc.Calculate();
	py::list motif_counters = convertVectorOfVectorsTo2DList(res);
	for (auto p : *res) {
		delete p;
	}
	delete res;
	return motif_counters;

}
