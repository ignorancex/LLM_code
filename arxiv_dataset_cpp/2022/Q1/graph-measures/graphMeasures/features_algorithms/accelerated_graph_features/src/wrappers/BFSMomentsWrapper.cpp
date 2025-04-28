/*
 * BFSMomentsWrapper.cpp
 *
 *  Created on: Nov 12, 2018
 *
 */

#include "BFSMomentsWrapper.h"

void BoostDefBFSMoments() {
	def("bfs_moments",BFSMomentWrapper);
}

py::list tupleVectorToPythonList(const std::vector<floatTuple>& v){
	py::list l;
	for(int i=0;i<v.size();i++){
		std::tuple<float,float> current = v[i];

		py::tuple py_tuple = py::make_tuple<float,float>(std::get<0>(current),std::get<1>(current));
		l.append<py::tuple>(py_tuple);
	}

	return l;
}


py::list BFSMomentWrapper(dict converted_dict) {

	ConvertedGNXReciever reciever(converted_dict);
	BfsMomentsCalculator calc;
	calc.setGraph(reciever.getCacheGraph());
	std::vector<std::tuple<float, float>> resVec = calc.Calculate();
	return tupleVectorToPythonList(resVec);

}
