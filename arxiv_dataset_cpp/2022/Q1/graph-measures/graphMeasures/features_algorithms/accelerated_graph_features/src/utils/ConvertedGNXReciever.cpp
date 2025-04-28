/*
 * ConvertedGNXReciever.cpp
 *
 *  Created on: Oct 28, 2018
 *
 */

#include "../includes/ConvertedGNXReciever.h"

ConvertedGNXReciever::ConvertedGNXReciever(dict converted_graph) {

	list offsetList = extract<list>(converted_graph["indices"]);
	list neighborList = extract<list>(converted_graph["neighbors"]);
	bool withWeights = extract<bool>(converted_graph["with_weights"]);
	bool directed = extract<bool>(converted_graph["directed"]);
	list weightsList;
	if(withWeights)
		weightsList = extract<list>(converted_graph["weights"]);


	this->offsets = new std::vector<int64>();
	this->offsets->reserve(len(offsetList));
	this->neighbors = new std::vector<unsigned int>();
	this->neighbors->reserve(len(neighborList));
	this->weights = new std::vector<double>();

	for (int i = 0; i < len(offsetList); ++i) {
		int64 currentOffset;
		currentOffset =
				static_cast<int64>(extract<unsigned int>(offsetList[i]));

		this->offsets->push_back(currentOffset);
	}

	for (int i = 0; i < len(neighborList); ++i) {
		unsigned int currentNeighbor = extract<unsigned int>(neighborList[i]);
		this->neighbors->push_back(currentNeighbor);
	}
	this->mGraph = new CacheGraph(directed);
	if(withWeights){
		for (int i = 0; i < len(weightsList); ++i) {
				double currentNeighbor = extract<double>(weightsList[i]);
				this->weights->push_back(currentNeighbor);
			}
		mGraph->Assign(*offsets, *neighbors,*weights);

	}else{
		mGraph->Assign(*offsets, *neighbors);

	}



}

ConvertedGNXReciever::~ConvertedGNXReciever() {

	delete offsets;
	delete neighbors;
	delete mGraph;
}

