#include "../includes/BfsMomentsCalculator.h"

BfsMomentsCalculator::BfsMomentsCalculator() {
}

vector<floatTuple> BfsMomentsCalculator::Calculate() {

	const int numOfNodes = mGraph->GetNumberOfNodes();
	vector<floatTuple> features(numOfNodes);

	for (int i = 0; i < numOfNodes; i++) {

		//calculate BFS distances
		std::vector<unsigned int> distances = DistanceUtils::BfsSingleSourceShortestPath(
				mGraph, i);

//count the number of times each distance exists
		std::unordered_map<unsigned int, int> distCounter;

		for (int j = 0; j < distances.size(); j++) {
			if (distCounter.find(distances[j]) == distCounter.end())
				//distance[j] hasn't been counted before
				distCounter[distances[j]] = 0;
			distCounter[distances[j]] += 1;
		}

		std::vector<float> dists(distCounter.size()), weights(
				distCounter.size());

		for (const auto& n : distCounter) {
			dists.push_back((float) n.first + 1); // the key is the distance, which needs adjustment
			weights.push_back((float) n.second); //the value is the number of times it has been counted
		}

		features[i] = std::make_tuple(
				MathUtils::calculateWeightedAverage(dists, weights, dists.size()),
				MathUtils::calculateWeightedStd(dists, weights, dists.size()));

	}

	return features;
}

BfsMomentsCalculator::~BfsMomentsCalculator() {
}
