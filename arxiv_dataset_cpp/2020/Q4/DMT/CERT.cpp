#include "CERT.h"

using namespace std;

CERT::CERT()
{
}

CERT::CERT(std::vector<dlib::shape_predictor> SD, double D, std::vector<std::pair<dlib::full_object_detection, double>> WB, dlib::matrix<float, 0, 1> IS) : SubDivisions(SD), TotalDeviation(D), WeightedBins(WB), InitShape(IS)
{
	setInitShape(IS); // Ensure all subdivisions have the same initshape
}


CERT::~CERT()
{
}

void CERT::serialize(std::ostream& out)
{
	dlib::serialize(this->SubDivisions, out);
	dlib::serialize(this->WeightedBins, out);
	dlib::serialize(this->InitShape, out);

}

void CERT::deserialize(std::istream& in)
{
	dlib::deserialize(this->SubDivisions, in);
	dlib::deserialize(this->WeightedBins, in);
	dlib::deserialize(this->InitShape, in);
}

void CERT::setSubDivisions(std::vector<dlib::shape_predictor> NSD)
{
	SubDivisions = NSD;
	setInitShape(getInitShape()); // Ensure all subdivisions have the same initshape. If the Initshapoe of the new SubDivisions must be used, call the setinit shape first, or again later using the wanted initshape.
}

std::vector<dlib::shape_predictor> CERT::getSubDivisions()
{
	return SubDivisions;
}


void CERT::setWeightedBins(std::vector<std::pair<dlib::full_object_detection, double>>  NWB)
{
	WeightedBins = NWB;
}

std::vector<std::pair<dlib::full_object_detection, double>> CERT::getWeightedBins()
{
	return WeightedBins;
}

void CERT::setInitShape(dlib::matrix<float, 0, 1> NIS)
{
	InitShape = NIS;
	for (int i = 0; i < SubDivisions.size(); i++)
	{
		dlib::shape_predictor TempERT = dlib::shape_predictor(NIS, SubDivisions[i].get_forests() , SubDivisions[i].get_pixel_coords()); // Create temp model using new InitShape
		SubDivisions[i] = TempERT; // Replace subdivision with model with new InitShape.
	}
}

dlib::matrix<float, 0, 1> CERT::getInitShape()
{
	return InitShape;
}

dlib::full_object_detection CERT::PredictFinalShape(dlib::array2d<unsigned char> &Image, dlib::rectangle Dect)
{
	//cout << "TEST" << endl;
	dlib::full_object_detection FinalShape;
	std::vector<dlib::point> Shape;
	for (int i = 0; i < SubDivisions.size(); i++) // Get sub results from all subdivisions
	{
		WeightedBins[i].first = (SubDivisions[i](Image, Dect)); // get result of subdivision in bin
	}
	//cout << "TEST2" << endl;
	for (int u = 0; u < WeightedBins[0].first.num_parts(); u++)
	{
		//cout << u << endl;
		long x = 0;
		long y = 0;
		for (int o = 0; o < SubDivisions.size(); o++) // adding all values of all subdivisons
		{
			if (WeightedBins[o].first.part(u).x() > 0)
				x += (WeightedBins[o].first.part(u).x() * WeightedBins[o].second + 0.5); // Add weighted bin to the raw result of X. 
			if (WeightedBins[o].first.part(u).y() > 0)
				y += (WeightedBins[o].first.part(u).y() * WeightedBins[o].second + 0.5); // Add weighted bin to the raw result of Y. 
		}
		x /= TotalDeviation; // Use total deviation to calculate final results
		y /= TotalDeviation;
		Shape.push_back(dlib::point(x, y));
		//cout <<"PART: " << u << "X: " << x << "  Y: " << y << "TotalDiv: "<< TotalDeviation << endl; // Debug code to  see final x and y results
	}
	FinalShape = dlib::full_object_detection(Dect, Shape);

	return FinalShape;

}
