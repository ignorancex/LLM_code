
#include <NTL/BasicThreadPool.h>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <algorithm>

#include "Methods.h"
#include "Tools.h"

#include "SerializationUtils.h"  // error: ‘SerializationUtils’ has not been declared

#include <cstdio>

using namespace std;
using namespace NTL;

int main(int argc, char **argv) {

	SetNumThreads(98);

	//string trainfile = "../data/Original_MNIST_testdata.csv";
	//string trainfile = "../data/REGNET_X_400MF_MNIST_TRAININGfirst8192.csv";
	string trainfile = "../data/REGNET_X_400MF_MNIST_TRAININGfirst128.csv";
	string testfile  = "../data/REGNET_X_400MF_MNIST_TESTING.csv";

	long trainSampleDim = 0, testSampleDim = 0, trainfactorDim = 0,
			testfactorDim = 0;
	double **traindataset, **testdataset;
	int *traindatalabel, *testdatalabel;

	Tools::dataFromFile(trainfile, trainfactorDim, trainSampleDim, traindataset,
			traindatalabel);
	Tools::dataFromFile(testfile, testfactorDim, testSampleDim, testdataset,
			testdatalabel);

	for (int n = 0; n < 5; n++) {
		cout << endl << testdatalabel[n] << endl;

		for (int i = 0; i < 20; ++i) {
			for (int j = 0; j < 20; ++j)
				if (traindataset[n][i * 20 + j] < 1e-2)
					cout << "     ";
				else
					printf("%5.2f", int(testdataset[n][i * 20 + j]));
			cout << endl;
		}
	}

	for (int n = 0; n < 5; n++) {
		cout << endl << testdatalabel[n] << endl;

		for (int i = 0; i < 20; ++i) {
			for (int j = 0; j < 20; ++j)
				if (testdataset[n][i * 20 + j] < 1e-2)
					cout << "     ";
				else
					printf("%5.2f", testdataset[n][i * 20 + j]);
			cout << endl;
		}
	}

	cout << endl;
	cout << "trainSampleDim : " << trainSampleDim << endl;
	cout << "testSampleDim  : " << testSampleDim << endl;
	cout << "trainfactorDim : " << trainfactorDim << endl;
	cout << "testfactorDim  : " << testfactorDim << endl;
	cout << endl << endl;

	//  traindata:=[label,datarow]  >> [1,datarow]
	Tools::zDataFromFile(trainfile, trainfactorDim, trainSampleDim,
			traindataset, traindatalabel);
	Tools::zDataFromFile(testfile, testfactorDim, testSampleDim, testdataset,
			testdatalabel);
	cout << endl << endl;
	cout << "---------------------------------" << "zDataFromFile"
			<< "---------------------------------" << endl;
	cout << endl << endl;

	for (int n = 0; n < 5; n++) {
		cout << endl << testdatalabel[n] << endl;

		for (int i = 0; i < 20; ++i) {
			for (int j = 0; j < 20; ++j)
				if (testdataset[n][i * 20 + j] < 1e-2)
					cout << "     ";
				else
					printf("%5.2f", testdataset[n][i * 20 + j]);
			cout << endl;
		}
	}

	cout << endl;
	cout << "trainSampleDim : " << trainSampleDim << endl;
	cout << "testSampleDim  : " << testSampleDim << endl;
	cout << "trainfactorDim : " << trainfactorDim << endl;
	cout << "testfactorDim  : " << testfactorDim << endl;
	cout << endl << endl;

	cout
			<< "-------------------------------------------------------------------------------------"
			<< endl;
	cout
			<< "    CNNtraining(traindataset, traindatalabel, trainfactorDim, trainSampleDim, testdataset, testdatalabel, testfactorDim, testSampleDim);"
			<< endl;
	cout
			<< "-------------------------------------------------------------------------------------"
			<< endl;
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	CNNtraining(traindataset, traindatalabel, trainfactorDim, trainSampleDim,
			testdataset, testdatalabel, testfactorDim, testSampleDim);
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	cout << endl << "END OF THE PROGRAMM" << endl;
	cout << endl << "It is surely going to be all right !! " << endl;
	return 0;
}
