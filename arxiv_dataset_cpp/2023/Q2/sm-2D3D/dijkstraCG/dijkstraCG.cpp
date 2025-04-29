/**
 * Shape Cut for Conjugate Product Graph
 *
 */

#include "mex.h"

#include <exception>
#include <fstream>
#include <list>
#include <iostream>
#include <string>
#include <cmath>

#include "buildMeshCG.hpp"
#include "helperCG.hpp"
#include "manifoldDijkstraCG.hpp"
#include "MinHeapCG.hpp"
#include "shapeCutCG.hpp"
#include "dijkstraCG_main.hpp"

using namespace std;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[])
{
// check and read mex input
	if (nrhs != 3 || nlhs != 3)
		mexErrMsgTxt("Usage: [energy, path, correspondence] = dijkstra(vertices, triangles, contour). Transform indices in triangles to initial 0.");

	// read vertices
	const double* const vertices = mxGetPr(prhs[0]);
	const long nVertices = long( mxGetN(prhs[0]) );
	const int dimVertices = int( mxGetM(prhs[0]) );

	// read triangles
	const double* const triangles = mxGetPr(prhs[1]);
	const long nTriangles = long( mxGetN(prhs[1]) );
	const int dimTriangles = int( mxGetM(prhs[1]) );

	// read contour
	const double* const contour = mxGetPr(prhs[2]);
	const long nContour = long( mxGetN(prhs[2]) ) - 1; // hackerman *_*
	const int dimContour = int( mxGetM(prhs[2]) );
    
    // cost mode 
    const double costMode = 13;

	if (dimTriangles != 3)
		mexErrMsgTxt("Triangles must be given in 3xm.");

	//if (dimContour != dimVertices)
	//	mexErrMsgTxt("Descriptors must have the same dimension. (Remove this constraint if you changed the energy function to handle different dimensional features.)");

try
{

    std::vector<std::tuple<long,long>> matching;
	const double energy = dijkstra_main(vertices, nVertices, dimVertices, triangles, nTriangles, contour, nContour, dimContour, matching);

	// transform results for mex

	// return loop
	plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(matching.size(), 1, mxREAL);
	plhs[2] = mxCreateDoubleMatrix(matching.size(), 1, mxREAL);
	double* e = mxGetPr(plhs[0]);
	double* resultPath = mxGetPr(plhs[1]);
	double* resultCorr = mxGetPr(plhs[2]);

	// energy
	e[0] = energy;

	// path
    int i = 0;
	for (const auto& match : matching) {
		resultPath[i] = std::get<0>(match); // 3d vertex id
		resultCorr[i] = std::get<1>(match); // 2d vertex id
        i++;
	}

} catch (std::exception& e)
	{
		const std::string msg = "Exception caught: " + std::string(e.what());
		mexErrMsgTxt(msg.c_str());
	}

} // close mexFunction
