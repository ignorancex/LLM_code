#include "DiGraph.h"
#include "Hypergraph.h"
#include "DTARRS.h"
#include "DTIC.h"
#include <cstring>
#include <getopt.h>
#include <chrono>
#include <random>
#include <cstdlib>
#include <vector>
#include <queue>
#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>
#include <cmath>
#include <map>
#include <list>
#include "omp.h"

using namespace std;
using namespace std::chrono;

DiGraph * transpose(DiGraph & G);
DiGraph * reduceMax(DiGraph & G);
DiGraph * reduceAvg(DiGraph & G);
DiGraph * reduceMin(DiGraph & G);
//DiGraph * readDiGraph(string path, double p, int nInterval);

DiGraph * readDiGraph_overall(string path, int nInterval);
int NODE_NUM;

void computePois_Infection_force(string path);

void computePairsTimeDist(string path);

void computePoisNumTimewindow(string path);
//map<int, unordered_set<int>>  max_K_degree;
map<int, unordered_set<int>>  random_K_nodes;
list<unordered_set<int>> max_K_degree;


map<int, map<int, unordered_set<int>>>  all_random_K_nodes;
int select_nodes_num = 0;
char * graphPath = NULL;
int c, aSize, nInterval, nSimulation, nThread;
double p;

int largestTimewindow;
int timwindowIncre;

int life_span;
int iterNum;
float stepProb;
float stepSize;
float prob0_a;
float prob0_b;
float prob0_c;
float a_para;
float rho_1=0.0;
float b_para=0.0;
float rho_2=0.0;
int largestK;
int batch_K;
int distance_threshold;
map<string, map<int, int>> pairsTimeDist;


int timeWindowNum;

string runMode;//basic, a=people_connection_prob, b=time_decaying_prob, overall=a+b
map<int, unordered_set<int>> nodesPoisList;
map<int, map<int, int>> poisTimeWindowNum;
//
int randomIterNum;
int timeDecay;
int timewindowBatch;
int valNum;
int decayWindow;
map<string, map<int, double>> pairs_timewindow_infection_force;


int main(int argc, char** argv) {

	string data(argv[1]);// data path
	runMode = (argv[2]);// overall

	string dataset_name(argv[3]);// dataset name: BBC, TKY, NYC, etc
	iterNum = atoi(argv[4]);// the number of iteration
	int t1 = 0, t2 = atoi(argv[5]);//the largest time window
	timewindowBatch = atoi(argv[6]);// the number of time unit in a T
	largestTimewindow = t2;
	randomIterNum =  atoi(argv[7]);//default = 20

	largestK = atoi(argv[8]);//the largest K
	batch_K = atoi(argv[9]);//incremental to select k

	distance_threshold = atoi(argv[10]);
	a_para = atof(argv[11]);// a parameter
	rho_1 = atof(argv[12]);// rho_1 parameter

	if (dataset_name!="BBC")
	{
		b_para = atof(argv[13]);// b parameter
		rho_2 = atof(argv[14]);// rho_2 parameter
		decayWindow = atof(argv[15]);// decayWindow default to be 14
		computePoisNumTimewindow(data);//venueID, timeID, num of people
//    	computePairsTimeDist(data);//pairs, timeID, dist
		computePois_Infection_force(data);
	}else
	{
		computePairsTimeDist(data);//pairs, timeID, dist
	}
	string backTracing = argv[16];

	//
	t2 = ceil(t2*1.0/timewindowBatch*1.0);
	printf("t2=%d", t2);
	DiGraph & G = *readDiGraph_overall(data, t2 - t1 + 1);

    printf("ceil(t2/timewindowBatch)=%d", t2);

    DiGraph & GT = *transpose(G);

    DTARRS * dtarrsG = new DTARRS(G);
    dtarrsG->generateHypergraph(t1, t2, iterNum);
    printf("HG complete\n");
    list<unordered_set<int>> temp;
    string summaryResult = "./experiment_result/" + runMode + "_" + dataset_name +"_"+backTracing;

    omp_set_num_threads(100);
    list<unordered_set<int>> AtopK = dtarrsG->computeTopK(timewindowBatch, largestTimewindow, summaryResult);

    return 0;
}

DiGraph * transpose(DiGraph & G) {
    DiGraph * GT = new DiGraph();
    for (int u : G.V) {
        for (auto e : G[u]) {
            int v = e.first;
            vector<double> & w = e.second;
            GT->addEdge(v, u, w);
        }
    }
    return GT;
}

DiGraph * reduceMax(DiGraph & G) {
    DiGraph * GMax = new DiGraph();
    vector<double> Wp;
    for (int u : G.V) {
        for (auto e : G[u]) {
            int v = e.first;
            vector<double> & W = e.second;
            double max = 0;
            for (double w : W) {
                if (max < w)
                    max = w;
            }
            Wp.clear();
            Wp.push_back(max);
            GMax->addEdge(u, v, Wp);
        }
    }
    return GMax;
}

DiGraph * reduceAvg(DiGraph & G) {
    DiGraph * GAvg = new DiGraph();
    vector<double> Wp;
    for (int u : G.V) {
        for (auto e : G[u]) {
            int v = e.first;
            vector<double> & W = e.second;
            double avg = 0;
            for (double w : W) {
                avg += w;
            }
            avg /= (double) W.size();
            Wp.clear();
            Wp.push_back(avg);
            GAvg->addEdge(u, v, Wp);
        }
    }
    return GAvg;
}

DiGraph * reduceMin(DiGraph & G) {
    DiGraph * GMin = new DiGraph();
    vector<double> Wp;
    for (int u : G.V) {
        for (auto e : G[u]) {
            int v = e.first;
            vector<double> & W = e.second;
            double min = 1;
            for (double w : W) {
                if (min >  w)
                	min = w;
            }
            Wp.clear();
            Wp.push_back(min);
            GMin->addEdge(u, v, Wp);
        }
    }
    return GMin;
}

DiGraph * readDiGraph_overall(string path, int nInterval){
		int V, E;
	    long seed = chrono::high_resolution_clock::now().time_since_epoch().count();
	    DiGraph * G = new DiGraph();
	    ifstream infile(path);
	    string line;
	    priority_queue<pair<int, int> > pq;
	    getline(infile, line);
	    istringstream iss(line);
	    iss >> V >> E;
	    printf("readDiGraph_overall function \n");
	    printf("V=%d,E=%d",V,E);
	    int reduced_connection = 0;

	    int seperateTimeWindows = ceil(largestTimewindow/timewindowBatch);

	    map<int, map<int, unordered_set<int>>> susceptible_list;//nodeid, timewindow, count[V+1][seperateTimeWindows];
	    int timeID, u, v;
	    double probability;
	    int poisPeopleNum;
	    vector<double> W(seperateTimeWindows);
	    vector<double> W_1(seperateTimeWindows);

	    int count_num = 0;
	    while (getline(infile, line)) {
	        istringstream iss(line);
	        count_num = count_num + 1;
	        iss >> timeID >> u >> v>>probability;
	        printf("count_num = %d\n", count_num);
	        int weight_timeID = ceil(1.0*timeID/timewindowBatch);
	        if (timeID>=largestTimewindow)
	        {
	        	continue;
	        }

	        if (G->existEdge(u - 1, v - 1) )
	        {
	        	vector<double> temp_W;
        		temp_W= G->returnWeight(u - 1, v - 1);
	        	double startProb = probability;
	        	temp_W[weight_timeID-1] = startProb;
	        	printf("seperateTimeWindows = %d, weight_timeID-1 = %d, temp_W[weight_timeID-1] = %f\n",seperateTimeWindows,weight_timeID-1, startProb);
	        	G->addEdge(u - 1, v - 1, temp_W);

	        }else
	        {
	        	W.clear();
	        	W = W_1;
	        	double startProb = probability;
	        	printf("seperateTimeWindows = %d, weight_timeID-1 = %d, W[weight_timeID-1] = %f\n",seperateTimeWindows,weight_timeID-1, startProb);

	        	W[weight_timeID-1] = startProb;
	        	G->addEdge(u - 1, v - 1, W);
	        }

	  	   map<int, map<int, unordered_set<int>>>::iterator findNode = susceptible_list.find(u-1);
		   if(findNode!=susceptible_list.end())
		   {
			   map<int, unordered_set<int>> timeNum = susceptible_list[u-1];
			   map<int, unordered_set<int>>::iterator timeNumIter = timeNum.find(weight_timeID-1);
			   if(timeNumIter!=timeNum.end())
			   {

				   timeNum[weight_timeID-1].insert(timeNum[weight_timeID-1].size()+1);
			   }else
			   {
				   timeNum[weight_timeID-1].insert(1);
			   }
			   susceptible_list[u-1] = timeNum;
		   }else
		   {
			   map<int, unordered_set<int>> timeNum;
			   timeNum[weight_timeID-1].insert(1);
			   printf("insert timeID-1 = %d, v-1=%d", weight_timeID-1, u-1);
			   //timeNum[beginTime-1] = 1;
			   susceptible_list[u-1] = timeNum;
		   }
	    }


	    for (int i = 1; i <= V; i++) {
			//map<int, map<int,int>>::iterator findNode = susceptible_list.find(i-1);
			map<int, unordered_set<int>> timeNum = susceptible_list[i-1];
			int deg = 0;
			for(int j = 1; j<=seperateTimeWindows; j++)
			{
				deg = deg + timeNum[j-1].size();


			}
			 pq.push(make_pair(deg, i));
		 }


	    int batchNum = largestK/batch_K;

	     for(int i = 0; i < batchNum; i++)
	     {
	        	int mapID = (i+1)*batch_K;
	        	unordered_set<int> tempNodesList;
	        	while(tempNodesList.size()<batch_K)
	        	{
	        		pair<int, int> top = pq.top();
	        		pq.pop();
	        		tempNodesList.insert(top.second);
	        	}
	        	max_K_degree.push_back(tempNodesList);
	     }

	    for(int i = 1; i<=randomIterNum; i++)
	    {
	    	unordered_set<int> oneIterRandomNodesList;
	    	for(int j = 0; j < batchNum; j++)
	    	{
	    		int mapID = (j+1)*batch_K;
	    		unordered_set<int> tempNodesList;
	    		while(oneIterRandomNodesList.size()<mapID)
	    		{
	    			int temp_rand =  rand() / (double)RAND_MAX *(G->nvtx - 1) + 1;
	    			tempNodesList.insert(temp_rand);
	    			oneIterRandomNodesList.insert(temp_rand);
	    		}

	    		random_K_nodes[mapID] = tempNodesList;
	    	}
	    	all_random_K_nodes[i] = random_K_nodes;
	    	random_K_nodes.erase(random_K_nodes.begin(),random_K_nodes.end());

	    }

	    printf("%d %d %d\n", G->nvtx, G->nedge, nInterval);

	    return G;
}


void computePois_Infection_force(string path){
	ifstream infile(path);
	string line;
	priority_queue<pair<int, int> > pq;
	getline(infile, line);
	istringstream iss(line);
	int V, E;
	iss >> V >> E;
	cout<< "V=" << V << ", E="<< E <<endl;
//	map<string, map<int, double>> pairs_timewindow_infection_force;
	int count = 0;
	int timeID, u,v,distance,venueID;
	while (getline(infile, line)) {
		istringstream iss(line);
		count = count + 1;
		iss >> timeID >> u >> v>>venueID;
		string pair_tag;
		if (timeID>largestTimewindow)
		{
			continue;
		}
		int timeWindow_ID = floor(timeID/timewindowBatch);
		if (u>v)
		{
			pair_tag = to_string(v) + "," + to_string(u);
		}else{
			pair_tag = to_string(u) + "," + to_string(v);
		}
		map<int, int> venueOne =  poisTimeWindowNum[venueID];
		int numPeople = venueOne[timeID];

		double temp_enforce = b_para*exp((-1/rho_2)*(1/numPeople));
		map<string, map<int, double>>::iterator pairs_infection_force = pairs_timewindow_infection_force.find(pair_tag);
		if(pairs_infection_force!=pairs_timewindow_infection_force.end())
		{
			map<int, double> timeDay_force = pairs_timewindow_infection_force[pair_tag];
			map<int, double>::iterator iters = timeDay_force.find(timeWindow_ID);
			if(iters!=timeDay_force.end())
			{
				timeDay_force[timeWindow_ID] = timeDay_force[timeWindow_ID] + temp_enforce;
			}else{
				timeDay_force[timeWindow_ID] = temp_enforce;

			}

			pairs_timewindow_infection_force[pair_tag] = timeDay_force;

		}else{
			map<int, double> timeDay_force;
			timeDay_force[timeWindow_ID] = temp_enforce;
			pairs_timewindow_infection_force[pair_tag] = timeDay_force;
		}
	}
}


void computePoisNumTimewindow(string path){

	ifstream infile(path);
	string line;
	priority_queue<pair<int, int> > pq;
	getline(infile, line);
	istringstream iss(line);
	int V, E;
	iss >> V >> E;
	cout<< "V=" << V << ", E="<< E <<endl;

	int timeID, u,v,distance,venueID;

	while (getline(infile, line)) {
		istringstream iss(line);

		iss >> timeID >> u >> v>>venueID;
//	        int begin_time = ceil(time_start*1.0/timewindowBatch*1.0);
		if (timeID>largestTimewindow)
		{
			continue;
		}
		map<int, map<int,int>>::iterator iterPois = poisTimeWindowNum.find(venueID);
		if(iterPois!=poisTimeWindowNum.end())
		{
			map<int,int> timeIDNum =  iterPois->second;

			map<int,int>::iterator itertimeIDNum = timeIDNum.find(timeID);
			if(itertimeIDNum!=timeIDNum.end())
			{
				timeIDNum[timeID] = timeIDNum[timeID] + 1;
			}else
			{
				timeIDNum[timeID] = 1;
			}
			poisTimeWindowNum[venueID] = timeIDNum;

		}else
		{
			map<int,int> timeIDNum;
			timeIDNum[timeID] = 1;
			poisTimeWindowNum[venueID] = timeIDNum;
		}
		}

}

void computePairsTimeDist(string path){


	ifstream infile(path);
	string line;
	priority_queue<pair<int, int> > pq;
	getline(infile, line);
	istringstream iss(line);
	int V, E;
	iss >> V >> E;
	cout<< "line = " << line << endl;
	cout<< "V=" << V << ", E="<< E <<endl;

	int timeID, u,v,distance,venueID;

	while (getline(infile, line)) {
	        istringstream iss(line);

	        iss >> timeID>> u >> v >> distance >>venueID;
	        if (timeID>largestTimewindow)
			{
				continue;
			}
	        string pair_tag;
	        if (u>v)
	        {
	        	pair_tag = to_string(v) + "," + to_string(u);
	        }else{
	        	pair_tag = to_string(u) + "," + to_string(v);
	        }
	        map<string, map<int,int>>::iterator iterPairs =pairsTimeDist.find(pair_tag);

	        if(iterPairs!=pairsTimeDist.end())
	        {
	        	map<int,int> timeIDDistance =  iterPairs->second;
	        	map<int,int>::iterator itertimeIDDistance = timeIDDistance.find(timeID);// timeID,distance
	        	if(itertimeIDDistance!=timeIDDistance.end())
	        	{
	        		timeIDDistance[timeID] = distance;
	        		pairsTimeDist[pair_tag] = timeIDDistance;
	        	}

	        }else
	        {
	        	map<int,int> timeIDDistance;
	        	timeIDDistance[timeID] = distance;
	        	pairsTimeDist[pair_tag] = timeIDDistance;
	        }
	}

}
