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
#include <algorithm>
using namespace std;
using namespace std::chrono;

DiGraph * transpose(DiGraph & G);
//DiGraph * readDiGraph(string path, double p, int nInterval);
DiGraph * readDiGraph_overall(string path, int nInterval);
DiGraph * reduced_DiGraph_random(string path, int nInterval, float reduceprob);
DiGraph * reduced_DiGraph_pois(string path, int nInterval, float reduceprob);
void computePois_Infection_force(string path);




unordered_set<int> generateRandomSelectNodes(DiGraph & G, int setNum);
void getPoisDistInformation(string path);

void computePoisNumTimewindow(string path);
map<int, unordered_set<int>>  random_K_nodes;
list<unordered_set<int>> max_K_degree;


map<int, map<int, unordered_set<int>>>  all_random_K_nodes;
int select_nodes_num = 0;
char * graphPath = NULL;
int c, aSize, nInterval, nSimulation, nThread;
double p;
int life_span;
int iterNum;
float stepProb;
float stepSize;
float prob0_a;
float prob0_b;
float prob0_c;
int timeWindowNum;
int largestTimewindow;
int timwindowIncre;

float a_para;
float rho_1=0.0;
float b_para=0.0;
float rho_2=0.0;
int largestK;
int batch_K;
//int timewindowBatch;
int distance_threshold;



string runMode;//basic, a=people_connection_prob, b=time_decaying_prob, overall=a+b
map<int, unordered_set<int>> nodesPoisList;
map<int, map<int, int>> poisTimeWindowNum;
float reduceprob;
//
int randomIterNum;
int randSetsize;
int lockdownStart;
int lockdownEnd;
int influencePois;
string distanceFile;
int poisDistNum;
priority_queue<pair<int, int> > poisRank;
map<int, int> poisCount;
priority_queue<pair<float, int> > poisDistRank;
int pois_delete_num = 0;
int random_delete_num = 0;
int poisdist_detele_num = 0;
int decayWindow;
map<string, map<int, double>> pairs_timewindow_infection_force;


int timewindowBatch = 1;
int main(int argc, char** argv) {

	string data(argv[1]);// data path
	runMode = (argv[2]);// overall

	string dataset_name(argv[3]);// dataset name: BBC, TKY, NYC, etc
	iterNum = atoi(argv[4]);// the number of iteration
	int t1 = 0, t2 = atoi(argv[5]);//the largest time window
	timewindowBatch = atoi(argv[6]);// the number of time unit in a T
	largestTimewindow = t2;
	randomIterNum =  atoi(argv[7]);//default = 20


	randSetsize = atoi(argv[8]);
	distance_threshold = atoi(argv[9]);
	a_para = atof(argv[10]);// a parameter
	rho_1 = atof(argv[11]);// rho_1 parameter

	if (dataset_name!="BBC")
	{
		b_para = atof(argv[12]);// b parameter
		rho_2 = atof(argv[13]);// rho_2 parameter
		decayWindow = atof(argv[14]);// decayWindow default to be 14
    	computePoisNumTimewindow(data);//venueID, timeID, num of people
//    	computePairsTimeDist(data);//pairs, timeID, dist
    	computePois_Infection_force(data);
	}

	reduceprob = atof(argv[15]);
	influencePois = atoi(argv[16]);

	t2 = ceil(t2*1.0/timewindowBatch*1.0);

    DiGraph & G = *readDiGraph_overall(data, t2 - t1 + 1);
    DiGraph & GT = *transpose(G);
    DiGraph & G_reduced_pois = *reduced_DiGraph_pois(data, t2 - t1 + 1, reduceprob);
    DiGraph & GT_reduced_pois = *transpose(G_reduced_pois);

    DiGraph & G_reduced_random = *reduced_DiGraph_random(data, t2 - t1 + 1, reduceprob);
    DiGraph & GT_reduced_random = *transpose(G_reduced_random);


    string G_influence_str;

    vector<unordered_set<int>> testNodeset;
    for(int i = 0; i < randomIterNum; i++)
    {
    	printf("oneNodeset:%d\n",i);
        	unordered_set<int> oneNodeset;
        	while(oneNodeset.size()<randSetsize)
        	{
        		int temp_rand =  rand() / (double)RAND_MAX *(G_reduced_pois.nvtx - 1) + 1;
        		unordered_set<int>::iterator vertexIter;
        		vertexIter = G_reduced_pois.V.find(temp_rand);
        		if(vertexIter==G_reduced_pois.V.end())
        		{
        		    continue;
        		}

        		vertexIter = G_reduced_random.V.find(temp_rand);
        		if(vertexIter==G_reduced_random.V.end())
        		{
        			continue;
        		}

        		oneNodeset.insert(temp_rand);
        	}
        	testNodeset.push_back(oneNodeset);
    }



    string temp_summaryResult = "./experiment_result/"+ runMode + "/reduce_" + dataset_name + "_time_window_" + to_string(t2) +
                	"_largestTimewindow_" + to_string(largestTimewindow) + "_timwindowIncre_" + to_string(timwindowIncre)+
                	"_iterNum_" + to_string(iterNum) +
       			"_reduceprob_" + to_string(reduceprob) +
       			"_randSetsize_" + to_string(randSetsize) +
       			"_influencePois_" + to_string(influencePois) +
       			".txt";
	string summaryResult = "./experiment_result/"+ runMode + "/reduce_" + dataset_name +
				"_rprob_" + to_string(reduceprob) +
				"_rsize_" + to_string(randSetsize) +
				"_numPois_" + to_string(influencePois) +
				".txt";
    char str_filenamesummaryResult[250];
    strcpy(str_filenamesummaryResult, summaryResult.c_str());
    FILE *fpsummaryResult;
    fpsummaryResult=fopen(str_filenamesummaryResult,"a+");
    G_influence_str = "";
    printf(("summaryResult= "+summaryResult).c_str());

    fprintf(fpsummaryResult,("temp_summaryResult," + temp_summaryResult + "\n").c_str());


    for(int i = 0; i < randomIterNum; i++)
    {
    	printf("GT influence %d\n", i);
    	unordered_set<int> oneNodeset = testNodeset[i];

        double influence_reduce_A = DTIC(t1, t2, oneNodeset, GT, iterNum);
        G_influence_str = G_influence_str + to_string(influence_reduce_A) + ",";
    }
   fprintf(fpsummaryResult,("GT influence," + G_influence_str + "\n").c_str());
   printf(("GT influence," + G_influence_str + "\n").c_str());


    G_influence_str = "";
    for(int i = 0; i < randomIterNum; i++)
    {
    	unordered_set<int> oneNodeset = testNodeset[i];
    	printf("GT_reduced_pois influence, %d\n", i);
    	double influence_reduce_A = DTIC(t1, t2, oneNodeset, GT_reduced_pois, iterNum);
    	G_influence_str = G_influence_str + to_string(influence_reduce_A) + ",";
    	printf(G_influence_str.c_str());
    }
    printf(summaryResult.c_str());
    fprintf(fpsummaryResult,("GT_reduced_pois influence," + G_influence_str + "\n").c_str());
    printf(("GT_reduced_pois influence," + G_influence_str + "\n").c_str());
    fprintf(fpsummaryResult,("GT_reduced_pois Delete Edge Num," + to_string(pois_delete_num) + "\n").c_str());


    G_influence_str = "";
    for(int i = 0; i < randomIterNum; i++)
    {
    	unordered_set<int> oneNodeset = testNodeset[i];
    	printf("G_reduced_random influence, %d\n", i);

        double influence_reduce_A = DTIC(t1, t2, oneNodeset, G_reduced_random, iterNum);
        G_influence_str = G_influence_str + to_string(influence_reduce_A) + ",";
    }
    fprintf(fpsummaryResult,("G_reduced_random influence," + G_influence_str + "\n").c_str());
    printf("random_delete_num = %d \n", random_delete_num);
    fprintf(fpsummaryResult,("G_reduced_random Delete Edge Num,," + to_string(random_delete_num) + "\n").c_str());
    G_influence_str = "";

    printf("poisdist_detele_num = %d \n", poisdist_detele_num);
    fclose(fpsummaryResult);

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

DiGraph * reduced_DiGraph_overall(DiGraph & G){

	DiGraph * reduceG = new DiGraph();

	//change the density distribution here


	return reduceG;
}


unordered_set<int> generateRandomSelectNodes(DiGraph & G, int setNum){

	unordered_set<int> selectSet;


	return selectSet;
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
	   cout<< "V=" << V << ", E="<< E <<endl;

	   int seperateTimeWindows = ceil(largestTimewindow/timewindowBatch);
	   vector<double> W(seperateTimeWindows);
	   vector<double> W_1(seperateTimeWindows);
	    //   map<int, int> poisCount;
	   int timeID, u, v, venueID;
	    int poisPeopleNum;

	    while (getline(infile, line)) {
	        istringstream iss(line);

	        iss >> timeID >> u >> v>>venueID;
	        if (timeID>largestTimewindow)
			{
				continue;
			}
			int timeWindow_ID = floor(timeID/timewindowBatch);
			if(timeWindow_ID==seperateTimeWindows)
			{
				timeWindow_ID = timeWindow_ID -1;
			}


	        map<int, int>::iterator itr = poisCount.find(venueID);
	      	if(itr!=poisCount.end())
	      	{
	      		poisCount[venueID] = poisCount[venueID] + 1;
	      	}
	      	else
	      	{
	      		poisCount[venueID] = 1;
	      	}
	        map<int,int> tempTimeWindowNum = poisTimeWindowNum[venueID];
	        poisPeopleNum = tempTimeWindowNum[timeWindow_ID];

	        map<int, unordered_set<int>>::iterator iterPois = nodesPoisList.find(u-1);
	        if(iterPois!=nodesPoisList.end())
	        {
	        	unordered_set<int> tempPoisList =  iterPois->second;
	        	tempPoisList.insert(venueID);
	        	nodesPoisList[u-1] = tempPoisList;
	        }else
	        {
	        	unordered_set<int> tempPoisList;
	        	tempPoisList.insert(venueID);
	        	nodesPoisList[u-1] = tempPoisList;
	        }

	        map<int, unordered_set<int>>::iterator iterPoisV = nodesPoisList.find(v-1);
	        if(iterPoisV!=nodesPoisList.end())
	        {
	              	unordered_set<int> tempPoisList =  iterPoisV->second;
	              	tempPoisList.insert(venueID);
	              	nodesPoisList[v-1] = tempPoisList;
	         }else
	         {
	              	unordered_set<int> tempPoisList;
	              	tempPoisList.insert(venueID);
	              	nodesPoisList[v-1] = tempPoisList;
	         }

	        string pair_tag;
			if (u>v){
				pair_tag = to_string(v) + "," + to_string(u);
			}else{
				pair_tag = to_string(u) + "," + to_string(v);
			}
			map<int, double> pairs_infection_force = pairs_timewindow_infection_force[pair_tag];
			float accumulate_enforce = 0;

			int timeBegin = timeWindow_ID-decayWindow;
			if (timeBegin <0)
			{
				timeBegin = 0;
			}

			for(int i = timeBegin; i<=timeWindow_ID;i++)
			{
				map<int, double>::iterator iters = pairs_infection_force.find(i);
				if(iters!=pairs_infection_force.end()){
					accumulate_enforce = accumulate_enforce + pairs_infection_force[i];
				}
			}

			if (G->existEdge(u - 1, v - 1)|| G->existEdge(v - 1, u - 1))
			{
				vector<double> temp_W;
				temp_W= G->returnWeight(u - 1, v - 1);
				double startProb = 1- exp(-accumulate_enforce);
				temp_W[timeWindow_ID] = startProb;
				int timeEnd = timeWindow_ID+decayWindow;
				if(timeEnd>seperateTimeWindows)
				{
					timeEnd= seperateTimeWindows;
				}
				if(timeWindow_ID+1<seperateTimeWindows){
					for(int i = timeWindow_ID+1; i < timeEnd; i++)
					{
						temp_W[i] = startProb;
					}
				}
				G->addEdge(u - 1, v - 1, temp_W);
				G->addEdge(v - 1, u - 1, temp_W);
			}else
			{
				W.clear();
				W = W_1;
				double startProb = 1- exp(-accumulate_enforce);
				W[timeWindow_ID] = startProb;
				int timeEnd = timeWindow_ID+decayWindow;
				if(timeEnd>seperateTimeWindows)
				{
					timeEnd= seperateTimeWindows;
				}
				if(timeWindow_ID+1<seperateTimeWindows){
					for(int i = timeWindow_ID+1; i < timeEnd; i++)
					{
						W[i] = startProb;
					}
				}
				G->addEdge(u - 1, v - 1, W);
				G->addEdge(v - 1, u - 1, W);
			}
	    }

		map<int, int>::iterator readPoisCount;

		for(readPoisCount=poisCount.begin(); readPoisCount!=poisCount.end(); readPoisCount++)
		{
			int venueID = readPoisCount->first;
			int count = readPoisCount->second;
			poisRank.push(make_pair(count, venueID));
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
	int countNum = 0;
	int timeID, u,v,distance,venueID;
	while (getline(infile, line)) {
		istringstream iss(line);

		countNum = countNum + 1;
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
DiGraph * reduced_DiGraph_random(string path, int nInterval, float reduceprob){
	int V, E;


		long seed = chrono::high_resolution_clock::now().time_since_epoch().count();
		DiGraph * G = new DiGraph();

		mt19937 mt_rand(seed);
		uniform_real_distribution<double> real_rand = uniform_real_distribution<double>(0, p);
		uniform_real_distribution<double> real_rand2 = uniform_real_distribution<double>(0, 1);

		ifstream infile(path);
		string line;
		getline(infile, line);
		istringstream iss(line);
		iss >> V >> E;
		cout<< "V=" << V << ", E="<< E <<endl;
		int eValue = E;
		int deleteEdgeNum = (int)(eValue*reduceprob);
		vector<int> poisDistList;

		map<int, float> reducedPois;

		int poisPeopleNum=0;
		int countAddEdge = 0;

		int poisDistrealReducedCount = 0;
		int seperateTimeWindows = ceil(largestTimewindow/timewindowBatch);
	   vector<double> W(seperateTimeWindows);
	   vector<double> W_1(seperateTimeWindows);
	   int timeID, u, v, venueID;
	   int realReducedCount = 0;
	   int removeNumEdge = 0;
		float localProb;
	while (getline(infile, line)) {
		istringstream iss(line);

		iss >> timeID >> u >> v>>venueID;
		if (timeID>largestTimewindow)
		{
			continue;
		}
		int timeWindow_ID = floor(timeID/timewindowBatch);
		if(timeWindow_ID==seperateTimeWindows)
		{
			timeWindow_ID = timeWindow_ID -1;
		}

		double filterProb = real_rand2(mt_rand);
		cout<<"filterProb ="<<filterProb<<", reduceprob="<<reduceprob<<endl;
		if (filterProb<=reduceprob*2)
		{
			removeNumEdge = removeNumEdge + 1;
			continue;
		}


		string pair_tag;
		if (u>v){
			pair_tag = to_string(v) + "," + to_string(u);
		}else{
			pair_tag = to_string(u) + "," + to_string(v);
		}
		map<int, double> pairs_infection_force = pairs_timewindow_infection_force[pair_tag];
		float accumulate_enforce = 0;

		int timeBegin = timeWindow_ID-decayWindow;
		if (timeBegin <0)
		{
			timeBegin = 0;
		}

		for(int i = timeBegin; i<=timeWindow_ID;i++)
		{
			map<int, double>::iterator iters = pairs_infection_force.find(i);
			if(iters!=pairs_infection_force.end()){
				accumulate_enforce = accumulate_enforce + pairs_infection_force[i];
			}
		}

		if (G->existEdge(u - 1, v - 1)|| G->existEdge(v - 1, u - 1))
		{
			vector<double> temp_W;
			temp_W= G->returnWeight(u - 1, v - 1);
			double startProb = 1- exp(-accumulate_enforce);
			temp_W[timeWindow_ID] = startProb;
			int timeEnd = timeWindow_ID+decayWindow;
			if(timeEnd>seperateTimeWindows)
			{
				timeEnd= seperateTimeWindows;
			}
			if(timeWindow_ID+1<seperateTimeWindows){
				for(int i = timeWindow_ID+1; i < timeEnd; i++)
				{
					temp_W[i] = startProb;
				}
			}
			G->addEdge(u - 1, v - 1, temp_W);
			G->addEdge(v - 1, u - 1, temp_W);
		}else
		{
			W.clear();
			W = W_1;
			double startProb = 1- exp(-accumulate_enforce);
			W[timeWindow_ID] = startProb;
			int timeEnd = timeWindow_ID+decayWindow;
			if(timeEnd>seperateTimeWindows)
			{
				timeEnd= seperateTimeWindows;
			}
			if(timeWindow_ID+1<seperateTimeWindows){
				for(int i = timeWindow_ID+1; i < timeEnd; i++)
				{
					W[i] = startProb;
				}
			}
			G->addEdge(u - 1, v - 1, W);
			G->addEdge(v - 1, u - 1, W);
		}

	}
	printf("poisDistrealReducedCount = %d \n", poisDistrealReducedCount);
	random_delete_num = removeNumEdge;

	printf("%d %d %d\n", G->nvtx, G->nedge, nInterval);
	return G;

}


DiGraph * reduced_DiGraph_pois(string path, int nInterval, float reduceprob){
	//poisRank influencePois

	int V, E;


	long seed = chrono::high_resolution_clock::now().time_since_epoch().count();
	DiGraph * G = new DiGraph();

	mt19937 mt_rand(seed);
	uniform_real_distribution<double> real_rand = uniform_real_distribution<double>(0, p);
	uniform_real_distribution<double> real_rand2 = uniform_real_distribution<double>(0, 1);

	ifstream infile(path);
	string line;
	getline(infile, line);
	istringstream iss(line);
	iss >> V >> E;
	cout<< "V=" << V << ", E="<< E <<endl;
	int eValue = E;
	int deleteEdgeNum = (int)(eValue*reduceprob);
	vector<int> poisDistList;
	printf("poisDistRank:%d\n",poisRank.size());
	pair<float, int> top = poisRank.top();
	poisRank.pop();


	map<int, float> reducedPois;

	int poisPeopleNum=0;
	int countAddEdge = 0;
	int poisInfluenceedge = 0;

	poisInfluenceedge = poisInfluenceedge + top.first;
	cout<<top.first <<","<<top.second<<",poisInfluenceedge="<<poisInfluenceedge<<endl;
	reducedPois[top.second] = float(top.first*1.0);
	cout<<"reducedPois.size() = "<<reducedPois.size()<<", influencePois ="<<influencePois<<endl;
	while(reducedPois.size()<influencePois)
	{
		cout<<"reducedPois.size() = "<<reducedPois.size()<<", influencePois ="<<influencePois<<endl;
		top = poisRank.top();
		poisRank.pop();
		poisInfluenceedge = poisInfluenceedge + top.first;

		reducedPois[top.second] = float(top.first*1.0);
	}

	map<int, float>::iterator poisIter;
	map<int, float> reducedPoisProb;

	map<int, map<int, int>> newpoisTimeWindowNum;
	for(poisIter = reducedPois.begin(); poisIter!=reducedPois.end(); poisIter++)
	{
		int id = poisIter->first;
		float num = poisIter ->second;
		float newProb = deleteEdgeNum*(1.0*num/(1.0*poisInfluenceedge))/(num/2);
		cout<< "newProb="<<newProb<<endl;
		reducedPoisProb.insert(make_pair(id, newProb));
	}


	int poisDistrealReducedCount = 0;
	int seperateTimeWindows = ceil(largestTimewindow/timewindowBatch);
   vector<double> W(seperateTimeWindows);
   vector<double> W_1(seperateTimeWindows);
   int timeID, u, v, venueID;
   int realReducedCount = 0;
	float localProb;
	while (getline(infile, line)) {
		istringstream iss(line);

		iss >> timeID >> u >> v>>venueID;
		if (timeID>largestTimewindow)
		{
			continue;
		}
		int timeWindow_ID = floor(timeID/timewindowBatch);
		if(timeWindow_ID==seperateTimeWindows)
		{
			timeWindow_ID = timeWindow_ID -1;
		}

		double filterProb = real_rand2(mt_rand);
		map<int, float>::iterator probIter = reducedPoisProb.find(venueID);
//		cout<<"filterProb = "<<filterProb<<endl;
		if(probIter!=reducedPoisProb.end())
		{
			float prob = probIter->second;
//			cout<<"prob = "<<prob<<endl;
			  if (filterProb<=prob)
			  {
				  realReducedCount = realReducedCount + 1;
				  continue;
			  }
		}
		string pair_tag;
		if (u>v){
			pair_tag = to_string(v) + "," + to_string(u);
		}else{
			pair_tag = to_string(u) + "," + to_string(v);
		}
		map<int, double> pairs_infection_force = pairs_timewindow_infection_force[pair_tag];
		float accumulate_enforce = 0;

		int timeBegin = timeWindow_ID-decayWindow;
		if (timeBegin <0)
		{
			timeBegin = 0;
		}

		for(int i = timeBegin; i<=timeWindow_ID;i++)
		{
			map<int, double>::iterator iters = pairs_infection_force.find(i);
			if(iters!=pairs_infection_force.end()){
				accumulate_enforce = accumulate_enforce + pairs_infection_force[i];
			}
		}

		if (G->existEdge(u - 1, v - 1)|| G->existEdge(v - 1, u - 1))
		{
			vector<double> temp_W;
			temp_W= G->returnWeight(u - 1, v - 1);
			double startProb = 1- exp(-accumulate_enforce);
			temp_W[timeWindow_ID] = startProb;
			int timeEnd = timeWindow_ID+decayWindow;
			if(timeEnd>seperateTimeWindows)
			{
				timeEnd= seperateTimeWindows;
			}
			if(timeWindow_ID+1<seperateTimeWindows){
				for(int i = timeWindow_ID+1; i < timeEnd; i++)
				{
					temp_W[i] = startProb;
				}
			}
			G->addEdge(u - 1, v - 1, temp_W);
			G->addEdge(v - 1, u - 1, temp_W);
		}else
		{
			W.clear();
			W = W_1;
			double startProb = 1- exp(-accumulate_enforce);
			W[timeWindow_ID] = startProb;
			int timeEnd = timeWindow_ID+decayWindow;
			if(timeEnd>seperateTimeWindows)
			{
				timeEnd= seperateTimeWindows;
			}
			if(timeWindow_ID+1<seperateTimeWindows){
				for(int i = timeWindow_ID+1; i < timeEnd; i++)
				{
					W[i] = startProb;
				}
			}
			G->addEdge(u - 1, v - 1, W);
			G->addEdge(v - 1, u - 1, W);
		}

	}
	printf("poisDistrealReducedCount = %d \n", poisDistrealReducedCount);
	poisdist_detele_num = realReducedCount;

	printf("%d %d %d\n", G->nvtx, G->nedge, nInterval);
	return G;


}

void getPoisDistInformation(string path){

	ifstream infile(path);
	string line;
	getline(infile, line);
	istringstream iss(line);


	int venueID, numVist;
	float  allDist, AvgDist;

		while (getline(infile, line)) {
		        istringstream iss(line);

		        iss >> venueID >> numVist>>allDist>>AvgDist;

		        poisDistRank.push(make_pair(AvgDist, venueID));
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

	int timeID, u, v, venueID;
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



