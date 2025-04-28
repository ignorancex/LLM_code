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

DiGraph * readDiGraph_overall(string path, int nInterval);


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
string runMode;//basic, a=people_connection_prob, b=time_decaying_prob, overall=a+b
map<int, unordered_set<int>> nodesPoisList;
map<int, map<int, int>> poisTimeWindowNum;
//
int randomIterNum;
int timeDecay;
int timewindowBatch;
int main(int argc, char** argv) {

    string data(argv[1]);
    runMode = (argv[2]);
    int t1 = 0, t2 = atoi(argv[3]);
    timewindowBatch = atoi(argv[4]);
    string dataset_name(argv[5]);

    iterNum = atoi(argv[6]);
    stepProb = atof(argv[7]);
    stepSize = atof(argv[8]);
    prob0_a = atof(argv[9]);
    prob0_b = atof(argv[10]);
    prob0_c = atof(argv[11]);
    randomIterNum = atoi(argv[12]);
    timeWindowNum = t2;
   timeDecay = atoi(argv[13]);


    computePoisNumTimewindow(data);
    t2 = ceil(t2*1.0/timewindowBatch*1.0);
    printf("t2=%d", t2);
    DiGraph & G = *readDiGraph_overall(data, t2 - t1 + 1);

    DiGraph & GT = *transpose(G);

    DTARRS * dtarrsG = new DTARRS(G);
    clock_t startTime,endTime;
    startTime = clock();
    dtarrsG->generateHypergraph(t1, t2, iterNum);
    endTime = clock();
    double hyperNets_running_time = (double)(endTime - startTime) / CLOCKS_PER_SEC;

    printf("HG complete\n");


    string timeEvaluate = "./experiment_result/"+ runMode + "/" + dataset_name + "_time_window_" + to_string(t2) +
             	"_iterNum_" + to_string(iterNum) + ".txt";
    char str_filenamerandomResult[250];
    strcpy(str_filenamerandomResult, timeEvaluate.c_str());
    FILE *fprandomResult;
    fprandomResult  =fopen(str_filenamerandomResult,"a+");


    fprintf(fprandomResult,"Running Time: \n");
    fprintf(fprandomResult,(to_string(hyperNets_running_time) +"\n").c_str());

    fclose(fprandomResult);

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
	        printf("V=%d,E=%d",V,E);

	       int reduced_connection = 0;

	    int seperateTimeWindows = ceil(timeWindowNum*1.0/timewindowBatch*1.0);
	    printf("seperateTimeWindows=%d\n",seperateTimeWindows);

	    map<int, map<int, unordered_set<int>>> susceptible_list;//nodeid, timewindow, count[V+1][seperateTimeWindows];

	    int u, v, time_start, time_end, venueID;
	    int poisPeopleNum;
	    vector<double> W;
	    W.reserve(seperateTimeWindows);
	    int count_num = 0;
	    while (getline(infile, line)) {
	        istringstream iss(line);
	        count_num = count_num + 1;
	        iss >> u >> v>>time_start>>time_end>>venueID;
	        int time_endNum = ceil(time_start*1.0/timewindowBatch*1.0) + timeDecay;
	        map<int,int> tempTimeWindowNum = poisTimeWindowNum[venueID];
	        poisPeopleNum = tempTimeWindowNum[ceil(time_start*1.0/timewindowBatch*1.0)];

	        map<int, unordered_set<int>>::iterator iterPois = nodesPoisList.find(u);
	        if(iterPois!=nodesPoisList.end())
	        {
	        	unordered_set<int> tempPoisList =  iterPois->second;
	        	tempPoisList.insert(venueID);
	        	nodesPoisList[u] = tempPoisList;
	        }else
	        {
	        	unordered_set<int> tempPoisList;
	        	tempPoisList.insert(venueID);
	        	nodesPoisList[u] = tempPoisList;
	        }

	        map<int, unordered_set<int>>::iterator iterPoisV = nodesPoisList.find(v);
	        if(iterPoisV!=nodesPoisList.end())
	        {
	              	unordered_set<int> tempPoisList =  iterPoisV->second;
	              	tempPoisList.insert(venueID);
	              	nodesPoisList[v] = tempPoisList;
	         }else
	         {
	              	unordered_set<int> tempPoisList;
	              	tempPoisList.insert(venueID);
	              	nodesPoisList[v] = tempPoisList;
	         }
	        int beginTime = ceil(time_start*1.0/timewindowBatch*1.0);
	        int endTime = seperateTimeWindows;
	        if (G->existEdge(u - 1, v - 1))
	        {
	        	W.clear();
	        	vector<double> temp_W= G->returnWeight(u - 1, v - 1);
	        	double startProb = 1/(prob0_a + prob0_b*(exp(-prob0_c*poisPeopleNum)));

	        	for (int j = beginTime-1; j <= endTime-1; j++)
	        	{
	        		if(j>=time_endNum)
	        		{
	        			temp_W[j] = 0;
	        			continue;
	        		}

	        		if (j>=seperateTimeWindows)
	        		{
	        			break;
	        		}
	        		double timeGap = (j*1.0 - (beginTime*1.0 - 1.0));
	        		double scaler = 1.0;
	        		if(runMode=="basic"){
						if (timeGap==0)
						{
							scaler = 1.0;
						}else
						{
							scaler = 0;
						}
	        		}else
	        		{
	        			scaler = 1.0;
	        		}

	        		if(runMode=="onlyf"){
	        			scaler = 0;
	        		}

	        		double currProb = scaler*startProb*pow((stepProb - stepSize*timeGap), timeGap);
	        		temp_W[j] = currProb;
	        	}
	        	G->addEdge(u - 1, v - 1, temp_W);
	        }else
	        {
	        	W.clear();
	        	double startProb = 1/(prob0_a + prob0_b*(exp(-prob0_c*poisPeopleNum)));
	        	for (int j = 0; j < nInterval; j++) {
	        		if(j>=beginTime-1 & j<= time_endNum-1){
	         			double timeGap = (j*1.0 - (beginTime*1.0 - 1.0));
	        			double scaler = 1.0;
	        			if(runMode=="basic"){
	        				if (timeGap==0)
	        				{
	        					scaler = 1.0;
	        				}else
	        				{
	        					scaler = 0;
	        				}
	        			}else
	        			{
	        				scaler = 1.0;
	        			}

	        			if(runMode=="onlyf"){
	        				scaler = 0;
	        			}
	        			double currProb = scaler*startProb*pow((stepProb - stepSize*timeGap), timeGap);
	        			W.push_back(currProb);
	        		}
	        	    else{
	        	    	W.push_back(0);
	        	    }
	        	  }
	        	G->addEdge(u - 1, v - 1, W);
	        }

	        map<int, map<int, unordered_set<int>>>::iterator findNode = susceptible_list.find(v-1);
	       if(findNode!=susceptible_list.end())
	       {
	    	   map<int, unordered_set<int>> timeNum = susceptible_list[v-1];
	    	   map<int, unordered_set<int>>::iterator timeNumIter = timeNum.find(beginTime-1);
	    	   if(timeNumIter!=timeNum.end())
	    	   {
	    		   timeNum[beginTime-1].insert(v-1);
	    	   }else
	    	   {
	    		   timeNum[beginTime-1].insert(v-1);
	    	   }
	    	   susceptible_list[v-1] = timeNum;
	       }else
	       {
	    	   map<int, unordered_set<int>> timeNum;
	    	   timeNum[beginTime-1].insert(v-1);
	    	   susceptible_list[v-1] = timeNum;
	       }


	    }

	    for (int i = 1; i <= V; i++) {
	 	    	map<int, unordered_set<int>> timeNum = susceptible_list[i-1];
	 	    	int deg = 0;
	 	    	for(int j = 1; j<=seperateTimeWindows; j++)
	 	    	{
	 	    		deg = deg + timeNum[j-1].size();


	 	    	}
	 	    	 pq.push(make_pair(deg, i));
	 	     }


	    int batchNum = largestTimewindow/timwindowIncre;

	     for(int i = 0; i < batchNum; i++)
	     {

	        	int mapID = (i+1)*timwindowIncre;
	        	unordered_set<int> tempNodesList;
	        	while(tempNodesList.size()<=timwindowIncre)
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
	    		int mapID = (j+1)*timwindowIncre;
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




void computePoisNumTimewindow(string path){


	ifstream infile(path);
	string line;
	priority_queue<pair<int, int> > pq;
	getline(infile, line);
	istringstream iss(line);
	int V, E;
	iss >> V >> E;
	cout<< "V=" << V << ", E="<< E <<endl;

	int u, v, time_start, time_end, venueID;
	int seperateTimeWindows = ceil(timeWindowNum/timewindowBatch);
	while (getline(infile, line)) {
	        istringstream iss(line);

	        iss >> u >> v>>time_start>>time_end>>venueID;
	        int begin_time = ceil(time_start*1.0/timewindowBatch*1.0);
	        map<int, map<int,int>>::iterator iterPois = poisTimeWindowNum.find(venueID);
	        if(iterPois!=poisTimeWindowNum.end())
	        {
	        	map<int,int> timeIDNum =  iterPois->second;

	        	map<int,int>::iterator itertimeIDNum = timeIDNum.find(begin_time);
	        	if(itertimeIDNum!=timeIDNum.end())
	        	{
	        		timeIDNum[begin_time] = timeIDNum[begin_time] + 1;
	        	}else
	        	{
	        		timeIDNum[begin_time] = 1;
	        	}
	        	poisTimeWindowNum[venueID] = timeIDNum;

	        }else
	        {
	        	map<int,int> timeIDNum;
	        	timeIDNum[begin_time] = 1;
	        	poisTimeWindowNum[venueID] = timeIDNum;
	        }
	}



}
