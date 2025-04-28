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
#include <cmath>

#include "omp.h"

using namespace std;
using namespace std::chrono;

DiGraph * transpose(DiGraph & G);
DiGraph * reduceMax(DiGraph & G);
DiGraph * reduceAvg(DiGraph & G);
DiGraph * reduceMin(DiGraph & G);
//DiGraph * readDiGraph(string path, double p, int nInterval);
void computePois_Infection_force(string path);
DiGraph * readDiGraph_overall(string path, int nInterval);
void readImmunization_set(string path, int largestK, int batchK);
void readDynamicIM_set(string path, int largestK, int batchK);
vector<string> split(const string& str, const string& delim);
void computePairsTimeDist(string path);
void computePoisNumTimewindow(string path);
//map<int, unordered_set<int>>  max_K_degree;
map<int, unordered_set<int>>  random_K_nodes;
list<unordered_set<int>> max_K_degree;
//list<unordered_set<int>> random_K_nodes;


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
float a_para;
float rho_1;
float b_para;
float rho_2;
int largestK;
int batch_K;
//int timewindowBatch;
int distance_threshold;


int timeWindowNum;
//int largestTimewindow;
int timwindowIncre;
int NODE_NUM;
string runMode;//basic, a=people_connection_prob, b=time_decaying_prob, overall=a+b
map<int, unordered_set<int>> nodesPoisList;
map<int, map<int, int>> poisTimeWindowNum;
map<string, map<int, int>> pairsTimeDist;
map<string, map<int, double>> pairs_timewindow_infection_force;
//
int randomIterNum;
int timeDecay;
int timewindowBatch;
int largestTimewindow;
int valNum;
map<int, unordered_set<int>> T_IMMU;
map<int, map<int,unordered_set<int>>> Dynamic_IM_set;
int decayWindow;
string writeName;


int main(int argc, char** argv) {

	 	string data(argv[1]);// data path
		runMode = (argv[2]);// overall

		string dataset_name(argv[3]);// dataset name: BBC, TKY, NYC, etc
		iterNum = atoi(argv[4]);// the number of iteration
		int t1 = 0, t2 = atoi(argv[5]);//the largest time window
		timewindowBatch = atoi(argv[6]);// the number of time unit in a T
		largestTimewindow = t2;
		randomIterNum =  atoi(argv[7]);//default = 20
		writeName = dataset_name;
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

		string dynamic_solution_file(argv[16]);
		string TImmuni_solution_file(argv[17]);

//
    t2 = ceil(t2*1.0/timewindowBatch*1.0);
    printf("t2=%d", t2);
    DiGraph & G = *readDiGraph_overall(data, t2 - t1 + 1);
    printf("ceil(t2/timewindowBatch)=%d", t2);
//    T_IMMU Dynamic_IM_set
//    T_IMMU =
    DTARRS * dtarrsG = new DTARRS(G);
    DiGraph & GT = *transpose(G);
	   printf("HG before\n");
	   printf("HG before AAAA\n");
	   printf("t1=%d, t2=%d, iterNum=%d \n", t1, t2, iterNum);
   dtarrsG->generateHypergraph(t1, t2, iterNum);
    printf("read Immunization set! \n");
    list<unordered_set<int>> AtopK = dtarrsG->computeTopK(batch_K, largestK);
    list<unordered_set<int>> A = dtarrsG->computeMaxSpread(batch_K, largestK);
   	readImmunization_set(TImmuni_solution_file, largestK, batch_K);
   //    Dynamic_IM_set =
    printf("read Dynamic set! \n");
	readDynamicIM_set(dynamic_solution_file, largestK, batch_K);






    omp_set_num_threads(100);

    int batchKNum = largestK/batch_K;
    string summaryResult = "./experiment_result/"+ runMode + "/summary_" + dataset_name + "_time_window_" + to_string(t2) +
         "a_para" + to_string(a_para)+ "_b_para_" + to_string(b_para)+
         	"_iterNum_" + to_string(iterNum) + ".txt";
    char str_filenamesummaryResult[250];
    strcpy(str_filenamesummaryResult, summaryResult.c_str());
    FILE *fpsummaryResult;
    fpsummaryResult=fopen(str_filenamesummaryResult,"a+");


    string randomResult = "./experiment_result/"+ runMode + "/random_" + dataset_name + "_time_window_" + to_string(t2) +
    					"a_para" + to_string(a_para)+
    		         	"_iterNum_" + to_string(iterNum) + ".txt";
    char str_filenamerandomResult[250];
    strcpy(str_filenamerandomResult, randomResult.c_str());
    FILE *fprandomResult;
    fprandomResult  =fopen(str_filenamerandomResult,"a+");


    string ouputResult = "./experiment_result/"+ runMode + "/output_" + dataset_name + "_time_window_" + to_string(t2) +
    		"a_para" + to_string(a_para)+
    		         	"_iterNum_" + to_string(iterNum) + ".txt";
      char str_filenameouputResult[250];
      strcpy(str_filenameouputResult, ouputResult.c_str());
      FILE *fpouputResult;
      fpouputResult=fopen(str_filenameouputResult,"a+");


      string poisResult = "./experiment_result/"+ runMode + "/calPois_" + dataset_name + "_time_window_" + to_string(t2) +
    		  "a_para" + to_string(a_para)+
    		           	"_iterNum_" + to_string(iterNum) + ".txt";

      char str_filenamepoisResult[250];
      strcpy(str_filenamepoisResult, poisResult.c_str());
      FILE *fppoisResult;
      fppoisResult=fopen(str_filenamepoisResult,"a+");

    string RSM_content_influence = "RSM Influence,";
    string ESM_content_influence = "ESM Influence,";
    string TIMMU_content_influence = "TIMMU Influence,";
    string dynamic_content_influence = "dynamic Influence,";
    string head_content_influence = "k,";


    string RSM_content_expected_vertices = "RSM Expected Vertices,";
	string ESM_content_expected_vertices = "ESM Expected Vertices,";
    string TIMMU_content_expected_vertices = "TIMMU Expected Vertices,";
    string dynamic_content_expected_vertices = "dynamic Expected Vertices,";
    //string head_content_expected_vertices = "";

    string RSM_content_binary_success = "RSM Binary Success,";
	string ESM_content_binary_success = "ESM Binary Success,";
    string TIMMU_content_binary_success = "TIMMU Binary Success,";
    string dynamic_content_binary_success = "dynamic Binary Success,";
   // string head_content_binary_success = "";


    string TIMMU_content_poisNum = "TIMMU POIS Num,";
    printf("BBBBBBBBB\n");
    printf("batchKNum=%d\n",batchKNum);

    unordered_set<int>  greedy_batchset;

    unordered_set<int>  RSM;
    unordered_set<int>  ESM;
    unordered_set<int> TIMMU_batch;
    unordered_set<int> dynamic_batch;
    map<int, unordered_set<int>>  IterrandomMethod;

//    int ntrials = valNum;
//    int ntrials = 1000;
    int ntrials = 1000;
//    iterNum = valNum;
    for(int i = 0; i < batchKNum; i++)
    {
    	printf("batchKNum=%d, i=%d \n", batchKNum, i);
    	int k_value = (i+1)*batch_K;
    	head_content_influence = head_content_influence + to_string(k_value) + ",";
    	int countNum = 0;
    	int mapID = (i+1);
//    	map<int, unordered_set<int>> T_IMMU;
//    	map<int, map<int,unordered_set<int>>> Dynamic_IM_set;
    	unordered_set<int> temp = T_IMMU[mapID];
    	printf("T_IMMU mapID = %d", mapID);
    	TIMMU_batch.insert(temp.begin(), temp.end());
    	printf("TIMMU element:");
    	unordered_set<int>::iterator it;
		for(it = TIMMU_batch.begin();it!=TIMMU_batch.end();it++){
			printf("ele = %d, ", *it);
		}
		printf("\n");
		double influence_A = DTIC(t1, t2, TIMMU_batch, GT, ntrials);
		TIMMU_content_influence = TIMMU_content_influence + to_string(influence_A) + ",";
		printf("TIMMU influence %f\n", influence_A);
		printf("TIMMU Num %d\n", TIMMU_batch.size());
		//number of pois
		unordered_set<int> poisSet_A;
		for (auto e : TIMMU_batch) {
			fprintf(fppoisResult, "TIMMU_batch --- People Node---%d:", e);
			unordered_set<int> readPois = nodesPoisList[e];
			for (unordered_set<int>::iterator w = readPois.begin(); w != readPois.end(); w++) {
				int poisVal = *w;
				fprintf(fppoisResult, "%d, ", poisVal);
				poisSet_A.insert(poisVal);
			}
			fprintf(fppoisResult, "\n");
		}

		TIMMU_content_poisNum = TIMMU_content_poisNum + to_string(poisSet_A.size()) +",";
		fprintf(fppoisResult, "TIMMU pois unique size %d:\n", poisSet_A.size());
		for (unordered_set<int>::iterator w = poisSet_A.begin(); w != poisSet_A.end(); w++) {
			int poisVal = *w;
			fprintf(fppoisResult, "%d, ", poisVal);
		}
		fprintf(fppoisResult, "\n");



	list<unordered_set<int>>::iterator listIter;
	for(listIter = A.begin(); listIter != A.end(); ++listIter)
	{
		countNum = countNum + 1;
		unordered_set<int> temp = *listIter;
		RSM.insert(temp.begin(), temp.end());
//    	    unordered_set<int> A_val = *it_A;
		unordered_set<int>::iterator it;
		printf("RSM element:");
		for(it = RSM.begin();it!=RSM.end();it++){
			printf("ele = %d, ", *it);
		}
		printf("\n");
		if (countNum==i+1)
		{
			double influence_A = DTIC(t1, t2, RSM, GT, ntrials);
			RSM_content_influence = RSM_content_influence + to_string(influence_A) + ",";
			printf("RSM influence %f\n", influence_A);
			printf("RSM Num %d\n", RSM.size());
			//number of pois
			unordered_set<int> poisSet_A;
			for (auto e : RSM) {
				fprintf(fppoisResult, "RSM --- People Node---%d:", e);
				unordered_set<int> readPois = nodesPoisList[e];
				for (unordered_set<int>::iterator w = readPois.begin(); w != readPois.end(); w++) {
					int poisVal = *w;
					fprintf(fppoisResult, "%d, ", poisVal);
					poisSet_A.insert(poisVal);
				}
				fprintf(fppoisResult, "\n");
			}


					break;
	   }

	}

	int countNumESM = 0;
	list<unordered_set<int>>::iterator listIterESM;
	for(listIterESM = AtopK.begin(); listIterESM != AtopK.end(); ++listIterESM)
	{
		countNumESM = countNumESM + 1;
		unordered_set<int> temp = *listIterESM;
		ESM.insert(temp.begin(), temp.end());
		unordered_set<int>::iterator it;
		printf("ESM element:");
		for(it = ESM.begin();it!=ESM.end();it++){
			printf("ele = %d, ", *it);
		}
		if (countNumESM==i+1)
		{
			double influence_A = DTIC(t1, t2, ESM, GT, ntrials);
			ESM_content_influence = ESM_content_influence + to_string(influence_A) + ",";
			printf("ESM influence %f\n", influence_A);
			printf("ESM Num %d\n", ESM.size());
			//number of pois
			unordered_set<int> poisSet_A;
			for (auto e : ESM) {
				fprintf(fppoisResult, "ESM --- People Node---%d:", e);
				unordered_set<int> readPois = nodesPoisList[e];
				for (unordered_set<int>::iterator w = readPois.begin(); w != readPois.end(); w++) {
					int poisVal = *w;
					fprintf(fppoisResult, "%d, ", poisVal);
					poisSet_A.insert(poisVal);
				}
				fprintf(fppoisResult, "\n");
			}


			break;
		}
	}







//dynamic_batch
//		t2,Dynamic_IM_set
		double dynamic_average_influence = 0.0;
		map<int, unordered_set<int>> dynamic_batchsize_list;
		for(int timeBatch = 0; timeBatch<t2;timeBatch++)
		{
			map<int, unordered_set<int>> tempIM_set = Dynamic_IM_set[timeBatch];
			for(int batchIter = 1; batchIter <=mapID; batchIter++)
			{
				unordered_set<int> temp = tempIM_set[batchIter];
				dynamic_batch.insert(temp.begin(), temp.end());
			}

			double influence_A = DTIC(timeBatch, timeBatch, dynamic_batch, GT, ntrials);
			dynamic_average_influence = dynamic_average_influence + influence_A;
			dynamic_batchsize_list[timeBatch] = dynamic_batch;

			dynamic_batch.clear();
		}
		dynamic_average_influence = dynamic_average_influence/t2;

		dynamic_content_influence = dynamic_content_influence + to_string(dynamic_average_influence) + ",";
		printf("dynamic influence %f\n", dynamic_average_influence);
		printf("dynamic Num %d\n", dynamic_batch.size());

	    long seed = chrono::high_resolution_clock::now().time_since_epoch().count();
	    mt19937 mt_rand(seed);
	    uniform_int_distribution<int> int_rand1 = uniform_int_distribution<int>(1, 1);
	    uniform_int_distribution<int> int_rand2 = uniform_int_distribution<int>(1, G.nvtx);
	    int nintersect1 = 0, nintersect2 = 0,
	    		binary1 = 0, binary2 = 0;
	    double detection_rate1 = 0.0, detection_rate2 = 0.0;
	    select_nodes_num = k_value;
	    for (int iA = 0; iA < ntrials; iA++) {
	    	unordered_set<int> S;
	    	int size = int_rand1(mt_rand);
	    	for (int jA = 0; jA < size; jA++) {
	    		S.insert(int_rand2(mt_rand));
	    	}
	    	int nintersect;
//	    	new_DTIC3(t1, t2, S, RSM, ESM,MaxDegree,IterrandomMethod, G);
	    	vector<int> v = dynamic_TIMMU_DTIC(t1, t2, S, TIMMU_batch,dynamic_batchsize_list,G);

	    	nintersect = v[0];
	    	nintersect1 += nintersect;
	    	if (nintersect > 0){
	    		binary1++;
	    		detection_rate1 = detection_rate1 + (double)nintersect/select_nodes_num;
	    	}

	    	for(int dynamicIter = 1; dynamicIter<=t2; dynamicIter++)
	    	{
	    		nintersect = v[dynamicIter];
				nintersect2 += nintersect;
				if (nintersect > 0){
					binary2++;
					detection_rate2 = detection_rate2 + (double)nintersect/select_nodes_num;
				}
	    	}




	    	S.clear();
	    }

	    TIMMU_content_expected_vertices = TIMMU_content_expected_vertices + to_string((double) nintersect1 / ntrials) + ",";
	    dynamic_content_expected_vertices = dynamic_content_expected_vertices + to_string((double) nintersect2 / ntrials/t2) + ",";

	    TIMMU_content_binary_success = TIMMU_content_binary_success + to_string((double) binary1 / ntrials) + ",";
	    dynamic_content_binary_success = dynamic_content_binary_success + to_string((double) binary2 / ntrials/t2) + ",";
    }

    fprintf(fpsummaryResult,"Reverse Influence \n");
    fprintf(fpsummaryResult,(head_content_influence+"\n").c_str());
    fprintf(fpsummaryResult,(TIMMU_content_influence+"\n").c_str());
    fprintf(fpsummaryResult,(dynamic_content_influence+"\n").c_str());

    fprintf(fpsummaryResult,"Expected Number of Vertices\n");
    fprintf(fpsummaryResult,(head_content_influence+"\n").c_str());
    fprintf(fpsummaryResult,(TIMMU_content_expected_vertices+"\n").c_str());
    fprintf(fpsummaryResult,(dynamic_content_expected_vertices+"\n").c_str());

    fprintf(fpsummaryResult,"Binary Success Rate\n");
    fprintf(fpsummaryResult,(head_content_influence+"\n").c_str());
    fprintf(fpsummaryResult,(TIMMU_content_binary_success+"\n").c_str());
    fprintf(fpsummaryResult,(dynamic_content_binary_success+"\n").c_str());

    fprintf(fpsummaryResult,"Pois Num\n");
    fprintf(fpsummaryResult,(head_content_influence+"\n").c_str());

    fclose(fpouputResult);
    fclose(fppoisResult);
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
	    NODE_NUM = V;
		printf("readDiGraph_overall function \n");
		cout<<"NODE_NUM="<<NODE_NUM<<endl;
		cout<<"largestTimewindow = "<<largestTimewindow<<endl;
		cout<<"timewindowBatch = "<<timewindowBatch<<endl;

	    int seperateTimeWindows = ceil(largestTimewindow/timewindowBatch);

	    map<int, map<int, unordered_set<int>>> susceptible_list;//nodeid, timewindow, count[V+1][seperateTimeWindows];
	    cout<<"seperateTimeWindows =" << seperateTimeWindows<<endl;
	//prob0_a prob0_b prob0_c
	    int timeID, u, v, venueID;
	    //	    int u, v, time_start, time_end, venueID;
	    int poisPeopleNum;
	    vector<double> W(seperateTimeWindows);
	    vector<double> W_1(seperateTimeWindows);
	//    nodesPoisList
//	    W.reserve(seperateTimeWindows);
	    int count_num = 0;
	    while (getline(infile, line)) {
	        istringstream iss(line);
	        count_num = count_num + 1;
	        iss >> timeID >> u >> v>>venueID;
//	        printf("count_num = %d\n", count_num);
	        cout<< "u = " << u << ", v = " << v << ", count_num = " << count_num << endl;
//	        int weight_timeID = ceil(1.0*timeID/timewindowBatch);
	        if (timeID>largestTimewindow)
	        {
	        	continue;
	        }
//	        map<string, map<int, double>> pairs_timewindow_infection_force;
	        int timeWindow_ID = floor(timeID/timewindowBatch);
			if(timeWindow_ID==seperateTimeWindows)
			{
				timeWindow_ID = timeWindow_ID -1;
			}
			cout<<"timeWindow_ID = "<<timeWindow_ID<<endl;
			string pair_tag;
			if (u>v){
				pair_tag = to_string(v) + "," + to_string(u);
			}else{
				pair_tag = to_string(u) + "," + to_string(v);
			}
			cout<<"pari_tag = "<<pair_tag<<endl;
			map<int, double> pairs_infection_force = pairs_timewindow_infection_force[pair_tag];
			//	        map<string, map<int,int>>::iterator iterPairs =pairsTimeDist.find(pair_tag);
			float accumulate_enforce = 0;

			int timeBegin = timeWindow_ID-decayWindow;
			if (timeBegin <0)
			{
				timeBegin = 0;
			}
			cout<<"timeBegin="<<timeBegin<<endl;
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
	        	cout<<"A timeEnd = "<<timeEnd<<", timeWindow_ID+1 ="<< timeWindow_ID+1<<",decayWindow="<<decayWindow<< endl;
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
//				cout<<"B timeEnd = "<<timeEnd<<", timeWindow_ID+1 ="<< timeWindow_ID+1<<",decayWindow="<<decayWindow<< endl;
				if(timeWindow_ID+1<seperateTimeWindows){
					for(int i = timeWindow_ID+1; i <= timeEnd; i++)
					{
						W[i] = startProb;
					}
				}
	        	G->addEdge(u - 1, v - 1, W);
	        	G->addEdge(v - 1, u - 1, W);
	        }
	    }

//	    string write_prob_file = writeName+"_new_proba_assignment_file.csv";
//		char str_filenamepoisResult[250];
//		strcpy(str_filenamepoisResult, write_prob_file.c_str());
//		FILE *fppoisResult;
//		fppoisResult=fopen(str_filenamepoisResult,"a+");
//
//		map<string, map<int, double>>::iterator pairsIter;
//		for(pairsIter=pairs_timewindow_infection_force.begin();pairsIter!=pairs_timewindow_infection_force.end();pairsIter++)
//		{
//			string pairName = pairsIter->first;
//			vector<string> pairIDs = split(pairName, ",");
//			int u = stoi(pairIDs[0]);
//			int v = stoi(pairIDs[1]);
//			vector<double> tempWeight = G->returnWeight(u - 1, v - 1);
//			for(int i = 0; i < tempWeight.size(); i++)
//			{
//				double weight = tempWeight[i];
//				if(weight>0){
//					fprintf(fppoisResult, "%d %d %d %f\n", i,u,v,weight);
//				}
//			}
//			tempWeight = G->returnWeight(v - 1, u - 1);
//			for(int i = 0; i < tempWeight.size(); i++)
//			{
//				double weight = tempWeight[i];
//				if(weight>0){
//				fprintf(fppoisResult, "%d %d %d %f\n", i,v,u,weight);
//				}
//			}
//		}
//
//		fclose(fppoisResult);

	    return G;
}



void computePoisNumTimewindow(string path){
//foursquare datasets

	ifstream infile(path);
	string line;
	priority_queue<pair<int, int> > pq;
	getline(infile, line);
	istringstream iss(line);
	int V, E;
	iss >> V >> E;
	cout<< "V=" << V << ", E="<< E <<endl;

	int timeID, u,v,distance,venueID;
//	int u, v, time_start, time_end, venueID;
//	int seperateTimeWindows = ceil(timeWindowNum/timewindowBatch);
//	poisTimeWindowNum:venue id, timeID, number of people
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



void readImmunization_set(string path, int largestK, int batchK)
{
//	map<int, unordered_set<int>> T_IMMU;
	ifstream infile1(path);
	string line;
	printf("Go to readImmunization_set!\n");
	cout<<"path = " << path << endl;
//	getline(infile1, line);
	printf("here check path! \n");
	printf("line=\n");
	cout << "line = " << line << endl;
	while (getline(infile1, line)) {
//		cout<<"Apath = " << path << endl;
		vector<string> seperateList = split(line,",");
		printf("read Immunization file:");
		cout << line << endl;
		unordered_set<int> tempbatchK;
		for(int i = 1; i<=largestK; i++)
		{
			int node = stoi(seperateList[i-1]);
			printf("node id = %d", node);
			if(i%batchK==0)
			{
				int batchNum = i/batchK;
				tempbatchK.insert(node);
				T_IMMU[batchNum] = tempbatchK;
				cout<<"inset node = "<<node<<",batchNum = "<< batchNum << ", tempbatchK.size()="<<tempbatchK.size()<<endl;
				tempbatchK.clear();
			}else{
				tempbatchK.insert(node);
			}
		}
		printf("\n");
	}

}


void readDynamicIM_set(string path, int largestK, int batchK)
{
//	map<int, map<int,unordered_set<int>>> Dynamic_IM_set;
	ifstream infile(path);
	string line;

	while (getline(infile, line)) {
		vector<string> seperateList = split(line," ");
		int timeID = stoi(seperateList[0]);
		if (timeID>largestTimewindow)
		{
			continue;
		}
		map<int, map<int, unordered_set<int>>>::iterator iterOne = Dynamic_IM_set.find(timeID);
		map<int, unordered_set<int>> batchMap;
		if(iterOne!=Dynamic_IM_set.end())
		{
			batchMap = Dynamic_IM_set[timeID];
		}

		unordered_set<int> tempbatchK;
		for(int i = 1; i<=largestK; i++)
		{
			int node = stoi(seperateList[i]);

			if(i%batchK==0)
			{
				int batchNum = i/batchK;
				tempbatchK.insert(node);
				batchMap[batchNum] = tempbatchK;
				tempbatchK.clear();
				Dynamic_IM_set[timeID] = batchMap;
			}else{
				tempbatchK.insert(node);
			}
		}
	}
}

vector<string> split(const string& str, const string& delim) {
	vector<string> res;
	if("" == str) return res;
	char * strs = new char[str.length() + 1] ;
	strcpy(strs, str.c_str());

	char * d = new char[delim.length() + 1];
	strcpy(d, delim.c_str());

	char *p = strtok(strs, d);
	while(p) {
		string s = p;
		res.push_back(s);
		p = strtok(NULL, d);
	}

	return res;
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

		iss >> timeID >> u >> v>>venueID;
		count = count + 1;
//		cout<<"infector force compute = "<< count<<endl;
		if (timeID>largestTimewindow)
		{
			continue;
		}

		string pair_tag;
//		cout<< "timeID = "<<timeID<<endl;
//		cout<<" floor(timeID/timewindowBatch)="<< floor(timeID/timewindowBatch)<<endl;
		int timeWindow_ID = floor(timeID/timewindowBatch);
//		cout<<"after timeWindow_ID="<<timeWindow_ID<<endl;
		if (u>v)
		{
			pair_tag = to_string(v) + "," + to_string(u);
		}else{
			pair_tag = to_string(u) + "," + to_string(v);
		}
//
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
//	map<string, map<int, double>> pairs_timewindow_infection_force;
	while (getline(infile, line)) {
		istringstream iss(line);

		iss >> timeID>> u >> v >> distance ;
//	        int begin_time = ceil(time_start*1.0/timewindowBatch*1.0);
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
		map<string, map<int,int>>::iterator iterPairs =pairsTimeDist.find(pair_tag);

		double temp_force = 0.0;
		if(distance <= distance_threshold){
			temp_force = a_para*exp(-distance/rho_1);
		}
		if(iterPairs!=pairsTimeDist.end())
		{
			map<int,int> timeIDDistance =  iterPairs->second;
			map<int,int>::iterator itertimeIDDistance = timeIDDistance.find(timeWindow_ID);// timeID,distance
			if(itertimeIDDistance!=timeIDDistance.end())
			{
				timeIDDistance[timeWindow_ID] = distance;
				pairsTimeDist[pair_tag] = timeIDDistance;
			}else{
				timeIDDistance[timeWindow_ID] = distance;
				pairsTimeDist[pair_tag] = timeIDDistance;
			}
			map<int, double> map_infection_force = pairs_timewindow_infection_force[pair_tag];
			map_infection_force[timeWindow_ID] = map_infection_force[timeWindow_ID] + temp_force;
			pairs_timewindow_infection_force[pair_tag] = map_infection_force;
		}else
		{
			map<int,int> timeIDDistance;
			timeIDDistance[timeWindow_ID] = distance;
			pairsTimeDist[pair_tag] = timeIDDistance;
			map<int, double> map_infection_force;
			map_infection_force[timeWindow_ID] = temp_force;
			pairs_timewindow_infection_force[pair_tag] = map_infection_force;
		}
}}
