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
vector<string> split(const string& str, const string& delim);

DiGraph * transpose(DiGraph & G);
DiGraph * reduceMax(DiGraph & G);
DiGraph * reduceAvg(DiGraph & G);
DiGraph * reduceMin(DiGraph & G);

DiGraph * readDiGraph_overall(string path, int nInterval);
void readImmunization_set(string path, int largestK, int batchK);
void readDynamicIM_set(string path, int largestK, int batchK);

void computePairsTimeDist(string path);
void computePoisNumTimewindow(string path);
void computePois_Infection_force(string path);
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
float a_para;
float rho_1=0.0;
float b_para=0.0;
float rho_2=0.0;
int largestK;
int batch_K;
//int timewindowBatch;
int distance_threshold;


int timeWindowNum;
//int largestTimewindow;
int timwindowIncre;
string runMode;//basic, a=people_connection_prob, b=time_decaying_prob, overall=a+b
map<int, unordered_set<int>> nodesPoisList;
map<int, map<int, int>> poisTimeWindowNum;
map<string, map<int, int>> pairsTimeDist;
//
int NODE_NUM;
int randomIterNum;
int timeDecay;
int timewindowBatch;
int largestTimewindow;
int valNum;
int decayWindow;
map<int, map<int,unordered_set<int>>> reduced_greedyset;
map<int, unordered_set<int>> T_IMMU;
map<int, map<int,unordered_set<int>>> Dynamic_IM_set;

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
    decayWindow = atof(argv[15]);
    if (dataset_name!="BBC")
    {
    	b_para = atof(argv[13]);// b parameter
    	rho_2 = atof(argv[14]);// rho_2 parameter
    	// decayWindow default to be 14
    	computePoisNumTimewindow(data);//venueID, timeID, num of people
//    	computePairsTimeDist(data);//pairs, timeID, dist
    	computePois_Infection_force(data);
    }else
    {
    	computePairsTimeDist(data);//pairs, timeID, dist
    }


//
    t2 = ceil(t2*1.0/timewindowBatch*1.0);
    DiGraph & G = *readDiGraph_overall(data, t2 - t1 + 1);

    DiGraph & GT = *transpose(G);

    DTARRS * dtarrsG = new DTARRS(G);


    printf("t1=%d, t2=%d, iterNum=%d \n", t1, t2, iterNum);
    dtarrsG->generateHypergraph(t1, t2, iterNum);
    printf("HG complete\n");
    list<unordered_set<int>> temp;
    list<unordered_set<int>> AtopK = dtarrsG->computeTopK(batch_K, largestK);
    list<unordered_set<int>> A = dtarrsG->computeMaxSpread(batch_K, largestK);

    list<unordered_set<int>>  kmax_degree_nodes = max_K_degree;
    map<int, map<int, unordered_set<int>>>* all_random_K_nodes_List = &all_random_K_nodes;
    string dynamic_solution_file(argv[17]);
    string TImmuni_solution_file(argv[16]);
    readImmunization_set(TImmuni_solution_file, largestK, batch_K);
	printf("read Dynamic set! \n");
	readDynamicIM_set(dynamic_solution_file, largestK, batch_K);

    printf("Greedy solution:\n");

    omp_set_num_threads(100);

    int batchKNum = largestK/batch_K;
    string summaryResult = "./experiment_result/"+ runMode + "/summary_" + dataset_name + "_time_window_" + to_string(t2) +
         "a_para" + to_string(a_para)+"b_para" + to_string(b_para)+
		 "rho_1" + to_string(rho_1)+"rho_2" + to_string(rho_2)+
         	"_iterNum_" + to_string(iterNum) + ".txt";
    char str_filenamesummaryResult[250];
    strcpy(str_filenamesummaryResult, summaryResult.c_str());
    FILE *fpsummaryResult;
    fpsummaryResult=fopen(str_filenamesummaryResult,"a+");


    string randomResult = "./experiment_result/"+ runMode + "/random_" + dataset_name + "_time_window_" + to_string(t2) +
    					"b_para" + to_string(b_para)+
    		         	"_iterNum_" + to_string(iterNum) + ".txt";
    char str_filenamerandomResult[250];
    strcpy(str_filenamerandomResult, randomResult.c_str());
    FILE *fprandomResult;
    fprandomResult  =fopen(str_filenamerandomResult,"a+");


    string ouputResult = "./experiment_result/"+ runMode + "/output_" + dataset_name + "_time_window_" + to_string(t2) +
    		"b_para" + to_string(b_para)+
    		         	"_iterNum_" + to_string(iterNum) + ".txt";
      char str_filenameouputResult[250];
      strcpy(str_filenameouputResult, ouputResult.c_str());
      FILE *fpouputResult;
      fpouputResult=fopen(str_filenameouputResult,"a+");


      string poisResult = "./experiment_result/"+ runMode + "/calPois_" + dataset_name + "_time_window_" + to_string(t2) +
    		  "b_para" + to_string(b_para)+
    		           	"_iterNum_" + to_string(iterNum) + ".txt";

      char str_filenamepoisResult[250];
      strcpy(str_filenamepoisResult, poisResult.c_str());
      FILE *fppoisResult;
      fppoisResult=fopen(str_filenamepoisResult,"a+");


    string RSM_content_influence = "RSM Influence,";
    string ESM_content_influence = "ESM Influence,";
    string TIMMU_content_influence = "TIMMU Influence,";
	string dynamic_content_influence = "dynamic Influence,";
    string MaxDegree_content_influence = "MaxDegree Influence,";
    string greedy_content_influence = "greedy Influence,";
    string random_content_influence = "Random Influence,";
    string head_content_influence = "k,";

    string RSM_content_expected_vertices = "RSM Expected Vertices,";
    string ESM_content_expected_vertices = "ESM Expected Vertices,";
    string MaxDegree_content_expected_vertices = "MaxDegree Expected Vertices,";
    string TIMMU_content_expected_vertices = "TIMMU Expected Vertices,";
	string dynamic_content_expected_vertices = "dynamic Expected Vertices,";
    string greedy_content_expected_vertices = "greedy Expected Vertices,";
    string random_content_expected_vertices = "Random Expected Vertices,";
    //string head_content_expected_vertices = "";

    string RSM_content_binary_success = "RSM Binary Success,";
    string ESM_content_binary_success = "ESM Binary Success,";
    string TIMMU_content_binary_success = "TIMMU Binary Success,";
	string dynamic_content_binary_success = "dynamic Binary Success,";
    string MaxDegree_content_binary_success = "MaxDegree Binary Success,";
    string greedy_content_binary_success = "greedy Binary Success,";
    string random_content_binary_success = "Random Binary Success,";

    string RSM_content_poisNum = "RSM POIS Num,";
    string ESM_content_poisNum = "ESM POIS Num,";
    string MaxDegree_content_poisNum = "MaxDegree POIS Num,";
    string random_content_poisNum = "Random POIS Num,";
    printf("batchKNum=%d\n",batchKNum);
    unordered_set<int>  RSM;
    unordered_set<int>  ESM;
    unordered_set<int>  MaxDegree;
    unordered_set<int>  reduced_greedy_batch;
    unordered_set<int> TIMMU_batch;
	unordered_set<int> dynamic_batch;
    map<int, unordered_set<int>>  IterrandomMethod;


    int ntrials = 1000;
    for(int i = 0; i < batchKNum; i++)
    {
    	printf("batchKNum=%d, i=%d \n", batchKNum, i);
    	int k_value = (i+1)*batch_K;
    	head_content_influence = head_content_influence + to_string(k_value) + ",";
    	int countNum = 0;
    	int mapID = (i+1)*batch_K;
    	list<unordered_set<int>>::iterator listIter;
    	for(listIter = A.begin(); listIter != A.end(); ++listIter)
    	{
    		countNum = countNum + 1;
    	    unordered_set<int> temp = *listIter;
    	    RSM.insert(temp.begin(), temp.end());
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

    	        RSM_content_poisNum = RSM_content_poisNum + to_string(poisSet_A.size()) +",";
    	        fprintf(fppoisResult, "RSM pois unique size %d:\n", poisSet_A.size());
    	        for (unordered_set<int>::iterator w = poisSet_A.begin(); w != poisSet_A.end(); w++) {
    	        	int poisVal = *w;
    	        	fprintf(fppoisResult, "%d, ", poisVal);
    	        }
    	        fprintf(fppoisResult, "\n");
    	        		//num of expected vertices.
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

    	        ESM_content_poisNum = ESM_content_poisNum + to_string(poisSet_A.size()) +",";
    	        fprintf(fppoisResult, "ESM pois unique size %d:\n", poisSet_A.size());
    	        for (unordered_set<int>::iterator w = poisSet_A.begin(); w != poisSet_A.end(); w++) {
    	        	int poisVal = *w;
    	            fprintf(fppoisResult, "%d, ", poisVal);
    	        }
    	        fprintf(fppoisResult, "\n");
    	        break;
    	    }
    	}


    	int countNumMaxDegree = 0;
    	list<unordered_set<int>>::iterator listIterMaxDegree;
    	for(listIterMaxDegree = kmax_degree_nodes.begin(); listIterMaxDegree != kmax_degree_nodes.end(); ++listIterMaxDegree)
    	{
    		countNumMaxDegree = countNumMaxDegree + 1;
    	    unordered_set<int> temp = *listIterMaxDegree;
    	    MaxDegree.insert(temp.begin(), temp.end());
    	    unordered_set<int>::iterator it;
    	    printf("MaxDegree element:");
    	    for(it = MaxDegree.begin();it!=MaxDegree.end();it++){
    	        printf("ele = %d, ", *it);
    	    }
    	    if (countNumMaxDegree==i+1)
    	    {
    	    	double influence_A = DTIC(t1, t2, MaxDegree, GT, ntrials);
    	    	printf("MaxDegree influence %f\n", influence_A);
    	    	 printf("MaxDegree Num %d\n", MaxDegree.size());
    	    	MaxDegree_content_influence = MaxDegree_content_influence + to_string(influence_A) + ",";
    	    	//number of pois
    	    	unordered_set<int> poisSet_A;
    	    	for (auto e : MaxDegree) {
    	    		fprintf(fppoisResult, "MaxDegree --- People Node---%d:", e);
    	    	    unordered_set<int> readPois = nodesPoisList[e];
    	    	    for (unordered_set<int>::iterator w = readPois.begin(); w != readPois.end(); w++) {
    	    	    	int poisVal = *w;
    	    	        fprintf(fppoisResult, "%d, ", poisVal);
    	    	        poisSet_A.insert(poisVal);
    	    	    }
    	    	    fprintf(fppoisResult, "\n");
    	    	}
    	    	MaxDegree_content_poisNum = MaxDegree_content_poisNum + to_string(poisSet_A.size()) +",";
    	    	fprintf(fppoisResult, "MaxDegree pois unique size %d:\n", poisSet_A.size());
    	    	for (unordered_set<int>::iterator w = poisSet_A.begin(); w != poisSet_A.end(); w++) {
    	    		int poisVal = *w;
    	    		fprintf(fppoisResult, "%d, ", poisVal);
    	    	}
    	    	fprintf(fppoisResult, "\n");
    	    	break;
    	    }
    	}

    	int newmapID = i + 1;
    	unordered_set<int> temp = T_IMMU[newmapID];
		printf("T_IMMU mapID = %d", newmapID);
		TIMMU_batch.insert(temp.begin(), temp.end());
		cout<<"TIMMU_batch.size()="<<TIMMU_batch.size()<<endl;
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


		double dynamic_average_influence = 0.0;
		map<int, unordered_set<int>> dynamic_batchsize_list;
		cout<<"get  dyamic set:"<<endl;
		for(int timeBatch = 0; timeBatch<t2;timeBatch++)
		{
			cout<<"timeBatch="<<timeBatch<<endl;
			map<int, unordered_set<int>> tempIM_set = Dynamic_IM_set[timeBatch];
			for(int batchIter = 1; batchIter <=newmapID; batchIter++)
			{
				unordered_set<int> temp = tempIM_set[batchIter];
				dynamic_batch.insert(temp.begin(), temp.end());
			}

			double influence_A = DTIC(t1, t2, dynamic_batch, GT, ntrials);
			dynamic_average_influence = dynamic_average_influence + influence_A;
			dynamic_batchsize_list[timeBatch] = dynamic_batch;

			dynamic_batch.clear();
		}
		dynamic_average_influence = dynamic_average_influence/t2;

		dynamic_content_influence = dynamic_content_influence + to_string(dynamic_average_influence) + ",";
		printf("dynamic influence %f\n", dynamic_average_influence);
		printf("dynamic Num %d\n", dynamic_batch.size());



		for(int randNum = 1; randNum <= randomIterNum; randNum++)
		{
			printf("Here random randNum = %d \n", randNum);
			unordered_set<int> tempRandomList;// = all_random_K_nodes_List[randNum][mapID];
			map<int, map<int, unordered_set<int>>>::iterator keyA = all_random_K_nodes_List->find(randNum);
			map<int, unordered_set<int>> temp_random_K_nodes_List = keyA->second;
		    unordered_set<int>::iterator it;
		    printf("mapID = %d \n", mapID);
		    for(it = temp_random_K_nodes_List[mapID].begin();it!=temp_random_K_nodes_List[mapID].end();it++){
		    	printf("random elemt = %d, ", *it);
		    }
		    printf("\n");
			tempRandomList.insert(temp_random_K_nodes_List[mapID].begin(), temp_random_K_nodes_List[mapID].end());
			map<int, unordered_set<int>>::iterator iterPois = IterrandomMethod.find(randNum);
			if(iterPois!=IterrandomMethod.end())
			{
				unordered_set<int> tempPoisList =  iterPois->second;
				tempPoisList.insert(tempRandomList.begin(), tempRandomList.end());
				IterrandomMethod[randNum] = tempPoisList;
			}else
			{
				unordered_set<int> tempPoisList;
				tempPoisList.insert(tempRandomList.begin(), tempRandomList.end());
				IterrandomMethod[randNum] = tempPoisList;
			}
		}


		string randomInfluence_list = "Random List Influence:";
	    double randomAvgInfluence = 0;
	    int countPoisRandom = 0;
	    for (int readRand = 1; readRand<=randomIterNum; readRand++)
	    {
	    	unordered_set<int> * randomRound = &IterrandomMethod[readRand];
	    	unordered_set<int>::iterator it;

	    	double influence_A = DTIC(t1, t2, *randomRound, GT, ntrials);
	    	randomAvgInfluence = randomAvgInfluence + influence_A;
	    	randomInfluence_list = randomInfluence_list + to_string(influence_A) + ",";
	    	unordered_set<int> poisSet_Random;
	    	printf("randomRound.size = %d \n", randomRound->size());
	    	for (auto e : *randomRound){
	    		unordered_set<int> readPois = nodesPoisList[e];
	    		for (unordered_set<int>::iterator w = readPois.begin(); w != readPois.end(); w++) {
	    			int poisVal = *w;
	    		    poisSet_Random.insert(poisVal);
	    		}
	    	}
	    	countPoisRandom = countPoisRandom + poisSet_Random.size();
	    }
	    double avgRandomPois = 1.0*countPoisRandom/(randomIterNum*1.0);
	    random_content_poisNum = random_content_poisNum + to_string(avgRandomPois) + ",";
	    printf("avgRandomPois:%f\n", avgRandomPois);

	    randomAvgInfluence = randomAvgInfluence/randomIterNum;
	    printf("randomAvgInfluence:%f\n", randomAvgInfluence);
	    fprintf(fprandomResult, randomInfluence_list.c_str());
	    random_content_influence = random_content_influence + to_string(randomAvgInfluence) + ",";

		fprintf(fpouputResult, "RSM:");
	    for (auto e : RSM) {
	    	fprintf(fpouputResult, "%d,", e);
	    }
	    fprintf(fpouputResult, "\n");
	    fprintf(fpouputResult, "ESM:");
	    for (auto e : ESM) {
	    	fprintf(fpouputResult, "%d,", e);
	    }
	    fprintf(fpouputResult, "\n");
	    fprintf(fpouputResult, "MaxDegree");
	    for (auto e : MaxDegree) {
	    	fprintf(fpouputResult, "%d,", e);
	    }
	    fprintf(fpouputResult, "\n");


	    long seed = chrono::high_resolution_clock::now().time_since_epoch().count();
	    mt19937 mt_rand(seed);
	    uniform_int_distribution<int> int_rand1 = uniform_int_distribution<int>(1, 1);
	    uniform_int_distribution<int> int_rand2 = uniform_int_distribution<int>(1, G.nvtx);
	    int nintersect1 = 0, nintersect2 = 0, nintersect3 = 0, nintersect4 = 0,nintersect5 = 0,nintersect6 = 0,
	    		binary1 = 0, binary2 = 0, binary3 = 0, binary4 = 0, binary5 = 0, binary6 = 0;
	    double detection_rate1 = 0.0, detection_rate2 = 0.0, detection_rate3 = 0.0, detection_rate4 = 0.0, detection_rate5 = 0.0, detection_rate6 = 0.0;
	    select_nodes_num = k_value;
	    for (int iA = 0; iA < ntrials; iA++) {
	    	unordered_set<int> S;
	    	int size = int_rand1(mt_rand);
	    	for (int jA = 0; jA < size; jA++) {
	    		S.insert(int_rand2(mt_rand));
	    	}
	    	int nintersect;

	    	vector<int> v = alldynamic_TIMMU_DTIC(t1, t2, S, RSM, ESM,MaxDegree,IterrandomMethod, TIMMU_batch, dynamic_batchsize_list,G);
	    	nintersect = v[0];
	    	nintersect1 += nintersect;
	    	if (nintersect > 0){
	    		binary1++;
	    		detection_rate1 = detection_rate1 + (double)nintersect/select_nodes_num;
	    	}
	    	nintersect = v[1];
	    	nintersect2 += nintersect;
	    	if (nintersect > 0){
	    		binary2++;
	    		detection_rate2 = detection_rate2 + (double)nintersect/select_nodes_num;
	    	}
	    	nintersect = v[2];
	    	nintersect3 += nintersect;
	    	if (nintersect > 0){
	    		binary3++;
	    		detection_rate3 = detection_rate3 + (double)nintersect/select_nodes_num;
	        }

	    	nintersect = v[3];
			nintersect4 += nintersect;
			if (nintersect > 0){
				binary4++;
				detection_rate4 = detection_rate4 + (double)nintersect/select_nodes_num;
			}


			nintersect = v[4];
			nintersect5 += nintersect;
			if (nintersect > 0){
				binary5++;
				detection_rate5 = detection_rate5 + (double)nintersect/select_nodes_num;
			}

			for(int dynamicIter = 5; dynamicIter< 5+t2; dynamicIter++)
			{
				nintersect = v[dynamicIter];
				nintersect6 += nintersect;
				if (nintersect > 0){
					binary6++;
					detection_rate6 = detection_rate6 + (double)nintersect/select_nodes_num;
				}
			}

	    	S.clear();
	    }

	    RSM_content_expected_vertices = RSM_content_expected_vertices + to_string((double) nintersect1 / ntrials) + ",";
	    ESM_content_expected_vertices = ESM_content_expected_vertices + to_string((double) nintersect2 / ntrials) + ",";
	    MaxDegree_content_expected_vertices = MaxDegree_content_expected_vertices + to_string((double) nintersect3 / ntrials) + ",";
//	    greedy_content_expected_vertices = greedy_content_expected_vertices + to_string((double) nintersect4 / ntrials/t2) + ",";
	    random_content_expected_vertices = random_content_expected_vertices + to_string((double) nintersect4 / ntrials) + ",";

	    TIMMU_content_expected_vertices = TIMMU_content_expected_vertices + to_string((double) nintersect5 / ntrials) + ",";
		dynamic_content_expected_vertices = dynamic_content_expected_vertices + to_string((double) nintersect6 / ntrials/t2) + ",";

		TIMMU_content_binary_success = TIMMU_content_binary_success + to_string((double) binary5 / ntrials) + ",";
		dynamic_content_binary_success = dynamic_content_binary_success + to_string((double) binary6 / ntrials/t2) + ",";


	    RSM_content_binary_success = RSM_content_binary_success + to_string((double) binary1 / ntrials) + ",";
	    ESM_content_binary_success = ESM_content_binary_success + to_string((double) binary2 / ntrials) + ",";
	    MaxDegree_content_binary_success = MaxDegree_content_binary_success + to_string((double) binary3 / ntrials) + ",";


	    //	    greedy_content_binary_success = greedy_content_binary_success + to_string((double) binary4 / ntrials/t2) + ",";

	    random_content_binary_success = random_content_binary_success + to_string((double) binary4 / ntrials) + ",";
    }

    fprintf(fpsummaryResult,"Reverse Influence \n");
    fprintf(fpsummaryResult,(head_content_influence+"\n").c_str());
    fprintf(fpsummaryResult,(RSM_content_influence+"\n").c_str());
    fprintf(fpsummaryResult,(ESM_content_influence+"\n").c_str());
    fprintf(fpsummaryResult,(MaxDegree_content_influence+"\n").c_str());
    fprintf(fpsummaryResult,(greedy_content_influence+"\n").c_str());
    fprintf(fpsummaryResult,(TIMMU_content_influence+"\n").c_str());
    fprintf(fpsummaryResult,(dynamic_content_influence+"\n").c_str());

    fprintf(fpsummaryResult,(greedy_content_influence+"\n").c_str());
    fprintf(fpsummaryResult,(random_content_influence+"\n").c_str());
    fprintf(fpsummaryResult,"Expected Number of Vertices\n");
    fprintf(fpsummaryResult,(head_content_influence+"\n").c_str());
    fprintf(fpsummaryResult,(RSM_content_expected_vertices+"\n").c_str());
    fprintf(fpsummaryResult,(ESM_content_expected_vertices+"\n").c_str());
    fprintf(fpsummaryResult,(MaxDegree_content_expected_vertices+"\n").c_str());
    fprintf(fpsummaryResult,(greedy_content_expected_vertices+"\n").c_str());
    fprintf(fpsummaryResult,(TIMMU_content_expected_vertices+"\n").c_str());
	fprintf(fpsummaryResult,(dynamic_content_expected_vertices+"\n").c_str());

    fprintf(fpsummaryResult,(random_content_expected_vertices+"\n").c_str());
    fprintf(fpsummaryResult,"Binary Success Rate\n");
    fprintf(fpsummaryResult,(head_content_influence+"\n").c_str());
    fprintf(fpsummaryResult,(RSM_content_binary_success+"\n").c_str());
    fprintf(fpsummaryResult,(ESM_content_binary_success+"\n").c_str());
    fprintf(fpsummaryResult,(MaxDegree_content_binary_success+"\n").c_str());
    fprintf(fpsummaryResult,(TIMMU_content_binary_success+"\n").c_str());
	fprintf(fpsummaryResult,(dynamic_content_binary_success+"\n").c_str());

    fprintf(fpsummaryResult,(greedy_content_binary_success+"\n").c_str());
    fprintf(fpsummaryResult,(random_content_binary_success+"\n").c_str());
    fprintf(fpsummaryResult,"Pois Num\n");
    fprintf(fpsummaryResult,(head_content_influence+"\n").c_str());
    fprintf(fpsummaryResult,(RSM_content_poisNum+"\n").c_str());
    fprintf(fpsummaryResult,(ESM_content_poisNum+"\n").c_str());
    fprintf(fpsummaryResult,(MaxDegree_content_poisNum+"\n").c_str());
    fprintf(fpsummaryResult,(random_content_poisNum+"\n").c_str());
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
	    int seperateTimeWindows = ceil(largestTimewindow/timewindowBatch);

	    map<int, map<int, unordered_set<int>>> susceptible_list;//nodeid, timewindow, count[V+1][seperateTimeWindows];

	    int timeID, u, v, venueID;
	    int poisPeopleNum;
	    vector<double> W(seperateTimeWindows);
	    vector<double> W_1(seperateTimeWindows);

	    int count_num = 0;
	    while (getline(infile, line)) {
	        istringstream iss(line);
	        count_num = count_num + 1;
	        iss >> timeID >> u >> v>>venueID;
	        printf("count_num = %d\n", count_num);
	        if (timeID>largestTimewindow)
	        {
	        	continue;
	        }
	        int timeWindow_ID = floor(timeID/timewindowBatch);
			if(timeWindow_ID==seperateTimeWindows)
			{
				timeWindow_ID = timeWindow_ID -1;
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
	        	printf("accumulate_enforce = %f, startProb = %f, timeEnd = %d\n", accumulate_enforce, startProb,timeEnd);
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
				printf("accumulate_enforce = %f, startProb = %f, timeWindow_ID = %d,  timeEnd = %d\n", accumulate_enforce, startProb,timeWindow_ID, timeEnd);
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


	        map<int, map<int, unordered_set<int>>>::iterator findNode = susceptible_list.find(v-1);
	        if(findNode!=susceptible_list.end())
	        {
			   map<int, unordered_set<int>> timeNum = susceptible_list[v-1];
			   map<int, unordered_set<int>>::iterator timeNumIter = timeNum.find(timeWindow_ID);
			   if(timeNumIter!=timeNum.end())
			   {
				   timeNum[timeWindow_ID].insert(timeNum[timeWindow_ID].size()+1);
			   }else
			   {
				   timeNum[timeWindow_ID].insert(1);

			   }
			   susceptible_list[v-1] = timeNum;
	        }else
	        {
				   map<int, unordered_set<int>> timeNum;
				   timeNum[timeWindow_ID].insert(1);
				   susceptible_list[v-1] = timeNum;
	        }
	        findNode = susceptible_list.find(u-1);
		   if(findNode!=susceptible_list.end())
		   {
			   map<int, unordered_set<int>> timeNum = susceptible_list[u-1];
			   map<int, unordered_set<int>>::iterator timeNumIter = timeNum.find(timeWindow_ID);
			   if(timeNumIter!=timeNum.end())
			   {
				   //timeNum[beginTime-1] = timeNum[beginTime-1] + 1;
					timeNum[timeWindow_ID].insert(timeNum[timeWindow_ID].size()+1);
			   }else
			   {
				   timeNum[timeWindow_ID].insert(1);
			   }
					susceptible_list[u-1] = timeNum;
		  }else
		  {
			  map<int, unordered_set<int>> timeNum;
			  timeNum[timeWindow_ID].insert(1);
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

	int timeID, u,v,distance,venueID;
	while (getline(infile, line)) {
		istringstream iss(line);

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
	printf("Go to readDynamicIM_set!\n");
	cout<<"path = " << path << endl;
	getline(infile, line);
	cout<<"line1 = "<<line<<endl;
	int seperateTimeWindows = ceil(largestTimewindow/timewindowBatch);
	while (getline(infile, line)) {
		if (line.size()==0)
		{
			continue;
		}
		cout<<"line.size()= "<<line.size()<<endl;
		cout<<"dhynamic line = "<<line<<endl;
		vector<string> seperateList = split(line," ");
		int timeID = stoi(seperateList[0]);
		int timeWindow_ID = floor(timeID/timewindowBatch);

//		int timeWindow_ID = floor(timeID/timewindowBatch);
		if(timeWindow_ID==seperateTimeWindows)
		{
			timeWindow_ID = timeWindow_ID -1;
		}
		cout<<"readDynamicIM_set timeWindow_ID = "<<timeWindow_ID<<endl;
		if (timeID>largestTimewindow)
		{
			continue;
		}

		map<int, map<int, unordered_set<int>>>::iterator iterOne = Dynamic_IM_set.find(timeWindow_ID);
		map<int, unordered_set<int>> batchMap;
		if(iterOne!=Dynamic_IM_set.end())
		{
			batchMap = Dynamic_IM_set[timeWindow_ID];
		}

		unordered_set<int> tempbatchK;
		for(int i = 1; i<=largestK; i++)
		{
			int node = stoi(seperateList[i]);
			cout<<"node value = "<<node<<endl;
			if(i%batchK==0)
			{
				int batchNum = i/batchK;
				tempbatchK.insert(node);
				batchMap[batchNum] = tempbatchK;
				cout<<"dynamic batchKsize = "<<tempbatchK.size()<<endl;
				cout<<"dynamic insert batchNum = "<<batchNum<<endl;
				tempbatchK.clear();
				Dynamic_IM_set[timeWindow_ID] = batchMap;
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
}

}
