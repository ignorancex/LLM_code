/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Hypergraph.cpp
 * Author: gunduz
 * 
 * Created on February 13, 2018, 1:01 AM
 */

#include "Hypergraph.h"

Hypergraph::Hypergraph() : nvtx(0), nedge(0) {
}

Hypergraph::Hypergraph(const Hypergraph& orig) {
}

Hypergraph::~Hypergraph() {
}

void Hypergraph::addNet(unordered_set<int> * net) {
    nets.insert(net);
    for (int u : *net) {
        vertices[u].insert(net);
    }
}

void Hypergraph::removeVertexAndConnectedNets(int u) {
    for (unordered_set<int> * net : vertices[u]) {
        net->erase(u);
        for (int v : *net) {
            vertices[v].erase(net);
        }
        nets.erase(net);
    }
    vertices.erase(u);
}

int Hypergraph::getMaxDegreeVertex() {
    int maxDegreeVertex, maxDegree = 0;
    int countH = 0;
    for (auto e : vertices) {
        int u = e.first;
        countH = countH + 1;
//        printf("getMaxDegreeVertex vertice u = %d, countH = %d \n", u,countH);
        if (maxDegree < vertices[u].size()) {
            maxDegree = vertices[u].size();
            maxDegreeVertex = u;
//            printf("maxDegreeVertex = %d, maxDegree = %d \n", maxDegreeVertex,maxDegree);
        }
    }
    return maxDegreeVertex;
}

list<unordered_set<int>> Hypergraph::computeMaxCoverage(int timwindowIncre, int largestTimewindow) {

	list<unordered_set<int>> allMaxMap;

    int batchNum = largestTimewindow/timwindowIncre;
//    printf("computeMaxCoverage->\n");
    for(int i = 0; i < batchNum; i++)
    {
    	int mapID = (i+1)*timwindowIncre;
    	unordered_set<int> tempNodesList;
    	for(int j = 0; j <timwindowIncre; j++)
    	{
    		if (vertices.empty())
    		{
    			printf("vertices are empty! \n");
    			break;
    		}
    		int u = getMaxDegreeVertex();
    		removeVertexAndConnectedNets(u);
//    		printf("remove node u = %d \n",u);
    		tempNodesList.insert(u);
//    		printf("insert node u = %d:\n", u);
    	}

    	allMaxMap.push_back(tempNodesList);

    }
    return allMaxMap;
}

list<unordered_set<int>> Hypergraph::computeTopK(int timwindowIncre, int largestTimewindow) {


	list<unordered_set<int>> allTopkMap;
    priority_queue<pair<int, int> > pq;
    for (auto e : vertices) {
        int u = e.first;
        int deg = vertices[u].size();
        pq.push(make_pair(deg, u));
    }

    int batchNum = largestTimewindow/timwindowIncre;

    for(int i = 0; i < batchNum; i++)
    {
    	int mapID = (i+1)*timwindowIncre;
    	unordered_set<int> tempNodesList;
    	for(int j = 0; j <timwindowIncre; j++)
    	{
    		 pair<int, int> top = pq.top();
    		 pq.pop();
    		 tempNodesList.insert(top.second);
    	}
    	allTopkMap.push_back(tempNodesList);
    }
    return allTopkMap;
}

list<unordered_set<int>> Hypergraph::computeTopK(int timwindowIncre, int largestTimewindow, string outputFile) {

	cout<<"computeTopK outputFile = "<<outputFile<<endl;
	string allrankList = outputFile + "_allRankList.csv";
	char str_filenamesummaryResult[250];
	strcpy(str_filenamesummaryResult, allrankList.c_str());
	FILE *fpsummaryResult;
	fpsummaryResult=fopen(str_filenamesummaryResult,"a+");

	cout<<"str_filenamesummaryResult = "<<str_filenamesummaryResult<<endl;
	list<unordered_set<int>> allTopkMap;
    priority_queue<pair<int, int> > pq;
    for (auto e : vertices) {
        int u = e.first;
        int deg = vertices[u].size();
        pq.push(make_pair(deg, u));
        fprintf(fpsummaryResult, "%d,%d", u, deg);
        fprintf(fpsummaryResult, "\n");
    }

    fclose(fpsummaryResult);


    string traceNodeList = outputFile + "_traceNodeList.csv";
    char filenamesummaryResult[250];
    strcpy(filenamesummaryResult, traceNodeList.c_str());
    FILE *summaryResult;
    summaryResult=fopen(filenamesummaryResult,"a+");

    int batchNum = largestTimewindow/timwindowIncre;

    for(int i = 0; i < batchNum; i++)
    {
    	int mapID = (i+1)*timwindowIncre;
    	unordered_set<int> tempNodesList;
    	for(int j = 0; j <timwindowIncre; j++)
    	{
    		 pair<int, int> top = pq.top();
    		 pq.pop();
    		 tempNodesList.insert(top.second);
    		 fprintf(summaryResult, "Node----%d\n", top.second);

    		 for (unordered_set<int> * net : vertices[top.second]) {
    			 for (int v : *net) {
    				 fprintf(summaryResult, "%d,", v);
    		     }
    			 fprintf(summaryResult, "\n");
    		 }
    	}
    	allTopkMap.push_back(tempNodesList);
    }

    fclose(summaryResult);
    return allTopkMap;
}
void Hypergraph::print() {
    printf("Nets:\n");
    for (auto net : nets) {
        for (int u : *net) {
            printf("%d ", u);
        }
        printf("\n");
    }

    for (auto e : vertices) {
        printf("%d:\n", e.first);

        for (unordered_set<int> * net : e.second) {
            for (int u : *net) {
                printf("%d ", u);
            }
            printf("\n");
        }
        printf("\n");
    }

}

