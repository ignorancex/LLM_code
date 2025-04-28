/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   DiGraph.cpp
 * Author: gunduz
 * 
 * Created on February 12, 2018, 7:42 PM
 */

#include "DiGraph.h"

DiGraph::DiGraph() : nvtx(0), nedge(0) {
}

DiGraph::DiGraph(const DiGraph& orig) {
}

DiGraph::~DiGraph() {
}

void DiGraph::addEdge(int u, int v, vector<double> w) {
    if (adj.find(u) == adj.end()
            || adj[u].find(v) == adj[u].end()) {
        nedge++;
    }
    adj[u][v] = w;
    V.insert(u);
    V.insert(v);
    nvtx = V.size();
}

vector<double> DiGraph::returnWeight(int u, int v){

	return adj[u][v];

}

bool DiGraph::existEdge(int u, int v){

	if (adj.find(u) != adj.end())
	{
		if (adj[u].find(v) != adj[u].end()){
			return true;
		}else
		{
			return false;
		}
	}else
	{
		return false;
	}

}


void DiGraph::print() {
    printf("nvtx : %d nedge : %d\n", nvtx, nedge);
    printf("V : ");
    for (int u : V) {
        printf("%d ", u);
    }
    printf("\n");

    for (auto u : adj) {
        printf("%d : ", u.first);
        for (auto v : u.second) {
            printf("(%d", v.first);
            for (double w : v.second) {
                printf(", %g", w);
            }
            printf(")");
        }
        printf("\n");
    }

}
