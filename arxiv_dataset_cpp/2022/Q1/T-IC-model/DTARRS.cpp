/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   DTARRS.cpp
 * Author: gunduz
 * 
 * Created on February 15, 2018, 8:44 PM
 */

#include "DTARRS.h"

DTARRS::DTARRS(DiGraph & GT) : GT(GT) {

    long seed = chrono::high_resolution_clock::now().time_since_epoch().count();
    real_rand = new uniform_real_distribution<double>(0, 1.0);
    dice_rand = new uniform_int_distribution<int>(0, GT.nvtx - 1);
    mt_rand = new mt19937(seed);

    vtxarr.insert(vtxarr.end(), GT.V.begin(), GT.V.end());

}

DTARRS::~DTARRS() {
}

unordered_set<int> * DTARRS::generateRRS(int t1, int t2) {

    int s = vtxarr[dice_rand->operator()(*mt_rand)];
    unordered_set<int> * S = new unordered_set<int>();
    S->insert(s);
//    printf("generateRRS->\n");
    int count_U = 0;
    for (int i = t1; i <= t2; i++) {
        visited.clear();

        for (int u : *S) {
            visited.insert(u);
            Q.push(u);
            count_U = count_U + 1;
//            printf("t1 = %d, t2=%d, RRS u =%d, count_U = %d \n", t1, t2, u,count_U);
        }


        while (!Q.empty()) {
            int u = Q.front();
            Q.pop();
            for (auto e : GT[u]) {
                int v = e.first;
//                printf("value V =%d ", v);
                vector<double> & w = e.second;
                if (visited.find(v) != visited.end())
                    continue;
                double x = real_rand->operator()(*mt_rand);
//                printf("probability x = %f, w[i] = %f \n", x, w[i]);
                if (x <= w[i]) {
                    visited.insert(v);
                    Q.push(v);
//                    printf("RRS V =%d ", v);
                }
            }
        }

        S->insert(visited.begin(), visited.end());
    }

    return S;
}

void DTARRS::generateHypergraph(int t1, int t2, int N) {
    for (int n = 0; n < N; n++) {
    	if(n%10==0)
    	{
    		printf("t1=%d, t2=%d, n=%d,\n", t1, t2, n);
    	}

    	H.addNet(generateRRS(t1, t2));
    }
}

list<unordered_set<int>> DTARRS::computeMaxSpread(int timwindowIncre, int largestTimewindow) {
    return H.computeMaxCoverage(timwindowIncre,largestTimewindow);
}


list<unordered_set<int>> DTARRS::computeTopK(int timwindowIncre, int largestTimewindow) {
    return H.computeTopK(timwindowIncre,largestTimewindow);
}

list<unordered_set<int>> DTARRS::computeTopK(int timwindowIncre, int largestTimewindow, string summaryResult) {
    return H.computeTopK(timwindowIncre,largestTimewindow,summaryResult);
}
