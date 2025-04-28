#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "CacheGraph.h"
#include "MotifCalculator.h"
#include "FeatureCalculator.h"


extern "C"
void get_edge_motifs(unsigned int nodes, unsigned int edges, unsigned int* neighbors, int64* offsets, 
                     int level, bool directed, unsigned int* output) {
    
    // for (int k = 0; k <= nodes; k++) {
    //     std::cout << offsets[k] << std::endl;
    // }

    // std::cout << "\n\n";
    
    // for (int k = 0; k < edges; k++) {
    //     std::cout << neighbors[k] << std::endl;
    // }

    // std::cout << "nodes: " << nodes << std::endl;
    // std::cout << "edges: " << edges << std::endl;
    // std::cout << "level: " << level << std::endl;
    // std::cout << "directed: " << directed << std::endl;

    CacheGraph* graph = new CacheGraph(directed, nodes, edges, neighbors, offsets);
    auto mc = new MotifCalculator(level, directed);
    mc->setGraph(graph);
    std::vector<std::vector<unsigned int>*>* result = mc->Calculate();

    int num_motifs = result->at(0)->size();
    
    for (int i = 0; i < edges; i++) {
        std::vector<unsigned int>* vec = result->at(i);
        // std::cout << "\n";
        for (int j = 0; j < vec->size(); j++) {
            // run for all the motifs
            output[num_motifs * i + j] = vec->at(j);
            // std::cout << vec->at(j) << " ";
        }
    }
}

int main() {
    unsigned int neighbors[] = {1, 3, 3, 4, 5, 2, 0, 1};
    int64 offsets[] = {0, 2, 3, 5, 6, 7, 8};

    CacheGraph* graph = new CacheGraph(true, 6, 8, neighbors, offsets);
    MotifCalculator* mc = new MotifCalculator(3, true);
    mc->setGraph(graph);
    std::vector<std::vector<unsigned int>*>* result = mc->Calculate();

    int num_motifs = result->at(0)->size();
    int edges = 8;

    for (int i = 0; i < edges; i++) {
        std::vector<unsigned int>* vec = result->at(i);
        for (int j = 0; j < vec->size(); j++) {
            std::cout << vec->at(j) << " ";
        }
        std::cout << std::endl;
    }

}