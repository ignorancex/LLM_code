/*
 * File:   preimages.cpp
 * Author: Nikolay Ulyanov (ulyanick@gmail.com)
 *
 * TODO:
 *
 */

#include "preimages.h"

#include <set>
#include <vector>

using namespace std;


namespace ExpPreimages {

/*********************************Methods*********************************/

void preimage_full_cycles(const Graph& petersen_graph, Graph& graph) {
    graph.full_cycles_from_petersen.clear();
    for (int col_idx = 0; col_idx < graph.petersen_colourings.size(); ++col_idx) {
        for (const auto c : petersen_graph.all_full_cycles) {
            Mask cycle_mask = 0;
            for (int e = 0; e < graph.number_of_edges; ++e) {
                int petersen_edge = graph.petersen_colourings[col_idx][e];
                if (c & BIT(petersen_edge)) {
                    cycle_mask += BIT(e);
                }
            }
            graph.full_cycles_from_petersen.insert(cycle_mask);
        }
    }
}

void preimage_all_cycles(const Graph& petersen_graph, Graph& graph) {
    graph.all_cycles_from_petersen.clear();
    for (int col_idx = 0; col_idx < graph.petersen_colourings.size(); ++col_idx) {
        for (const auto c : petersen_graph.all_cycles) {
            Mask cycle_mask = 0;
            for (int e = 0; e < graph.number_of_edges; ++e) {
                int petersen_edge = graph.petersen_colourings[col_idx][e];
                if (c & BIT(petersen_edge)) {
                    cycle_mask += BIT(e);
                }
            }
            graph.all_cycles_from_petersen[cycle_mask].insert(c);
        }
    }
}

void preimage_dominating_circuits(const Graph& petersen_graph, Graph& graph) {
  graph.all_pet_dominating_circuits.clear();
  for (int col_idx = 0; col_idx < graph.petersen_colourings.size(); ++col_idx) {
    for (const auto c : petersen_graph.all_cycles) {
      Mask cycle_mask = 0;
      for (int e = 0; e < graph.number_of_edges; ++e) {
        int petersen_edge = graph.petersen_colourings[col_idx][e];
        if (c & BIT(petersen_edge)) {
          cycle_mask += BIT(e);
        }
      }
      if (graph.all_dominating_circuits.find(cycle_mask) != graph.all_dominating_circuits.end()) {
        graph.all_pet_dominating_circuits.insert(cycle_mask);
      }
    }
  }
}

// TODO: split into several functions
void preimage_6c4c_5cdc_cycles(const Graph& petersen_graph, Graph& graph) {
    for (int col_idx = 0; col_idx < graph.petersen_colourings.size(); ++col_idx) {
        vector<set<Mask>> cur_5cdcs;
        for (const auto& petersen_5cdc : petersen_graph.all_5cdc) {
            set<Mask> cur_5cdc;
            for (const auto& c : petersen_5cdc) {
                Mask cycle_mask = 0;
                for (int e = 0; e < graph.number_of_edges; ++e) {
                    int petersen_edge = graph.petersen_colourings[col_idx][e];
                    if (c & BIT(petersen_edge)) {
                        cycle_mask += BIT(e);
                    }
                }
                cur_5cdc.insert(cycle_mask);
            }
            cur_5cdcs.push_back(cur_5cdc);
            graph.petersen_5cdc.insert(cur_5cdc);
            graph.petersen_5cdc_to_colouring_idx[cur_5cdc].push_back(col_idx);
            graph.petersen_5cdc_to_pet_5cdc[cur_5cdc].push_back(petersen_5cdc);
        }

        for (const auto& petersen_6c4c : petersen_graph.all_6c4c) {
            multiset<Mask> cur_6c4c;
            for (const auto& c : petersen_6c4c) {
                Mask cycle_mask = 0;
                for (int e = 0; e < graph.number_of_edges; ++e) {
                    int petersen_edge = graph.petersen_colourings[col_idx][e];
                    if (c & BIT(petersen_edge)) {
                        cycle_mask += BIT(e);
                    }
                }
                cur_6c4c.insert(cycle_mask);
            }
            graph.petersen_6c4c.insert(cur_6c4c);
            graph.petersen_6c4c_to_colouring_idx[cur_6c4c].push_back(col_idx);
            for (const auto cur_5cdc : cur_5cdcs) {
                graph.petersen_6c4c_5cdc_pairs[cur_6c4c].insert(cur_5cdc);
                graph.petersen_5cdc_6c4c_pairs[cur_5cdc].insert(cur_6c4c);
            }
        }
    }
}

} // ExpPreimages
