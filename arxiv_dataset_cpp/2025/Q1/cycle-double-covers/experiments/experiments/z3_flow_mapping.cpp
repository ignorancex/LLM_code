/*
 * File:   z3_flow_mapping.cpp
 * Author: Nikolay Ulyanov (ulyanick@gmail.com)
 *
 * TODO:
 *
 * Created on June 1, 2019
 *
 */

#include "z3_flow_mapping.h"

//TODO: remove some includes
#include <assert.h>
#include <iostream>
#include <cstdlib>
#include <string>

namespace ExpZ3FlowMapping {

int z3_vertex_flow[MAX_VERTEX];
int z3_edge_flow[MAX_EDGE];

int edge_image[MAX_EDGE];
int edge_orientation[MAX_EDGE];
bool edge_is_rich[MAX_EDGE];

/*********************************Methods*********************************/

vector<int> find_fast_edge_order(const Graph& graph, int start_edge) {
  vector<int> edge_order;
  edge_order.push_back(start_edge);
  unordered_set<int> visited_edges;
  visited_edges.insert(start_edge);
  for (int i = 0; i < graph.number_of_edges; ++i) {
    int eo = edge_order[i];
    for (int vi = 0; vi < 2; ++vi) {
      int v = graph.e2v[eo][vi];
      for (int j = 0; j < MAX_DEG; ++j) {
        int e = graph.v2e[v][j];
        if (visited_edges.find(e) == visited_edges.end()) {
          edge_order.push_back(e);
          visited_edges.insert(e);
        }
      }
    }
  }
  return edge_order;
}

// only for petersen graph
bool gen_all_z3_flows(Graph& graph, int cur_edge_idx) {
  if (cur_edge_idx == graph.number_of_edges) {
      for (int v = 0; v < graph.number_of_vertices; ++v) {
        if ((z3_vertex_flow[v]) % 3 != 0) {
          return false;
        }
      }
      vector<int> flow;
      for (int e = 0; e < graph.number_of_edges; ++e) {
        flow.push_back(z3_edge_flow[e]);
      }
      if (graph.all_z3_flows.find(flow) != graph.all_z3_flows.end()) {
        return false;
      }
      set<vector<int>> prev_flows = graph.all_z3_flows;
      for (int c1 = 1; c1 < 3; ++c1) {
        for (const auto& prev_flow : prev_flows) {
          vector<int> comb_flow;
          for (int e = 0; e < graph.number_of_edges; ++e) {
            comb_flow.push_back((c1 * flow[e] + prev_flow[e]) % 3);
          }
          assert(graph.all_z3_flows.find(comb_flow) == graph.all_z3_flows.end());
          graph.all_z3_flows.insert(comb_flow);
        }
      }
      return false;
  }

  int cur_edge = graph.faster_edge_order[cur_edge_idx];

  int v1 = graph.e2v[cur_edge][0];
  int v2 = graph.e2v[cur_edge][1];
  for (int f = 0; f < 3; ++f) {
    z3_edge_flow[cur_edge] = f;
    z3_vertex_flow[v1] = z3_vertex_flow[v1] + 3 - f;
    z3_vertex_flow[v2] = z3_vertex_flow[v2] + f;
    if (gen_all_z3_flows(graph, cur_edge_idx + 1)) {
      return true;
    }
    z3_vertex_flow[v1] = z3_vertex_flow[v1] + f;
    z3_vertex_flow[v2] = z3_vertex_flow[v2] + 3 - f;
  }
  z3_edge_flow[cur_edge] = 0;
  return false;
}

void gen_all_z3_flows(Graph& graph) {
  for (int v = 0; v < graph.number_of_vertices; ++v) {
    z3_vertex_flow[v] = 0;
  }
  vector<int> zero_flow;
  for (int e = 0; e < graph.number_of_edges; ++e) {
    z3_edge_flow[e] = 0;
    zero_flow.push_back(0);
  }
  graph.all_z3_flows.insert(zero_flow);
  gen_all_z3_flows(graph, 0);
}

bool z3_map_vertices(Graph& graph, int cur_edge_idx, const Graph& petersen_graph) {
  if (cur_edge_idx == graph.number_of_edges) {
    // set<int> image;
    // for (int e = 0; e < graph.number_of_edges; ++e) {
    //   image.insert(edge_image[e]);
    // }
    // if (image.size() < 15) {
    //   return false;
    // }

    map<set<int>, int> petersen_triples;
    for (int v = 0; v < petersen_graph.number_of_vertices; ++v) {
      set<int> triple;
      for (int j = 0; j < MAX_DEG; ++j) {
        triple.insert(petersen_graph.v2e[v][j]);
      }
      petersen_triples[triple] = v;
    }

    int rich_edge = 0; // FIXME
    for (int e = 0; e < graph.number_of_edges; ++e) {
      edge_orientation[e] = 0;
      edge_is_rich[e] = false;
    }

    vector<int> edge_order = find_fast_edge_order(graph, rich_edge);
    edge_orientation[edge_order[0]] = 1;
    for (int ei = 1; ei < graph.number_of_edges; ++ei) {
      int e = edge_order[ei];
      set<int> images;
      int neib_orientation = 0;
      vector<int> triples_idx;
      for (int vi = 0; vi < 2; ++vi) {
        int v = graph.e2v[e][vi];
        set<int> triple;
        for (int ei = 0; ei < MAX_DEG; ++ei) {
          triple.insert(edge_image[graph.v2e[v][ei]]);
        }
        assert(petersen_triples.find(triple) != petersen_triples.end());
        triples_idx.push_back(petersen_triples[triple]);
      }
      int def_orientation = 0;
      if (triples_idx[1] < triples_idx[0]) {
        def_orientation = -1;
      } else {
        def_orientation = 1;
      }
      for (int e2i = 0; e2i < 2 * MAX_DEG - 2; ++e2i) {
        int e2 = graph.e2e[e][e2i];
        images.insert(edge_image[e2]);
        if (edge_orientation[e2] != 0) {
          neib_orientation = edge_orientation[e2];
        }
      }
      assert(neib_orientation != 0);
      if (images.size() == 2) { // poor edge
        edge_orientation[e] = -neib_orientation;
      } else {
        edge_orientation[e] = neib_orientation;
      }
    }
    bool all_good = true;
    for (const auto& z3_flow : petersen_graph.all_z3_flows) {

    }
    if (all_good) {

    }
    return false;
  }

  int cur_edge = graph.faster_edge_order[cur_edge_idx];
  set<int> image_candidates = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  for (int e2i = 0; e2i < 2 * MAX_DEG - 2; ++e2i) {
    int e2 = graph.e2e[cur_edge][e2i];
    int e2_image = edge_image[e2];
    if (e2_image != NONE) {
      for (auto it = image_candidates.begin(); it != image_candidates.end(); ) {
        if (!petersen_graph.edges_are_neibs[cur_edge][*it]) {
          it = image_candidates.erase(it);
        } else {
          ++it;
        }
      }

    }
  }

  for (const auto& image : image_candidates) {
    edge_image[cur_edge] = image;
    if (z3_map_vertices(graph, cur_edge_idx + 1, petersen_graph)) {
      return true;
    }
  }
  edge_image[cur_edge] = NONE;
  return false;
}

void find_all_z3_flow_mappings(Graph& graph, const Graph& petersen_graph) {
  graph.has_z3_flow_mapping = false;

  for (int e = 0; e < graph.number_of_edges; ++e) {
    edge_image[e] = NONE;
  }

  for (int v = 0; v < graph.number_of_vertices; ++v) {
  }

  z3_map_vertices(graph, 0, petersen_graph);
  cerr << "number of z3 mappings: " << graph.z3_edge_mappings.size() << endl;
}

} // ExpZ3FlowMapping
