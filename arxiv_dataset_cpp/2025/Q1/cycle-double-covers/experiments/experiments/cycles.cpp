/*
 * File:   cycles.cpp
 * Author: Nikolay Ulyanov (ulyanick@gmail.com)
 *
 * TODO:
 *
 */

#include "cycles.h"
#include "graph.h"

#include <vector>

using namespace std;


namespace ExpCycles {

bool vertex_in_cycle[MAX_VERTEX];
bool edge_in_cycle[MAX_EDGE];
int cycle_length;

int start_vertex[MAX_VERTEX];
int number_of_circuits = 0;
int separate_circuits[MAX_VERTEX][2 * MAX_VERTEX];
int separate_circuits_length[MAX_VERTEX];
int cycle_count_for_edges[MAX_EDGE];
int vertex_colors[MAX_VERTEX];

/*********************************Methods*********************************/

bool cycle_is_nonseparating(const Graph& graph) {
  for (int v = 0; v < graph.number_of_vertices; ++v) {
    vertex_colors[v] = NONE;
  }
  vector<int> q;
  q.push_back(0);
  vertex_colors[0] = 0;
  int idx = 0;
  while (idx < q.size()) {
    int v = q[idx];
    ++idx;
    for (int j = 0; j < MAX_DEG; ++j) {
      int e = graph.v2e[v][j];
      if (!edge_in_cycle[e]) {
        int v2 = graph.v2v[v][j];
        if (vertex_colors[v2] == NONE) {
          q.push_back(v2);
          vertex_colors[v2] = 0;
        }
      }
    }
  }
  for (int v = 0; v < graph.number_of_vertices; ++v) {
    if (vertex_colors[v] == NONE) {
      return false;
    }
  }
  return true;
}

void build_cycle(int cur_vertex, int min_possible_edge, Graph& graph, bool hamiltonian) {
  if (cur_vertex == start_vertex[number_of_circuits]) {
    Mask bit_cycle = 0;
    for (int e = 0; e < graph.number_of_edges; ++e) {
      if (edge_in_cycle[e]) {
        bit_cycle += BIT(e);
        ++cycle_count_for_edges[e];
      }
    }
    graph.all_cycles.insert(bit_cycle);
    if (cycle_length == graph.number_of_vertices) {
      if (!hamiltonian || number_of_circuits == 0) {
        graph.all_full_cycles.insert(bit_cycle);
      }
    }
    if (number_of_circuits == 0) {
      graph.all_circuits.insert(bit_cycle);
    }
    bool is_dominating = true;
    for (int e = 0; e < graph.number_of_edges; ++e) {
      if (!vertex_in_cycle[graph.e2v[e][0]] && !vertex_in_cycle[graph.e2v[e][1]]) {
        is_dominating = false;
        break;
      }
    }

    if (is_dominating) {
      graph.all_dominating_cycles.insert(bit_cycle);
      Mask ignored = 0;
      for (int v = 0; v < graph.number_of_vertices; ++v) {
        if (!vertex_in_cycle[v]) {
          // bool has_neib_not_in_cycle = false;
          // for (int j = 0; j < MAX_DEG; ++j) {
          //   if (!vertex_in_cycle[graph.v2v[v][j]]) {
          //     has_neib_not_in_cycle = true;
          //     break;
          //   }
          // }
          // if (!has_neib_not_in_cycle) {
          ignored += BIT(v);
          // }
        }
      }
      graph.ignored_vertices_by_dominating_cycle[bit_cycle] = ignored;

      if (number_of_circuits == 0) {
        graph.all_dominating_circuits.insert(bit_cycle);
        Mask vertex_set = 0;
        for (int v = 0; v < graph.number_of_vertices; ++v) {
          if (vertex_in_cycle[v]) {
            vertex_set += BIT(v);
          }
        }
        graph.all_dominating_vertex_sets.insert(vertex_set);
      }
    }

    if (cycle_is_nonseparating(graph)) {
      graph.all_nonseparating_cycles.insert(bit_cycle);
    }

    int odd_cycle_count = 0;
    for (int i = 0; i <= number_of_circuits; ++i) {
      if (separate_circuits_length[i] % 2 == 1) {
        ++odd_cycle_count;
      }
    }
    if (odd_cycle_count == 0) {
      graph.all_even_cycles.insert(bit_cycle);
      if (cycle_length == graph.number_of_vertices - 4) {
        graph.all_even_v_minus_4_cycles.insert(bit_cycle);
      }
    }
    if (cycle_length == graph.number_of_vertices) {
      graph.oddness = min(graph.oddness, odd_cycle_count);
    }

    vector<vector<int>> circuits;
    vector<Mask> circuit_masks;
    for (int i = 0; i <= number_of_circuits; ++i) {
      vector<int> circuit;
      for (int j = 0; j < separate_circuits_length[i]; ++j) {
        circuit.push_back(separate_circuits[i][j]);
      }
      Mask circuit_mask = 0;
      circuit.push_back(separate_circuits[i][0]);
      for (int j = 0; j < circuit.size() - 1; ++j) {
        int v1 = circuit[j];
        int v2 = circuit[j + 1];
        int ei = graph.vv2e[v1][v2];
        circuit_mask += BIT(ei);
      }
      circuits.push_back(circuit);
      circuit_masks.push_back(circuit_mask);
    }
    graph.cycles_as_circuits[bit_cycle] = circuits;
    graph.cycles_as_circuit_masks[bit_cycle] = circuit_masks;

    // section for Petersen graph
    if (graph.number_of_vertices == 10) {
      /*if (cycle_length <= 6) {
        check_tcm_decomposition_in_petersen_graph(graph);
      }*/
    }
  }
  if (cur_vertex == start_vertex[number_of_circuits] && separate_circuits_length[number_of_circuits] > 0) {
    start_or_continue_build_cycle(min_possible_edge + 1, graph, hamiltonian);
    return;
  }

  for (int j = 0; j < MAX_DEG; ++j) {
    int next_vertex = graph.v2v[cur_vertex][j];
    int ei = graph.v2e[cur_vertex][j];

    // conditions
    if (ei < min_possible_edge || edge_in_cycle[ei] || vertex_in_cycle[next_vertex]) {
      continue;
    }

    // initialization
    separate_circuits[number_of_circuits][separate_circuits_length[number_of_circuits]] = next_vertex;
    ++cycle_length;
    ++separate_circuits_length[number_of_circuits];
    vertex_in_cycle[next_vertex] = true;
    edge_in_cycle[ei] = true;

    // recursion
    build_cycle(next_vertex, min_possible_edge, graph, hamiltonian);

    // undo
    --cycle_length;
    --separate_circuits_length[number_of_circuits];
    vertex_in_cycle[next_vertex] = false;
    edge_in_cycle[ei] = false;
  }
}

void start_or_continue_build_cycle(int possible_edge_lower_bound, Graph& graph, bool hamiltonian) {
  for (int min_possible_edge = possible_edge_lower_bound; min_possible_edge < graph.number_of_edges; ++min_possible_edge) {
    int v1 = graph.e2v[min_possible_edge][0];
    int v2 = graph.e2v[min_possible_edge][1];

    // conditions
    if (edge_in_cycle[min_possible_edge] || vertex_in_cycle[v1] || vertex_in_cycle[v2]) {
      continue;
    }

    // initialization
    ++number_of_circuits;
    ++cycle_length;
    separate_circuits[number_of_circuits][0] = v2;
    separate_circuits_length[number_of_circuits] = 1;
    start_vertex[number_of_circuits] = v1;
    vertex_in_cycle[v2] = true;
    edge_in_cycle[min_possible_edge] = true;

    // recursion
    build_cycle(v2, min_possible_edge, graph, hamiltonian);

    // undo
    --number_of_circuits;
    --cycle_length;
    vertex_in_cycle[v2] = false;
    edge_in_cycle[min_possible_edge] = false;
  }
}

void find_all_cycles(Graph& graph, bool hamiltonian) {
  graph.oddness = MAX_VERTEX;
  graph.all_cycles.clear();
  for (int e = 0; e < graph.number_of_edges; ++e) {
    cycle_count_for_edges[e] = 0;
  }
  cycle_length = 0;
  separate_circuits_length[0] = 0;
  for (int v = 0; v < graph.number_of_vertices; ++v) {
    vertex_in_cycle[v] = false;
  }
  for (int e = 0; e < graph.number_of_edges; ++e) {
    edge_in_cycle[e] = false;
  }
  number_of_circuits = -1;

  start_or_continue_build_cycle(0, graph, hamiltonian);
}

} // ExpCycles
