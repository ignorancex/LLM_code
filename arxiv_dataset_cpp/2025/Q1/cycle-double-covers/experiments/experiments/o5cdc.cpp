/*
 * File:   o5cdc.cpp
 * Author: Nikolay Ulyanov (ulyanick@gmail.com)
 *
 * TODO:
 *
 */

#include "o5cdc.h"
#include "graph.h"
#include "constants.h"

#include <cassert>
#include <set>
#include <iostream>
#include <cstdlib>
#include <vector>

using namespace std;


namespace Exp5cdc {

const vector<vector<int>> u33pp_indices = {{0, 1, 2, 3, 4}, {0, 2, 1, 3, 4}, {0, 3, 1, 2, 4},
                                           {0, 1, 2, 4, 3}, {0, 2, 1, 4, 3}, {0, 4, 1, 2, 3},
                                           {0, 1, 3, 4, 2}, {0, 3, 1, 4, 2}, {0, 4, 1, 3, 2},
                                           {0, 2, 3, 4, 1}, {0, 3, 2, 4, 1}, {0, 4, 2 ,3, 1},
                                           {1, 2, 3, 4, 0}, {1, 3, 2, 4, 0}, {1, 4, 2, 3, 0}};

const vector<vector<int>> nz5_from_33pp_coefs = {{3, 1}, {3, -1}, {1, 3}, {1, -3},
                                                 {-3, -1}, {-3, 1}, {-1, -3}, {-1, 3}};

set<pair<Mask, Mask>> u33pp_pairs;

set<set<Mask>> all_5cdc;
set<Mask> cur_5cdc;

set<Mask> u333pp_cycles_from_o5cdc;
set<Mask> u333pp_even_cycles_from_o5cdc;

set<Mask> all_cycles_from_5cdc;
set<Mask> all_full_cycles_from_5cdc;
set<Mask> all_even_cycles_from_5cdc;
set<Mask> all_circuits_from_5cdc;

int start_vertex_in_5cdc[5][MAX_VERTEX];
bool vertex_in_5cdc[5][MAX_VERTEX];
bool edge_in_5cdc[5][MAX_EDGE];
//int cycle_length_in_5cdc[5];
int number_of_circuits_in_5cdc[5];
int separate_circuits_in_5cdc[5][MAX_VERTEX][2 * MAX_VERTEX];
int separate_circuits_length_in_5cdc[5][MAX_VERTEX];
int vertex_count_in_5cdc[MAX_VERTEX];
int edge_count_in_5cdc[MAX_EDGE];
int edge_pair_count_in_5cdc[MAX_EDGE][MAX_EDGE];
int total_edge_count_in_5cdc;
//int edge_pair_count_in_5cdc[MAX_DEG * MAX_VERTEX];
Mask u5cdc_cycles[5];

int o5cdc_aggregated_solutions;
int all_o5cdc_solutions;
int all_5cdc_solutions;
int same_cycles_different_orientations_in_5cdc;
vector<vector<int>> all_circuits_in_5cdc;
int edge_orientation_count_in_5cdc[MAX_EDGE][2];
int orientations_in_5cdc[5 * MAX_VERTEX];
int layer_in_5cdc[5 * MAX_VERTEX];
int layer_flow_in_5cdc[5][MAX_EDGE];

int u33pp_count;

/*********************************Methods*********************************/

void gen_33pp_from_o5cdc(Graph& graph) {
    for (int i = 0; i < u33pp_indices.size(); ++i) {
        Mask m1 = 0;
        Mask m2 = 0;
        for (int e = 0; e < graph.number_of_edges; ++e) {
            bool in[5];
            for (int j = 0; j < 5; ++j) {
                in[j] = (BIT(e) & u5cdc_cycles[u33pp_indices[i][j]]) > 0;
            }
            if ((in[0] && in[1]) || (in[2] && in[3]) || in[4]) {
                m1 |= BIT(e);
                m2 |= BIT(e);
            }
            if ((in[0] && in[2]) || (in[1] && in[3])) {
                m1 |= BIT(e);
            }
            if ((in[0] && in[3]) || (in[1] && in[2])) {
                m2 |= BIT(e);
            }
        }
        if (m1 > m2) {
            swap(m1, m2);
        }
        u33pp_pairs.insert(make_pair(m1, m2));
    }
    for (int i = 0; i < 5; ++i) {
        u333pp_cycles_from_o5cdc.insert(u5cdc_cycles[i]);
        if (graph.all_even_cycles.find(u5cdc_cycles[i]) != graph.all_even_cycles.end()) {
            u333pp_even_cycles_from_o5cdc.insert(u5cdc_cycles[i]);
        }
    }
}

void winding(Graph& graph, int dom_idx) {
  cerr << "o5cdc" << endl;

  Mask dom;
  for (int i = 0; i < all_circuits_in_5cdc.size(); ++i) {
    cerr << "layer: " << layer_in_5cdc[i] << "; circ: " << i << "; or: " << orientations_in_5cdc[i] << "; ";
    if (layer_in_5cdc[i] == dom_idx) {
      cerr << "DOM (";
      dom = u5cdc_cycles[layer_in_5cdc[i]];
      cerr << dom << "); ";
    }

    int di = 1;
    int first = 0;
    int last = all_circuits_in_5cdc[i].size() - 1;
    if (orientations_in_5cdc[i] == 1) {
      di = -1;
      first = last;
      last = 0;
    }

    int w = 0;
    int init_v_idx = -1;
    int cur_v_idx = -1;
    for (int vi = first; vi != last; vi += di) {
      int v = all_circuits_in_5cdc[i][vi];
      cerr << v << " ";
    }
    cerr << all_circuits_in_5cdc[i][first] << endl;
  }

  bool in_dom[MAX_VERTEX];
  int v_idx[MAX_VERTEX];
  for (int v = 0; v < graph.number_of_vertices; ++v) {
    in_dom[v] = false;
    v_idx[v] = -1;
  }
  int len = -1;
  for (int i = 0; i < all_circuits_in_5cdc.size(); ++i) {
    if (layer_in_5cdc[i] != dom_idx) {
      continue;
    }
    len = all_circuits_in_5cdc[i].size() - 1;
    for (int vi = 0; vi < len; ++vi) {
      int v = all_circuits_in_5cdc[i][vi];
      in_dom[v] = true;
      assert(v_idx[v] == -1);
      v_idx[v] = vi;
      if (orientations_in_5cdc[i] == 0) {
        v_idx[v] = len - 1 - vi;
      }
    }
  }

  int total_w = 0;
  for (int i = 0; i < all_circuits_in_5cdc.size(); ++i) {
    if (layer_in_5cdc[i] == dom_idx) {
      continue;
    }

    int di = 1;
    int first = 0;
    int last = all_circuits_in_5cdc[i].size() - 1;
    if (orientations_in_5cdc[i] == 1) {
      di = -1;
      first = last;
      last = 0;
    }

    int w = 0;
    cerr << "ws: ";
    int init_v_idx = -1;
    int cur_v_idx = -1;
    for (int vi = first; vi != last; vi += di) {
      int v = all_circuits_in_5cdc[i][vi];
      if (!in_dom[v]) {
        continue;
      }
      if (init_v_idx == -1) {
        init_v_idx = v_idx[v];
        cur_v_idx = init_v_idx;
      } else {
        int next_v_idx = v_idx[v];
        int d1 = (len + init_v_idx - cur_v_idx) % len;
        int d2 = (len + next_v_idx - cur_v_idx) % len;
        assert(d2 != 0);
        assert(d1 != d2);
        if (d1 != 0 && d1 < d2) {
          ++w;
        }
        //cerr << len << " " << init_v_idx << " " << cur_v_idx << " " << next_v_idx << ": " << d1 << " " << d2 << endl;
        cur_v_idx = next_v_idx;
      }
    }
    cerr << w << " ";
    total_w += w;
  }
  cerr << "; total: " << total_w << endl << endl;
  graph.windings[dom][total_w] += 1;
}

bool orient_5cdc(Graph& graph, int cur_circuit) {
    if (cur_circuit == all_circuits_in_5cdc.size()) {
        for (int i = 0; i < 5; ++i) {
            if (graph.all_dominating_circuits.find(u5cdc_cycles[i]) != graph.all_dominating_circuits.end()) {
              winding(graph, i);
            }
        }

        set<pair<int, int>> edge_pairs;
        set<pair<int, int>> edge_pairs_op;
        set<int> first_edge;
        set<int> second_edge;
        for (int i = 0; i < all_circuits_in_5cdc.size(); ++i) {
          int last_idx = all_circuits_in_5cdc[i].size() - 1;
          assert(all_circuits_in_5cdc[i][0] == all_circuits_in_5cdc[i][last_idx]);
          for (int vi = 0; vi < last_idx; ++vi) {
            int v0 = all_circuits_in_5cdc[i][vi];
            int v1 = -1;
            if (vi > 0) {
              v1 = all_circuits_in_5cdc[i][vi - 1];
            } else {
              v1 = all_circuits_in_5cdc[i][last_idx - 1];
            }
            int v2 = all_circuits_in_5cdc[i][vi + 1];
            assert(v0 != v1);
            assert(v0 != v2);
            int e1 = graph.vv2e[v0][v1];
            int e2 = graph.vv2e[v0][v2];
            if (orientations_in_5cdc[i] == 1) {
              swap(e1, e2);
            }
            edge_pairs.insert(make_pair(e1, e2));
            edge_pairs_op.insert(make_pair(e2, e1));
            first_edge.insert(e1);
            second_edge.insert(e2);
          }
        }

        assert(edge_pairs.size() == 3 * graph.number_of_vertices);
        assert(first_edge.size() == graph.number_of_edges);
        assert(second_edge.size() == graph.number_of_edges);
        graph.all_o5cdc_edge_pairs.insert(edge_pairs);
        graph.all_o5cdc_edge_pairs.insert(edge_pairs_op);
        ++same_cycles_different_orientations_in_5cdc;
        assert(same_cycles_different_orientations_in_5cdc == 1);
        gen_33pp_from_o5cdc(graph);
        return false;//true;
    }
    int max_orientation = 1;
    if (cur_circuit == 0) {
        max_orientation = 0;
    }
    for (int orientation = 0; orientation <= max_orientation; ++orientation) {
        orientations_in_5cdc[cur_circuit] = orientation;
        int vi = 0;
        while (vi < all_circuits_in_5cdc[cur_circuit].size() - 1) {
            int v1 = all_circuits_in_5cdc[cur_circuit][vi];
            int v2 = all_circuits_in_5cdc[cur_circuit][vi + 1];
            int ei = graph.vv2e[v1][v2];
            int cur_edge_orientation = orientation;
            if (v1 > v2) {
                cur_edge_orientation = 1 - cur_edge_orientation;
            }
            if (edge_orientation_count_in_5cdc[ei][cur_edge_orientation] == 1) {
                break;
            }
            ++edge_orientation_count_in_5cdc[ei][cur_edge_orientation];
            ++vi;
        }
        if (vi == all_circuits_in_5cdc[cur_circuit].size() - 1) {
            if (orient_5cdc(graph, cur_circuit + 1)) {
                return true;
            }
        }
        --vi;
        while (vi >= 0) {
            int v1 = all_circuits_in_5cdc[cur_circuit][vi];
            int v2 = all_circuits_in_5cdc[cur_circuit][vi + 1];
            int ei = graph.vv2e[v1][v2];
            int cur_edge_orientation = 0;
            if (v1 > v2) {
                cur_edge_orientation = 1 - cur_edge_orientation;
            }
            if (orientation == 1) {
                cur_edge_orientation = 1 - cur_edge_orientation;
            }
            --edge_orientation_count_in_5cdc[ei][cur_edge_orientation];
            --vi;
        }
    }
    return false;
}

bool check_orientability_5cdc(Graph& graph) {
    vector<vector<int>> circuits;
    all_circuits_in_5cdc.clear();
    for (int i = 0; i < 5; ++i) {
        circuits = graph.cycles_as_circuits[u5cdc_cycles[i]];
        for (const auto& circuit : circuits) {
            layer_in_5cdc[all_circuits_in_5cdc.size()] = i;
            all_circuits_in_5cdc.push_back(circuit);
        }
    }
    //cerr << "all circuits: " << all_circuits_in_5cdc.size() << endl;
    for (int e = 0; e < graph.number_of_edges; ++e) {
        for (int orientation = 0; orientation < 2; ++orientation) {
            edge_orientation_count_in_5cdc[e][orientation] = 0;
        }
    }
    same_cycles_different_orientations_in_5cdc = 0;
    orient_5cdc(graph, 0);
    if (same_cycles_different_orientations_in_5cdc > 0) {
        ++o5cdc_aggregated_solutions;
        all_o5cdc_solutions += same_cycles_different_orientations_in_5cdc;
        graph.all_o5cdc.insert(cur_5cdc);
    }
    /*if (same_cycles_different_orientations_in_5cdc > 1) {
        cerr << "found " << same_cycles_different_orientations_in_5cdc << " same cycles different orientations for 5cdc" << endl;
    }*/
    return false;
}

void gen_33pp_from_5cdc_with_orientations(Graph& graph) {
    // TODO: I count every solution here 4 times
    // remove this redundancy
    for (int i = 0; i < u33pp_indices.size(); ++i) {
        Mask common_cycle = 0;
        Mask m1 = 0;
        Mask m2 = 0;
        int flows[MAX_EDGE][4];
        int f1[MAX_EDGE];
        int f2[MAX_EDGE];
        bool edge_fail = false;
        int failed_edge = NONE;
        for (int e = 0; e < graph.number_of_edges; ++e) {
            for (int j = 0; j < 4; ++j) {
                flows[e][j] = layer_flow_in_5cdc[u33pp_indices[i][j]][e];
            }
            f1[e] = (flows[e][0] - flows[e][1]) + (flows[e][2] - flows[e][3]);
            if (f1[e] != 0) {
                m1 |= BIT(e);
            }
            f2[e] = (flows[e][0] - flows[e][1]) - (flows[e][2] - flows[e][3]);
            if (f2[e] != 0) {
                m2 |= BIT(e);
            }

            if (abs(f1[e]) > 2 || abs(f2[e]) > 2) {
              edge_fail = true;
              failed_edge = e;
            }
            //assert(abs(f1[e]) <= 2 && abs(f2[e]) <= 2);

            if (abs(f1[e]) == 1 &&
                abs((flows[e][0] - flows[e][1]) - (flows[e][2] - flows[e][3])) == 1) {
              common_cycle |= BIT(e);
            }
        }

        if (edge_fail) {
          graph.print();
          cerr << "cycles" << endl;
          for (int ii = 0; ii < 5; ++ii) {
            const auto& circuits = graph.cycles_as_circuits[u5cdc_cycles[ii]];
            for (int idx = 0; idx < circuits.size(); ++idx) {
              for (const auto& v : circuits[idx]) {
                cerr << v << " ";
              }
              cerr << "; ";
            }
            if (u5cdc_cycles[ii] == common_cycle) {
              cerr << "is common!" << endl;
            }
            cerr << endl;
          }
          for (int e = 0; e < graph.number_of_edges; ++e) {
            cerr << "e: " << e << " (" << graph.e2v[e][0] << ", " << graph.e2v[e][1] << ") " <<
                ((common_cycle & BIT(e)) > 0) << " " << ((m1 & BIT(e)) > 0) << " " <<
                ((m2 & BIT(e)) > 0) << "; " << f1[e] << " " << f2[e] << "; " <<
                flows[e][0] << " " << flows[e][1] << " " << flows[e][2] << " " << flows[e][3] << "; " <<
                edge_count_in_5cdc[e] << " ";
            if (e == failed_edge) {
              cerr << "ZIS";
            }
            cerr << endl;
          }
        }
        assert(!edge_fail);

        if ((m1 | m2) != BIT(graph.number_of_edges) - 1) {
            continue;
        }
        if (m1 > m2) {
            swap(m1, m2);
        }
        pair<Mask, pair<Mask, Mask>> u33pp = make_pair(common_cycle, make_pair(m1, m2));

        const Mask ignored_vertices = graph.ignored_vertices_by_dominating_cycle[common_cycle];
        for (int coef_pair = 0; coef_pair < nz5_from_33pp_coefs.size(); ++coef_pair) {
          vector<int> f;
          vector<int> f_mod;
          for (int e = 0; e < graph.number_of_edges; ++e) {
            int e_flow = (nz5_from_33pp_coefs[coef_pair][0] * f1[e] + nz5_from_33pp_coefs[coef_pair][1] * f2[e]) / 2;
            f.push_back(e_flow);
            assert(e_flow != 0);
            assert(abs(e_flow) < 5);
            int e_mod_flow = e_flow;
            if (e_mod_flow < 0) {
              e_mod_flow = e_mod_flow + 5;
            }
            f_mod.push_back(e_mod_flow);
          }
          graph.ignored_vertices_by_nz5[f].insert(ignored_vertices);
          graph.ignored_vertices_by_nzmod5[f_mod].insert(ignored_vertices);

          graph.from_nz5_to_33pp[f].insert(u33pp);
        }

        bool has_common_cycle = false;
        for (int i = 0; i < 5; ++i) {
            Mask cyc = u5cdc_cycles[i];
            if (cyc == common_cycle) {
              has_common_cycle = true;
            }
            if (graph.all_dominating_cycles.find(cyc) != graph.all_dominating_cycles.end()) {
              graph.u5cdc_with_domcyc_with_33pp_relaxed.insert(cur_5cdc);
            }
            if (graph.all_dominating_circuits.find(cyc) != graph.all_dominating_circuits.end()) {
              graph.u5cdc_with_domcirc_with_33pp_relaxed.insert(cur_5cdc);
            }
        }
        assert(has_common_cycle);

        graph.u5cdc_with_33pp.insert(cur_5cdc);
        if (graph.all_dominating_cycles.find(common_cycle) != graph.all_dominating_cycles.end()) {
          graph.u5cdc_with_domcyc_with_33pp.insert(cur_5cdc);
        }
        if (graph.all_dominating_circuits.find(common_cycle) != graph.all_dominating_circuits.end()) {
          graph.u5cdc_with_domcirc_with_33pp.insert(cur_5cdc);
        }

        graph.u5cdc_from_33pp[u33pp].insert(cur_5cdc);
        ++u33pp_count;
        graph.all_33pp_dominating_circuits.insert(common_cycle);

        if (graph.petersen_5cdc.find(cur_5cdc) != graph.petersen_5cdc.end()) {
          for (const int colouring_idx : graph.petersen_5cdc_to_colouring_idx[cur_5cdc]) {
            graph.dominating_petersens.insert(colouring_idx);
          }
          graph.pet_5cdc_dominating_circuits.insert(common_cycle);
        }

        /*if (u33pp_count == 1) {
          graph.print();
          for (int e = 0; e < graph.number_of_edges; ++e) {
            cerr << "e: " << e << " (" << graph.e2v[e][0] << ", " << graph.e2v[e][1] << ") " <<
                ((common_cycle & BIT(e)) > 0) << " " << ((m1 & BIT(e)) > 0) << " " << ((m2 & BIT(e)) > 0) << endl;
          }

          cerr << "petersen structure" << endl;
          for (int e = 0; e < graph.number_of_edges; ++e) {
            cerr << "e: " << e << " (" << graph.e2v[e][0] << ", " << graph.e2v[e][1] << "); flows: ";
            for (int j = 0; j < 4; ++j) {
              cerr << flows[e][j] << " ";
            }
            cerr << endl;
          }
          cerr << "colours" << endl;
          const auto colouring_idx = graph.petersen_5cdc_to_colouring_idx[cur_5cdc].begin();
          for (int e = 0; e < graph.number_of_edges; ++e) {
            cerr << "e: " << e << " (" << graph.e2v[e][0] << ", " << graph.e2v[e][1] << ") " <<
                "; colour: " << graph.normal5_colourings[*colouring_idx][e] << endl;
          }
          cerr << endl << endl;
        }*/
    }
}

void gen_33pp_from_5cdc(Graph& graph, int cur_circuit) {
    if (cur_circuit == all_circuits_in_5cdc.size()) {
        gen_33pp_from_5cdc_with_orientations(graph);
        return;
    }
    int max_orientation = 1;
    if (cur_circuit == 0) {
        max_orientation = 0;
    }
    for (int orientation = 0; orientation <= max_orientation; ++orientation) {
        orientations_in_5cdc[cur_circuit] = orientation;
        int vi = 0;
        while (vi < all_circuits_in_5cdc[cur_circuit].size() - 1) {
            int v1 = all_circuits_in_5cdc[cur_circuit][vi];
            int v2 = all_circuits_in_5cdc[cur_circuit][vi + 1];
            int ei = graph.vv2e[v1][v2];
            int cur_edge_orientation = orientation;
            if (v1 > v2) {
                cur_edge_orientation = 1 - cur_edge_orientation;
            }
            //if (edge_orientation_count_in_5cdc[ei][cur_edge_orientation] == 1) {
                // TODO: log this event as inconsistency between layers
            //}
            if (cur_edge_orientation == 0) {
                layer_flow_in_5cdc[layer_in_5cdc[cur_circuit]][ei] += 1;
            } else {
                layer_flow_in_5cdc[layer_in_5cdc[cur_circuit]][ei] -= 1;
            }

            ++edge_orientation_count_in_5cdc[ei][cur_edge_orientation];
            ++vi;
        }

        gen_33pp_from_5cdc(graph, cur_circuit + 1);

        --vi;
        while (vi >= 0) {
            int v1 = all_circuits_in_5cdc[cur_circuit][vi];
            int v2 = all_circuits_in_5cdc[cur_circuit][vi + 1];
            int ei = graph.vv2e[v1][v2];
            int cur_edge_orientation = 0;
            if (v1 > v2) {
                cur_edge_orientation = 1 - cur_edge_orientation;
            }
            if (orientation == 1) {
                cur_edge_orientation = 1 - cur_edge_orientation;
            }
            if (cur_edge_orientation == 0) {
                layer_flow_in_5cdc[layer_in_5cdc[cur_circuit]][ei] -= 1;
            } else {
                layer_flow_in_5cdc[layer_in_5cdc[cur_circuit]][ei] += 1;
            }
            --edge_orientation_count_in_5cdc[ei][cur_edge_orientation];
            --vi;
        }
    }
    return;
}

bool build_5cdc(int cur_vertex, int min_possible_edge, Graph& graph, int cur_cycle_layer, int prev_edge, bool only_find) {
    // maybe TODO
    // fasten 5cdc generation
    // by changing edge order in every vertex
    // such that we generate circuits as fast as possible

    if (cur_vertex == start_vertex_in_5cdc[cur_cycle_layer][number_of_circuits_in_5cdc[cur_cycle_layer]]) {
        bool has_leaf = false;
        for (int v = 0; v < graph.number_of_vertices; ++v) {
            int deg = MAX_DEG;
            for (int j = 0; j < MAX_DEG; ++j) {
                if (edge_count_in_5cdc[graph.v2e[v][j]] == 2) {
                    --deg;
                }
            }
            if (deg == 1) {
                has_leaf = true;
                break;
            }
        }
        if (!has_leaf) {
            Mask bit_cycle = 0;
            for (int e = 0; e < graph.number_of_edges; ++e) {
                if (edge_in_5cdc[cur_cycle_layer][e]) {
                    bit_cycle += BIT(e);
                }
            }
            assert(graph.all_cycles.find(bit_cycle) != graph.all_cycles.end());

            u5cdc_cycles[cur_cycle_layer] = bit_cycle;
            if (start_or_continue_build_5cdc(0, graph, cur_cycle_layer + 1, only_find)) {
              return true;
            }
        }
    }
    const int circuit_index = number_of_circuits_in_5cdc[cur_cycle_layer];
    const int cur_circuit_len = separate_circuits_length_in_5cdc[cur_cycle_layer][circuit_index];
    if (cur_vertex == start_vertex_in_5cdc[cur_cycle_layer][circuit_index] && cur_circuit_len > 0) {
        if (start_or_continue_build_5cdc(min_possible_edge + 1, graph, cur_cycle_layer, only_find)) {
          return true;
        }
        return false;
    }

    for (int j = 0; j < MAX_DEG; ++j) {
        const int next_vertex = graph.v2v[cur_vertex][j];
        const int ei = graph.v2e[cur_vertex][j];

        // conditions
        if (ei < min_possible_edge || edge_in_5cdc[cur_cycle_layer][ei] || vertex_in_5cdc[cur_cycle_layer][next_vertex]) {
            continue;
        }
        // specific 5cdc conditions
        if (edge_count_in_5cdc[ei] == 2) {
            continue;
        }
        if (edge_pair_count_in_5cdc[prev_edge][ei] == 1) {
            continue;
        }

        // initialization
        separate_circuits_in_5cdc[cur_cycle_layer][circuit_index][cur_circuit_len] = next_vertex;
        //++cycle_length_in_5cdc[cur_cycle_layer];
        ++separate_circuits_length_in_5cdc[cur_cycle_layer][circuit_index];
        vertex_in_5cdc[cur_cycle_layer][next_vertex] = true;
        edge_in_5cdc[cur_cycle_layer][ei] = true;
        ++vertex_count_in_5cdc[next_vertex];
        ++edge_count_in_5cdc[ei];
        ++edge_pair_count_in_5cdc[prev_edge][ei];
        ++edge_pair_count_in_5cdc[ei][prev_edge];
        ++total_edge_count_in_5cdc;

        // recursion
        if (build_5cdc(next_vertex, min_possible_edge, graph, cur_cycle_layer, ei, only_find)) {
          return true;
        }

        // undo
        //--cycle_length_in_5cdc[cur_cycle_layer];
        --separate_circuits_length_in_5cdc[cur_cycle_layer][circuit_index];
        vertex_in_5cdc[cur_cycle_layer][next_vertex] = false;
        edge_in_5cdc[cur_cycle_layer][ei] = false;
        --vertex_count_in_5cdc[next_vertex];
        --edge_count_in_5cdc[ei];
        --edge_pair_count_in_5cdc[prev_edge][ei];
        --edge_pair_count_in_5cdc[ei][prev_edge];
        --total_edge_count_in_5cdc;
    }
    return false;
}

bool start_or_continue_build_5cdc(int possible_edge_lower_bound, Graph& graph, int cur_cycle_layer, bool only_find) {
    if (cur_cycle_layer == 4) {
        u5cdc_cycles[4] = (BIT(graph.number_of_edges) - 1) * 2 -
            (u5cdc_cycles[0] + u5cdc_cycles[1] + u5cdc_cycles[2] + u5cdc_cycles[3]);
        if (graph.all_cycles.find(u5cdc_cycles[4]) == graph.all_cycles.end()) {
            return false;
        }

        // VERY IMPORTANT TO FILTER OUT NON-SOLUTIONS
        // TODO: write a better checker
        for (int e = 0; e < graph.number_of_edges; ++e) {
          if ((BIT(e) & u5cdc_cycles[4]) != 0) {
            if (edge_count_in_5cdc[e] != 1) {
              return false;
            }
          } else {
            if (edge_count_in_5cdc[e] != 2) {
              return false;
            }
          }
        }

        cur_5cdc.clear();

        for (int i = 0; i < 5; ++i) {
            cur_5cdc.insert(u5cdc_cycles[i]);
        }

        if (all_5cdc.find(cur_5cdc) != all_5cdc.end()) {
            return false;
        }

        for (int i = 0; i < 5; ++i) {
          for (int j = i + 1; j < 5; ++j) {
            for (int k = j + 1; k < 5; ++k) {
              set<Mask> triple;
              triple.insert(u5cdc_cycles[i]);
              triple.insert(u5cdc_cycles[j]);
              triple.insert(u5cdc_cycles[k]);
              //cerr << "triple: " << u5cdc_cycles[i] << " " << u5cdc_cycles[j] << " " << u5cdc_cycles[k] << endl;
              graph.all_5cdc_by_triples[triple].insert(cur_5cdc);
            }
          }
        }

        all_5cdc.insert(cur_5cdc);
        graph.all_5cdc.insert(cur_5cdc);

        if (only_find) {
            return false;
        }

        vector<vector<int>> circuits;
        all_circuits_in_5cdc.clear();
        for (int i = 0; i < 5; ++i) {
            circuits = graph.cycles_as_circuits[u5cdc_cycles[i]];
            for (const auto& circuit : circuits) {
                layer_in_5cdc[all_circuits_in_5cdc.size()] = i;
                all_circuits_in_5cdc.push_back(circuit);
            }
        }

        for (int e = 0; e < graph.number_of_edges; ++e) {
            for (int i = 0; i < 5; ++i) {
                layer_flow_in_5cdc[i][e] = 0;
            }
        }

        u33pp_count = 0;

        for (int i = 0; i < 5; ++i) {
          Mask cyc = u5cdc_cycles[i];
          if (graph.all_dominating_cycles.find(cyc) != graph.all_dominating_cycles.end()) {
            graph.all_5cdc_with_domcyc.insert(cur_5cdc);
          }
          if (graph.all_dominating_circuits.find(cyc) != graph.all_dominating_circuits.end()) {
            graph.all_5cdc_with_domcirc.insert(cur_5cdc);
          }
        }

        // FIXME: why do i have some faulty solutions here?

        gen_33pp_from_5cdc(graph, 0);
        return false; // FIXME

        /*if (graph.petersen_5cdc.find(cur_5cdc) != graph.petersen_5cdc.end()) {
            gen_33pp_from_5cdc(graph, 0);
            if (u33pp_count > 0) {
              //cerr << "pet 5cdc: " << u33pp_count << endl;
              //if (graph.dominating_petersens.size() == graph.normal5_colourings.size()) {
              if (graph.pet_5cdc_dominating_circuits.size() == graph.all_dominating_circuits.size()) {
                return true;
              }
            }
        }*/

        // cycle metrics
        for (int i = 0; i < 5; ++i) {
            all_cycles_from_5cdc.insert(u5cdc_cycles[i]);
            if (graph.all_full_cycles.find(u5cdc_cycles[i]) != graph.all_full_cycles.end()) {
                all_full_cycles_from_5cdc.insert(u5cdc_cycles[i]);
            }
            if (graph.all_even_cycles.find(u5cdc_cycles[i]) != graph.all_even_cycles.end()) {
                all_even_cycles_from_5cdc.insert(u5cdc_cycles[i]);
            }
            if (graph.all_circuits.find(u5cdc_cycles[i]) != graph.all_circuits.end()) {
                all_circuits_from_5cdc.insert(u5cdc_cycles[i]);
            }

        }

        if (check_orientability_5cdc(graph)) {
          return true;
        }
        return false;
    }

    if (possible_edge_lower_bound == 0) {
        if (2 * graph.number_of_edges - total_edge_count_in_5cdc > graph.number_of_vertices * (5 - cur_cycle_layer)) {
            return false;
        }

        // specific 5cdc conditions
        if (cur_cycle_layer == 3) {
            for (int v = 0; v < graph.number_of_vertices; ++v) {
                if (vertex_count_in_5cdc[v] == 0) {
                    return false;
                }
            }
        }
    }

    for (int min_possible_edge = possible_edge_lower_bound; min_possible_edge < graph.number_of_edges; ++min_possible_edge) {
        int v1 = graph.e2v[min_possible_edge][0];
        int v2 = graph.e2v[min_possible_edge][1];

        // conditions
        if (edge_in_5cdc[cur_cycle_layer][min_possible_edge] || vertex_in_5cdc[cur_cycle_layer][v1] || vertex_in_5cdc[cur_cycle_layer][v2]) {
            continue;
        }
        // specific 5cdc conditions
        if (edge_count_in_5cdc[min_possible_edge] == 2) {
            continue;
        }

        // TODO: if (possible_edge_lower_bound == 0) {
        // remember this min_possible_edge
        // for this layer
        // and use it later for cycle comparison
        // with same min_possible_edge
        // }

        // initialization
        ++number_of_circuits_in_5cdc[cur_cycle_layer];
        //++cycle_length_in_5cdc[cur_cycle_layer];
        separate_circuits_in_5cdc[cur_cycle_layer][number_of_circuits_in_5cdc[cur_cycle_layer]][0] = v2;
        separate_circuits_length_in_5cdc[cur_cycle_layer][number_of_circuits_in_5cdc[cur_cycle_layer]] = 1;
        start_vertex_in_5cdc[cur_cycle_layer][number_of_circuits_in_5cdc[cur_cycle_layer]] = v1;
        vertex_in_5cdc[cur_cycle_layer][v2] = true;
        edge_in_5cdc[cur_cycle_layer][min_possible_edge] = true;
        ++vertex_count_in_5cdc[v2];
        ++edge_count_in_5cdc[min_possible_edge];
        ++total_edge_count_in_5cdc;

        // recursion
        if (build_5cdc(v2, min_possible_edge, graph, cur_cycle_layer, min_possible_edge, only_find)) {
          return true;
        }

        // undo
        --number_of_circuits_in_5cdc[cur_cycle_layer];
        //--cycle_length_in_5cdc[cur_cycle_layer];
        vertex_in_5cdc[cur_cycle_layer][graph.e2v[min_possible_edge][1]] = false;
        edge_in_5cdc[cur_cycle_layer][min_possible_edge] = false;
        --vertex_count_in_5cdc[v2];
        --edge_count_in_5cdc[min_possible_edge];
        --total_edge_count_in_5cdc;

        if (possible_edge_lower_bound == 0) {
            break;
        }
    }
    return false;
}

bool prepare_build_5cdc(Graph& graph, bool only_find) {
    for (int layer = 0; layer < 5; ++layer) {
        //cycle_length_in_5cdc[layer] = 0;
        separate_circuits_length_in_5cdc[layer][0] = 0;

        for (int v = 0; v < graph.number_of_vertices; ++v) {
            vertex_in_5cdc[layer][v] = false;
        }
        for (int e = 0; e < graph.number_of_edges; ++e) {
            edge_in_5cdc[layer][e] = false;
        }
        number_of_circuits_in_5cdc[layer] = -1;
    }

    for (int v = 0; v < graph.number_of_vertices; ++v) {
        vertex_count_in_5cdc[v] = 0;
    }
    for (int e = 0; e < graph.number_of_edges; ++e) {
        edge_count_in_5cdc[e] = 0;
        for (int e2 = 0; e2 < graph.number_of_edges; ++e2) {
          edge_pair_count_in_5cdc[e][e2] = 0;
        }
    }
    total_edge_count_in_5cdc = 0;

    return start_or_continue_build_5cdc(0, graph, 0, only_find);
}

void find_all_o5cdc(Graph& graph, bool only_find) {
    graph.u5cdc_from_33pp.clear();
    all_5cdc.clear();
    all_5cdc_solutions = 0;
    all_o5cdc_solutions = 0;
    u33pp_pairs.clear();
    u333pp_cycles_from_o5cdc.clear();
    u333pp_even_cycles_from_o5cdc.clear();

    all_cycles_from_5cdc.clear();
    all_full_cycles_from_5cdc.clear();
    all_even_cycles_from_5cdc.clear();
    all_circuits_from_5cdc.clear();

    prepare_build_5cdc(graph, only_find);

    cerr << "all 5cdc solutions: " << graph.all_5cdc.size() << endl;
    cerr << "with 33pp: " << graph.u5cdc_with_33pp.size() << endl;

    cerr << "all 5cdc solutions with domcyc: " << graph.all_5cdc_with_domcyc.size() << endl;
    cerr << "with 33pp: " << graph.u5cdc_with_domcyc_with_33pp.size() << endl;
    cerr << "with 33pp: " << graph.u5cdc_with_domcyc_with_33pp_relaxed.size() << endl;

    cerr << "all 5cdc solutions with domcirc: " << graph.all_5cdc_with_domcirc.size() << endl;
    cerr << "with 33pp: " << graph.u5cdc_with_domcirc_with_33pp.size() << endl;
    cerr << "with 33pp: " << graph.u5cdc_with_domcirc_with_33pp_relaxed.size() << endl;

    //cerr << "all o5cdc solutions: " << graph.all_o5cdc.size() << endl;
}

} // Exp5cdc
