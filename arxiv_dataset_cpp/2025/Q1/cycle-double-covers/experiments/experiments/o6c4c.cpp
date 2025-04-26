/*
 * File:   o6c4c.cpp
 * Author: Nikolay Ulyanov (ulyanick@gmail.com)
 *
 * TODO:
 *
 */

#include "graph.h"

#include "experiments/mnk_flows.h"

#include <cassert>
#include <unordered_set>
#include <unordered_map>
#include <set>
#include <map>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>
#include <random>

// #include <boost/graph/adjacency_list.hpp>
// #include <boost/graph/boyer_myrvold_planar_test.hpp>

using namespace std;


namespace Exp6c4c {

using namespace ExpMNKFlows; // TODO: rewrite?

//int mindiff1;
//int mindiff2;

int graphs_to_write = 3; // FIXME
Mask dom_circ;
int sol_count;
// int nz5_or_count;
// int nz5_or244_count;
// int nz5_or_not244_count;
// int nz5_unor244_count;
// int nz5_unor_not244_count;
set<vector<int>> nz5_flow_types;
set<vector<int>> nz5_flow_types244;
set<vector<int>> nz5_flow_types_not244;
set<vector<int>> nz5_flow_types_ors;
set<vector<int>> nz5_flow_types_ors244;
set<vector<int>> nz5_flow_types_ors_not244;
set<vector<int>> nz5_flow_types_unors;
set<vector<int>> nz5_flow_types_unors244;
set<vector<int>> nz5_flow_types_unors_not244;

set<int> nz5_sums;
set<int> nz5_sums244;
set<int> nz5_sums_not244;
set<int> oriented_nz5_sums;
set<int> oriented244_nz5_sums;
set<int> oriented_not244_nz5_sums;
set<int> unoriented_nz5_sums;
set<int> unoriented244_nz5_sums;
set<int> unoriented_not244_nz5_sums;

set<int> selected_triple;
set<set<int>> selected_triples;

map<set<int>, vector<bool>> vertex_in_244_cycle;
map<set<int>, vector<bool>> edge_in_244_cycle;

vector<int> rich_244_counts;

multiset<Mask> cur_6c4c;

unordered_set<Mask> v_minus_4_6c4c;
unordered_set<Mask> v_minus_4_unvertex_neib_masks;

vector<Mask> bit_cycles;
int o6c4c_aggregated_solutions;
int all_o6c4c_solutions;
int all_6c4c_solutions;
int same_cycles_different_orientations;
int all_nz_mod5_from_o6c4c;
int all_nz_mod6_from_o6c4c;
vector<vector<int>> all_circuits_in_6c4c;
vector<Mask> all_circuit_masks_in_6c4c;
int layer_vertex_to_circuit[6][MAX_VERTEX];
int edge_orientation_count[MAX_EDGE][2];
int orientations[6 * MAX_VERTEX];
int layer[6 * MAX_VERTEX];
vector<vector<int>> cur_flows;
int cur_flow[MAX_EDGE];
int cur_flow_mod5[MAX_EDGE];
int cur_flow_modb[MAX_EDGE];
int max_layer[MAX_EDGE];
int layer_flow[6][MAX_EDGE];
set<set<int>> layer_pairings[MAX_VERTEX];
vector<int> layer_weights = {0, 0, 0, 0, 0, 0};
vector<int> layer_weights_nz_mod5 = {0, 0, 0, 0, 0, 0};
vector<int> layer_weights_nz_mod6 = {0, 0, 0, 0, 0, 0};
vector<int> layer_weights_nz_modb = {0, 0, 0, 0, 0, 0};
int u6c4c_oriented_parity;
int s0, s1, s2, s2uu;
int total_rich_count, total_poor_count, even_rich_matchings;
int odd_poor_2_factors, odd_poor_comps_2_factors;
int odd_rich_2_factors, odd_rich_comps_2_factors;
int odd_poor_matchings, odd_poor_comps_matchings;
int odd_rich_matchings, odd_rich_comps_matchings;
set<set<set<int>>> vertex_types;
map<set<int>, int> oriented_vertices_to_s0_parity;

int poor_component[MAX_VERTEX];
int rich_component[MAX_VERTEX];
int edge_poor_component[MAX_EDGE];
int edge_rich_component[MAX_EDGE];
int total_poor_comps, total_rich_comps;

string prof;
unordered_map<Mask, int> two_factor_count;

bool has_nz_mod5_flow = false;
bool has_nz_mod6_flow = false;
bool has_nz_modb_flow = false;
bool has_int_nz5_flow = false;
bool has_int_nz6_flow = false;
//bool has_both_nz_mod_flow = false;
bool has_dominating_circuit = false;
map<pair<int, int>, int> edge_pairs;
map<pair<int, int>, set<int>> edge_pair_layers;
set<int> oriented_vertices;
Mask oriented_vertices_mask;
set<set<int>> all_oriented_vertices;
set<int> ors;
bool u6c4c_edge_is_poor[MAX_EDGE];
Mask poor_mask = 0;
int edge_type[MAX_EDGE];

vector<int> any_chords_frequency;
vector<int> poor_chords_frequency;
vector<int> rich_chords_frequency;
vector<int> t1_chords_frequency;
vector<int> t2_chords_frequency;
vector<int> t3_chords_frequency;
vector<int> t4_chords_frequency;
vector<int> circuit_count_by_layer;
vector<int> chord_count_by_layer;
vector<int> t4_chord_count_by_layer;
vector<int> antichord_count_by_layer;

bool o6c4c_always_1[MAX_EDGE];
bool o6c4c_always_2[MAX_EDGE];
bool o6c4c_always_3[MAX_EDGE];
bool o6c4c_always_4[MAX_EDGE];
bool o6c4c_never_1[MAX_EDGE];
bool o6c4c_never_2[MAX_EDGE];
bool o6c4c_never_3[MAX_EDGE];
bool o6c4c_never_4[MAX_EDGE];

map<vector<int>, int> ch0l;
map<vector<int>, int> ch1l;
map<vector<int>, int> ch1chl;
map<vector<int>, int> ch2l;

set<pair<int, int>> u6c4c_edge_pair_counts[MAX_EDGE];

int u33pp_solutions;
int all_33pp_solutions;

set<Mask> u333pp_cycles_from_o6c4c;

bool has_33pp_from_3pm;
bool has_333pp_from_3pm;

set<vector<int>> common_triples;
set<vector<int>> all_33pp_triples;

bool u6c4c_always_poor[MAX_EDGE];
bool u6c4c_always_rich[MAX_EDGE];
int u6c4c_min_poor;
int u6c4c_max_poor;

bool has_or_comb = false;
bool has_un_comb = false;

set<set<Mask>> all_6c4c_triples;
set<set<Mask>> all_o6c4c_triples;


int cdc_cover_count[MAX_EDGE];
int total_cdc_need_cover_count = 0;
bool two_cdcs_taken[MAX_VERTEX * MAX_DEG];
bool taken[MAX_VERTEX * MAX_DEG];
int taken_c1, taken_c2;
set<int> layers_c1;
set<int> layers_c2;
bool has_2cdcs;
int count_2cdcs;

set<map<int, set<int>>> two_cdcs_vertices;

vector<int> rich_edges_frequency;
vector<int> poor_edges_frequency;

map<multiset<multiset<int>>, int> all_ve_types;


/*********************************Methods*********************************/

// todo: merge with int flow function
bool find_nz_mod_from_o6c4c(Graph& graph, int mod, int cur_layer, bool both) {
  if (cur_layer == 6) {
    for (int e = 0; e < graph.number_of_edges; ++e) {
      if (cur_flow[e] % mod == 0) {
        return false;
      }
      if (both && (cur_flow[e] % (mod + 1) == 0)) {
        return false;
      }
    }
    if (mod == 5) {
      for (int c = 0; c < 6; ++c) {
        if (!both) {
          layer_weights_nz_mod5[c] = layer_weights[c];
        } else {
          layer_weights_nz_modb[c] = layer_weights[c];
        }
      }
      for (int e = 0; e < graph.number_of_edges; ++e) {
        if (!both) {
          cur_flow_mod5[e] = cur_flow[e];
        } else {
          cur_flow_modb[e] = cur_flow[e];
        }
      }

      ++all_nz_mod5_from_o6c4c;
      if (!both) {
        has_nz_mod5_flow = true;
      } else {
        has_nz_modb_flow = true;
      }
      graph.has_nz_mod5_from_o6c4c = true;

      vector<int> cur_flow_vec;
      for (int e = 0; e < graph.number_of_edges; ++e) {
        cur_flow_vec.push_back((mod * 6 + cur_flow[e]) % mod);
      }
      graph.oriented_vertices_by_nzmod5[cur_flow_vec].insert(oriented_vertices_mask);
    } else {
      assert(mod == 6);
      for (int c = 0; c < 6; ++c) {
        layer_weights_nz_mod6[c] = layer_weights[c];
      }
      ++all_nz_mod6_from_o6c4c;
      has_nz_mod6_flow = true;
    }
    // FIXMEFIXME
    // return false;
    return true;
  }

  for (int w = 0; w < mod; ++w) {
    layer_weights[cur_layer] = w;
    bool no_problems_with_flow = true;
    for (int e = 0; e < graph.number_of_edges; ++e) {
      if (w > 0) {
        cur_flow[e] += layer_flow[cur_layer][e];
      }
      if ((max_layer[e] == cur_layer) && (cur_flow[e] % mod == 0)) {
        no_problems_with_flow = false;
      }
    }
    if (no_problems_with_flow && find_nz_mod_from_o6c4c(graph, mod, cur_layer + 1, both)) {
      return true;
    }
  }
  for (int e = 0; e < graph.number_of_edges; ++e) {
    cur_flow[e] -= layer_flow[cur_layer][e] * (mod - 1);
  }
  return false;
}

bool find_int_nz_from_o6c4c_cleaner(Graph& graph, int mod, int cur_layer) {
  if (cur_layer == 6) {
    for (int e = 0; e < graph.number_of_edges; ++e) {
      if ((cur_flow[e] == 0) || (abs(cur_flow[e]) >= mod)) {
        return false;
      }
    }
    if (mod == 5) {
      has_int_nz5_flow = true;
      vector<int> cur_flow_vec;
      for (int e = 0; e < graph.number_of_edges; ++e) {
        cur_flow_vec.push_back(cur_flow[e]);
      }
      cur_flows.push_back(cur_flow_vec);
      graph.oriented_vertices_by_nz5[cur_flow_vec].insert(oriented_vertices_mask);
      return true;

      // int or_sum = 0;
      // int unor_sum = 0;
      // vector<int> types;
      // for (int v = 0; v < graph.number_of_vertices; ++v) {
      //   int max_f = 0;
      //   multiset<int> abses;
      //   for (int j = 0; j < MAX_DEG; ++j) {
      //     int v2 = graph.v2v[v][j];
      //     int f = cur_flow_vec[graph.v2e[v][j]];
      //     if (v2 < v) {
      //       f = -f;
      //     }
      //     if (abs(f) > abs(max_f)) {
      //       max_f = f;
      //     }
      //     abses.insert(abs(f));
      //   }
      //   int type = 0;
      //   for (const auto& t : abses) {
      //     type = type * 10 + t;
      //   }
      //   if (max_f < 0) {
      //     type = -type;
      //   }
      //   types.push_back(type);
      //   if ((oriented_vertices_mask & BIT(v)) == 0) {
      //     unor_sum += type;
      //   } else {
      //     or_sum += type;
      //   }
      // }
      // if (nz5_flow_types.find(types) != nz5_flow_types.end()) {
      //   return false;
      // }

      // oriented_nz5_sums.insert(or_sum);
      // unoriented_nz5_sums.insert(unor_sum);
    } else {
      assert(mod == 6);
      has_int_nz6_flow = true;
    }
    return true;
  }

  int threshold = (mod - 1);
  for (int e = 0; e < graph.number_of_edges; ++e) {
    cur_flow[e] += layer_flow[cur_layer][e] * (-(threshold + 1));
  }
  for (int w = -threshold; w <= threshold; ++w) {
    layer_weights[cur_layer] = w;
    bool no_problems_with_flow = true;
    for (int e = 0; e < graph.number_of_edges; ++e) {
      cur_flow[e] += layer_flow[cur_layer][e];
      if (no_problems_with_flow && (max_layer[e] == cur_layer) && ((cur_flow[e] == 0) || (abs(cur_flow[e]) >= mod))) {
        no_problems_with_flow = false;
      }
    }
    if (no_problems_with_flow && find_int_nz_from_o6c4c_cleaner(graph, mod, cur_layer + 1)) {
      return true;
    }
  }
  for (int e = 0; e < graph.number_of_edges; ++e) {
    cur_flow[e] -= layer_flow[cur_layer][e] * threshold;
  }
  return false;
}

bool find_int_nz_from_o6c4c(Graph& graph, int mod, int cur_layer) {
  if (cur_layer == 6) {
    for (int e = 0; e < graph.number_of_edges; ++e) {
      if ((cur_flow[e] == 0) || (abs(cur_flow[e]) >= mod)) {
        return false;
      }
    }
    if (mod == 5) {
      has_int_nz5_flow = true;
      vector<int> cur_flow_vec;
      for (int e = 0; e < graph.number_of_edges; ++e) {
        cur_flow_vec.push_back(cur_flow[e]);
      }
      cur_flows.push_back(cur_flow_vec);
      graph.oriented_vertices_by_nz5[cur_flow_vec].insert(oriented_vertices_mask);

      int or_sum = 0;
      int unor_sum = 0;
      vector<int> types;
      for (int v = 0; v < graph.number_of_vertices; ++v) {
        int max_f = 0;
        multiset<int> abses;
        for (int j = 0; j < MAX_DEG; ++j) {
          int v2 = graph.v2v[v][j];
          int f = cur_flow_vec[graph.v2e[v][j]];
          if (v2 < v) {
            f = -f;
          }
          if (abs(f) > abs(max_f)) {
            max_f = f;
          }
          abses.insert(abs(f));
        }
        int type = 0;
        for (const auto& t : abses) {
          type = type * 10 + t;
        }
        if (max_f < 0) {
          type = -type;
        }
        types.push_back(type);
        if ((oriented_vertices_mask & BIT(v)) == 0) {
          unor_sum += type;
        } else {
          or_sum += type;
        }
      }
      if (nz5_flow_types.find(types) != nz5_flow_types.end()) {
        return false;
      }

      oriented_nz5_sums.insert(or_sum);
      unoriented_nz5_sums.insert(unor_sum);
      // FIXME
      if (or_sum == 0) {//} || selected_triple.size() == 0) {
        return true;
      }

      if (or_sum != 0) {
        map<int, int> weight_counts;
        for (int c = 0; c < 6; ++c) {
          weight_counts[layer_weights[c]] += 1;
        }
        assert(weight_counts.size() == 4);
        int triple_weight = mod;
        for (const auto& wc : weight_counts) {
          if (wc.second == 3) {
            triple_weight = wc.first;
            break;
          }
        }
        assert(triple_weight != mod);
        // FIXME
        if (triple_weight != 0) {
          return false;
        }
        selected_triple.clear();
        for (int c = 0; c < 6; ++c) {
          if (layer_weights[c] == triple_weight) {
            selected_triple.insert(c);
          }
        }
        selected_triples.insert(selected_triple);
      }

      Mask unor244_mask = 0;
      Mask or244_mask = 0;
      Mask both244_mask = 0;
      int or_count_on_244_cycle = 0;
      for (int v = 0; v < graph.number_of_vertices; ++v) {
        if (!vertex_in_244_cycle[selected_triple][v]) {
          continue;
        }
        both244_mask += BIT(v);
        if ((oriented_vertices_mask & BIT(v)) != 0) {
          or244_mask += BIT(v);
          ++or_count_on_244_cycle;
        } else {
          unor244_mask += BIT(v);
        }
      }

      set<int> both_not244_abs_types;
      set<int> or_not244_abs_types;
      set<int> unor_not244_abs_types;
      for (int v = 0; v < graph.number_of_vertices; ++v) {
        if ((both244_mask & BIT(v)) != 0) {
          continue;
        }
        both_not244_abs_types.insert(abs(types[v]));
        if ((oriented_vertices_mask & BIT(v)) != 0) {
          or_not244_abs_types.insert(abs(types[v]));
        } else {
          unor_not244_abs_types.insert(abs(types[v]));
        }
      }
      assert(or_not244_abs_types.size() == 1);
      assert(unor_not244_abs_types.size() <= 2);
      assert(both_not244_abs_types.size() <= 2);

      set<int> or_abs_types;
      for (int v = 0; v < graph.number_of_vertices; ++v) {
        if ((oriented_vertices_mask & BIT(v)) == 0) {
          continue;
        }
        or_abs_types.insert(abs(types[v]));
      }
      if (or_abs_types.size() > 2) {
        cerr << "WOW2" << endl; // never seen this
      }
      // TODO: group oriented vertices by types
      // check that there is 1 or 2 abs-types
      // if there is 1 type - all of the vertices are outside the cycle, and difference
      // between counts of positive and negative values for this type is divisible by 3
      // if there is additional type - it has 2 vertices, both are on the cycle
      assert(or_count_on_244_cycle % 2 == 0);

      int both244_sum = 0;
      int both_not244_sum = 0;
      int or244_sum = 0;
      int or_not244_sum = 0;
      int unor244_sum = 0;
      int unor_not244_sum = 0;
      for (int v = 0; v < graph.number_of_vertices; ++v) {
        if ((both244_mask & BIT(v)) != 0) {
          both244_sum += types[v];
        } else {
          both_not244_sum += types[v];
        }
        if ((oriented_vertices_mask & BIT(v)) != 0) {
          if ((or244_mask & BIT(v)) != 0) {
            or244_sum += types[v];
          } else {
            or_not244_sum += types[v];
          }
        } else {
          if ((unor244_mask & BIT(v)) != 0) {
            unor244_sum += types[v];
          } else {
            unor_not244_sum += types[v];
          }
        }
      }
      assert(or244_sum == 0);
      cerr << "allsums: " <<
          both244_sum + both_not244_sum << " " << both244_sum << " " << both_not244_sum << " " <<
          or_sum << " " << or244_sum << " " << or_not244_sum << " " <<
          unor_sum << " " << unor244_sum << " " << unor_not244_sum << endl;

      nz5_sums.insert(both244_sum + both_not244_sum);
      nz5_sums244.insert(both244_sum);
      nz5_sums_not244.insert(both_not244_sum);
      oriented244_nz5_sums.insert(or244_sum);
      oriented_not244_nz5_sums.insert(or_not244_sum);
      unoriented244_nz5_sums.insert(unor244_sum);
      unoriented_not244_nz5_sums.insert(unor_not244_sum);

      vector<int> flow_types;
      vector<int> flow_types244;
      vector<int> flow_types_not244;
      vector<int> flow_types_ors;
      vector<int> flow_types_ors244;
      vector<int> flow_types_ors_not244;
      vector<int> flow_types_unors;
      vector<int> flow_types_unors244;
      vector<int> flow_types_unors_not244;
      // nz5_or_count = 0;
      // nz5_or244_count = 0;
      // nz5_or_not244_count = 0;
      // nz5_unor244_count = 0;
      // nz5_unor_not244_count = 0;
      for (int v = 0; v < graph.number_of_vertices; ++v) {
        flow_types.push_back(types[v]);
        if ((both244_mask & BIT(v)) != 0) {
          flow_types244.push_back(types[v]);
        } else {
          flow_types_not244.push_back(types[v]);
        }
        if ((oriented_vertices_mask & BIT(v)) != 0) {
          flow_types_ors.push_back(types[v]);
          // ++nz5_or_count;
          if ((or244_mask & BIT(v)) != 0) {
            // ++nz5_or244_count;
            flow_types_ors244.push_back(types[v]);
          } else {
            // ++nz5_or_not244_count;
            flow_types_ors_not244.push_back(types[v]);
          }
        } else {
          flow_types_unors.push_back(types[v]);
          if ((unor244_mask & BIT(v)) != 0) {
            // ++nz5_unor244_count;
            flow_types_unors244.push_back(types[v]);
          } else {
            // ++nz5_unor_not244_count;
            flow_types_unors_not244.push_back(types[v]);
          }
        }
      }
      nz5_flow_types.insert(flow_types);
      nz5_flow_types244.insert(flow_types244);
      nz5_flow_types_not244.insert(flow_types_not244);
      nz5_flow_types_ors.insert(flow_types_ors);
      nz5_flow_types_ors244.insert(flow_types_ors244);
      nz5_flow_types_ors_not244.insert(flow_types_ors_not244);
      nz5_flow_types_unors.insert(flow_types_unors);
      nz5_flow_types_unors244.insert(flow_types_unors244);
      nz5_flow_types_unors_not244.insert(flow_types_unors_not244);
      cerr << graph.number << ":" << sol_count << " - weights: ";
      multiset<int> weights;
      for (int c = 0; c < 6; ++c) {
        cerr << layer_weights[c] << " ";
      }
      cerr << endl;
      cerr << graph.number << ":" << sol_count << " - types: ";
      for (int v = 0; v < graph.number_of_vertices; ++v) {
        if ((oriented_vertices_mask & BIT(v)) != 0) {
          cerr << "or_";
        }
        if ((both244_mask & BIT(v)) != 0) {
          cerr << "cyc_";
        }

        cerr << v << ":" << types[v] << " ";
      }
      cerr << endl;
      cerr << graph.number << ":" << sol_count << " - ortypes: ";
      for (int v = 0; v < graph.number_of_vertices; ++v) {
        if ((oriented_vertices_mask & BIT(v)) == 0) {
          continue;
        }
        cerr << v << ":" << types[v] << " ";
      }
      cerr << endl;
      cerr << graph.number << ":" << sol_count << " - or244types: ";
      for (int v = 0; v < graph.number_of_vertices; ++v) {
        if ((or244_mask & BIT(v)) == 0) {
          continue;
        }
        cerr << v << ":" << types[v] << " ";
      }
      cerr << endl;
      cerr << graph.number << ":" << sol_count << " - or_not244types: ";
      for (int v = 0; v < graph.number_of_vertices; ++v) {
        if ((oriented_vertices_mask & BIT(v)) == 0) {
          continue;
        }
        if ((or244_mask & BIT(v)) != 0) {
          continue;
        }
        cerr << v << ":" << types[v] << " ";
      }
      cerr << endl;

      cerr << graph.number << ":" << sol_count << " - unortypes: ";
      for (int v = 0; v < graph.number_of_vertices; ++v) {
        if ((oriented_vertices_mask & BIT(v)) != 0) {
          continue;
        }
        cerr << v << ":" << types[v] << " ";
      }
      cerr << endl;
      cerr << graph.number << ":" << sol_count << " - unor244types: ";
      for (int v = 0; v < graph.number_of_vertices; ++v) {
        if ((unor244_mask & BIT(v)) == 0) {
          continue;
        }
        cerr << v << ":" << types[v] << " ";
      }
      cerr << endl;
      cerr << graph.number << ":" << sol_count << " - unor_not244types: ";
      for (int v = 0; v < graph.number_of_vertices; ++v) {
        if ((oriented_vertices_mask & BIT(v)) != 0) {
          continue;
        }
        if ((unor244_mask & BIT(v)) != 0) {
          continue;
        }
        cerr << v << ":" << types[v] << " ";
      }
      cerr << endl;
    } else {
      assert(mod == 6);
      has_int_nz6_flow = true;
    }
    return false;//true;
  }

  for (int e = 0; e < graph.number_of_edges; ++e) {
    cur_flow[e] += layer_flow[cur_layer][e] * (-mod);
  }
  for (int w = -(mod - 1); w < mod; ++w) {
    layer_weights[cur_layer] = w;
    bool no_problems_with_flow = true;
    for (int e = 0; e < graph.number_of_edges; ++e) {
      cur_flow[e] += layer_flow[cur_layer][e];
      if ((max_layer[e] == cur_layer) && ((cur_flow[e] == 0) || (abs(cur_flow[e]) >= mod))) {
        no_problems_with_flow = false;
      }
    }
    if (no_problems_with_flow && find_int_nz_from_o6c4c(graph, mod, cur_layer + 1)) {
      return true;
    }
  }
  for (int e = 0; e < graph.number_of_edges; ++e) {
    cur_flow[e] -= layer_flow[cur_layer][e] * (mod - 1);
  }
  return false;
}

// checks whether there exists dominating circuit, which goes through all oriented vertices
bool has_compatible_dominating_circuit(Graph& graph) {
  /*unordered_set<int> ignored_vertices;
  for (int v = 0; v < graph.number_of_vertices; ++v) {
    bool all_oriented = true;
    for (int j = 0; j < MAX_DEG; ++j) {
      if (oriented_vertices.find(graph.v2v[v][j]) == oriented_vertices.end()) {
        all_oriented = false;
        break;
      }
    }
    if (all_oriented && oriented_vertices.find(v) != oriented_vertices.end()) {
      cerr << "wut" << endl;
    } else if (all_oriented) {
      ignored_vertices.insert(v);
    }
  }*/
  int count = 0;
  for (const auto& c : graph.all_dominating_circuits) {
    bool has_all_oriented_vertices = true;
    //cerr << "oriented vertices size: " << oriented_vertices.size() << endl;
    for (const auto& v : oriented_vertices) {
      bool has_edge = false;
      for (int j = 0; j < MAX_DEG; ++j) {
        if ((BIT(graph.v2e[v][j]) & c) > 0) {
          has_edge = true;
          break;
        }
      }
      if (!has_edge) {
        has_all_oriented_vertices = false;
        break;
      }
    }
    if (!has_all_oriented_vertices) {
      continue;
    }

    /*bool has_ignored_vertex = false;
    for (const auto& v : ignored_vertices) {
      bool has_edge = false;
      for (int j = 0; j < MAX_DEG; ++j) {
        if ((BIT(graph.v2e[v][j]) & c) > 0) {
          has_edge = true;
          break;
        }
      }
      if (has_edge) {
        has_ignored_vertex = true;
        break;
      }
    }
    if (has_ignored_vertex) {
      continue;
    }*/

    ++count;
    has_dominating_circuit = true;
    //return false;
    dom_circ = c;
    return true;
  }
  /*if (oriented_vertices.size() * 2 >= graph.number_of_vertices) {
    cerr << "dominat: " << count << endl;
  }*/
  /*if (count < 5) { // never happens
    cerr << "ov: " << oriented_vertices.size() << "; dominat: " << count << endl;
  }*/
  return false;
}

// TODO: why 2 functions for finding dominating circuit?
void check_compatible_dominating_circuit(Graph& graph) {
  int glob_min_c = graph.number_of_vertices;
  for (const auto& c : graph.all_dominating_circuits) {
    bool has_all_oriented_vertices = true;

    for (const auto& v : oriented_vertices) {
      bool has_edge = false;
      for (int j = 0; j < MAX_DEG; ++j) {
        if ((BIT(graph.v2e[v][j]) & c) > 0) {
          has_edge = true;
          break;
        }
      }
      if (!has_edge) {
        has_all_oriented_vertices = false;
        break;
      }
    }
    if (!has_all_oriented_vertices) {
      continue;
    }

    map<int, set<int>> neibs;
    int first_v = MAX_VERTEX;
    for (int e = 0; e < graph.number_of_edges; ++e) {
      if (BIT(e) & c) {
        int v1 = graph.e2v[e][0];
        int v2 = graph.e2v[e][1];
        neibs[v1].insert(v2);
        neibs[v2].insert(v1);
        first_v = min(first_v, min(v1, v2));
      }
    }

    set<int> visited;
    vector<int> vertices;
    visited.insert(first_v);
    vertices.push_back(first_v);
    int cur_v = first_v;
    do {
      bool found = false;
      for (const auto& v : neibs[cur_v]) {
        if (visited.find(v) == visited.end()) {
          vertices.push_back(v);
          cur_v = v;
          visited.insert(cur_v);
          found = true;
          break;
        }
      }
      if (!found) {
        break;
      }
    } while (true);
    vertices.push_back(first_v);
    vertices.push_back(vertices[1]);

    int c1 = 0;
    int c2 = 0;
    for (int vi = 1; vi < vertices.size() - 1; ++vi) {
      int v0 = vertices[vi];
      if (oriented_vertices.find(v0) == oriented_vertices.end()) {
        continue;
      }
      int v1 = vertices[vi - 1];
      int v2 = vertices[vi + 1];
      int e1 = graph.vv2e[v0][v1];
      int e2 = graph.vv2e[v0][v2];
      if (edge_pairs[make_pair(e1, e2)] == 2) {
        ++c1;
      } else {
        ++c2;
      }
    }
    glob_min_c = min(glob_min_c, min(c1, c2));
    //cerr << "dominat dispar: " << min(c1, c2) << " " << max(c1, c2) << endl;
  }
  cerr << "dominat min dispar: " << glob_min_c << endl;
}

void check_oriented_244_dominating_cycle1(Graph& graph) {
  bool found_cycle = false;
  bool found_dominating_circuit = false;
  for (const auto& triple : selected_triples) {
    Mask or_and_244_mask = 0;
    for (int v = 0; v < graph.number_of_vertices; ++v) {
      if ((oriented_vertices_mask & BIT(v)) != 0) {
        or_and_244_mask += BIT(v);
      } else if (vertex_in_244_cycle[triple][v]) {
        or_and_244_mask += BIT(v);
      }
    }

    for (const auto& c : graph.all_cycles) {
      bool is_circuit = false;
      if (graph.all_dominating_circuits.find(c) != graph.all_dominating_circuits.end()) {
        is_circuit = true;
      }
      if (found_cycle && !is_circuit) {
        continue;
      }
      bool same_vertices = true;
      bool has_all_vertices = true;
      for (int v = 0; v < graph.number_of_vertices; ++v) {
        bool has_edge = false;
        for (int j = 0; j < MAX_DEG; ++j) {
          if ((BIT(graph.v2e[v][j]) & c) > 0) {
            has_edge = true;
            break;
          }
        }
        bool in_or_cycle = ((or_and_244_mask & BIT(v)) != 0);
        if (in_or_cycle != has_edge) {
          same_vertices = false;
        }
        if (!has_edge && in_or_cycle) {
          has_all_vertices = false;
          break;
        }
      }
      if (!has_all_vertices) {
        continue;
      }
      if (same_vertices) {
        found_cycle = true;
      }
      if (is_circuit) {
        found_dominating_circuit = true;
        break;
      }
    }
    if (found_cycle && found_dominating_circuit) {
      break;
    }
  }
  if (found_dominating_circuit && found_cycle) {
    cerr << "ALL FOUND; DOMINATING1, AND CYCLE" << endl;
  } else if (found_dominating_circuit && !found_cycle) {
    cerr << "DOMINATING1 CIRCUIT FOUND; BUT NO CYCLE" << endl;
  } else if (!found_dominating_circuit && found_cycle) {
    cerr << "NO DOMINATING1 CIRCUIT, BUT CYCLE FOUND" << endl;
  } else {
    cerr << "NO DOMINATING1 OR CYCLE FOUND" << endl;
  }
}

void check_oriented_244_dominating_cycle2(Graph& graph) {
  for (const auto& triple : selected_triples) {
    int doms = 0;
    bool found_cycle = false;
    bool found_dominating_circuit = false;
    Mask or_and_244_mask = 0;
    for (int v = 0; v < graph.number_of_vertices; ++v) {
      if ((oriented_vertices_mask & BIT(v)) != 0) {
        or_and_244_mask += BIT(v);
      } else if (vertex_in_244_cycle[triple][v]) {
        or_and_244_mask += BIT(v);
      }
    }

    for (const auto& c : graph.all_cycles) {
      bool is_circuit = false;
      if (graph.all_dominating_circuits.find(c) != graph.all_dominating_circuits.end()) {
        is_circuit = true;
      }
      if (found_cycle && !is_circuit) {
        continue;
      }
      bool same_vertices = true;
      bool has_all_vertices = true;
      for (int v = 0; v < graph.number_of_vertices; ++v) {
        bool has_edge = false;
        for (int j = 0; j < MAX_DEG; ++j) {
          if ((BIT(graph.v2e[v][j]) & c) > 0) {
            has_edge = true;
            break;
          }
        }
        bool in_or_cycle = ((or_and_244_mask & BIT(v)) != 0);
        if (in_or_cycle != has_edge) {
          same_vertices = false;
        }
        if (!has_edge && in_or_cycle) {
          has_all_vertices = false;
          break;
        }
      }
      if (!has_all_vertices) {
        continue;
      }
      if (same_vertices) {
        found_cycle = true;
      }
      if (is_circuit) {
        found_dominating_circuit = true;
        dom_circ = c;
        doms += 1;
        //break;
      }
    }

    // FIXME: commented out cerr
    // if (doms > 0) {
    //   cerr << "doms: " << doms << "/" << graph.all_dominating_circuits.size() << endl;
    // }
    // if (found_dominating_circuit && found_cycle) {
    //   cerr << "ALL FOUND; DOMINATING2, AND CYCLE" << endl;
    // } else if (found_dominating_circuit && !found_cycle) {
    //   cerr << "DOMINATING2 CIRCUIT FOUND; BUT NO CYCLE" << endl;
    // } else if (!found_dominating_circuit && found_cycle) {
    //   cerr << "NO DOMINATING2 CIRCUIT, BUT CYCLE FOUND" << endl;
    // } else {
    //   cerr << "NO DOMINATING2 OR CYCLE FOUND" << endl;
    // }
  }
}

void print_graphviz(const Graph& graph, Mask dom_circ,
    const set<int>& oriented_vertices, const set<int>& poor_edges) {
  if (graphs_to_write == 0) {
    return;
  }
  --graphs_to_write;

  // retrieving vertices and edges of 244-cycles
  vector<bool> edges_in_244_cycle;
  vector<bool> vertices_in_244_cycle;
  vector<int> permutation;
  for (const auto& triple : selected_triples) {
    edges_in_244_cycle = edge_in_244_cycle[triple];
    vertices_in_244_cycle = vertex_in_244_cycle[triple];
    for (const auto& i : triple) {
      permutation.push_back(i);
    }
    for (int i = 0; i < 6; ++i) {
      if (triple.find(i) == triple.end()) {
        permutation.push_back(i);
      }
    }
    break;
  }
  if (permutation.size() == 0) {
    permutation = {0, 1, 2, 3, 4, 5};
  }

  // find vertices in dominating circuit
  vector<int> dom_vertices;
  map<int, set<int>> neibs;
  int first_v = MAX_VERTEX;
  for (int e = 0; e < graph.number_of_edges; ++e) {
    if (BIT(e) & dom_circ) {
      int v1 = graph.e2v[e][0];
      int v2 = graph.e2v[e][1];
      neibs[v1].insert(v2);
      neibs[v2].insert(v1);
      first_v = min(first_v, min(v1, v2));
    }
  }
  set<int> visited;
  visited.insert(first_v);
  dom_vertices.push_back(first_v);
  int cur_v = first_v;
  do {
    bool found = false;
    for (const auto& v : neibs[cur_v]) {
      if (visited.find(v) == visited.end()) {
        cur_v = v;
        visited.insert(cur_v);
        dom_vertices.push_back(cur_v);
        found = true;
        break;
      }
    }
    if (!found) {
      break;
    }
  } while (true);
  vector<int> not_dom_vertices;
  for (int v = 0; v < graph.number_of_vertices; ++v) {
    if (visited.find(v) == visited.end()) {
      not_dom_vertices.push_back(v);
    }
  }

  int dom_radius = 3;
  string oriented_color = "aquamarine3";
  string unoriented_in244_color = "burlywood1";
  string unoriented_not_in244_color = "hotpink";
  string poor_edge_color = "brown1";
  string rich_edge_color_in_matching = "cornsilk4";
  string poor_edge_color_in_matching = "orange1";
  string background_color = "white";

  // set colors
  vector<string> vertex_color;
  for (int v = 0; v < graph.number_of_vertices; ++v) {
    if (oriented_vertices.find(v) != oriented_vertices.end()) {
      vertex_color.push_back(oriented_color);
    } else {
      if ((vertices_in_244_cycle.size() > 0) && vertices_in_244_cycle[v]) {
        vertex_color.push_back(unoriented_in244_color);
      } else {
        vertex_color.push_back(unoriented_not_in244_color);
      }
    }
  }

  // set coords
  vector<vector<float>> coords;
  for (int v = 0; v < graph.number_of_vertices; ++v) {
    coords.push_back({0, 0});
  }
  for (int vi = 0; vi < dom_vertices.size(); ++vi) {
    int v = dom_vertices[vi];
    float angle = (float) vi / dom_vertices.size() * 2 * M_PI;
    float x = sin(angle) * dom_radius;
    float y = cos(angle) * dom_radius;
    coords[v] = {x, y};
  }
  vector<vector<float>> not_dom_coords;
  for (int vi = 0; vi < not_dom_vertices.size(); ++vi) {
    int v = not_dom_vertices[vi];
    float x = 0;
    float y = 0;
    for (int j = 0; j < MAX_DEG; ++j) {
      int v2 = graph.v2v[v][j];
      x += coords[v2][0];
      y += coords[v2][1];
    }
    x /= MAX_DEG;
    y /= MAX_DEG;
    bool found_nearby = false;
    float vertex_radius = 0.35;
    random_device rd;
    mt19937 rng(rd());
    std::uniform_real_distribution<float> uni(-vertex_radius/1.5, vertex_radius/1.5);
    do {
      found_nearby = false;
      for (const auto& prev_coord : not_dom_coords) {
        if ((abs(prev_coord[0] - x) + abs(prev_coord[1] - y)) < vertex_radius * 2) {
          found_nearby = true;
          break;
        }
      }
      if (found_nearby) {
        x += uni(rng);
        y += uni(rng);
      }
    } while (found_nearby);
    coords[v] = {x, y};
    not_dom_coords.push_back({x, y});
  }

  ofstream out;
  out.open("graph" + to_string(graph.number_of_vertices) +
      "g" + to_string(graph.number) + "-" + to_string(sol_count) + ".dot");
  out << "digraph {" << endl;
  out << "  graph [fontname=\"georgia\" pad=\"0.212,0.055\" bgcolor=" << background_color << " " <<
      "outputorder=\"edgesfirst\" splines=spline dpi=300]" << endl;
  out << "  node [fontname=\"georgia\" style=\"filled\" shape=circle fixedsize=shape width=0.35]" << endl;
  for (int v = 0; v < graph.number_of_vertices; ++v) {
    out << "  n0_" << v << " [label=\"" << v << "\" " <<
        "color=\"" << background_color << "\" " <<
        "fillcolor=\"" << vertex_color[v] << "\" " <<
        "pos=\"" << coords[v][0] << "," << coords[v][1] << "!\"]" << endl;
  }
  for (int e = 0; e < graph.number_of_edges; ++e) {
    out << "  n0_" << graph.e2v[e][0] << " -> n0_" << graph.e2v[e][1];
    out << " [dir=none";
    if (edges_in_244_cycle.size() > 0 && edges_in_244_cycle[e]) {
      out << " penwidth=3";
    }
    // if (edge_type[e] == 2) {
    //   out << " penwidth=3";
    // }
    if (poor_edges.find(e) != poor_edges.end()) {
      out << " color=\"" << poor_edge_color << "\"";
    }
    out << "]";
    out << endl;
  }

  for (int lyr_idx = 0; lyr_idx < 6; ++lyr_idx) {
    float shift_x = 0;
    float shift_y = -2 * (dom_radius + 1);;
    if (lyr_idx >= 3) {
      shift_y += -2 * (dom_radius + 1);
    }
    if (lyr_idx % 3 == 0) {
      shift_x = -2 * (dom_radius + 1);
    } else if (lyr_idx % 3 == 2) {
      shift_x = 2 * (dom_radius + 1);
    }

    int lyr = permutation[lyr_idx];
    out << endl;
    for (int v = 0; v < graph.number_of_vertices; ++v) {
      out << "  n" << lyr + 1 << "_" << v << " [label=\"" << v << "\" " <<
          "color=\"" << background_color << "\" " <<
          "fillcolor=\"" << vertex_color[v] << "\" " <<
          "pos=\"" << coords[v][0] + shift_x << "," << coords[v][1] + shift_y << "!\"]" << endl;
    }

    set<int> edges_in_layer;
    for (int c = 0; c < all_circuits_in_6c4c.size(); ++c) {
      if (layer[c] != lyr) {
        continue;
      }
      vector<int> circuit_vertices;
      if (orientations[c] == 0) {
        for (int vi = 0; vi < all_circuits_in_6c4c[c].size(); ++vi) {
          int v1 = all_circuits_in_6c4c[c][vi];
          circuit_vertices.push_back(v1);
        }
      } else {
        for (int vi = all_circuits_in_6c4c[c].size() - 1; vi >= 0; --vi) {
          int v1 = all_circuits_in_6c4c[c][vi];
          circuit_vertices.push_back(v1);
        }
      }
      for (int vi = 0; vi < circuit_vertices.size() - 1; ++vi) {
        int v1 = circuit_vertices[vi];
        int v2 = circuit_vertices[vi + 1];
        int e = graph.vv2e[v1][v2];
        edges_in_layer.insert(e);
        out << "  n" << lyr + 1 << "_" << v1 << " -> n" << lyr + 1 << "_" << v2;
        if (poor_edges.find(e) != poor_edges.end()) {
          out << " [color=\"" << poor_edge_color << "\"]";
        }
        out << endl;
      }
    }
    for (int e = 0; e < graph.number_of_edges; ++e) {
      if (edges_in_layer.find(e) != edges_in_layer.end()) {
        continue;
      }
      out << "  n" << lyr + 1 << "_" << graph.e2v[e][0] << " -> n" << lyr + 1 << "_" << graph.e2v[e][1];
      out << " [dir=none style=\"dashed\"";
      if (poor_edges.find(e) != poor_edges.end()) {
        out << " color=\"" << poor_edge_color_in_matching << "\"";
      } else {
        out << " color=\"" << rich_edge_color_in_matching << "\"";
      }
      out << "]";
      out << endl;
    }
  }

  out << "}";
  out.close();
}

void check_compatible_o5cdc(Graph& graph) {
  for (const auto& o5cdc_edge_pairs : graph.all_o5cdc_edge_pairs) {
    bool has_all_oriented_vertices = true;

    for (const auto& v : oriented_vertices) {
      int e0 = graph.v2e[v][0];
      int e1 = graph.v2e[v][1];
      pair<int, int> ep = make_pair(e0, e1);
      if (edge_pairs[ep] == 0) {
        ep = make_pair(e1, e0);
      }
      assert(edge_pairs[ep] != 0);
      if (o5cdc_edge_pairs.find(ep) == o5cdc_edge_pairs.end()) {
        has_all_oriented_vertices = false;
        break;
      }
    }

    if (!has_all_oriented_vertices) {
      continue;
    }
    cerr << "has" << endl;
    return;
  }
  cerr << "hasn't" << endl;
}

bool orient_6c4c(Graph& graph, int cur_circuit, bool first_time) {
    if (cur_circuit == all_circuits_in_6c4c.size()) {
        if (first_time) {
          ++same_cycles_different_orientations;
        }
        if (same_cycles_different_orientations == 1) {
          // FIXME: commenting out cerrs
          // cerr << "NEW 6C4C with o6c4c" << endl;
          graph.o6c4c_profiles.insert(prof);
        }
        sol_count += 1;

        for (int e = 0; e < graph.number_of_edges; ++e) {
            for (int i = 0; i < 6; ++i) {
                layer_flow[i][e] = 0;
            }
        }

        edge_pairs.clear();
        edge_pair_layers.clear();

        for (int c = 0; c < all_circuits_in_6c4c.size(); ++c) {
            for (int vi = 0; vi < all_circuits_in_6c4c[c].size() - 1; ++vi) {
                int v1 = all_circuits_in_6c4c[c][vi];
                int v2 = all_circuits_in_6c4c[c][vi + 1];
                int ei = graph.vv2e[v1][v2];
                max_layer[ei] = c;
                int cur_edge_orientation = orientations[c];

                int e1 = -1;
                int e2 = -1;
                int v0 = -1;
                if (vi > 0) {
                    v0 = all_circuits_in_6c4c[c][vi - 1];
                } else {
                    v0 = all_circuits_in_6c4c[c][all_circuits_in_6c4c[c].size() - 2];
                }
                int prev_ei = graph.vv2e[v0][v1];
                if (orientations[c] == 0) {
                    e1 = prev_ei;
                    e2 = ei;
                } else {
                    e1 = ei;
                    e2 = prev_ei;
                }
                ++edge_pairs[make_pair(e1, e2)];
                edge_pair_layers[make_pair(e1, e2)].insert(layer[c]);

                if (v1 > v2) {
                    cur_edge_orientation = 1 - cur_edge_orientation;
                }
                if (cur_edge_orientation == 0) {
                    layer_flow[layer[c]][ei] += 1;
                } else {
                    layer_flow[layer[c]][ei] -= 1;
                }
            }
        }

        for (int v = 0; v < graph.number_of_vertices; ++v) {
          layer_pairings[v].clear();
        }
        oriented_vertices.clear();
        oriented_vertices_mask = 0;
        for (int v = 0; v < graph.number_of_vertices; ++v) {
          int e1 = graph.v2e[v][0];
          int e2 = graph.v2e[v][1];
          if (edge_pairs[make_pair(e1, e2)] != 1) { // so, it's 0 or 2
            oriented_vertices.insert(v);
            oriented_vertices_mask += BIT(v);
          }
          for (int j1 = 0; j1 < MAX_DEG; ++j1) {
            int e1 = graph.v2e[v][j1];
            for (int j2 = 0; j2 < MAX_DEG; ++j2) {
              if (j1 == j2) {
                continue;
              }
              int e2 = graph.v2e[v][j2];
              const auto pair = make_pair(e1, e2);
              if (edge_pair_layers.find(pair) != edge_pair_layers.end()) {
                layer_pairings[v].insert(edge_pair_layers[pair]);
              }
            }
          }
        }
        all_oriented_vertices.insert(oriented_vertices);
        if (first_time) {
          ors.insert(oriented_vertices.size());
        }
        assert(oriented_vertices.size() != 1); // TODO
        if (same_cycles_different_orientations == 1) {
          u6c4c_oriented_parity = oriented_vertices.size() % 2;
        }
        assert(u6c4c_oriented_parity == (oriented_vertices.size() % 2)); // this is true and easy to prove
        //cerr << "or: " << oriented_vertices.size() << "; ";
        //return true;

        // FIXMEFIXME
        if (first_time) {
          dom_circ = NONE;
        }

        if (first_time) {
          return false;
        }

        // FIXMEFIXMEFIXME
        // if (ors.find(0) == ors.end()) {
        //   cerr << endl;
        //   return false;
        // }

        // FIXMEFIXMEFIXME
        // if (ors.size() == 1) {
        // // if (oriented_vertices.size() > 0) {
        //   cerr << endl;
        //   return false;
        // }

        graph.possible_oriented_vertices.insert(oriented_vertices.size());

        int t1 = 0;
        int t2 = 0;
        int t3 = 0;
        int t4 = 0;
        for (int e = 0; e < graph.number_of_edges; ++e) {
            bool v1_is_oriented = (oriented_vertices.find(graph.e2v[e][0]) != oriented_vertices.end());
            bool v2_is_oriented = (oriented_vertices.find(graph.e2v[e][1]) != oriented_vertices.end());
            int oriented_count = 0;
            if (v1_is_oriented) {
                ++oriented_count;
            }
            if (v2_is_oriented) {
                ++oriented_count;
            }
            if (u6c4c_edge_pair_counts[e].size() == 2) {
                o6c4c_always_3[e] = false;
                o6c4c_always_4[e] = false;
                if (oriented_count == 2) {
                    o6c4c_always_2[e] = false;
                    o6c4c_never_1[e] = false;
                    edge_type[e] = 0;
                    ++t1;
                } else if (oriented_count == 0) {
                    o6c4c_always_1[e] = false;
                    o6c4c_never_2[e] = false;
                    edge_type[e] = 1;
                    ++t2;
                } else {
                  assert(false);
                  cerr << "wat5" << endl;
                }
            } else {
                o6c4c_always_1[e] = false;
                o6c4c_always_2[e] = false;
                if (oriented_count == 1) {
                    o6c4c_always_4[e] = false;
                    o6c4c_never_3[e] = false;
                    edge_type[e] = 2;
                    ++t3;
                } else if (oriented_count == 0) {
                    o6c4c_always_3[e] = false;
                    o6c4c_never_4[e] = false;
                    edge_type[e] = 3;
                    ++t4;
                } else {
                  assert(false);
                  cerr << "wat6" << endl;
                }
            }
        }
        assert(t1 + t3 != 7); // TODO

        // =========================================================

        vector<int> chords_frequency = {0, 0, 0};
        t1_chords_frequency = {0, 0, 0};
        t2_chords_frequency = {0, 0, 0};
        t3_chords_frequency = {0, 0, 0};
        t4_chords_frequency = {0, 0, 0};
        vector<int> chords0_by_layer = {0, 0, 0, 0, 0, 0};
        vector<int> chords1_by_layer = {0, 0, 0, 0, 0, 0};
        vector<int> chords1_by_chord_layer = {0, 0, 0, 0, 0, 0};
        vector<int> chords2_by_layer = {0, 0, 0, 0, 0, 0};
        bool is_antichord[6][MAX_EDGE];
        for (int i = 0; i < 6; ++i) {
          chords0_by_layer[i] = 0;
          chords1_by_layer[i] = 0;
          chords1_by_chord_layer[i] = 0;
          chords2_by_layer[i] = 0;
        }
        int oriented_chords = 0;
        int poor_antichords = 0;
        vector<int> local_oriented_chords = {0, 0, 0, 0, 0, 0};
        for (int e = 0; e < graph.number_of_edges; ++e) {
          int chord_count = 0;
          vector<int> chord_layers;
          vector<int> layers;
          for (int i = 0; i < 6; ++i) {
            is_antichord[i][e] = false;
            if ((u6c4c_cycles[i] & BIT(e)) == 0) { // edge is from matching
              const int v1 = graph.e2v[e][0];
              const int v2 = graph.e2v[e][1];
              layers.push_back(i);
              if (layer_vertex_to_circuit[i][v1] == layer_vertex_to_circuit[i][v2]) {
                ++chord_count;
                chord_layers.push_back(i);
              } else {
                is_antichord[i][e] = true;
                if (u6c4c_edge_is_poor[e]) {
                  ++poor_antichords;
                }
              }
              bool v1_is_oriented = (oriented_vertices.find(v1) != oriented_vertices.end());
              bool v2_is_oriented = (oriented_vertices.find(v2) != oriented_vertices.end());
              if ((v1_is_oriented && !v2_is_oriented) || (!v1_is_oriented && v2_is_oriented)) {
                if (u6c4c_edge_is_poor[e]) {
                  ++local_oriented_chords[i];
                }
              }
            }
          }
          ++chords_frequency[chord_count];
          if (edge_type[e] == 0) {
            t1_chords_frequency[chord_count] += 1;
          } else if (edge_type[e] == 1) {
            t2_chords_frequency[chord_count] += 1;
          } else if (edge_type[e] == 2) {
            t3_chords_frequency[chord_count] += 1;
          } else {
            assert(edge_type[e] == 3);
            t4_chords_frequency[chord_count] += 1;
          }
          if (chord_count == 0) {
            chords0_by_layer[layers[0]] += 1;
            chords0_by_layer[layers[1]] += 1;
          }
          if (chord_count == 1) {
            chords1_by_layer[layers[0]] += 1;
            chords1_by_layer[layers[1]] += 1;

            chords1_by_chord_layer[chord_layers[0]] += 1;
          }
          if (chord_count == 2) {
            chords2_by_layer[layers[0]] += 1;
            chords2_by_layer[layers[1]] += 1;
          }
        }
        sort(chords0_by_layer.begin(), chords0_by_layer.end());
        sort(chords1_by_layer.begin(), chords1_by_layer.end());
        sort(chords1_by_chord_layer.begin(), chords1_by_chord_layer.end());
        sort(chords2_by_layer.begin(), chords2_by_layer.end());

        t4_chord_count_by_layer = {0, 0, 0, 0, 0, 0};
        for (int e = 0; e < graph.number_of_edges; ++e) {
          for (int i = 0; i < 6; ++i) {
            if ((u6c4c_cycles[i] & BIT(e)) == 0) { // edge is from matching
              const int v1 = graph.e2v[e][0];
              const int v2 = graph.e2v[e][1];
              if (layer_vertex_to_circuit[i][v1] == layer_vertex_to_circuit[i][v2]) {
                if (edge_type[e] == 3) {
                  ++t4_chord_count_by_layer[i];
                }
              }
            }
          }
        }

        /*if (ch0l.find(chords0_by_layer) != ch0l.end()) {
          if (ch0l[chords0_by_layer] != (oriented_vertices.size() % 2)) {
            cerr << "bad0; ";
          }
        }
        ch0l[chords0_by_layer] = oriented_vertices.size() % 2;

        if (ch1l.find(chords0_by_layer) != ch1l.end()) {
          if (ch1l[chords1_by_layer] != (oriented_vertices.size() % 2)) {
            cerr << "bad1; ";
          }
        }
        ch1l[chords1_by_layer] = oriented_vertices.size() % 2;

        if (ch1chl.find(chords1_by_chord_layer) != ch1chl.end()) {
          if (ch1chl[chords1_by_chord_layer] != (oriented_vertices.size() % 2)) {
            cerr << "badch; ";
          }
        }
        ch1chl[chords1_by_chord_layer] = oriented_vertices.size() % 2;

        if (ch2l.find(chords2_by_layer) != ch2l.end()) {
          if (ch2l[chords2_by_layer] != (oriented_vertices.size() % 2)) {
            cerr << "bad2; ";
          }
        }
        ch2l[chords2_by_layer] = oriented_vertices.size() % 2;*/

        for (int i = 0; i < 6; ++i) {
          if (local_oriented_chords[i] % 2 == 0) {
            ++oriented_chords;
          }
        }

        set<int> vertices_neib_to_poor_edge;
        for (int e = 0; e < graph.number_of_edges; ++e) {
          if (u6c4c_edge_is_poor[e]) {
            const int v1 = graph.e2v[e][0];
            const int v2 = graph.e2v[e][1];
            vertices_neib_to_poor_edge.insert(v1);
            vertices_neib_to_poor_edge.insert(v2);
          }
        }

        int even_poor_comps_matchings = 0;
        for (const auto& c : cur_6c4c) {
          set<int> poor_comps_edges;
          for (int e = 0; e < graph.number_of_edges; ++e) {
            if ((c & BIT(e)) == 0) { // edge is from matching
              if (u6c4c_edge_is_poor[e]) {
                poor_comps_edges.insert(edge_poor_component[e]);
              }
            }
          }
          if (poor_comps_edges.size() % 2 == 0) {
            ++even_poor_comps_matchings;
          }
        }
        assert(even_poor_comps_matchings % 2 == 0);
        even_poor_comps_matchings /= 2;

        int circuits_even_or = 0;
        int circuits_even_len = 0;
        int circuits_odd_len = 0;
        int circuits_even_rich = 0;
        int circuits_even_poor = 0;
        int circuits_odd_poor = 0;
        int circuits_one_poor = 0;
        vector<int> circuit_lens;
        //int rich_circuits = 0; // BAD
        bool is_even_rich_circuit[6 * MAX_VERTEX];
        bool is_even_poor_circuit[6 * MAX_VERTEX];
        for (int c = 0; c < all_circuits_in_6c4c.size(); ++c) {
          is_even_rich_circuit[c] = false;
          is_even_poor_circuit[c] = false;
          int poor_count = 0;
          int rich_count = 0;
          int len = 0;
          int delta = 1;
          int or_count = 0;
          for (int vi = 0; vi < all_circuits_in_6c4c[c].size() - 1; ++vi) {
            int v1 = all_circuits_in_6c4c[c][vi];
            if (oriented_vertices.find(v1) != oriented_vertices.end()) {
              ++or_count;
            }
            int v2 = all_circuits_in_6c4c[c][vi + 1];
            int ei = graph.vv2e[v1][v2];
            if (u6c4c_edge_is_poor[ei]) {
              ++poor_count;
              delta *= -1;
            } else {
              ++rich_count;
              len += delta;
            }
          }
          if (or_count % 2 == 0) {
            ++circuits_even_or;
          }
          if ((rich_count + poor_count) % 2 == 0) {
            ++circuits_even_len;
          } else {
            ++circuits_odd_len;
          }
          if (rich_count % 2 == 0) {
            ++circuits_even_rich;
            is_even_rich_circuit[c] = true;
          }
          if (poor_count == 1) {
            ++circuits_one_poor;
          }
          if (poor_count % 2 == 0) {
            ++circuits_even_poor;
            is_even_poor_circuit[c] = true;
            if (len < 0) {
              len = -len;
            }
            circuit_lens.push_back(len);
          } else {
            ++circuits_odd_poor;
            circuit_lens.push_back(-1);
          }
          /*if (poor_count == 0) { // BAD
            ++rich_circuits;
          }*/
        }

        int croo = 0;
        int cree = 0;
        int aroo = 0;
        int areo = 0;
        int aree = 0;
        for (int i = 0; i < 6; ++i) {
          for (int e = 0; e < graph.number_of_edges; ++e) {
            if ((u6c4c_cycles[i] & BIT(e)) != 0) {
              continue;
            }
            // edge is from matching
            const int v1 = graph.e2v[e][0];
            const int v2 = graph.e2v[e][1];
            int c1 = layer_vertex_to_circuit[i][v1];
            int c2 = layer_vertex_to_circuit[i][v2];
            bool is_even_rich1 = is_even_rich_circuit[v1];
            bool is_even_rich2 = is_even_rich_circuit[v2];
            if (c1 != c2) { // antichord
              if (is_even_rich1 && is_even_rich2) {
                ++aree;
              } else if (!is_even_rich1 && !is_even_rich2) {
                ++aroo;
              } else {
                ++areo;
              }
            } else { // chord
              if (is_even_rich1) {
                ++cree;
              } else {
                ++croo;
              }
            }
          }
        }

        int cpoo = 0;
        int cpee = 0;
        int apoo = 0;
        int apeo = 0;
        int apee = 0;
        for (int i = 0; i < 6; ++i) {
          for (int e = 0; e < graph.number_of_edges; ++e) {
            if ((u6c4c_cycles[i] & BIT(e)) != 0) {
              continue;
            }
            // edge is from matching
            const int v1 = graph.e2v[e][0];
            const int v2 = graph.e2v[e][1];
            int c1 = layer_vertex_to_circuit[i][v1];
            int c2 = layer_vertex_to_circuit[i][v2];
            bool is_even_poor1 = is_even_poor_circuit[v1];
            bool is_even_poor2 = is_even_poor_circuit[v2];
            if (c1 != c2) { // antichord
              if (is_even_poor1 && is_even_poor2) {
                ++apee;
              } else if (!is_even_poor1 && !is_even_poor2) {
                ++apoo;
              } else {
                ++apeo;
              }
            } else { // chord
              if (is_even_poor1) {
                ++cpee;
              } else {
                ++cpoo;
              }
            }
          }
        }

        vector<int> chords_by_type = {0, 0, 0, 0};
        vector<int> antichords_by_type = {0, 0, 0, 0};

        for (int i = 0; i < 6; ++i) {
          for (int e = 0; e < graph.number_of_edges; ++e) {
            if ((u6c4c_cycles[i] & BIT(e)) != 0) {
              continue;
            }
            // edge is from matching
            const int v1 = graph.e2v[e][0];
            const int v2 = graph.e2v[e][1];
            int c1 = layer_vertex_to_circuit[i][v1];
            int c2 = layer_vertex_to_circuit[i][v2];
            if (c1 != c2) { // antichord
              ++antichords_by_type[edge_type[e]];
            } else { // chord
              ++chords_by_type[edge_type[e]];
            }
          }
        }

        int circuits_even_oriented_vertices = 0;
        int circuits_without_oriented_vertices = 0;
        int opoo = 0;
        int opeo = 0;
        int epoo = 0;
        int epeo = 0;
        for (int c = 0; c < all_circuits_in_6c4c.size(); ++c) {
          int oriented_vertices_count = 0;
          int poor_count = 0;
          for (int vi = 0; vi < all_circuits_in_6c4c[c].size() - 1; ++vi) {
             int v1 = all_circuits_in_6c4c[c][vi];
             bool v1_is_oriented = (oriented_vertices.find(v1) != oriented_vertices.end());
             if (v1_is_oriented) {
               ++oriented_vertices_count;
             }
             int v2 = all_circuits_in_6c4c[c][vi + 1];
             int ei = graph.vv2e[v1][v2];
             if (u6c4c_edge_is_poor[ei]) {
               ++poor_count;
             }
          }
          if (oriented_vertices_count == 0) {
            ++circuits_without_oriented_vertices;
          }
          if (oriented_vertices_count % 2 == 0) {
            ++circuits_even_oriented_vertices;
          }

          if (oriented_vertices_count % 2 == 0) {
            if (poor_count % 2 == 0) {
              ++epeo;
            } else {
              ++opeo;
            }
          } else {
            if (poor_count % 2 == 0) {
              ++epoo;
            } else {
              ++opoo;
            }
          }
        }

        vector<int> rich_unoriented_vertices_frequency = {0, 0, 0, 0};
        vector<int> rich_oriented_vertices_frequency = {0, 0, 0, 0};
        for (int v = 0; v < graph.number_of_vertices; ++v) {
          bool v_is_oriented = (oriented_vertices.find(v) != oriented_vertices.end());
          int rich_edge_count = 0;
          for (int j = 0; j < MAX_DEG; ++j) {
            int e = graph.v2e[v][j];
            if (!u6c4c_edge_is_poor[e]) {
              ++rich_edge_count;
            }
          }
          if (v_is_oriented) {
            ++rich_oriented_vertices_frequency[rich_edge_count];
          } else {
            ++rich_unoriented_vertices_frequency[rich_edge_count];
          }
        }

        int circuits_oriented_frequencies[MAX_VERTEX];
        for (int i = 0; i < graph.number_of_vertices; ++i) {
          circuits_oriented_frequencies[i] = 0;
        }
        for (int c = 0; c < all_circuits_in_6c4c.size(); ++c) {
          int oriented_vertices_count = 0;
          for (int vi = 0; vi < all_circuits_in_6c4c[c].size() - 1; ++vi) {
             int v1 = all_circuits_in_6c4c[c][vi];
             bool v1_is_oriented = (oriented_vertices.find(v1) != oriented_vertices.end());
             if (v1_is_oriented) {
               ++oriented_vertices_count;
             }
          }
          ++circuits_oriented_frequencies[oriented_vertices_count];
        }

        int next[6][MAX_VERTEX];
        int prev[6][MAX_VERTEX];
        bool e_in_layer[6][MAX_EDGE];
        for (int l = 0; l < 6; ++l) {
          for (int e = 0; e < graph.number_of_edges; ++e) {
            e_in_layer[l][e] = false;
          }
        }
        for (int c = 0; c < all_circuits_in_6c4c.size(); ++c) {
          int l = layer[c];
          for (int vi = 0; vi < all_circuits_in_6c4c[c].size() - 1; ++vi) {
            int v1 = all_circuits_in_6c4c[c][vi];
            int v2 = all_circuits_in_6c4c[c][vi + 1];
            int e = graph.vv2e[v1][v2];
            e_in_layer[l][e] = true;
            if (orientations[c] == 0) {
              next[l][v1] = v2;
              prev[l][v2] = v1;
            } else {
              prev[l][v1] = v2;
              next[l][v2] = v1;
            }
          }
        }

        // int almost_poor_circuits_count = 0;
        // for (int c = 0; c < all_circuits_in_6c4c.size(); ++c) {
        //   int cur_circuit_rich_count = 0;
        //   for (int vi = 0; vi < all_circuits_in_6c4c[c].size() - 1; ++vi) {
        //     int v1 = all_circuits_in_6c4c[c][vi];
        //     int v2 = all_circuits_in_6c4c[c][vi + 1];
        //     int ei = graph.vv2e[v1][v2];
        //     if (!u6c4c_edge_is_poor[ei]) {
        //       cur_circuit_rich_count += 1;
        //     }
        //   }
        //   if (cur_circuit_rich_count <= 1) {
        //     almost_poor_circuits_count += 1;
        //   }
        // }

        /*map<Mask, int> poor_circuits;
        map<Mask, map<int, int>> poor_edges_ors;
        bool found_same_or = false;
        for (int c = 0; c < all_circuits_in_6c4c.size(); ++c) {
          bool has_rich = false;
          Mask m = 0;
          for (int vi = 0; vi < all_circuits_in_6c4c[c].size() - 1; ++vi) {
            int v1 = all_circuits_in_6c4c[c][vi];
            int v2 = all_circuits_in_6c4c[c][vi + 1];
            int ei = graph.vv2e[v1][v2];
            m += BIT(ei);
            if (!u6c4c_edge_is_poor[ei]) {
              has_rich = true;
              break;
            }
          }
          if (!has_rich) {
            poor_circuits[m] += 1;

            for (int vi = 0; vi < all_circuits_in_6c4c[c].size() - 1; ++vi) {
              int v1 = all_circuits_in_6c4c[c][vi];
              int v2 = all_circuits_in_6c4c[c][vi + 1];
              int ei = graph.vv2e[v1][v2];
              int e_ori = 1;
              if (v1 > v2) {
                e_ori *= -1;
              }
              if (orientations[c] == 1) {
                e_ori *= -1;
              }
              poor_edges_ors[m][ei] += e_ori;
            }
          }
        }

        for (const auto& pc : poor_circuits) {
          if (pc.second < 2) {
            continue;
          }
          Mask m = pc.first;
          for (const auto& ec : poor_edges_ors[m]) {
            if (ec.second != 0) {
              found_same_or = true;
              break;
            }
          }
        }*/


        int poor_breaking_orientation = 0;
        for (int e = 0; e < graph.number_of_edges; ++e) {
          if (!u6c4c_edge_is_poor[e]) {
            continue;
          }
          int v1 = graph.e2v[e][0];
          int v2 = graph.e2v[e][1];
          for (int l = 0; l < 6; ++l) {
            if (!e_in_layer[l][e]) {
              int v11 = prev[l][v1];
              int v22 = next[l][v2];
              if (v11 > v22) {
                swap(v11, v22);
              }
              if (u6c4c_edge_pair_counts[e].find(make_pair(v11, v22)) == u6c4c_edge_pair_counts[e].end()) {
                ++poor_breaking_orientation;
              }
              break;
            }
          }
        }

        int even_oo_matchings = 0;
        int even_uu_matchings = 0;
        int even_ou_matchings = 0;
        for (const auto& c : cur_6c4c) {
          int oo_count = 0;
          int uu_count = 0;
          int ou_count = 0;
          for (int e = 0; e < graph.number_of_edges; ++e) {
            if ((c & BIT(e)) == 0) { // edge is from matching
              int v1 = graph.e2v[e][0];
              int v2 = graph.e2v[e][1];
              bool v1_is_oriented = (oriented_vertices.find(v1) != oriented_vertices.end());
              bool v2_is_oriented = (oriented_vertices.find(v2) != oriented_vertices.end());
              if (v1_is_oriented && v2_is_oriented) {
                ++oo_count;
              } else if (!v1_is_oriented && !v2_is_oriented) {
                ++uu_count;
              } else {
                ++ou_count;
              }
            }
          }
          if (oo_count % 2 == 0) {
            ++even_oo_matchings;
          }
          if (uu_count % 2 == 0) {
            ++even_uu_matchings;
          }
          if (ou_count % 2 == 0) {
            ++even_ou_matchings;
          }
        }
        assert(even_oo_matchings % 2 == 0);
        even_oo_matchings /= 2;
        assert(even_uu_matchings % 2 == 0);
        even_uu_matchings /= 2;

        assert(even_ou_matchings % 2 == 0);
        even_ou_matchings /= 2;
        assert(even_ou_matchings != 1 && even_ou_matchings != 2); // this is true
        assert((oriented_vertices.size() + even_ou_matchings) % 2 == 1); // this is true

        if (oriented_vertices.size() == 0) {
          /*for (const auto& c : cur_6c4c) {
            cerr << "poor edges: ";
            for (int e = 0; e < graph.number_of_edges; ++e) {
              if ((c & BIT(e)) == 0) { // edge is from matching
                if (u6c4c_edge_is_poor[e]) {
                  cerr << e << " ";
                }
              }
            }
            cerr << endl;
          }*/

          map<int, vector<int>> edge_to_layers;
          int layer_idx = 0;
          for (const auto& c : cur_6c4c) {
            cerr << "rich edges: ";
            for (int e = 0; e < graph.number_of_edges; ++e) {
              if ((c & BIT(e)) == 0) { // edge is from matching
                if (!u6c4c_edge_is_poor[e]) {
                  cerr << e << " ";
                  edge_to_layers[e].push_back(layer_idx);
                }
              }
            }
            cerr << endl;
            ++layer_idx;
          }
          int layer_pair_counts[6][6];
          for (int i = 0; i < 6; ++i) {
            for (int j = i + 1; j < 6; ++j) {
              layer_pair_counts[i][j] = 0;
            }
          }
          for (const auto& els : edge_to_layers) {
            ++layer_pair_counts[els.second[0]][els.second[1]];
          }
          vector<int> layer_counts = {0, 0, 0, 0, 0 ,0};
          for (int i = 0; i < 6; ++i) {
            for (int j = i + 1; j < 6; ++j) {
              if (layer_pair_counts[i][j] % 2 != 0) {
                ++layer_counts[i];
                ++layer_counts[j];
              }
            }
          }
          sort(layer_counts.begin(), layer_counts.end());
          cerr << "whats left: ";
          for (int i = 0; i < 6; ++i) {
            cerr << layer_counts[i] << " ";
          }
          cerr << endl;
        }

        has_nz_mod5_flow = false;
        for (int e = 0; e < graph.number_of_edges; ++e) {
          cur_flow[e] = 0;
        }
        find_nz_mod_from_o6c4c(graph, 5, 1, false);

        // FIXMEFIXMEFIXME
        has_nz_mod6_flow = false;
        for (int e = 0; e < graph.number_of_edges; ++e) {
          cur_flow[e] = 0;
        }
        find_nz_mod_from_o6c4c(graph, 6, 1, false);

        // both flows with same weights (and it's not same as having nz5 flow)
        has_nz_modb_flow = false;
        for (int e = 0; e < graph.number_of_edges; ++e) {
          cur_flow[e] = 0;
        }
        find_nz_mod_from_o6c4c(graph, 5, 1, true);


        vertex_in_244_cycle.clear();
        edge_in_244_cycle.clear();
        rich_244_counts.clear();
        Mask ms[3];
        // for (int i = 0; i < 6; ++i) { // FIXME?
        for (int i = 0; i < 1; ++i) {
          ms[0] = u6c4c_cycles[i];
          for (int j = i + 1; j < 6; ++j) {
            ms[1] = u6c4c_cycles[j];
            for (int k = j + 1; k < 6; ++k) {
              ms[2] = u6c4c_cycles[k];
              set<int> triple = {i, j, k};
              vector<bool> vertex_in_cycle;
              vector<bool> edge_in_cycle;
              int rich_count = 0;
              for (int v = 0; v < graph.number_of_vertices; ++v) {
                vertex_in_cycle.push_back(false);
              }
              for (int e = 0; e < graph.number_of_edges; ++e) {
                edge_in_cycle.push_back(false);
              }
              for (int e = 0; e < graph.number_of_edges; ++e) {
                int edge_count = 0;
                for (int part = 0; part < 3; ++part) {
                  if ((BIT(e) & ms[part]) > 0) {
                    ++edge_count;
                  }
                }
                if (edge_count != 2) {
                  edge_in_cycle[e] = true;
                  if (!u6c4c_edge_is_poor[e]) {
                    rich_count++;
                  }
                  for (int ii = 0; ii < 2; ++ii) {
                    vertex_in_cycle[graph.e2v[e][ii]] = true;
                  }
                }
              }
              vertex_in_244_cycle[triple] = vertex_in_cycle;
              edge_in_244_cycle[triple] = edge_in_cycle;
              rich_244_counts.push_back(rich_count);
            }
          }
        }

        int triples_without_ors = 0;
        selected_triple.clear();
        selected_triples.clear();
        // for (const auto& triple_and_vertices : vertex_in_244_cycle) {
        //   const auto& vertices = triple_and_vertices.second;
        //   bool no_ors = true;
        //   int or_count = 0;
        //   for (int v = 0; v < graph.number_of_vertices; ++v) {
        //     if ((oriented_vertices_mask & BIT(v)) != 0) {
        //       if (vertices[v]) {
        //         no_ors = false;
        //         or_count += 1;
        //       }
        //     }
        //   }
        //   if (no_ors) {
        //     ++triples_without_ors;
        //     selected_triple = triple_and_vertices.first;
        //   }
        //   if (or_count == 0) {
        //     selected_triples.insert(triple_and_vertices.first);
        //   }
        // }
        // //check_oriented_244_dominating_cycle1(graph);
        // selected_triples.clear();

        has_int_nz5_flow = false;
        for (int e = 0; e < graph.number_of_edges; ++e) {
          cur_flow[e] = 0;
        }

        nz5_sums.clear();
        nz5_sums244.clear();
        nz5_sums_not244.clear();
        oriented_nz5_sums.clear();
        oriented244_nz5_sums.clear();
        oriented_not244_nz5_sums.clear();
        unoriented_nz5_sums.clear();
        unoriented244_nz5_sums.clear();
        unoriented_not244_nz5_sums.clear();
        nz5_flow_types.clear();
        nz5_flow_types244.clear();
        nz5_flow_types_not244.clear();
        nz5_flow_types_ors.clear();
        nz5_flow_types_ors244.clear();
        nz5_flow_types_ors_not244.clear();
        nz5_flow_types_unors.clear();
        nz5_flow_types_unors244.clear();
        nz5_flow_types_unors_not244.clear();

        cur_flows.clear();

        // FIXMEFIXMEFIXME
        find_int_nz_from_o6c4c_cleaner(graph, 5, 0);

        // FIXMEFIXMEFIXME
        if (false) {
        // if (oriented_nz5_sums.size() > 0) {
          for (const auto& f : cur_flows) {
            cerr << "flow count: " << graph.from_nz5_to_33pp[f].size() << endl;
          }
          if (oriented_nz5_sums.size() > 1) {
            // check_oriented_244_dominating_cycle1(graph);
            check_oriented_244_dominating_cycle2(graph);
            cerr << "pairings:" << endl;
            for (const auto& v : oriented_vertices) {
              cerr << v << ": ";
              for (const auto& pair : layer_pairings[v]) {
                for (const auto& e : pair) {
                  cerr << e << ",";
                }
                cerr << " ";
              }
              cerr << endl;
            }
          }

          // check_oriented_244_dominating_cycle2(graph);
          // cerr << "dom_circ: " << dom_circ << endl;

          // FIXME: commenting out cerrs
          // cerr << "stats244: " <<
          //     nz5_sums.size() << " " <<
          //     nz5_sums244.size() << " " <<
          //     nz5_sums_not244.size() << " or " <<
          //     oriented_nz5_sums.size() << " " <<
          //     oriented244_nz5_sums.size() << " " <<
          //     oriented_not244_nz5_sums.size() << " un " <<
          //     unoriented_nz5_sums.size() << " " <<
          //     unoriented244_nz5_sums.size() << " " <<
          //     unoriented_not244_nz5_sums.size() << "  VS  " <<
          //     nz5_flow_types.size() << " " <<
          //     nz5_flow_types244.size() << " " <<
          //     nz5_flow_types_not244.size() << " or " <<
          //     nz5_flow_types_ors.size() << " " <<
          //     nz5_flow_types_ors244.size() << " " <<
          //     nz5_flow_types_ors_not244.size() << " un " <<
          //     nz5_flow_types_unors.size() << " " <<
          //     nz5_flow_types_unors244.size() << " " <<
          //     nz5_flow_types_unors_not244.size() << endl;

              // " " <<
              // nz5_or_count << " " <<
              // nz5_or244_count << " " <<
              // nz5_or_not244_count << " " <<
              // nz5_unor244_count << " " <<
              // nz5_unor_not244_count << endl;

          // FIXME: commenting out cerrs
          // cerr << "both sums: ";
          // for (const auto& s : nz5_sums) {
          //   cerr << s << " ";
          // }
          // cerr << endl;
          // cerr << "both244 sums: ";
          // for (const auto& s : nz5_sums244) {
          //   cerr << s << " ";
          // }
          // cerr << endl;
          // cerr << "both_not244 sums: ";
          // for (const auto& s : nz5_sums_not244) {
          //   cerr << s << " ";
          // }
          // cerr << endl;
          // cerr << "oriented sums: ";
          // for (const auto& s : oriented_nz5_sums) {
          //   cerr << s << " ";
          // }
          // cerr << endl;
          // cerr << "oriented244 sums: ";
          // for (const auto& s : oriented244_nz5_sums) {
          //   cerr << s << " ";
          // }
          // cerr << endl;
          // cerr << "oriented_not244 sums: ";
          // for (const auto& s : oriented_not244_nz5_sums) {
          //   cerr << s << " ";
          // }
          // cerr << endl;
          // cerr << "unoriented sums: ";
          // for (const auto& s : unoriented_nz5_sums) {
          //   cerr << s << " ";
          // }
          // cerr << endl;
          // cerr << "unoriented244 sums: ";
          // for (const auto& s : unoriented244_nz5_sums) {
          //   cerr << s << " ";
          // }
          // cerr << endl;
          // cerr << "unoriented_not244 sums: ";
          // for (const auto& s : unoriented_not244_nz5_sums) {
          //   cerr << s << " ";
          // }
          // cerr << endl << endl << endl;
        }

        // has_int_nz6_flow = false;
        // if (!has_int_nz5_flow) {
        //   for (int e = 0; e < graph.number_of_edges; ++e) {
        //     cur_flow[e] = 0;
        //   }
        //   find_int_nz_from_o6c4c(graph, 6, 0);
        // } else {
        //   has_int_nz6_flow = true;
        // }

        // FIXME: where's find_both_nz_mod_from_o6c4c?
        // has_both_nz_mod_flow = false;
        // if (has_nz_mod5_flow && has_nz_mod6_flow) {
        //   for (int e = 0; e < graph.number_of_edges; ++e) {
        //     cur_flow[e] = 0;
        //   }
        //   find_both_nz_mod_from_o6c4c(graph, 0);
        // }

        vector<int> t1_edges_frequency = {0, 0, 0, 0};
        vector<int> t2_edges_frequency = {0, 0, 0, 0};
        vector<int> t3_edges_frequency = {0, 0, 0, 0};
        vector<int> t3o_edges_frequency = {0, 0, 0, 0};
        vector<int> t3u_edges_frequency = {0, 0, 0, 0};
        vector<int> t4_edges_frequency = {0, 0, 0, 0};
        set<int> t4_freq3_vertices;
        vector<int> single_t4_t2_edges_frequency = {0, 0, 0};
        for (int v = 0; v < graph.number_of_vertices; ++v) {
          int t1_edge_count = 0;
          int t2_edge_count = 0;
          int t3_edge_count = 0;
          int t4_edge_count = 0;
          for (int j = 0; j < MAX_DEG; ++j) {
            int e = graph.v2e[v][j];
            if (edge_type[e] == 0) {
              ++t1_edge_count;
            } else if (edge_type[e] == 1) {
              ++t2_edge_count;
            } else if (edge_type[e] == 2) {
              ++t3_edge_count;
            } else {
              assert(edge_type[e] == 3);
              ++t4_edge_count;
            }
          }
          ++t1_edges_frequency[t1_edge_count];
          ++t2_edges_frequency[t2_edge_count];
          ++t3_edges_frequency[t3_edge_count];
          if (oriented_vertices.find(v) != oriented_vertices.end()) {
            ++t3o_edges_frequency[t3_edge_count];
          } else {
            ++t3u_edges_frequency[t3_edge_count];
          }
          ++t4_edges_frequency[t4_edge_count];
          if (t4_edge_count == 1) {
            ++single_t4_t2_edges_frequency[t2_edge_count];
          }
          if (t4_edge_count == 3) {
            t4_freq3_vertices.insert(v);
          }
        }

        // if (total_poor_comps == 1) {
        //   assert(s0 % 2 == 0); // breaks on 28.05g611
        // }

        set<set<set<int>>> oriented_vertex_types;
        set<set<set<int>>> unoriented_vertex_types;
        for (int v = 0; v < graph.number_of_vertices; ++v) {
          set<set<int>> cur_vertex_type;
          for (int j = 0; j < MAX_DEG; ++j) {
            int e = graph.v2e[v][j];
            set<int> edge_matching_layers;
            for (int i = 0; i < 6; ++i) {
              if ((u6c4c_cycles[i] & BIT(e)) == 0) { // edge is from matching
                edge_matching_layers.insert(i);
              }
            }
            cur_vertex_type.insert(edge_matching_layers);
          }
          if (oriented_vertices.find(v) != oriented_vertices.end()) {
            oriented_vertex_types.insert(cur_vertex_type);
          } else {
            unoriented_vertex_types.insert(cur_vertex_type);
          }
        }
        set<set<set<int>>> both_vertex_types;
        for (const auto& vt: oriented_vertex_types) {
          if (unoriented_vertex_types.find(vt) != unoriented_vertex_types.end()) {
            both_vertex_types.insert(vt);
          }
        }

        // map<int, set<int>> v_to_c;
        /*int odd_or_circuits_count = 0;
        // vector<set<int>> odd_or_circuit_vertices;
        // set<int> odd_c;
        for (int c = 0; c < all_circuits_in_6c4c.size(); ++c) {
          int or_count = 0;
          set<int> ors;
          for (int vi = 0; vi < all_circuits_in_6c4c[c].size() - 1; ++vi) {
            int v = all_circuits_in_6c4c[c][vi];
            // v_to_c[v].insert(c);
            if (oriented_vertices.find(v) != oriented_vertices.end()) {
              ++or_count;
              ors.insert(v);
            }
          }
          if (or_count % 2 != 0) {
            ++odd_or_circuits_count;
            // odd_or_circuit_vertices.push_back(ors);
            // odd_c.insert(c);
          }
        }*/
        // int odd_or_odd_c_count = 0;
        // for (const auto& v : oriented_vertices) {
        //   int odd_c_count = 0;
        //   for (const auto& c : odd_c) {
        //     if (v_to_c[v].find(c) != v_to_c[v].end()) {
        //       ++odd_c_count;
        //     }
        //   }
        //   if (odd_c_count % 2 != 0) {
        //     ++odd_or_odd_c_count;
        //   }
        // }

        int odd_t1_2_factors = 0;
        int odd_t2_2_factors = 0;
        int odd_t3_2_factors = 0;
        int odd_t4_2_factors = 0;
        for (const auto& c : cur_6c4c) {
          int t1_count = 0;
          int t2_count = 0;
          int t3_count = 0;
          int t4_count = 0;
          for (int e = 0; e < graph.number_of_edges; ++e) {
            if ((c & BIT(e)) != 0) { // edge is from 2-factor
              if (edge_type[e] == 0) {
                ++t1_count;
              } else if (edge_type[e] == 1) {
                ++t2_count;
              } else if (edge_type[e] == 2) {
                ++t3_count;
              } else {
                ++t4_count;
              }
            }
          }
          if (t1_count % 2 != 0) {
            ++odd_t1_2_factors;
          }
          if (t2_count % 2 != 0) {
            ++odd_t2_2_factors;
          }
          if (t3_count % 2 != 0) {
            ++odd_t3_2_factors;
          }
          if (t4_count % 2 != 0) {
            ++odd_t4_2_factors;
          }
        }

        int even_t1_matchings = 0;
        int even_t2_matchings = 0;
        int even_t3_matchings = 0;
        int even_t4_matchings = 0;
        for (const auto& c : cur_6c4c) {
          int t1_count = 0;
          int t2_count = 0;
          int t3_count = 0;
          int t4_count = 0;
          for (int e = 0; e < graph.number_of_edges; ++e) {
            if ((c & BIT(e)) == 0) { // edge is from matching
              if (edge_type[e] == 0) {
                ++t1_count;
              } else if (edge_type[e] == 1) {
                ++t2_count;
              } else if (edge_type[e] == 2) {
                ++t3_count;
              } else {
                ++t4_count;
              }
            } else {
            }
          }
          if (t1_count % 2 == 0) {
            ++even_t1_matchings;
          }
          if (t2_count % 2 == 0) {
            ++even_t2_matchings;
          }
          if (t3_count % 2 == 0) {
            ++even_t3_matchings;
          }
          if (t4_count % 2 == 0) {
            ++even_t4_matchings;
          }
        }

        assert(odd_t1_2_factors % 2 == 0); // true
        odd_t1_2_factors /= 2;
        assert(odd_t2_2_factors % 2 == 0); // true
        odd_t2_2_factors /= 2;
        assert(even_t3_matchings % 2 == 0); // true
        even_t3_matchings /= 2;
        assert(even_t4_matchings % 2 == 0); // true
        even_t4_matchings /= 2;

        int t1_component[MAX_VERTEX];
        int t2_component[MAX_VERTEX];
        int t3_component[MAX_VERTEX];
        int t4_component[MAX_VERTEX];
        int edge_t1_component[MAX_EDGE];
        int edge_t2_component[MAX_EDGE];
        int edge_t3_component[MAX_EDGE];
        int edge_t4_component[MAX_EDGE];

        for (int v = 0; v < graph.number_of_vertices; ++v) {
          t1_component[v] = 0;
          t2_component[v] = 0;
          t3_component[v] = 0;
          t4_component[v] = 0;
        }

        for (int e = 0; e < graph.number_of_edges; ++e) {
          edge_t1_component[e] = 0;
          edge_t2_component[e] = 0;
          edge_t3_component[e] = 0;
          edge_t4_component[e] = 0;
        }

        set<int> vertices_neib_to_t1_edge;
        set<int> vertices_neib_to_t2_edge;
        set<int> vertices_neib_to_t3_edge;
        set<int> vertices_neib_to_t4_edge;
        for (int e = 0; e < graph.number_of_edges; ++e) {
          const int v1 = graph.e2v[e][0];
          const int v2 = graph.e2v[e][1];
          if (edge_type[e] == 0) {
            vertices_neib_to_t1_edge.insert(v1);
            vertices_neib_to_t1_edge.insert(v2);
          } else if (edge_type[e] == 1) {
            vertices_neib_to_t2_edge.insert(v1);
            vertices_neib_to_t2_edge.insert(v2);
          } else if (edge_type[e] == 2) {
            vertices_neib_to_t3_edge.insert(v1);
            vertices_neib_to_t3_edge.insert(v2);
          } else {
            assert(edge_type[e] == 3);
            vertices_neib_to_t4_edge.insert(v1);
            vertices_neib_to_t4_edge.insert(v2);
          }
        }

        int total_t1_comps = 0;
        for (int v = 0; v < graph.number_of_vertices; ++v) {
          if (vertices_neib_to_t1_edge.find(v) != vertices_neib_to_t1_edge.end()) {
            if (t1_component[v] != 0) {
              continue;
            }
            ++total_t1_comps;
            t1_component[v] = total_t1_comps;
            vector<int> queue;
            queue.push_back(v);
            int head = 0;
            while (head < queue.size()) {
              int v_head = queue[head];
              ++head;
              for (int j = 0; j < MAX_DEG; ++j) {
                int e = graph.v2e[v_head][j];
                if (edge_type[e] == 0) {
                  edge_t1_component[e] = total_t1_comps;
                  int v2 = graph.v2v[v_head][j];
                  if (t1_component[v2] == 0) {
                    t1_component[v2] = total_t1_comps;
                    queue.push_back(v2);
                  }
                }
              }
            }
          }
        }

        int total_t2_comps = 0;
        for (int v = 0; v < graph.number_of_vertices; ++v) {
          if (vertices_neib_to_t2_edge.find(v) != vertices_neib_to_t2_edge.end()) {
            if (t2_component[v] != 0) {
              continue;
            }
            ++total_t2_comps;
            t2_component[v] = total_t2_comps;
            vector<int> queue;
            queue.push_back(v);
            int head = 0;
            while (head < queue.size()) {
              int v_head = queue[head];
              ++head;
              for (int j = 0; j < MAX_DEG; ++j) {
                int e = graph.v2e[v_head][j];
                if (edge_type[e] == 1) {
                  edge_t2_component[e] = total_t2_comps;
                  int v2 = graph.v2v[v_head][j];
                  if (t2_component[v2] == 0) {
                    t2_component[v2] = total_t2_comps;
                    queue.push_back(v2);
                  }
                }
              }
            }
          }
        }

        int total_t3_comps = 0;
        for (int v = 0; v < graph.number_of_vertices; ++v) {
          if (vertices_neib_to_t3_edge.find(v) != vertices_neib_to_t3_edge.end()) {
            if (t3_component[v] != 0) {
              continue;
            }
            ++total_t3_comps;
            t3_component[v] = total_t3_comps;
            vector<int> queue;
            queue.push_back(v);
            int head = 0;
            while (head < queue.size()) {
              int v_head = queue[head];
              ++head;
              for (int j = 0; j < MAX_DEG; ++j) {
                int e = graph.v2e[v_head][j];
                if (edge_type[e] == 2) {
                  edge_t3_component[e] = total_t3_comps;
                  int v2 = graph.v2v[v_head][j];
                  if (t3_component[v2] == 0) {
                    t3_component[v2] = total_t3_comps;
                    queue.push_back(v2);
                  }
                }
              }
            }
          }
        }

        int total_t4_comps = 0;
        for (int v = 0; v < graph.number_of_vertices; ++v) {
          if (vertices_neib_to_t4_edge.find(v) != vertices_neib_to_t4_edge.end()) {
            if (t4_component[v] != 0) {
              continue;
            }
            ++total_t4_comps;
            t4_component[v] = total_t4_comps;
            vector<int> queue;
            queue.push_back(v);
            int head = 0;
            while (head < queue.size()) {
              int v_head = queue[head];
              ++head;
              for (int j = 0; j < MAX_DEG; ++j) {
                int e = graph.v2e[v_head][j];
                if (edge_type[e] == 3) {
                  edge_t4_component[e] = total_t4_comps;
                  int v2 = graph.v2v[v_head][j];
                  if (t4_component[v2] == 0) {
                    t4_component[v2] = total_t4_comps;
                    queue.push_back(v2);
                  }
                }
              }
            }
          }
        }

        int odd_t1_comps_2_factors = 0;
        int odd_t2_comps_2_factors = 0;
        int odd_t3_comps_2_factors = 0;
        int odd_t4_comps_2_factors = 0;
        int odd_t1_comps_matchings = 0;
        int odd_t2_comps_matchings = 0;
        int odd_t3_comps_matchings = 0;
        int odd_t4_comps_matchings = 0;
        for (const auto& c : cur_6c4c) {
          set<int> t1_comps_edges_in_2_factor;
          set<int> t1_comps_edges_in_matching;
          set<int> t2_comps_edges_in_2_factor;
          set<int> t2_comps_edges_in_matching;
          set<int> t3_comps_edges_in_2_factor;
          set<int> t3_comps_edges_in_matching;
          set<int> t4_comps_edges_in_2_factor;
          set<int> t4_comps_edges_in_matching;

          for (int e = 0; e < graph.number_of_edges; ++e) {
            if ((c & BIT(e)) == 0) { // edge is from matching
              if (edge_type[e] == 0) {
                t1_comps_edges_in_matching.insert(edge_t1_component[e]);
              } else if (edge_type[e] == 1) {
                t2_comps_edges_in_matching.insert(edge_t2_component[e]);
              } else if (edge_type[e] == 2) {
                t3_comps_edges_in_matching.insert(edge_t3_component[e]);
              } else {
                assert(edge_type[e] == 3);
                t4_comps_edges_in_matching.insert(edge_t4_component[e]);
              }
            } else { // edge is from full cycle
              if (edge_type[e] == 0) {
                t1_comps_edges_in_2_factor.insert(edge_t1_component[e]);
              } else if (edge_type[e] == 1) {
                t2_comps_edges_in_2_factor.insert(edge_t2_component[e]);
              } else if (edge_type[e] == 2) {
                t3_comps_edges_in_2_factor.insert(edge_t3_component[e]);
              } else {
                assert(edge_type[e] == 3);
                t4_comps_edges_in_2_factor.insert(edge_t4_component[e]);
              }
            }
          }

          if (t1_comps_edges_in_2_factor.size() % 2 != 0) {
            ++odd_t1_comps_2_factors;
          }
          if (t2_comps_edges_in_2_factor.size() % 2 != 0) {
            ++odd_t2_comps_2_factors;
          }
          if (t3_comps_edges_in_2_factor.size() % 2 != 0) {
            ++odd_t3_comps_2_factors;
          }
          if (t4_comps_edges_in_2_factor.size() % 2 != 0) {
            ++odd_t4_comps_2_factors;
          }

          if (t1_comps_edges_in_matching.size() % 2 != 0) {
            ++odd_t1_comps_matchings;
          }
          if (t2_comps_edges_in_matching.size() % 2 != 0) {
            ++odd_t2_comps_matchings;
          }
          if (t3_comps_edges_in_matching.size() % 2 != 0) {
            ++odd_t3_comps_matchings;
          }
          if (t4_comps_edges_in_matching.size() % 2 != 0) {
            ++odd_t4_comps_matchings;
          }
        }

        // assert(odd_poor_2_factors % 2 == 0); // this is true
        // odd_poor_2_factors /= 2;
        //
        // assert(odd_rich_2_factors % 2 == 0); // this is true
        // odd_rich_2_factors /= 2;
        //
        // assert(odd_poor_matchings % 2 == 0); // this is true
        // odd_poor_matchings /= 2;
        //
        // assert(odd_rich_matchings % 2 == 0); // this is true
        // odd_rich_matchings /= 2;
        //
        // assert(odd_poor_comps_2_factors % 2 == 0); // TODO
        // odd_poor_comps_2_factors /= 2;
        //
        // assert(odd_rich_comps_2_factors % 2 == 0); // TODO
        // odd_rich_comps_2_factors /= 2;
        //
        // assert(odd_poor_comps_matchings % 2 == 0); // TODO!
        // odd_poor_comps_matchings /= 2;

        // // this is false!
        // assert(odd_rich_comps_matchings % 2 == 0);
        // odd_rich_comps_matchings /= 2;

        vector<int> rich_rich_neibs = {0, 0, 0, 0, 0};
        vector<int> poor_rich_neibs = {0, 0, 0, 0, 0};
        for (int e = 0; e < graph.number_of_edges; ++e) {
          int rich_count = 0;
          for (int i = 0; i < 2; ++i) {
            int v = graph.e2v[e][i];
            for (int j = 0; j < MAX_DEG; ++j) {
              int e2 = graph.v2e[v][j];
              if (e2 == e) {
                continue;
              }
              if (!u6c4c_edge_is_poor[e2]) {
                ++rich_count;
              }
            }
          }
          if (u6c4c_edge_is_poor[e]) {
            ++poor_rich_neibs[rich_count];
          } else {
            ++rich_rich_neibs[rich_count];
          }
        }

        vector<tuple<int, int>> circuit_info;
        for (int c = 0; c < all_circuits_in_6c4c.size(); ++c) {
          int or_count = 0;
          for (int vi = 0; vi < all_circuits_in_6c4c[c].size() - 1; ++vi) {
            int v = all_circuits_in_6c4c[c][vi];
            bool v_is_oriented = (oriented_vertices.find(v) != oriented_vertices.end());
            if (v_is_oriented) {
              ++or_count;
            }
          }
          circuit_info.push_back(make_tuple(
            all_circuits_in_6c4c[c].size() - 1,
            or_count
          ));
        }
        sort(circuit_info.begin(), circuit_info.end());

        vector<int> or_or_neibs = {0, 0, 0, 0};
        vector<int> unor_or_neibs = {0, 0, 0, 0};
        for (int v = 0; v < graph.number_of_vertices; ++v) {
          bool v_is_oriented = (oriented_vertices.find(v) != oriented_vertices.end());
          int or_neibs_count = 0;
          for (int j = 0; j < MAX_DEG; ++j) {
            int v2 = graph.v2v[v][j];
            bool v2_is_oriented = (oriented_vertices.find(v2) != oriented_vertices.end());
            if (v2_is_oriented) {
              ++or_neibs_count;
            }
          }
          if (v_is_oriented) {
            ++or_or_neibs[or_neibs_count];
          } else {
            ++unor_or_neibs[or_neibs_count];
          }
        }

        map<int, pair<int, int>> edge_to_layers;
        for (int e = 0; e < graph.number_of_edges; ++e) {
          int l1 = NONE;
          int l2 = NONE;
          for (int i = 0; i < 6; ++i) {
            if ((u6c4c_cycles[i] & BIT(e)) == 0) { // edge is from matching
              if (l1 == NONE) {
                l1 = i;
              } else {
                l2 = i;
              }
            }
          }
          edge_to_layers[e] = make_pair(l1, l2);
        }
        // cerr << "debug" << endl;
        set<tuple<tuple<int, int>, vector<vector<tuple<int, int>>>>> rich_edge_layer_types;
        for (int e = 0; e < graph.number_of_edges; ++e) {
          if (u6c4c_edge_is_poor[e]) {
            continue;
          }
          vector<vector<tuple<int, int>>> fours_of_layers;
          for (int i = 0; i < 2; ++i) {
            vector<tuple<int, int>> l1l2s;
            int v = graph.e2v[e][i];
            for (int j = 0; j < MAX_DEG; ++j) {
              int e2 = graph.v2e[v][j];
              if (e2 == e) {
                continue;
              }
              l1l2s.push_back(edge_to_layers[e2]);
            }
            sort(l1l2s.begin(), l1l2s.end());
            fours_of_layers.push_back(l1l2s);
          }
          sort(fours_of_layers.begin(), fours_of_layers.end());
          rich_edge_layer_types.insert(make_pair(edge_to_layers[e], fours_of_layers));
          // cerr << edge_to_layers[e].first << " " << edge_to_layers[e].second << ": ";
          // cerr << get<0>(fours_of_layers[0][0]) << " " <<
          //         get<1>(fours_of_layers[0][0]) << " " <<
          //         get<0>(fours_of_layers[0][1]) << " " <<
          //         get<1>(fours_of_layers[0][1]) << " " <<
          //         get<0>(fours_of_layers[1][0]) << " " <<
          //         get<1>(fours_of_layers[1][0]) << " " <<
          //         get<0>(fours_of_layers[1][1]) << " " <<
          //         get<1>(fours_of_layers[1][1]) << endl;
        }

        if (graph.number > 0) {
          graph.u6c4c_profiles_to_orientations[prof].insert(oriented_vertices.size() % 2);
          if (graph.u6c4c_profiles_to_orientations[prof].size() == 2) { // e. g., 28.05g2869
            cerr << "SAME PROFILE, DIFFERENT ORIENTED VERTICES COUNT" << endl;
          }

          assert((s0 + circuits_even_rich) % 2 == 0); // this is true
          assert((s0 + circuits_even_oriented_vertices) % 2 == 0); // this is true

          if (oriented_vertices.size() == 0) {
            assert(s2 == 3); // TODO
          }

          cerr << "g" << graph.number << ": ";
          cerr << "new o6c4c: ";
          cerr << "or: ";
          if (oriented_vertices.size() < 10) {
            cerr << "0";
          }
          cerr << oriented_vertices.size() << "; ";

          cerr << "s0: " << s0 << "; ";
          cerr << "s1: " << s1 << "; ";
          cerr << "s2: " << s2 << "; ";

          // cerr << "rich244:";
          int rich_244_odd_count = 0;
          for (const auto& c : rich_244_counts) {
            // cerr << "_" << c;
            if (c % 2 != 0) {
              rich_244_odd_count += 1;
            } else {
            }
          }
          // cerr << "; ";
          cerr << "r244odd: " << rich_244_odd_count << "; ";

          // cerr << "s2uu: " << even_t4_matchings << "; ";

          int old_parity = (s0 + s1 + s2) % 2;
          // rules for creating new parity:
          // i can use or, s0, s1, s2, t1, ..., t5, ...
          // following are forbidden:
          // t3 is forbidden (== or (mod 2))
          // t4 is forbidden, t4 = s1 - t3
          // t2 is forbidden, t2 = e - t1 - t3 - t4
          // circuits_even_poor is forbidden (== s0 (mod 2))
          // and probably other circuits_even/odd_poor/rich counts also
          int parity     = (old_parity) % 2;
          cerr << "PAR: " << parity << "; ";// << " (" << old_parity << "); ";

          // int s0s1orsum = NONE;
          // if (even_t4_matchings == 3) {
          //   s0s1orsum = (s0 + s1 + oriented_vertices.size()) % 2;
          //   cerr << "s0s1orsum: " << s0s1orsum << "; ";
          // }

          int npar = (s0 + s1 + oriented_vertices.size() + even_t4_matchings) % 2;
          assert(npar == old_parity); // todo?
          // cerr << "npar: " << npar << "; ";

          cerr << "reors: " << same_cycles_different_orientations << "; ";
          cerr << "em:_" << even_t1_matchings << "_" << even_t2_matchings << "_" <<
            even_t3_matchings << "_" << even_t4_matchings << "; ";
          cerr << "o2:_" << odd_t1_2_factors << "_" << odd_t2_2_factors << "_" <<
            odd_t3_2_factors << "_" << odd_t4_2_factors << "; ";

          cerr << "t1+t3: ";
          if (t1 + t3 < 10) {
            cerr << "0";
          }
          cerr << t1 + t3 << "; ";

          if (t1 + t3 <= 8 && (t1 + t3 != 6)) {
            assert(s2 != 0); // TODO
          }

          // cerr << "o244: " << o244_triples.size() << "; ";

          // cerr << "pairings: ";
          // for (const auto& v : oriented_vertices) {
          //   for (const auto& pair : layer_pairings[v]) {
          //     for (const auto& v2 : pair) {
          //       cerr << v2 << "_";
          //     }
          //     cerr << "|";
          //   }
          //   cerr << "/";
          // }
          // cerr << "; ";

          map<set<set<int>>, int> or_pairings;
          for (const auto& v : oriented_vertices) {
            or_pairings[layer_pairings[v]] += 1;
          }
          cerr << "or_type_count: " << or_pairings.size() << "; ";
          cerr << "or_type_decomp:";
          vector<int> or_type_counts;
          for (const auto& p : or_pairings) {
            or_type_counts.push_back(p.second);
          }
          sort(or_type_counts.begin(), or_type_counts.end());
          for (const auto& c : or_type_counts) {
            cerr << "_" << c;
          }
          cerr << "; ";

          cerr << "rich_type_count: " << rich_edge_layer_types.size() << "; ";

          cerr << "less: " << (t1 + t3 <= oriented_vertices.size() * 2 + 2) << "; ";

          cerr << "or_counts:";
          for (const auto& c : ors) {
            cerr << "_" << c;
          }
          cerr << "; ";

          cerr << "t1: " << t1 << "; ";
          cerr << "t2: " << t2 << "; ";
          cerr << "t3: " << t3 << "; ";
          cerr << "t4: " << t4 << "; ";

          // cerr << "u_comps: " << total_poor_comps << " " << total_rich_comps << "; ";
          // cerr << "u_morecomps_undiv: " <<
          //   odd_poor_comps_2_factors << " " <<
          //   odd_rich_comps_2_factors << " " <<
          //   odd_poor_comps_matchings << " " <<
          //   odd_rich_comps_matchings << "; ";

          // cerr << "apcc: " << almost_poor_circuits_count << "; ";

          // cerr << "comps: " <<
          //     odd_t1_comps_2_factors << " " << odd_t2_comps_2_factors << " " <<
          //     odd_t3_comps_2_factors << " " << odd_t4_comps_2_factors << " " <<
          //     odd_t1_comps_matchings << " " << odd_t2_comps_matchings << " " <<
          //     odd_t3_comps_matchings << " " << odd_t4_comps_matchings << "; ";

          cerr << "rov: " << rich_oriented_vertices_frequency[0] << " " <<
                             rich_oriented_vertices_frequency[1] << " " <<
                             rich_oriented_vertices_frequency[2] << " " <<
                             rich_oriented_vertices_frequency[3] << "; ";

          cerr << "ruv: " << rich_unoriented_vertices_frequency[0] << " " <<
                             rich_unoriented_vertices_frequency[1] << " " <<
                             rich_unoriented_vertices_frequency[2] << " " <<
                             rich_unoriented_vertices_frequency[3] << "; ";

          // cerr << "dup_oon: " <<
          //   or_or_neibs[0] << " " <<
          //   or_or_neibs[1] << " " <<
          //   or_or_neibs[2] << " " <<
          //   or_or_neibs[3] << "; ";

          cerr << "uon: " <<
            unor_or_neibs[0] << " " <<
            unor_or_neibs[1] << " " <<
            unor_or_neibs[2] << " " <<
            unor_or_neibs[3] << "; ";

          cerr << "rrn: " <<
            rich_rich_neibs[0] << " " <<
            rich_rich_neibs[1] << " " <<
            rich_rich_neibs[2] << " " <<
            rich_rich_neibs[3] << " " <<
            rich_rich_neibs[4] << "; ";

          cerr << "prn: " <<
            poor_rich_neibs[0] << " " <<
            poor_rich_neibs[1] << " " <<
            poor_rich_neibs[2] << " " <<
            poor_rich_neibs[3] << " " <<
            poor_rich_neibs[4] << "; ";

          assert((rich_unoriented_vertices_frequency[0] + rich_unoriented_vertices_frequency[2]) % 2 == 0); // this is true
          assert((rich_oriented_vertices_frequency[0] + rich_oriented_vertices_frequency[2]) % 2 == 0); // this is true

          // FIXMEFIXME
          bool is_mismatch = false;
          if (has_nz_mod5_flow && !has_nz_mod6_flow) { // happens quite often
            is_mismatch = true;
            assert(s0 % 2 == 0); // TODO
            assert(oriented_vertices.size() > 2); // TODO
          }

          if (has_nz_mod5_flow && has_nz_mod6_flow && !has_nz_modb_flow) { // happens quite often
            is_mismatch = true;
            assert(s0 % 2 == 0); // TODO
            assert(oriented_vertices.size() > 2); // TODO
          }

          /**********/

          // cerr << "orverts: ";
          // for (const auto& v : oriented_vertices) {
          //   cerr << v << " ";
          // }
          // cerr << "; ";

          cerr << "has_nz5: " << has_int_nz5_flow << "; ";
          cerr << "mismatch: " << is_mismatch << "; ";
          cerr << "has_nzmod5: " << has_nz_mod5_flow << "; ";
          cerr << "has_nzmod6: " << has_nz_mod6_flow << "; ";
          cerr << "has_nzmodb: " << has_nz_modb_flow << "; ";

          if (has_nz_mod5_flow) {
            cerr << "flow5: ";
            // for (int e = 0; e < graph.number_of_edges; ++e) {
            //   cerr << cur_flow_mod5[e] << " ";
            // }
            // cerr << "; ";
            int or_sum = 0;
            for (int v = 0; v < graph.number_of_vertices; ++v) {
              bool v_is_oriented = oriented_vertices.find(v) != oriented_vertices.end();
              if (v_is_oriented) {
                cerr << "o";
              } else {
                cerr << "u";
              }
              vector<int> types;
              for (int j = 0; j < MAX_DEG; ++j) {
                int e = graph.v2e[v][j];
                int v2 = graph.v2v[v][j];
                int flow_val = cur_flow_mod5[e];
                if (v2 < v) {
                  flow_val = -flow_val;
                }
                flow_val = (5 * 6 + flow_val) % 5;
                types.push_back(flow_val);
              }
              sort(types.begin(), types.end());
              int type_as_int = 0;
              for (const auto& t : types) {
                cerr << t;
                type_as_int = type_as_int * 10 + t;
              }
              cerr << "_";
              if (v_is_oriented) {
                int v_type = 0;
                if (type_as_int == 122) {
                  v_type = 1;
                } else if (type_as_int == 244) {
                  v_type = 2;
                } else if (type_as_int == 113) {
                  v_type = 3;
                } else {
                  assert(type_as_int == 334);
                  v_type = 4;
                }
                or_sum += v_type;
              }
            }
            cerr << "; ";
            cerr << "or_sum: " << or_sum << "; ";
          } else {
            cerr << "flow5: nope; or_sum: nope; ";
          }

          // if (has_nz_mod5_flow) {
          //   cerr << "w_mod5:";
          //   for (int c = 0; c < 6; ++c) {
          //     cerr << " " << layer_weights_nz_mod5[c];
          //   }
          //   cerr << "; ";
          // }

          // if (has_nz_modb_flow) {
          //   cerr << "w_modb:";
          //   for (int c = 0; c < 6; ++c) {
          //     cerr << " " << layer_weights_nz_modb[c];
          //   }
          //   cerr << "; ";

          //   cerr << "flowb: ";
          //   for (int e = 0; e < graph.number_of_edges; ++e) {
          //     cerr << cur_flow_modb[e] << " ";
          //   }
          //   cerr << "; ";
          // }

          // FIXMEFIXME
          // if (oriented_vertices.size() == 0) {
          // cerr << "or0: ";
          // cerr << "sum: " << s1 - s0 - (rich_unoriented_vertices_frequency[0] + rich_unoriented_vertices_frequency[2]) << "; ";
          // cerr << "sum: " << s1 - s0 - rich_unoriented_vertices_frequency[3] << "; ";
          // cerr << "sum: " << (rich_unoriented_vertices_frequency[3]+rich_unoriented_vertices_frequency[1]) - s1 + s0 << "; ";
          // int genus = 1 - ((rich_unoriented_vertices_frequency[3]+rich_unoriented_vertices_frequency[1]) - s1 + s0)/2;
          // cerr << "genus: " << genus << "; ";
          // int g2 = 1 - ((rich_unoriented_vertices_frequency[3]+rich_unoriented_vertices_frequency[1]) - s1 + circuits_even_rich)/2;
          // cerr << "g2: " << g2 << "; ";
          // int g3 = 1 - ((rich_unoriented_vertices_frequency[3]+rich_unoriented_vertices_frequency[1]) - s1 + circuits_even_poor)/2;
          // cerr << "g3: " << g3 << "; ";
          // int g4 = 1 - ((rich_unoriented_vertices_frequency[3]+rich_unoriented_vertices_frequency[1]) - s1 + circuits_even_len)/2;
          // cerr << "g4: " << g4 << "; ";

          cerr << "rrn024: " << rich_rich_neibs[0] + rich_rich_neibs[2] + rich_rich_neibs[4] << "; ";
          cerr << "s1s0diff: " << s1-s0 << "; ";
          
          // s0 has same parity as following variables:
          // circuits_even_rich, circuits_even_poor, circuits_even_len
          // cerr << "sames: ";
          // cerr << "cer: " << circuits_even_rich << "; ";
          // cerr << "cep: " << circuits_even_poor << "; ";
          // cerr << "cel: " << circuits_even_len << "; ";
          // cerr << "ceo: " << circuits_even_or << "; ";
          // cerr << "cop: " << s0-circuits_even_poor << "; ";
          cerr << "ruv13: " << rich_unoriented_vertices_frequency[1] + rich_unoriented_vertices_frequency[3] << "; ";

          cerr << "evens: ";
          cerr << "col: " << s0-circuits_even_len << "; ";
          cerr << "rrn13: " << rich_rich_neibs[1] + rich_rich_neibs[3] << "; ";
          cerr << "ruv02: " << rich_unoriented_vertices_frequency[0] + rich_unoriented_vertices_frequency[2] << "; ";

          cerr << "chord_info: (" <<
            any_chords_frequency[0] << " " << any_chords_frequency[1] << " " << any_chords_frequency[2] << ") " <<
            "t1(" << t1_chords_frequency[0] << " " << t1_chords_frequency[1] << " " << t1_chords_frequency[2] << ") " <<
            "t2(" << t2_chords_frequency[0] << " " << t2_chords_frequency[1] << " " << t2_chords_frequency[2] << ") " <<
            "t3(" << t3_chords_frequency[0] << " " << t3_chords_frequency[1] << " " << t3_chords_frequency[2] << ") " <<
            "t4(" << t4_chords_frequency[0] << " " << t4_chords_frequency[1] << " " << t4_chords_frequency[2] << "); ";

          cerr << "done; ";

          if (s1 == graph.number_of_edges) {
            cerr << "onlyrich; ";
          }

          // vector<tuple<int, int, int>> chord_layers;
          // for (int i = 0; i < 6; ++i) {
          //   chord_layers.push_back(
          //     make_tuple(
          //       circuit_count_by_layer[i],
          //       chord_count_by_layer[i],
          //       t4_chord_count_by_layer[i]));
          // }
          // sort(chord_layers.begin(), chord_layers.end());
          // cerr << "chord_layers:";
          // for (int i = 0; i < 6; ++i) {
          //   cerr << " (" <<
          //     get<0>(chord_layers[i]) << " " <<
          //     get<1>(chord_layers[i]) << " " <<
          //     get<2>(chord_layers[i]) << ")";
          // }
          // cerr << "; ";

          // cerr << "circuit_info:";
          // for (int i = 0; i < circuit_info.size(); ++i) {
          //   cerr << " (" <<
          //     get<0>(circuit_info[i]) << " " <<
          //     get<1>(circuit_info[i]) << ")";
          // }
          // cerr << "; ";

          bool also_pet = false;
          if (graph.petersen_6c4c.find(cur_6c4c) != graph.petersen_6c4c.end()) {
            also_pet = true;
          }

          pair<int, int> p(s0, s1);
          if (also_pet) {
            graph.s0s1_pet.insert(p);
            assert(has_2cdcs == 1); // this is true
          }

          if (old_parity == 0) {
            graph.s0s1_par0.insert(p);
          }

          if (has_2cdcs) {
            graph.o6c4c_2xcdcs_profiles.insert(prof);
            cerr << "has_2cdcs; ";
          } else {
            cerr << "no_2cdcs; ";
          }

          if (has_2cdcs) {
            cerr << "taken: " << taken_c1 << " " << taken_c2 << "; ";
            assert(layers_c1.size() == 6); // TODO
            assert(layers_c2.size() == 6); // TODO
            assert(s0 % 2 == 0); // TODO
            assert((oriented_vertices.size() % 2) == (s1 % 2)); // TODO
            assert(odd_poor_2_factors == 0); // TODO
            assert((s2 != 1) && (s2 != 2)); // TODO

            map<pair<int, int>, int> counts;
            for (int ci = 0; ci < all_circuits_in_6c4c.size(); ++ci) {
              if (!two_cdcs_taken[ci]) {
                continue;
              }
              for (int vi = 0; vi < all_circuits_in_6c4c[ci].size() - 1; ++vi) {
                int v1 = all_circuits_in_6c4c[ci][vi];
                int v2 = all_circuits_in_6c4c[ci][vi + 1];
                if (orientations[ci] != 0) {
                  swap(v1, v2);
                }
                counts[make_pair(v1, v2)] += 1;
              }
            }
            int rich_doubly_oriented = 0;
            int poor_doubly_oriented = 0;
            for (const auto& p : counts) {
              if (p.second != 2) {
                continue;
              }
              int v1 = p.first.first;
              int v2 = p.first.second;
              int e = graph.vv2e[v1][v2];
              if (u6c4c_edge_is_poor[e]) {
                ++poor_doubly_oriented;
              } else {
                ++rich_doubly_oriented;
              }
            }

            int t5 = poor_doubly_oriented;
            cerr << "t5: " << t5 << " " << t2 - t5 << "; ";
            // cerr << "magic: " << (t1+t3+t2-t5+taken_c1) % 2 << "; ";
            assert(rich_doubly_oriented == t4); // this is true

            if (two_cdcs_vertices.size() > 0) {
              cerr << "2cdcs vertices: ";
              for (const auto& verts : two_cdcs_vertices) {
                cerr << "(";
                for (const auto& vv : verts) {
                  cerr << vv.first << ": ";
                  for (const auto& v : vv.second) {
                    cerr << v << " ";
                  }
                  cerr << "; ";
                }
                cerr << ") ";
              }
              cerr << "; ";
            }
          }

          // cerr << "vertices descriptions: ";
          // for (int v = 0; v < graph.number_of_vertices; ++v) {
          //   if (oriented_vertices.find(v) != oriented_vertices.end()) {
          //     cerr << "o";
          //   } else {
          //     cerr << "u";
          //   }
          //   cerr << " ";
          //   vector<int> types;
          //   for (int j = 0; j < MAX_DEG; ++j) {
          //     int e = graph.v2e[v][j];
          //     types.push_back(edge_type[e] + 1);
          //   }
          //   sort(types.begin(), types.end());
          //   for (const auto& t : types) {
          //     cerr << "t" << t << " ";
          //   }
          //   cerr << "; ";
          // }

          multiset<multiset<int>> ve_types;
          for (int v = 0; v < graph.number_of_vertices; ++v) {
            multiset<int> ve_type;
            if (oriented_vertices.find(v) != oriented_vertices.end()) {
              ve_type.insert(5);
            } else {
              ve_type.insert(6);
            }
            for (int j = 0; j < MAX_DEG; ++j) {
              int e = graph.v2e[v][j];
              ve_type.insert(edge_type[e]);
            }
            ve_types.insert(ve_type);
          }
          if (all_ve_types.find(ve_types) != all_ve_types.end()) {
            if (all_ve_types[ve_types] != (s0 % 2)) {
              cerr << "SAME TYPES DIFF S0 parity" << endl;
            }
          } else {
            all_ve_types[ve_types] = s0 % 2;
          }

          // FIXMEFIXMEFIXME
          // if (oriented_vertices.size() == 0 || has_2cdcs) {
          // if (has_2cdcs) {
          // FIXME: commenting out cerrs
          if (false) {
            cerr << "circuits:" << endl;
            for (int c = 0; c < all_circuits_in_6c4c.size(); ++c) {
              cerr << "layer: " << layer[c] << "; vertices: ";
              if (orientations[c] == 0) {
                for (int vi = 0; vi < all_circuits_in_6c4c[c].size(); ++vi) {
                  int v1 = all_circuits_in_6c4c[c][vi];
                  cerr << v1 << " ";
                }
              } else {
                for (int vi = all_circuits_in_6c4c[c].size() - 1; vi >= 0; --vi) {
                  int v1 = all_circuits_in_6c4c[c][vi];
                  cerr << v1 << " ";
                }
              }
              cerr << endl;
            }
            cerr << "matchings: " << endl;
            for (int i = 0; i < 6; ++i) {
              cerr << "layer: " << i << "; edges: ";
              for (int e = 0; e < graph.number_of_edges; ++e) {
                if ((u6c4c_cycles[i] & BIT(e)) == 0) { // edge is from matching
                  cerr << e << "(" << graph.e2v[e][0] << ", " << graph.e2v[e][1] << ") ";
                }
              }
              cerr << endl;
            }
            cerr << "circuits vertices descriptions:" << endl;
            for (int c = 0; c < all_circuits_in_6c4c.size(); ++c) {
              cerr << "layer: " << layer[c] << "; vertices:" << endl;
              for (int vi = 0; vi < all_circuits_in_6c4c[c].size() - 1; ++vi) {
                int v1 = all_circuits_in_6c4c[c][vi];
                //cerr << v1 << " ";
                if (oriented_vertices.find(v1) != oriented_vertices.end()) {
                  cerr << "o";
                } else {
                  cerr << "u";
                }
                cerr << " : ";
                for (int j = 0; j < MAX_DEG; ++j) {
                  int e = graph.v2e[v1][j];
                  if ((u6c4c_cycles[layer[c]] & BIT(e)) == 0) { // edge is from matching
                    // cerr << e << " (t" << edge_type[e] + 1 << ");   ";
                    cerr << "t" << edge_type[e] + 1 << " | ";
                    break;
                  }
                }
                vector<int> types;
                for (int j = 0; j < MAX_DEG; ++j) {
                  int e = graph.v2e[v1][j];
                  if ((u6c4c_cycles[layer[c]] & BIT(e)) != 0) { // edge is from 2-factor
                    types.push_back(edge_type[e] + 1);
                    //cerr << e << " (t" << edge_type[e] + 1 << ") ";
                  }
                }
                sort(types.begin(), types.end());
                for (const auto& t : types) {
                  cerr << "t" << t << " ";
                }
                cerr << endl;
              }
            }
          }

          // TODO: this whole if is unexplained, only also_pet is understood
          // actually - can remove also_pet (because there's has_2cdcs)

          if (
              has_2cdcs ||
              also_pet ||
              (s1 <= 18) ||
              (t1 + t3 <= 8) ||
              (
                (
                  (t1 + t3 <= oriented_vertices.size() * 2 + 2) || (t1 + t3 == 9)
                ) &&
                (
                  (s2 == 0) || (s2 == 3)
                )
              ))
          {
            cerr << "SEAL; ";
            assert(old_parity == 1);
            assert(npar == 1);
          }

          if (also_pet) {
            cerr << "also_pet! ";
          }

          /*vector<set<vector<int>>> triples(6);
          for (int c = 0; c < all_circuits_in_6c4c.size(); ++c) {
            int len = all_circuits_in_6c4c[c].size() - 1;
            for (int vi = 0; vi < len; ++vi) {
              int v1 = all_circuits_in_6c4c[c][vi];
              int v2 = all_circuits_in_6c4c[c][(vi + 1) % len];
              int v3 = all_circuits_in_6c4c[c][(vi + 2) % len];
              if (orientations[c] == 0) {
                vector<int> s = {v1, v2, v3};
                triples[layer[c]].insert(s);
              } else {
                vector<int> s = {v3, v2, v1};
                triples[layer[c]].insert(s);
              }
            }
          }

          set<pair<int, int>> allowed_pairs;
          for (int i = 0; i < 6; ++i) {
            for (int j = i + 1; j < 6; ++j) {
              allowed_pairs.insert(make_pair(i, j));
            }
          }

          for (int i = 0; i < 6; ++i) {
            for (int j = i + 1; j < 6; ++j) {
              for (const auto& t : triples[i]) {
                if (triples[j].find(t) != triples[j].end()) {
                  allowed_pairs.erase(make_pair(i, j));
                  break;
                }
              }
            }
          }

          vector<pair<int, int>> allowed_pairs_vec;
          for (const auto& p : allowed_pairs) {
            allowed_pairs_vec.push_back(p);
          }

          bool found_triple = false;

          for (int i = 0; i < allowed_pairs_vec.size(); ++i) {
            for (int j = i + 1; j < allowed_pairs_vec.size(); ++j) {
              for (int k = j + 1; k < allowed_pairs_vec.size(); ++k) {
                set<int> nums;
                nums.insert(allowed_pairs_vec[i].first);
                nums.insert(allowed_pairs_vec[i].second);
                nums.insert(allowed_pairs_vec[j].first);
                nums.insert(allowed_pairs_vec[j].second);
                nums.insert(allowed_pairs_vec[k].first);
                nums.insert(allowed_pairs_vec[k].second);
                if (nums.size() == 6) {
                  found_triple = true;
                }
              }
            }
          }
          if (found_triple) {
            cerr << "pair-partition; ";
            if (old_parity == 0) {
              cerr << "WATAFAK" << endl;
            }
          }*/

          // FIXME: commenting out cerrs
          // but this one is actually the most useful one
          cerr << endl;

          graph.u6c4c_profiles_to_parities[prof].insert(parity);
          // FIXME: commenting out cerrs
          // if (graph.u6c4c_profiles_to_parities[prof].size() == 2) {
          //   cerr << "same profile, different parity" << endl;
          // }

          // FIXMEFIXME
          if (dom_circ == NONE) {
            has_compatible_dominating_circuit(graph);
          }

          // FIXMEFIXME
          // if (dom_circ != NONE) {
          if ((npar == 0) && (dom_circ != NONE)) {
          // if ((s0s1orsum == 1) && (dom_circ != NONE)) {
          // if (has_2cdcs && (dom_circ != NONE)) {
          // if (s0 > any_chords_frequency[0] && (dom_circ != NONE)) {

          //if (old_parity == 0) {
          //if (true) {
          //if (prof == "rrrppprpprrrprrrrpprprrrrrrrrrppppprrrr") {
          //if (t1 + t3 == 14 && oriented_vertices.size() == 5) {
          //if (s1 <= 18) {
            set<int> poor_edges;
            for (int e = 0; e < graph.number_of_edges; ++e) {
              if (u6c4c_edge_is_poor[e]) {
                poor_edges.insert(e);
              }
            }
            // TODO: slow
            //graph.print(oriented_vertices, poor_edges);

            // FIXME: commenting out cerrs
            // graph.print(dom_circ);
            print_graphviz(graph, dom_circ, oriented_vertices, poor_edges);

            // FIXME: commenting out cerrs
            // cerr << "rich edges: ";
            // for (int e = 0; e < graph.number_of_edges; ++e) {
            //   if (!u6c4c_edge_is_poor[e]) {
            //     cerr << e << "(" << graph.e2v[e][0] << ", " << graph.e2v[e][1] << ") ";
            //   }
            // }
            // cerr << endl;
            // cerr << "poor edges: ";
            // for (int e = 0; e < graph.number_of_edges; ++e) {
            //   if (u6c4c_edge_is_poor[e]) {
            //     cerr << e << "(" << graph.e2v[e][0] << ", " << graph.e2v[e][1] << ") ";
            //   }
            // }
            // cerr << endl;
            // cerr << "oriented vertices: ";
            // for (const auto& v : oriented_vertices) {
            //   cerr << v << " ";
            // }
            // cerr << endl;
            // TODO: slow
            // FIXME: commenting out cerrs
            // cerr << "circuits:" << endl;
            // for (int c = 0; c < all_circuits_in_6c4c.size(); ++c) {
            //   cerr << "layer: " << layer[c] << "; vertices: ";
            //   if (orientations[c] == 0) {
            //     for (int vi = 0; vi < all_circuits_in_6c4c[c].size(); ++vi) {
            //       int v1 = all_circuits_in_6c4c[c][vi];
            //       cerr << v1 << " ";
            //     }
            //   } else {
            //     for (int vi = all_circuits_in_6c4c[c].size() - 1; vi >= 0; --vi) {
            //       int v1 = all_circuits_in_6c4c[c][vi];
            //       cerr << v1 << " ";
            //     }
            //   }
            //   cerr << endl;
            // }
            // cerr << "matchings: " << endl;
            // for (int i = 0; i < 6; ++i) {
            //   cerr << "layer: " << i << "; edges: ";
            //   for (int e = 0; e < graph.number_of_edges; ++e) {
            //     if ((u6c4c_cycles[i] & BIT(e)) == 0) { // edge is from matching
            //       cerr << e << "(" << graph.e2v[e][0] << ", " << graph.e2v[e][1] << ") ";
            //     }
            //   }
            //   cerr << endl;
            // }
          }

          //if (parity == 0) {
          //check_compatible_o5cdc(graph);
          //}
        }
        //check_compatible_dominating_circuit(graph);

        /*if (!graph.has_nz_mod5_from_o6c4c) {
          find_nz_mod_from_o6c4c(graph, 5, 1);
        }*/

        /*if (graph.o6c4c_profiles_min_oriented_vertices.find(prof) == graph.o6c4c_profiles_min_oriented_vertices.end()) {
          graph.o6c4c_profiles_min_oriented_vertices[prof] = oriented_vertices.size();
        } else {
          graph.o6c4c_profiles_min_oriented_vertices[prof] = min(graph.o6c4c_profiles_min_oriented_vertices[prof], oriented_vertices.size());
        }*/

        //cerr << endl;

        //has_compatible_dominating_circuit(graph);

        //find_nz_mod_from_o6c4c(graph, 5, 1); // skip first layer - for canonicity - it will always have weight 0
        //cerr << "find_nz_mod_from_o6c4c" << endl;
        //find_nz_mod_from_o6c4c(graph, 6, 1); // skip first layer - for canonicity - it will always have weight 0
        /*if (!has_dominating_circuit) {
            cerr << "wow" << endl;
        }
        has_dominating_circuit = false;*/

        return false;
    }
    int max_orientation = 1;
    if (cur_circuit == 0) {
        max_orientation = 0;
    }
    for (int orientation = 0; orientation <= max_orientation; ++orientation) {
        orientations[cur_circuit] = orientation;
        int vi = 0;
        while (vi < all_circuits_in_6c4c[cur_circuit].size() - 1) {
            int v1 = all_circuits_in_6c4c[cur_circuit][vi];
            int v2 = all_circuits_in_6c4c[cur_circuit][vi + 1];
            int ei = graph.vv2e[v1][v2];
            int cur_edge_orientation = orientation;
            if (v1 > v2) {
                cur_edge_orientation = 1 - cur_edge_orientation;
            }
            if (edge_orientation_count[ei][cur_edge_orientation] == 2) {
                break;
            }
            ++edge_orientation_count[ei][cur_edge_orientation];
            ++vi;
        }
        if (vi == all_circuits_in_6c4c[cur_circuit].size() - 1) {
            if (orient_6c4c(graph, cur_circuit + 1, first_time)) {
                return true;
            }
        }
        --vi;
        while (vi >= 0) {
            int v1 = all_circuits_in_6c4c[cur_circuit][vi];
            int v2 = all_circuits_in_6c4c[cur_circuit][vi + 1];
            int ei = graph.vv2e[v1][v2];
            int cur_edge_orientation = 0;
            if (v1 > v2) {
                cur_edge_orientation = 1 - cur_edge_orientation;
            }
            if (orientation == 1) {
                cur_edge_orientation = 1 - cur_edge_orientation;
            }
            --edge_orientation_count[ei][cur_edge_orientation];
            --vi;
        }
    }
    return false;
}

void contract_poor_edges(Graph& graph, bool order_is_random, mt19937& rng, set<int>& counts) {
  vector<bool> edge_is_poor;
  set<pair<int, int>> edge_pair_counts[MAX_EDGE];
  Graph g = Graph(graph.number_of_vertices);
  for (int e = 0; e < graph.number_of_edges; ++e) {
    g.add_edge(graph.e2v[e][0], graph.e2v[e][1]);
    edge_is_poor.push_back(u6c4c_edge_is_poor[e]);
    edge_pair_counts[e] = u6c4c_edge_pair_counts[e];
  }
  vector<int> con_layer;
  for (int c = 0; c < all_circuits_in_6c4c.size(); ++c) {
    con_layer.push_back(layer[c]);
  }
  vector<vector<int>> circuits = all_circuits_in_6c4c;
  int poor_count = total_poor_count;

  /*cerr << "circuits: " << endl;
  for (int c = 0; c < circuits.size(); ++c) {
    for (int vi = 0; vi < circuits[c].size(); ++vi) {
      cerr << circuits[c][vi] << " ";
    }
    cerr << endl;
  }*/

  bool bad_graph = false;
  while (poor_count > 0) {
    std::uniform_int_distribution<int> uni(0, poor_count - 1);
    //cerr << "cur poor count: " << poor_count << endl;
    int edge_to_contract_idx = 0;
    if (order_is_random) {
      edge_to_contract_idx = uni(rng);
    }
    int e_to_contract = -1;
    for (int e = 0; e < g.number_of_edges; ++e) {
      if (edge_is_poor[e]) {
        if (edge_to_contract_idx == 0) {
          e_to_contract = e;
          break;
        }
        --edge_to_contract_idx;
      }
    }
    assert(e_to_contract != -1);

    int v1_to_remove = g.e2v[e_to_contract][0];
    int v2_to_remove = g.e2v[e_to_contract][1];
    //cerr << "contracting edge " << v1_to_remove << " " << v2_to_remove << endl;

    set<int> neib_vs;
    for (const auto& p : edge_pair_counts[e_to_contract]) {
      neib_vs.insert(p.first);
      neib_vs.insert(p.second);
    }
    //cerr << "neibs count: " << neib_vs.size() << endl;
    assert((edge_pair_counts[e_to_contract].size() == 1 && neib_vs.size() == 2) ||
        (edge_pair_counts[e_to_contract].size() == 2 && (neib_vs.size() == 3 || neib_vs.size() == 4)));

    //cerr << "building new graph" << endl;
    Graph new_g(graph.number_of_vertices);
    for (int e = 0; e < g.number_of_edges; ++e) {
      int v1 = g.e2v[e][0];
      int v2 = g.e2v[e][1];
      if (v1 != v1_to_remove && v1 != v2_to_remove &&
          v2 != v1_to_remove && v2 != v2_to_remove) {
        new_g.add_edge(v1, v2);
      }
    }
    for (const auto& p : edge_pair_counts[e_to_contract]) {
      assert(p.first != p.second);
      if (new_g.vv2e[p.first][p.second] != NONE) {
        bad_graph = true;
        break;
      }
      new_g.add_edge(p.first, p.second);
    }
    if (bad_graph == true) {
      break;
    }

    // circuits
    //cerr << "circuits 1" << endl;
    // 1. convert circuits to layers
    bool new_layer_edges[6][MAX_EDGE];
    for (int l = 0; l < 6; ++l) {
      for (int e = 0; e < new_g.number_of_edges; ++e) {
        new_layer_edges[l][e] = false;
      }
    }
    for (int c = 0; c < circuits.size(); ++c) {
      int l = con_layer[c];
      set<int> cur_neibs;
      bool had_removed_edge = false;
      for (int vi = 0; vi < circuits[c].size() - 1; ++vi) {
        int v1 = circuits[c][vi];
        int v2 = circuits[c][vi + 1];
        if (v1 != v1_to_remove && v1 != v2_to_remove &&
            v2 != v1_to_remove && v2 != v2_to_remove) {
          new_layer_edges[l][new_g.vv2e[v1][v2]] = true;
        } else {
          if (v1 != v1_to_remove && v1 != v2_to_remove) {
            cur_neibs.insert(v1);
          }
          if (v2 != v1_to_remove && v2 != v2_to_remove) {
            cur_neibs.insert(v2);
          }
          if (g.vv2e[v1][v2] == e_to_contract) {
            had_removed_edge = true;
          }
        }
      }
      for (const auto& p : edge_pair_counts[e_to_contract]) {
        if (cur_neibs.find(p.first) != cur_neibs.end() &&
            (!had_removed_edge ||
             cur_neibs.find(p.second) != cur_neibs.end())) {
          new_layer_edges[l][new_g.vv2e[p.first][p.second]] = true;
        }
      }
    }

    vector<int> cover_count;
    for (int e = 0; e < new_g.number_of_edges; ++e) {
      cover_count.push_back(0);
    }
    for (int l = 0; l < 6; ++l) {
      for (int e = 0; e < new_g.number_of_edges; ++e) {
        if (new_layer_edges[l][e]) {
          ++cover_count[e];
        }
      }
    }
    for (int e = 0; e < new_g.number_of_edges; ++e) {
      if (cover_count[e] != 4) {
        bad_graph = true;
        break;
      }
      //assert(cover_count[e] == 4);
    }
    if (bad_graph) {
      break;
    }
    cover_count.clear();

    //cerr << "circuits 2" << endl;
    // 2. convert layers to circuits
    vector<vector<int>> new_circuits;
    vector<int> new_con_layer;
    for (int e = 0; e < new_g.number_of_edges; ++e) {
      cover_count.push_back(0);
    }
    for (int l = 0; l < 6; ++l) {
      bool visited_edge[MAX_EDGE];
      for (int e = 0; e < new_g.number_of_edges; ++e) {
        visited_edge[e] = false;
      }
      for (int e = 0; e < new_g.number_of_edges; ++e) {
        if (new_layer_edges[l][e] && !visited_edge[e]) {
          vector<int> circ;
          circ.push_back(new_g.e2v[e][0]);
          int cur_v = new_g.e2v[e][0];
          do {
            for (int j = 0; j < MAX_DEG; ++j) {
              int ee = new_g.v2e[cur_v][j];
              if (new_layer_edges[l][ee] && !visited_edge[ee]) {
                visited_edge[ee] = true;
                ++cover_count[ee];
                cur_v = new_g.v2v[cur_v][j];
                circ.push_back(cur_v);
                break;
              }
            }
          } while (cur_v != new_g.e2v[e][0]);
          assert(visited_edge[e]);
          new_circuits.push_back(circ);
          new_con_layer.push_back(l);
        }
      }
    }
    for (int e = 0; e < new_g.number_of_edges; ++e) {
      assert(cover_count[e] == 4);
    }

    //cerr << "rewriting" << endl;
    g = new_g;
    con_layer = new_con_layer;
    circuits = new_circuits;
    edge_is_poor.clear();
    poor_count = 0;
    for (int e = 0; e < graph.number_of_edges; ++e) {
      edge_pair_counts[e].clear();
    }
    /*cerr << "circuits: " << endl;
    for (int c = 0; c < circuits.size(); ++c) {
      for (int vi = 0; vi < circuits[c].size(); ++vi) {
        cerr << circuits[c][vi] << " ";
      }
      cerr << endl;
    }*/

    for (int c = 0; c < circuits.size(); ++c) {
      for (int vi = 0; vi < circuits[c].size() - 1; ++vi) {
        int v0 = -1;
        if (vi > 0) {
          v0 = circuits[c][vi - 1];
        } else {
          v0 = circuits[c][circuits[c].size() - 2];
        }
        int v1 = circuits[c][vi];
        int v2 = circuits[c][vi + 1];
        int v3 = -1;
        if (vi < circuits[c].size() - 2) {
          v3 = circuits[c][vi + 2];
        } else {
          v3 = circuits[c][1];
        }
        if (v0 > v3) {
          swap(v0, v3);
        }
        int ei = g.vv2e[v1][v2];
        edge_pair_counts[ei].insert(make_pair(v0, v3));
      }
    }

    for (int e = 0; e < g.number_of_edges; ++e) {
      /*if (edge_pair_counts[e].size() == 0 || edge_pair_counts[e].size() == 3) {
        cerr << "edge pair count problem in edge " << g.e2v[e][0] << " " << g.e2v[e][1] << endl;
      }*/
      assert(edge_pair_counts[e].size() == 1 || edge_pair_counts[e].size() == 2 || edge_pair_counts[e].size() == 4);
      if (edge_pair_counts[e].size() != 4) {
        edge_is_poor.push_back(true);
        ++poor_count;
      } else {
        edge_is_poor.push_back(false);
      }
    }
  }
  //cerr << "cur poor count: " << 0 << endl;
  //cerr << "bad graph: " << bad_graph << endl;
  if (!bad_graph) {
    counts.insert(circuits.size());
    //cerr << "circuits count: " << circuits.size() << endl;
    if (circuits.size() == 12) {
      /*bool all5 = true;
      for (const auto& c : circuits) {
        if (c.size() != 6) {
          all5 = false;
          break;
        }
      }
      if (all5) {
        cerr << "resulting graph: " << endl;
        for (int e = 0; e < g.number_of_edges; ++e) {
          cerr << "e: " << g.e2v[e][0] << ", " << g.e2v[e][1] << endl;
        }
        cerr << endl;
      }*/
    }
  }
}

bool find_2cdcs(Graph& graph, int cur_circuit, bool print_debug) {
  if (total_cdc_need_cover_count == 0) {
    has_2cdcs = true;
    ++count_2cdcs;
    //cerr << "find 2cdcs success" << endl;

    Mask u6c4c_cycles1[6];
    Mask u6c4c_cycles2[6];
    for (int l = 0; l < 6; ++l) {
      u6c4c_cycles1[l] = 0;
      u6c4c_cycles2[l] = 0;
    }

    int taken_count = 0;
    layers_c1.clear();
    layers_c2.clear();
    for (int ci = 0; ci < all_circuits_in_6c4c.size(); ++ci) {
      two_cdcs_taken[ci] = taken[ci];
      if (taken[ci]) {
        layers_c1.insert(layer[ci]);
        //cerr << "layer: " << layer[ci] << "; vertices: ";
        ++taken_count;
        u6c4c_cycles1[layer[ci]] += all_circuit_masks_in_6c4c[ci];
        //cerr << ci << ": ";
        for (int vi = 0; vi < all_circuits_in_6c4c[ci].size(); ++vi) {
          int v1 = all_circuits_in_6c4c[ci][vi];
          //cerr << v1 << " ";
          if (vi < all_circuits_in_6c4c[ci].size() - 1) {
            int v2 = all_circuits_in_6c4c[ci][vi + 1];
            int ei = graph.vv2e[v1][v2];
            /*if (u6c4c_edge_is_poor[ei]) {
              cerr << "p ";
            } else {
              cerr << "r ";
            }*/
          }
        }
        //cerr << endl;
      } else {
        layers_c2.insert(layer[ci]);
        u6c4c_cycles2[layer[ci]] += all_circuit_masks_in_6c4c[ci];
      }
    }

    bool has_some_5cdc1 = false;
    //cerr << "some1: ";
    int some1count = 0;
    int yep1 = 0;
    int ins1 = 0;
    int nop1 = 0;
    for (int i = 0; i < 6; ++i) {
      for (int j = i + 1; j < 6; ++j) {
        for (int k = j + 1; k < 6; ++k) {
          set<Mask> triple;
          triple.insert(u6c4c_cycles1[i]);
          triple.insert(u6c4c_cycles1[j]);
          triple.insert(u6c4c_cycles1[k]);
          if (graph.all_5cdc_by_triples.find(triple) != graph.all_5cdc_by_triples.end()) {
            map<int, set<int>> counts;
            for (int v = 0; v < graph.number_of_vertices; ++v) {
              vector<bool> ins = {false, false, false};
              for (int jj = 0; jj < MAX_DEG; ++jj) {
                int e = graph.v2e[v][jj];
                if ((BIT(e) & u6c4c_cycles1[i]) > 0) {
                  ins[0] = true;
                }
                if ((BIT(e) & u6c4c_cycles1[j]) > 0) {
                  ins[1] = true;
                }
                if ((BIT(e) & u6c4c_cycles1[k]) > 0) {
                  ins[2] = true;
                }
              }
              int count = 0;
              for (int ii = 0; ii < 3; ++ii) {
                if (ins[ii]) {
                  ++count;
                }
              }
              //if (count == 3) {
                counts[count].insert(v);
              //}
            }
            two_cdcs_vertices.insert(counts);

            Mask tff_cycle = u6c4c_cycles[i] ^ u6c4c_cycles[j] ^ u6c4c_cycles[k];
            assert(graph.all_cycles.find(tff_cycle) != graph.all_cycles.end());

            //cerr << "(";
            for (const auto& sol : graph.all_5cdc_by_triples[triple]) {
              if (sol.find(tff_cycle) != sol.end()) {
                //cerr << "yep ";
                ++yep1;
              } else {
                bool is_inside = false;
                for (const auto& cyc : sol) {
                  if ((cyc != u6c4c_cycles1[i]) &&
                      (cyc != u6c4c_cycles1[j]) &&
                      (cyc != u6c4c_cycles1[k]) &&
                      ((cyc & tff_cycle) == tff_cycle)) {
                    is_inside = true;
                  }
                }
                if (is_inside) {
                  //cerr << "ins ";
                  ++ins1;
                } else {
                  //cerr << "nop ";
                  ++nop1;
                }
              }
            }
            has_some_5cdc1 = true;
            //cerr << graph.all_5cdc_by_triples[triple].size() << ": " << i << " " << j << " " << k << "); ";
            ++some1count;
            //break;
          } else {
          }
        }
        /*(if (has_some_5cdc1) {
          break;
        }*/
      }
      /*if (has_some_5cdc1) {
        break;
      }*/
    }
    //cerr << "total: " << some1count << endl;

    bool has_some_5cdc2 = false;
    int some2count = 0;
    //cerr << "some2: ";
    int yep2 = 0;
    int ins2 = 0;
    int nop2 = 0;
    for (int i = 0; i < 6; ++i) {
      for (int j = i + 1; j < 6; ++j) {
        for (int k = j + 1; k < 6; ++k) {
          set<Mask> triple;
          triple.insert(u6c4c_cycles2[i]);
          triple.insert(u6c4c_cycles2[j]);
          triple.insert(u6c4c_cycles2[k]);
          if (graph.all_5cdc_by_triples.find(triple) != graph.all_5cdc_by_triples.end()) {
            Mask tff_cycle = u6c4c_cycles[i] ^ u6c4c_cycles[j] ^ u6c4c_cycles[k];
            assert(graph.all_cycles.find(tff_cycle) != graph.all_cycles.end());

            //cerr << "(";
            for (const auto& sol : graph.all_5cdc_by_triples[triple]) {
              if (sol.find(tff_cycle) != sol.end()) {
                //cerr << "yep ";
                ++yep2;
              } else {
                bool is_inside = false;
                for (const auto& cyc : sol) {
                  if ((cyc != u6c4c_cycles2[i]) &&
                      (cyc != u6c4c_cycles2[j]) &&
                      (cyc != u6c4c_cycles2[k]) &&
                      ((cyc & tff_cycle) == tff_cycle)) {
                    is_inside = true;
                  }
                }
                if (is_inside) {
                  //cerr << "ins ";
                  ++ins2;
                } else {
                  //cerr << "nop ";
                  ++nop2;
                }
              }
            }
            has_some_5cdc2 = true;
            //cerr << graph.all_5cdc_by_triples[triple].size() << ": " << i << " " << j << " " << k << "); ";
            ++some2count;
            //break;
          } else {
          }
        }
        /*(if (has_some_5cdc2) {
          break;
        }*/
      }
      /*if (has_some_5cdc2) {
        break;
      }*/
    }
    //cerr << endl;

    //cerr << "6to5cdc counts: " << min(some1count, some2count) << " vs " << max(some1count, some2count) << endl;
    //cerr << "yepinsnops: " << yep1 << " " << ins1 << " " << yep1 + ins1 << " vs " << yep2 << " " << ins2 << " " << yep2 + ins2 << endl;
    //cerr << "yepinsnopdiffs: " << yep1 - yep2 << " " << ins1 - ins2 << " " << yep1 + ins1 - (yep2 + ins2) << endl;

    /*cerr << "dom 1: ";
    int dom1count = 0;
    for (int i = 0; i < 6; ++i) {
      for (int j = i + 1; j < 6; ++j) {
        for (int k = j + 1; k < 6; ++k) {
          set<Mask> triple;
          triple.insert(u6c4c_cycles1[i]);
          triple.insert(u6c4c_cycles1[j]);
          triple.insert(u6c4c_cycles1[k]);
          if (graph.all_5cdc_by_triples.find(triple) != graph.all_5cdc_by_triples.end()) {
            cerr << "(";
            for (const auto& sol : graph.all_5cdc_by_triples[triple]) {
              bool is_dom = false;
              for (const auto& cyc : sol) {
                if ((cyc != u6c4c_cycles1[i]) &&
                    (cyc != u6c4c_cycles1[j]) &&
                    (cyc != u6c4c_cycles1[k]) &&
                    (graph.all_dominating_circuits.find(cyc) != graph.all_dominating_circuits.end())) {
                  is_dom = true;
                  break;
                }
              }
              if (is_dom) {
                cerr << "yep ";
                ++dom1count;
              } else {
                cerr << "nop ";
              }
            }
            cerr << graph.all_5cdc_by_triples[triple].size() << ": " << i << " " << j << " " << k << "); ";
          } else {
          }
        }
      }
    }
    cerr << endl;
    cerr << "dom 2: ";
    int dom2count = 0;
    for (int i = 0; i < 6; ++i) {
      for (int j = i + 1; j < 6; ++j) {
        for (int k = j + 1; k < 6; ++k) {
          set<Mask> triple;
          triple.insert(u6c4c_cycles2[i]);
          triple.insert(u6c4c_cycles2[j]);
          triple.insert(u6c4c_cycles2[k]);
          if (graph.all_5cdc_by_triples.find(triple) != graph.all_5cdc_by_triples.end()) {
            cerr << "(";
            for (const auto& sol : graph.all_5cdc_by_triples[triple]) {
              bool is_dom = false;
              for (const auto& cyc : sol) {
                if ((cyc != u6c4c_cycles2[i]) &&
                    (cyc != u6c4c_cycles2[j]) &&
                    (cyc != u6c4c_cycles2[k]) &&
                    (graph.all_dominating_circuits.find(cyc) != graph.all_dominating_circuits.end())) {
                  is_dom = true;
                  break;
                }
              }
              if (is_dom) {
                cerr << "yep ";
                ++dom2count;
              } else {
                cerr << "nop ";
              }
            }
            cerr << graph.all_5cdc_by_triples[triple].size() << ": " << i << " " << j << " " << k << "); ";
          } else {
          }
        }
      }
    }
    cerr << endl;
    cerr << "doms: " << min(dom1count, dom2count) << " vs " << max(dom1count, dom2count) << endl;*/

    taken_c1 = taken_count;
    taken_c2 = all_circuits_in_6c4c.size() - taken_c1;
    if (taken_c2 < taken_c1) {
      swap(taken_c1, taken_c2);
      swap(layers_c1, layers_c2);
      swap(has_some_5cdc1, has_some_5cdc2);
    }
    //cerr << "taken count: " << taken_count << " vs " << all_circuits_in_6c4c.size() - taken_count << endl;
    //cerr << "layers: " << layers_c1.size() << " vs " << layers_c2.size() << endl;
    /*if (graph.all_5cdc.size() > 0) {
      cerr << "has 5cdc with same triple: " << has_some_5cdc1 << " " << has_some_5cdc2 << endl;
    }*/
    return false;
  }

  if (cur_circuit >= all_circuits_in_6c4c.size()) {
    return false;
  }

  // FIXME: commenting out cerrs
  // if (print_debug) {
  //   cerr << total_cdc_need_cover_count << "; taken: ";
  //   for (int ci = 0; ci < cur_circuit; ++ci) {
  //     if (taken[ci]) {
  //       cerr << ci << " ";
  //     }
  //   }
  //   cerr << endl;
  // }

  bool can_be_used = true;
  vector<int> edges;
  for (int vi = 0; vi < all_circuits_in_6c4c[cur_circuit].size() - 1; ++vi) {
    int v1 = all_circuits_in_6c4c[cur_circuit][vi];
    int v2 = all_circuits_in_6c4c[cur_circuit][vi + 1];
    int ei = graph.vv2e[v1][v2];
    if (cdc_cover_count[ei] >= 2) {
      can_be_used = false;
      break;
    }
    edges.push_back(ei);
  }
  if (can_be_used) {
    taken[cur_circuit] = true;
    for (const auto& ei : edges) {
      ++cdc_cover_count[ei];
    }
    total_cdc_need_cover_count -= edges.size();
    if (find_2cdcs(graph, cur_circuit + 1, print_debug)) {
      return true;
    }
    total_cdc_need_cover_count += edges.size();
    for (const auto& ei : edges) {
      --cdc_cover_count[ei];
    }
  }
  taken[cur_circuit] = false;
  return find_2cdcs(graph, cur_circuit + 1, print_debug);
}

void find_2cdcs(Graph& graph, bool print_debug) {
  total_cdc_need_cover_count = 0;
  for (int e = 0; e < graph.number_of_edges; ++e) {
    cdc_cover_count[e] = 0;
    total_cdc_need_cover_count += 2;
  }
  for (int c = 0; c < all_circuits_in_6c4c.size(); ++c) {
    taken[c] = false;
  }

  has_2cdcs = false;
  count_2cdcs = 0;
  find_2cdcs(graph, 0, print_debug);
  assert(count_2cdcs % 2 == 0);
  count_2cdcs /= 2;
  // FIXME: commenting out cerrs
  // if (has_2cdcs) {
  //   cerr << "has 2 cdcs with " << count_2cdcs << "; ";
  // } else {
  //   cerr << "no 2 cdcs; ";
  // }
}

int min_diff, cur_min_diff;
vector<int> layer_circuit_count;
vector<int> layer_circuit_total_lens;
vector<vector<int>> layer_circuit_signs;
vector<vector<int>> layer_circuit_lens;
vector<vector<int>> layer_circuit_nums;
vector<vector<int>> best_layer_circuit_signs;
vector<int> first_part_edge_count;
vector<int> second_part_edge_count;
bool had_best;
bool check_for_has_four;

void find_min_diff(Graph& graph, int l);

void find_min_diff(Graph& graph, int l, int c) {
  if (min_diff == 0) {
    return;
  }
  if (c == layer_circuit_count[l]) {
    bool has_plus1 = false;
    bool has_minus1 = false;
    for (const auto& s : layer_circuit_signs[l]) {
      if (s == 1) {
        has_plus1 = true;
      } else {
        assert(s == -1);
        has_minus1 = true;
      }

      if (has_plus1 && has_minus1) {
        break;
      }
    }

    if (!has_plus1 || !has_minus1) {
      return;
    }

    for (int i = 0; i < layer_circuit_signs[l].size(); ++i) {
      cur_min_diff += layer_circuit_signs[l][i] * layer_circuit_lens[l][i];
      int cur_circuit = layer_circuit_nums[l][i];
      for (int vi = 0; vi < all_circuits_in_6c4c[cur_circuit].size() - 1; ++vi) {
        int v1 = all_circuits_in_6c4c[cur_circuit][vi];
        int v2 = all_circuits_in_6c4c[cur_circuit][vi + 1];
        int ei = graph.vv2e[v1][v2];
        if (layer_circuit_signs[l][i] == 1) {
          ++first_part_edge_count[ei];
        } else {
          ++second_part_edge_count[ei];
        }
      }
    }
    bool has_four = false;
    if (check_for_has_four) {
      for (int e = 0; e < graph.number_of_edges; ++e) {
        if (first_part_edge_count[e] == 4 || second_part_edge_count[e] == 4) {
          has_four = true;
          break;
        }
      }
    }
    if (!has_four) {
      find_min_diff(graph, l + 1);
    }
    for (int i = 0; i < layer_circuit_signs[l].size(); ++i) {
      cur_min_diff -= layer_circuit_signs[l][i] * layer_circuit_lens[l][i];
      int cur_circuit = layer_circuit_nums[l][i];
      for (int vi = 0; vi < all_circuits_in_6c4c[cur_circuit].size() - 1; ++vi) {
        int v1 = all_circuits_in_6c4c[cur_circuit][vi];
        int v2 = all_circuits_in_6c4c[cur_circuit][vi + 1];
        int ei = graph.vv2e[v1][v2];
        if (layer_circuit_signs[l][i] == 1) {
          --first_part_edge_count[ei];
        } else {
          --second_part_edge_count[ei];
        }
      }
    }
    return;
  }
  layer_circuit_signs[l][c] = 1;
  find_min_diff(graph, l, c + 1);
  layer_circuit_signs[l][c] = -1;
  find_min_diff(graph, l, c + 1);
}

void find_min_diff(Graph& graph, int l) {
  if (l == 6) {
    if (abs(cur_min_diff) < min_diff) {
      best_layer_circuit_signs = layer_circuit_signs;
      min_diff = abs(cur_min_diff);
      had_best = true;
    }
    return;
  }
  if (min_diff == 0) {
    return;
  }
  find_min_diff(graph, l, 0);
}

void find_min_diff(Graph& graph) {
  layer_circuit_signs = {{}, {}, {}, {}, {}, {}};
  layer_circuit_lens = {{}, {}, {}, {}, {}, {}};
  layer_circuit_nums = {{}, {}, {}, {}, {}, {}};
  layer_circuit_count = {0, 0, 0, 0, 0, 0};
  layer_circuit_total_lens = {0, 0, 0, 0, 0, 0};
  for (int c = 0; c < all_circuits_in_6c4c.size(); ++c) {
    int l = layer[c];
    layer_circuit_signs[l].push_back(1);
    layer_circuit_lens[l].push_back(all_circuits_in_6c4c[c].size() - 1);
    layer_circuit_nums[l].push_back(c);
    layer_circuit_count[l] += 1;
    layer_circuit_total_lens[l] += all_circuits_in_6c4c[c].size() - 1;
  }

  first_part_edge_count.clear();
  second_part_edge_count.clear();
  for (int e = 0; e < graph.number_of_edges; ++e) {
    first_part_edge_count.push_back(0);
    second_part_edge_count.push_back(0);
  }

  had_best = false;
  cur_min_diff = 0;
  min_diff = graph.number_of_vertices * 6;
  find_min_diff(graph, 0);

  assert(min_diff % 2 == 0);
  min_diff /= 2;

  /*if (had_best) {
    for (int l = 0; l < 6; ++l) {
      cerr << "layer " << l << ": ";
      for (int c = 0; c < best_layer_circuit_signs[l].size(); ++c) {
        if (best_layer_circuit_signs[l][c] == 1) {
          cerr << "+";
        } else {
          cerr << "-";
        }
        cerr << layer_circuit_lens[l][c] << " ";
      }
      cerr << endl;
    }
  }*/
}

bool check_orientability_6c4c(Graph& graph) {
    has_dominating_circuit = false;


    all_circuits_in_6c4c.clear();
    all_circuit_masks_in_6c4c.clear();
    for (int i = 0; i < 6; ++i) {
        const auto& circuits = graph.cycles_as_circuits[u6c4c_cycles[i]];
        const auto& circuit_masks = graph.cycles_as_circuit_masks[u6c4c_cycles[i]];
        for (int idx = 0; idx < circuits.size(); ++idx) {
            const int circuit_number = all_circuits_in_6c4c.size();
            all_circuits_in_6c4c.push_back(circuits[idx]);
            layer[circuit_number] = i;
            for (const auto& v : circuits[idx]) {
              layer_vertex_to_circuit[i][v] = circuit_number;
            }

            all_circuit_masks_in_6c4c.push_back(circuit_masks[idx]);
        }
    }

    for (int e = 0; e < graph.number_of_edges; ++e) {
        for (int orientation = 0; orientation < 2; ++orientation) {
            edge_orientation_count[e][orientation] = 0;
        }
    }

    for (int e = 0; e < graph.number_of_edges; ++e) {
        u6c4c_edge_pair_counts[e].clear();
    }

    for (int c = 0; c < all_circuits_in_6c4c.size(); ++c) {
        for (int vi = 0; vi < all_circuits_in_6c4c[c].size() - 1; ++vi) {
            int v0 = -1;
            if (vi > 0) {
                v0 = all_circuits_in_6c4c[c][vi - 1];
            } else {
                v0 = all_circuits_in_6c4c[c][all_circuits_in_6c4c[c].size() - 2];
            }
            int v1 = all_circuits_in_6c4c[c][vi];
            int v2 = all_circuits_in_6c4c[c][vi + 1];
            int v3 = -1;
            if (vi < all_circuits_in_6c4c[c].size() - 2) {
                v3 = all_circuits_in_6c4c[c][vi + 2];
            } else {
                v3 = all_circuits_in_6c4c[c][1];
            }
            if (v0 > v3) {
                swap(v0, v3);
            }
            int ei = graph.vv2e[v1][v2];
            u6c4c_edge_pair_counts[ei].insert(make_pair(v0, v3));
        }
    }

    same_cycles_different_orientations = 0;

    for (int e = 0; e < graph.number_of_edges; ++e) {
      if (u6c4c_edge_pair_counts[e].size() != 2 && u6c4c_edge_pair_counts[e].size() != 4) {
        cerr << "wut" << endl;
        continue;
      }
      if (u6c4c_edge_pair_counts[e].size() == 2) {
        u6c4c_edge_is_poor[e] = true;
      } else {
        u6c4c_edge_is_poor[e] = false;
      }
    }

    prof = "";
    for (int e = 0; e < graph.number_of_edges; ++e) {
      if (u6c4c_edge_is_poor[e]) {
        prof += "p";
      } else {
        prof += "r";
      }
    }

    total_rich_count = 0;
    for (int e = 0; e < graph.number_of_edges; ++e) {
      if (!u6c4c_edge_is_poor[e]) {
        ++total_rich_count;
      }
    }
    total_poor_count = graph.number_of_edges - total_rich_count;

    // FIXMEFIXMEFIXME
    // onlyrich graphs:
    // if (total_poor_count > 0) {
    //   return false;
    // }

    even_rich_matchings = 0;
    for (const auto& c : cur_6c4c) {
      int rich_count = 0;
      for (int e = 0; e < graph.number_of_edges; ++e) {
        if ((c & BIT(e)) == 0) { // edge is from matching
          if (!u6c4c_edge_is_poor[e]) {
            ++rich_count;
          }
        }
      }
      if (rich_count % 2 == 0) {
        ++even_rich_matchings;
      }
    }

    set<Mask> different_circuits;
    for (const auto& c : all_circuit_masks_in_6c4c) {
      different_circuits.insert(c);
    }

    set<int> vertices_neib_to_poor_edge;
    set<int> vertices_neib_to_rich_edge;
    for (int e = 0; e < graph.number_of_edges; ++e) {
      const int v1 = graph.e2v[e][0];
      const int v2 = graph.e2v[e][1];
      if (u6c4c_edge_is_poor[e]) {
        vertices_neib_to_poor_edge.insert(v1);
        vertices_neib_to_poor_edge.insert(v2);
      } else {
        vertices_neib_to_rich_edge.insert(v1);
        vertices_neib_to_rich_edge.insert(v2);
      }
    }

    for (int v = 0; v < graph.number_of_vertices; ++v) {
      poor_component[v] = 0;
      rich_component[v] = 0;
    }

    for (int e = 0; e < graph.number_of_edges; ++e) {
      edge_poor_component[e] = 0;
      edge_rich_component[e] = 0;
    }

    total_poor_comps = 0;
    for (int v = 0; v < graph.number_of_vertices; ++v) {
      if (vertices_neib_to_poor_edge.find(v) != vertices_neib_to_poor_edge.end()) {
        if (poor_component[v] != 0) {
          continue;
        }
        ++total_poor_comps;
        poor_component[v] = total_poor_comps;
        vector<int> queue;
        queue.push_back(v);
        int head = 0;
        while (head < queue.size()) {
          int v_head = queue[head];
          ++head;
          for (int j = 0; j < MAX_DEG; ++j) {
            int e = graph.v2e[v_head][j];
            if (u6c4c_edge_is_poor[e]) {
              edge_poor_component[e] = total_poor_comps;
              int v2 = graph.v2v[v_head][j];
              if (poor_component[v2] == 0) {
                poor_component[v2] = total_poor_comps;
                queue.push_back(v2);
              }
            }
          }
        }
      }
    }

    total_rich_comps = 0;
    for (int v = 0; v < graph.number_of_vertices; ++v) {
      if (vertices_neib_to_rich_edge.find(v) != vertices_neib_to_rich_edge.end()) {
        if (rich_component[v] != 0) {
          continue;
        }
        ++total_rich_comps;
        rich_component[v] = total_rich_comps;
        vector<int> queue;
        queue.push_back(v);
        int head = 0;
        while (head < queue.size()) {
          int v_head = queue[head];
          ++head;
          for (int j = 0; j < MAX_DEG; ++j) {
            int e = graph.v2e[v_head][j];
            if (!u6c4c_edge_is_poor[e]) {
              edge_rich_component[e] = total_rich_comps;
              int v2 = graph.v2v[v_head][j];
              if (rich_component[v2] == 0) {
                rich_component[v2] = total_rich_comps;
                queue.push_back(v2);
              }
            }
          }
        }
      }
    }

    odd_poor_2_factors = 0;
    odd_poor_comps_2_factors = 0;
    odd_rich_2_factors = 0;
    odd_rich_comps_2_factors = 0;
    odd_poor_matchings = 0;
    odd_poor_comps_matchings = 0;
    odd_rich_matchings = 0;
    odd_rich_comps_matchings = 0;
    for (const auto& c : cur_6c4c) {
      int poor_in_2_factor_count = 0;
      int poor_in_matching_count = 0;
      int rich_in_2_factor_count = 0;
      int rich_in_matching_count = 0;

      set<int> poor_comps_edges_in_2_factor;
      set<int> poor_comps_edges_in_matching;
      set<int> rich_comps_edges_in_2_factor;
      set<int> rich_comps_edges_in_matching;

      for (int e = 0; e < graph.number_of_edges; ++e) {
        if ((c & BIT(e)) == 0) { // edge is from matching
          if (u6c4c_edge_is_poor[e]) {
            poor_comps_edges_in_matching.insert(edge_poor_component[e]);
            ++poor_in_matching_count;
          } else {
            rich_comps_edges_in_matching.insert(edge_rich_component[e]);
            ++rich_in_matching_count;
          }
        } else { // edge is from full cycle
          if (u6c4c_edge_is_poor[e]) {
            poor_comps_edges_in_2_factor.insert(edge_poor_component[e]);
            ++poor_in_2_factor_count;
          } else {
            rich_comps_edges_in_2_factor.insert(edge_rich_component[e]);
            ++rich_in_2_factor_count;
          }
        }
      }

      if (poor_comps_edges_in_2_factor.size() % 2 != 0) {
        ++odd_poor_comps_2_factors;
      }
      if (poor_in_2_factor_count % 2 != 0) {
        ++odd_poor_2_factors;
      }

      if (rich_comps_edges_in_2_factor.size() % 2 != 0) {
        ++odd_rich_comps_2_factors;
      }
      if (rich_in_2_factor_count % 2 != 0) {
        ++odd_rich_2_factors;
      }

      if (poor_comps_edges_in_matching.size() % 2 != 0) {
        ++odd_poor_comps_matchings;
      }
      if (poor_in_matching_count % 2 != 0) {
        ++odd_poor_matchings;
      }

      if (rich_comps_edges_in_matching.size() % 2 != 0) {
        ++odd_rich_comps_matchings;
      }
      if (rich_in_matching_count % 2 != 0) {
        ++odd_rich_matchings;
      }
    }

    assert(odd_poor_2_factors % 2 == 0); // this is true
    odd_poor_2_factors /= 2;

    assert(odd_rich_2_factors % 2 == 0); // this is true
    odd_rich_2_factors /= 2;

    assert(odd_poor_matchings % 2 == 0); // this is true
    odd_poor_matchings /= 2;

    assert(odd_rich_matchings % 2 == 0); // this is true
    odd_rich_matchings /= 2;

    assert(odd_poor_comps_2_factors % 2 == 0); // TODO
    // FIXME
    // odd_poor_comps_2_factors /= 2;

    assert(odd_rich_comps_2_factors % 2 == 0); // TODO
    // FIXME
    // odd_rich_comps_2_factors /= 2;

    assert(odd_poor_comps_matchings % 2 == 0); // TODO!
    // FIXME
    // odd_poor_comps_matchings /= 2;

    // // this is false!
    // assert(odd_rich_comps_matchings % 2 == 0);
    // odd_rich_comps_matchings /= 2;


    vector<int> circuit_lens;
    int zero_lens = 0;
    int minus1_lens = 0;
    for (int c = 0; c < all_circuits_in_6c4c.size(); ++c) {
      int poor_count = 0;
      bool has_rich = false;
      int len = 0;
      int delta = 1;
      Mask m = 0;
      for (int vi = 0; vi < all_circuits_in_6c4c[c].size() - 1; ++vi) {
        int v1 = all_circuits_in_6c4c[c][vi];
        int v2 = all_circuits_in_6c4c[c][vi + 1];
        int ei = graph.vv2e[v1][v2];
        m += BIT(ei);
        if (u6c4c_edge_is_poor[ei]) {
          ++poor_count;
          delta *= -1;
        } else {
          has_rich = true;
          len += delta;
        }
      }
      if (!has_rich) {
        for (int vi = 0; vi < all_circuits_in_6c4c[c].size() - 1; ++vi) {
          int v1 = all_circuits_in_6c4c[c][vi];
          int v2 = all_circuits_in_6c4c[c][vi + 1];
          int ei = graph.vv2e[v1][v2];
        }
      }
      if (poor_count % 2 == 0) {
        if (len < 0) {
          len = -len;
        }
        circuit_lens.push_back(len);
        //if (len == 0 || len == 6 || len == 10) {
        if (len % 2 == 0) {
          ++zero_lens;
        }
      } else {
        circuit_lens.push_back(-1);
        ++minus1_lens;
      }
    }
    assert(minus1_lens % 2 == 0); // TODO ? or is it obvious?
    minus1_lens /= 2;

    any_chords_frequency = {0, 0, 0};
    poor_chords_frequency = {0, 0, 0};
    rich_chords_frequency = {0, 0, 0};
    circuit_count_by_layer = {0, 0, 0, 0, 0, 0};
    chord_count_by_layer = {0, 0, 0, 0, 0, 0};
    antichord_count_by_layer = {0, 0, 0, 0, 0, 0};
    for (int e = 0; e < graph.number_of_edges; ++e) {
      int chord_count = 0;
      for (int i = 0; i < 6; ++i) {
        if ((u6c4c_cycles[i] & BIT(e)) == 0) { // edge is from matching
          const int v1 = graph.e2v[e][0];
          const int v2 = graph.e2v[e][1];
          if (layer_vertex_to_circuit[i][v1] == layer_vertex_to_circuit[i][v2]) {
            ++chord_count;
            ++chord_count_by_layer[i];
          } else {
            ++antichord_count_by_layer[i];
          }
        }
      }
      ++any_chords_frequency[chord_count];
      if (u6c4c_edge_is_poor[e]) {
        ++poor_chords_frequency[chord_count];
      } else {
        ++rich_chords_frequency[chord_count];
      }
    }
    for (int c = 0; c < all_circuits_in_6c4c.size(); ++c) {
      ++circuit_count_by_layer[layer[c]];
    }

    vertex_types.clear();
    for (int v = 0; v < graph.number_of_vertices; ++v) {
      set<set<int>> cur_vertex_type;
      for (int j = 0; j < MAX_DEG; ++j) {
        int e = graph.v2e[v][j];
        set<int> edge_matching_layers;
        for (int i = 0; i < 6; ++i) {
          if ((u6c4c_cycles[i] & BIT(e)) == 0) { // edge is from matching
            edge_matching_layers.insert(i);
          }
        }
        cur_vertex_type.insert(edge_matching_layers);
      }
      vertex_types.insert(cur_vertex_type);
    }

    rich_edges_frequency = {0, 0, 0, 0};
    poor_edges_frequency = {0, 0, 0, 0};
    for (int v = 0; v < graph.number_of_vertices; ++v) {
      int rich_edge_count = 0;
      for (int j = 0; j < MAX_DEG; ++j) {
        int e = graph.v2e[v][j];
        if (!u6c4c_edge_is_poor[e]) {
          ++rich_edge_count;
        }
      }
      ++rich_edges_frequency[rich_edge_count];
      ++poor_edges_frequency[MAX_DEG - rich_edge_count];
    }

    s0 = all_circuits_in_6c4c.size();
    s1 = total_rich_count;
    s2 = even_rich_matchings / 2;
    assert(even_rich_matchings % 2 == 0); // this is true
    int parity = (s0 + s1 + s2) % 2;

    // FIXMEFIXMEFIXMEFIXME
    // s1s0diff_low
    // if (s1-s0 > 0) {
    //   return false;
    // }

    map<int, pair<int, int>> edge_to_layers;
    for (int e = 0; e < graph.number_of_edges; ++e) {
      int l1 = NONE;
      int l2 = NONE;
      for (int i = 0; i < 6; ++i) {
        if ((u6c4c_cycles[i] & BIT(e)) == 0) { // edge is from matching
          if (l1 == NONE) {
            l1 = i;
          } else {
            l2 = i;
          }
        }
      }
      edge_to_layers[e] = make_pair(l1, l2);
    }
    set<tuple<tuple<int, int>, vector<vector<tuple<int, int>>>>> rich_edge_layer_types;
    // so, it's a set of tuples
    // first in tuple is tuple<int, int> - pair of layers, where edge is from PM
    // second is fours_of_layers - vector<vector<tuple<int, int>>>>
    // it's a 2x2x2 construction
    // it has 2 l1l2s: vector<tuple<int, int>>
    // 2, because edge connects 2 vertices
    // each l1l2 has 2 tuples, each tuple is edge_to_layers
    // so, for each rich edge we track it's neighbour edges
    // each neighbour edge carries 2 numbers, edge_to_layers, pair of layers, where edge is from PM
    for (int e = 0; e < graph.number_of_edges; ++e) {
      if (u6c4c_edge_is_poor[e]) {
        continue;
      }
      vector<vector<tuple<int, int>>> fours_of_layers;
      for (int i = 0; i < 2; ++i) {
        vector<tuple<int, int>> l1l2s;
        int v = graph.e2v[e][i];
        for (int j = 0; j < MAX_DEG; ++j) {
          int e2 = graph.v2e[v][j];
          if (e2 == e) {
            continue;
          }
          l1l2s.push_back(edge_to_layers[e2]);
        }
        sort(l1l2s.begin(), l1l2s.end());
        fours_of_layers.push_back(l1l2s);
      }
      sort(fours_of_layers.begin(), fours_of_layers.end());
      rich_edge_layer_types.insert(make_pair(edge_to_layers[e], fours_of_layers));
      // cerr << edge_to_layers[e].first << " " << edge_to_layers[e].second << ": ";
      // cerr << get<0>(fours_of_layers[0][0]) << " " <<
      //         get<1>(fours_of_layers[0][0]) << " " <<
      //         get<0>(fours_of_layers[0][1]) << " " <<
      //         get<1>(fours_of_layers[0][1]) << " " <<
      //         get<0>(fours_of_layers[1][0]) << " " <<
      //         get<1>(fours_of_layers[1][0]) << " " <<
      //         get<0>(fours_of_layers[1][1]) << " " <<
      //         get<1>(fours_of_layers[1][1]) << endl;
    }

    rich_244_counts.clear();
    Mask ms[3];
    for (int i = 0; i < 1; ++i) {
      ms[0] = u6c4c_cycles[i];
      for (int j = i + 1; j < 6; ++j) {
        ms[1] = u6c4c_cycles[j];
        for (int k = j + 1; k < 6; ++k) {
          ms[2] = u6c4c_cycles[k];
          set<int> triple = {i, j, k};
          vector<bool> vertex_in_cycle;
          vector<bool> edge_in_cycle;
          int rich_count = 0;
          for (int e = 0; e < graph.number_of_edges; ++e) {
            int edge_count = 0;
            for (int part = 0; part < 3; ++part) {
              if ((BIT(e) & ms[part]) > 0) {
                ++edge_count;
              }
            }
            if (edge_count != 2) {
              if (!u6c4c_edge_is_poor[e]) {
                rich_count++;
              }
            }
          }
          rich_244_counts.push_back(rich_count);
        }
      }
    }

    rich_chords_frequency = {0, 0, 0};
    for (int e = 0; e < graph.number_of_edges; ++e) {
      int chord_count = 0;
      for (int i = 0; i < 6; ++i) {
        if ((u6c4c_cycles[i] & BIT(e)) == 0) { // edge is from matching
          const int v1 = graph.e2v[e][0];
          const int v2 = graph.e2v[e][1];
          if (layer_vertex_to_circuit[i][v1] == layer_vertex_to_circuit[i][v2]) {
            ++chord_count;
          }
        }
      }
      if (!u6c4c_edge_is_poor[e]) {
        ++rich_chords_frequency[chord_count];
      }
    }

    int odd_poor_2_factors = 0;
    int odd_rich_2_factors = 0;
    for (const auto& c : cur_6c4c) {
      int poor_count = 0;
      int rich_count = 0;
      for (int e = 0; e < graph.number_of_edges; ++e) {
        if ((c & BIT(e)) != 0) { // edge is from 2-factor
          if (!u6c4c_edge_is_poor[e]) {
            ++rich_count;
          } else {
            ++poor_count;
          }
        }
      }
      if (poor_count % 2 != 0) {
        ++odd_poor_2_factors;
      }
      if (rich_count % 2 != 0) {
        ++odd_rich_2_factors;
      }
    }

    int even_poor_matchings = 0;
    int even_rich_matchings = 0;
    for (const auto& c : cur_6c4c) {
      int poor_count = 0;
      int rich_count = 0;
      for (int e = 0; e < graph.number_of_edges; ++e) {
        if ((c & BIT(e)) == 0) { // edge is from matching
          if (!u6c4c_edge_is_poor[e]) {
            ++rich_count;
          } else {
            ++poor_count;
          }
        }
      }
      if (poor_count % 2 == 0) {
        ++even_poor_matchings;
      }
      if (rich_count % 2 == 0) {
        ++even_rich_matchings;
      }
    }

    // ors.clear();
    // all_oriented_vertices.clear();
    // orient_6c4c(graph, 0, true);
    // if (same_cycles_different_orientations == 0) {

    if (graph.number > 0) {
      // FIXME: commenting out cerrs
      // cerr << endl;
      // cerr << "g" << graph.number << ": ";
      // cerr << "another 6c4c: ";
      // // cerr << endl;
      // // cerr << "profile: " << prof << "; ";
      // cerr << "s0: " << s0 << "; ";
      // cerr << "s1: " << s1 << "; ";
      // cerr << "s2: " << s2 << "; ";

      // int rich_244_odd_count = 0;
      // for (const auto& c : rich_244_counts) {
      //   if (c % 2 != 0) {
      //     rich_244_odd_count += 1;
      //   }
      // }
      // cerr << "r244odd: " << rich_244_odd_count << "; ";

      // cerr << "par: " << parity << "; ";

      // cerr << "rich_type_count: " << rich_edge_layer_types.size() << "; ";
      // cerr << "reors: " << same_cycles_different_orientations << "; ";

      // cerr << "em:_" << even_rich_matchings << "_" << even_poor_matchings << "; ";
      // cerr << "o2:_" << odd_rich_2_factors << "_" << odd_poor_2_factors << "; ";

      // cerr << "rchords: " << rich_chords_frequency[0] << " " <<
      //   rich_chords_frequency[1] << " " << rich_chords_frequency[2] << "; ";

      // cerr << "rf: " << rich_edges_frequency[0] << " " <<
      //                   rich_edges_frequency[1] << " " <<
      //                   rich_edges_frequency[2] << " " <<
      //                   rich_edges_frequency[3] << "; ";
      // cerr << "pf: " << poor_edges_frequency[0] << " " <<
      //                   poor_edges_frequency[1] << " " <<
      //                   poor_edges_frequency[2] << " " <<
      //                   poor_edges_frequency[3] << "; ";

      // cerr << "vt: " << vertex_types.size() << "; ";
      // FIXME: commenting out cerrs
      // cerr << "u_op2f: " << odd_poor_2_factors << "; ";
      // cerr << "doubles: " << s0 - different_circuits.size() << "; ";

      // FIXME: commenting out cerrs
      // cerr << "u_comps: " << total_poor_comps << " " << total_rich_comps << "; ";
      if (total_poor_comps == 1) {
        assert((s0 % 2) != (parity % 2)); // TODO
      }
      // FIXME: commenting out cerrs
      // cerr << "u_morecomps: " << odd_poor_comps_2_factors << " " <<
      //                            odd_rich_comps_2_factors << " " <<
      //                            odd_poor_comps_matchings << "; ";

      // FIXME: commenting out cerrs
      // cerr << "circuit lens: ";
      // for (int c = 0; c < all_circuits_in_6c4c.size(); ++c) {
      //   if (c == 0 || layer[c] != layer[c - 1]) {
      //     if (c != 0) {
      //       cerr << ") ";
      //     }
      //     cerr << "l" << layer[c] << " (";
      //   }
      //   cerr << circuit_lens[c] << " ";
      // }
      // cerr << "); ";

      /*if (s1 == graph.number_of_edges - 1) {
        int poor_edge = -1;
        for (int e = 0; e < graph.number_of_edges; ++e) {
          if (u6c4c_edge_is_poor[e]) {
            poor_edge = e;
            break;
          }
        }
        int smallest_circuit_len = graph.number_of_vertices;
        for (const auto& c : graph.all_circuits) {
          if ((BIT(poor_edge) & c) > 0) {
            int len = 0;
            for (int e = 0; e < graph.number_of_edges; ++e) {
              if ((BIT(e) & c) > 0) {
                ++len;
              }
            }
            smallest_circuit_len = min(smallest_circuit_len, len);
          }
        }
        cerr << "small_circ: " << smallest_circuit_len << "; ";
      }*/

      /*if (s1 == graph.number_of_edges - 1) {
        int poor_edge = -1;
        for (int e = 0; e < graph.number_of_edges; ++e) {
          if (u6c4c_edge_is_poor[e]) {
            poor_edge = e;
            break;
          }
        }
        int odd_circuits = 0;
        for (const auto& c : graph.all_circuits) {
          if ((BIT(poor_edge) & c) > 0) {
            ++odd_circuits;
          }
        }
        cerr << "odd_circ: " << odd_circuits << "; ";
      }*/

      // FIXME: commenting out cerrs
      // if (graph.petersen_6c4c.find(cur_6c4c) != graph.petersen_6c4c.end()) {
      //   cerr << "also pet! ";// << endl;
      // }

      /*for (int e = 0; e < graph.number_of_edges; ++e) {
        if (edge_counts[e] > 1) {
          cerr << "has doubly poor edge" << endl;
        }
      }*/

      /*if (poor_circuits.size() > 0) {
        cerr << "poor circuits: ";
        for (const auto& pc : poor_circuits) {
          cerr << pc.second << " ";
        }
        cerr << endl;
      }*/

      /*cerr << "lens: ";
      for (const auto& len : circuit_lens) {
        cerr << len << " ";
      }
      cerr << endl;*/

      /*random_device rd;
      //mt19937 rng(42);
      mt19937 rng(rd());
      set<int> counts;
      contract_poor_edges(graph, true, rng, counts);
      contract_poor_edges(graph, true, rng, counts);
      contract_poor_edges(graph, true, rng, counts);
      if (counts.size() > 0) {
        if (graph.petersen_6c4c.find(cur_6c4c) != graph.petersen_6c4c.end()) {
          cerr << "also pet! ";// << endl;
        }

        cerr << "circuit counts: ";
        for (const int c : counts) {
          cerr << c << " ";
        }
        cerr << endl;
      }*/
    }

    two_cdcs_vertices.clear();

    // TODO: slow
    // FIXMEFIXMEFIXME
    find_2cdcs(graph, false);
    if (has_2cdcs) {
      // cerr << "has_2cdcs; ";
      assert(layers_c1.size() == 6); // TODO
      assert(layers_c2.size() == 6); // TODO
      assert(odd_poor_2_factors == 0); // TODO
      assert((s2 != 1) && (s2 != 2)); // TODO
      // TODO: so s2 parity depends on something?

      int vertex_part[6][MAX_VERTEX];
      for (int v = 0; v < graph.number_of_vertices; ++v) {
        for (int vertex_layer = 0; vertex_layer < 6; ++vertex_layer) {
          vertex_part[vertex_layer][v] = 2;
        }
      }
      for (int ci = 0; ci < all_circuits_in_6c4c.size(); ++ci) {
        if (!two_cdcs_taken[ci]) {
          continue;
        }
        int circuit_layer = layer[ci];
        for (int vi = 0; vi < all_circuits_in_6c4c[ci].size() - 1; ++vi) {
          int v = all_circuits_in_6c4c[ci][vi];
          vertex_part[circuit_layer][v] = 1;
        }
      }

      int odd_poor_pm_both = 0;
      int odd_rich_pm_both = 0;
      int odd_poor_pm_first = 0;
      int odd_rich_pm_first = 0;
      int odd_poor_pm_second = 0;
      int odd_rich_pm_second = 0;

      for (int i = 0; i < 6; ++i) {
        int poor_pm_both = 0;
        int rich_pm_both = 0;
        int poor_pm_first = 0;
        int rich_pm_first = 0;
        int poor_pm_second = 0;
        int rich_pm_second = 0;
        for (int e = 0; e < graph.number_of_edges; ++e) {
          if ((u6c4c_cycles[i] & BIT(e)) == 0) { // edge is from matching
            const int v1 = graph.e2v[e][0];
            const int v2 = graph.e2v[e][1];
            if (vertex_part[i][v1] != vertex_part[i][v2]) {
              if (u6c4c_edge_is_poor[e]) {
                ++poor_pm_both;
              } else {
                ++rich_pm_both;
              }
            } else if (vertex_part[i][v1] == 1) {
              if (u6c4c_edge_is_poor[e]) {
                ++poor_pm_first;
              } else {
                ++rich_pm_first;
              }
            } else {
              if (u6c4c_edge_is_poor[e]) {
                ++poor_pm_second;
              } else {
                ++rich_pm_second;
              }
            }
          }
        }
        if (poor_pm_both % 2 != 0) {
          ++odd_poor_pm_both;
        }
        if (rich_pm_both % 2 != 0) {
          ++odd_rich_pm_both;
        }
        if (poor_pm_first % 2 != 0) {
          ++odd_poor_pm_first;
        }
        if (rich_pm_first % 2 != 0) {
          ++odd_rich_pm_first;
        }
        if (poor_pm_second % 2 != 0) {
          ++odd_poor_pm_second;
        }
        if (rich_pm_second % 2 != 0) {
          ++odd_rich_pm_second;
        }
      }

      // vector<int> poor_both_halfs_frequency = {0, 0, 0};
      // vector<int> rich_both_halfs_frequency = {0, 0, 0};
      // for (int e = 0; e < graph.number_of_edges; ++e) {
      //   int freq = 0;
      //   for (int i = 0; i < 6; ++i) {
      //     if ((u6c4c_cycles[i] & BIT(e)) == 0) { // edge is from matching
      //       const int v1 = graph.e2v[e][0];
      //       const int v2 = graph.e2v[e][1];
      //       if (vertex_part[i][v1] != vertex_part[i][v2]) {
      //         ++freq;
      //       }
      //     }
      //   }
      //   if (u6c4c_edge_is_poor[e]) {
      //     ++poor_both_halfs_frequency[freq];
      //   } else {
      //     ++rich_both_halfs_frequency[freq];
      //   }
      // }

      // cerr << "stats: " << odd_poor_pm_both << " " << odd_rich_pm_both << " " <<
      //     odd_poor_pm_first << " " << odd_rich_pm_first << " " <<
      //     odd_poor_pm_second << " " << odd_rich_pm_second << "; ";

      // cerr << "stats: " << poor_both_halfs_frequency[0] << " " <<
      //     poor_both_halfs_frequency[1] << " " <<
      //     poor_both_halfs_frequency[2] << " " <<
      //     rich_both_halfs_frequency[0] << " " <<
      //     rich_both_halfs_frequency[1] << " " <<
      //     rich_both_halfs_frequency[2] << "; ";
    }

    ors.clear();

    // FIXME: commenting out cerrs
    // cerr << endl;

    // TODO: slow
    /*cerr << "circuits:" << endl;
    for (int c = 0; c < all_circuits_in_6c4c.size(); ++c) {
      cerr << "layer: " << layer[c] << "; vertices: ";
      int rich_count = 0;
      for (int vi = 0; vi < all_circuits_in_6c4c[c].size(); ++vi) {
        int v1 = all_circuits_in_6c4c[c][vi];
        cerr << v1 << " ";
      }

      int poor_count = 0;
      int len = 0;
      int delta = 1;
      int total_len = -1;
      for (int vi = 0; vi < all_circuits_in_6c4c[c].size() - 1; ++vi) {
        int v1 = all_circuits_in_6c4c[c][vi];
        int v2 = all_circuits_in_6c4c[c][vi + 1];
        int ei = graph.vv2e[v1][v2];
        if (u6c4c_edge_is_poor[ei]) {
          ++poor_count;
          delta *= -1;
        } else {
          len += delta;
        }
      }
      if (poor_count % 2 == 0) {
        if (len < 0) {
          len = -len;
        }
        total_len = len;
      }
      cerr << "; rich-poor len: " << total_len << endl;
    }*/


    /*if (!has_2cdcs) {
      check_for_has_four = false;
      find_min_diff(graph);
      mindiff1 = min_diff;*/
      /*check_for_has_four = true;
      find_min_diff(graph);
      mindiff2 = min_diff;
      if (!had_best) {
        mindiff2 = -1;
      }*/

      /*if (had_best) {
        cerr << "mindiff: " << min_diff << "; parity: " << parity << endl;
      } else {
        cerr << "no best; parity: " << parity << endl;
      }*/
    /*} else {
      //cerr << "mindiff: " << 0 << "; parity: " << parity << endl;
      mindiff1 = 0;
      //mindiff2 = 0;
    }*/

    // FIXME2024
    // cerr << endl;
    if (has_2cdcs) {
      all_oriented_vertices.clear();
      orient_6c4c(graph, 0, true);
      orient_6c4c(graph, 0, false);
    }

    // FIXME: commented out cerr
    // if (same_cycles_different_orientations == 0) {
    //   cerr << "vt: " << vertex_types.size() << "; ";
    //   cerr << "UNOR" << endl;
    // } else {
    //   cerr << "allorverts: ";
    //   for (const auto& orverts : all_oriented_vertices) {
    //     for (const auto& v : orverts) {
    //       cerr << v << " ";
    //     }
    //     cerr << "; ";
    //   }
    //   cerr << endl;
    // }

    // FIXME: commented out cerr
    // if (ors.size() > 0) {
    //   cerr << "ors: ";
    //   for (const auto& c : ors) {
    //     cerr << c << " ";
    //   }
    //   cerr << endl;
    // }

    if (same_cycles_different_orientations > 0) {
        ++o6c4c_aggregated_solutions;
        all_o6c4c_solutions += same_cycles_different_orientations;
        for (int i = 0; i < 6; ++i) {
            ++two_factor_count[u6c4c_cycles[i]];
        }
        graph.all_o6c4c.insert(cur_6c4c);
    }

    return false;
}

bool find_33pp_from_3pm(Graph& graph) { // TODO: rewrite using check_33pp
    bool vertex_in_33pp_cycle[MAX_VERTEX];
    bool edge_in_33pp_cycle[MAX_EDGE];
    for (int v = 0; v < graph.number_of_vertices; ++v) {
        vertex_in_33pp_cycle[v] = false;
    }
    Mask cycle_mask = 0;
    for (int e = 0; e < graph.number_of_edges; ++e) {
        int edge_count = 0;
        for (int i = 0; i < 3; ++i) {
            if ((BIT(e) & u3_inv_pm[i]) > 0) {
                ++edge_count;
            }
        }
        if (edge_count == 1 || edge_count == 3) {
            edge_in_33pp_cycle[e] = true;
            cycle_mask += BIT(e);
            for (int i = 0; i < 2; ++i) {
                vertex_in_33pp_cycle[graph.e2v[e][i]] = true;
            }
        } else {
            edge_in_33pp_cycle[e] = false;
        }
    }

    /*if (u333pp_cycles_from_o5cdc.find(cycle_mask) == u333pp_cycles_from_o5cdc.end()) { // has exceptions - 28.05g1422
        return false;
    }*/

    /*if ((poor_mask & cycle_mask) > 0) { // has exceptions - 22.05g3, g8
        return false;
    }*/

    /*bool has_some_oriented_vertices = false; // has exceptions - 10.05g1
    for (const auto& vs : all_oriented_vertices) {
        bool has_this = true;
        for (const auto& v : vs) {
            if (!vertex_in_33pp_cycle[v]) {
                has_this = false;
                break;
            }
        }
        if (has_this) {
            has_some_oriented_vertices = true;
            break;
        }
    }
    if (!has_some_oriented_vertices) {
        return false;
    }*/

    int three_flow_count = 0;
    Mask part_mask[3];
    bool has_3flow[3];
    for (int part = 0; part < 3; ++part) {
        part_mask[part] = 0;
        bool edge_in_cur_part[MAX_EDGE];
        bool vertex_in_cur_part[MAX_VERTEX];
        int vertex_colour[MAX_VERTEX];
        for (int v = 0; v < graph.number_of_vertices; ++v) {
            vertex_colour[v] = -1;
            vertex_in_cur_part[v] = false;
        }
        for (int e = 0; e < graph.number_of_edges; ++e) {
            if (edge_in_33pp_cycle[e]) {
                edge_in_cur_part[e] = true;
            } else {
                edge_in_cur_part[e] = ((BIT(e) & u3_inv_pm[part]) > 0);

                if (edge_in_cur_part[e]) {
                    for (int i = 0; i < 2; ++i) {
                       if (vertex_in_33pp_cycle[graph.e2v[e][i]]) {
                            vertex_in_cur_part[graph.e2v[e][i]] = true;
                        }
                    }
                }
            }
            if (edge_in_cur_part[e]) {
                part_mask[part] += BIT(e);
            }
        }

        bool edge_visited[MAX_EDGE];
        for (int e = 0; e < graph.number_of_edges; ++e) {
            edge_visited[e] = !edge_in_cur_part[e];
        }
        int queue[MAX_VERTEX];
        int queue_size;
        has_3flow[part] = true;
        for (int v = 0; v < graph.number_of_vertices; ++v) {
            if (vertex_in_cur_part[v] && vertex_colour[v] == -1) {
                vertex_colour[v] = 0;
                queue_size = 1;
                queue[0] = v;
                for (int cur_idx = 0; cur_idx < queue_size; ++cur_idx) {
                    int cur_vertex = queue[cur_idx];
                    for (int j = 0; j < MAX_DEG; ++j) {
                        if (edge_visited[graph.v2e[cur_vertex][j]]) {
                            continue;
                        }
                        edge_visited[graph.v2e[cur_vertex][j]] = true;
                        int v1 = cur_vertex;
                        int v2 = graph.v2v[cur_vertex][j];
                        while (!vertex_in_cur_part[v2]) {
                            for (int j2 = 0; j2 < MAX_DEG; ++j2) {
                                int e = graph.v2e[v2][j2];
                                int v3 = graph.v2v[v2][j2];
                                if (!edge_visited[e] && v3 != v1) {
                                    v1 = v2;
                                    v2 = v3;
                                    edge_visited[e] = true;
                                    break;
                                }
                            }
                        }
                        if (vertex_colour[v2] == -1) {
                            vertex_colour[v2] = 1 - vertex_colour[cur_vertex];
                            queue[queue_size] = v2;
                            ++queue_size;
                        } else if (vertex_colour[v2] == vertex_colour[cur_vertex]) {
                            has_3flow[part] = false;
                            break;
                        }
                    }
                    if (!has_3flow[part]) {
                        break;
                    }
                }
            }
            if (!has_3flow[part]) {
                break;
            }
        }
        if (has_3flow[part]) {
            ++three_flow_count;
        }
    }
    if (three_flow_count < 2) {
        return false;
    }

    has_33pp_from_3pm = true;
    if (three_flow_count == 3) {
        u333pp_cycles_from_o6c4c.insert(cycle_mask);
    }

    for (int part = 0; part < 3; ++part) {
        int i = 0;
        int j = 2;
        if (part == 0) {
            i = 1;
        }
        if (part == 2) {
            j = 1;
        }
        if (!has_3flow[i] || !has_3flow[j])
            continue;
        Mask m1 = part_mask[i];
        Mask m2 = part_mask[j];
        if (m1 > m2) {
            swap(m1, m2);
        }
        for (const auto& u5cdc : graph.u5cdc_from_33pp[make_pair(cycle_mask, make_pair(m1, m2))]) {
            graph.u244_6c4c_5cdc_pairs[cur_6c4c].insert(u5cdc);
            graph.u244_5cdc_6c4c_pairs[u5cdc].insert(cur_6c4c);
            /*cerr << "6c4c-5cdc:\t";
            for (const auto& c : cur_6c4c) {
                cerr << c << " ";
            }
            cerr << "\t-\t";
            for (const auto& c : u5cdc) {
                cerr << c << " ";
            }
            cerr << endl;*/
        }
        /*if (u33pp_pairs.find(make_pair(m1, m2)) == u33pp_pairs.end()) {
            graph.print();
            for (int e = 0; e < graph.number_of_edges; ++e) {
                cerr << e;
                if (edge_in_33pp_cycle[e]) {
                    cerr << ": in cycle";
                } else {
                    for (int part = 0; part < 3; ++part) {
                        if ((BIT(e) & u3_inv_pm[part]) == 0) {
                            cerr << ": p" << part;
                        }
                    }
                }
                cerr << endl;
            }

            //u33pp_solutions += 1;
            //return true;
            break;
        }*/
    }
    return false;
    //return false;
    /*
    if (three_flow_count == 2) {
        u33pp_solutions += 1;
    } else if (three_flow_count == 3) {
        u33pp_solutions += 3;
    }
    return false;*/
}

bool find_v_minus_4_cycle_from_6c4c(Graph& graph) {
    has_33pp_from_3pm = false;
    has_333pp_from_3pm = false;
    u33pp_solutions = 0;
    int min_edge_count = graph.number_of_edges;
    set<vector<int>> triples;
    for (int i = 0; i < 6; ++i) {
        u3_inv_pm[0] = u6c4c_cycles[i];
        for (int j = i + 1; j < 6; ++j) {
            u3_inv_pm[1] = u6c4c_cycles[j];
            for (int k = j + 1; k < 6; ++k) {
                u3_inv_pm[2] = u6c4c_cycles[k];
                int cycle_len = 0;
                Mask cycle = 0;
                bool vertex_in_cycle[MAX_VERTEX];
                for (int v = 0; v < graph.number_of_vertices; ++v) {
                    vertex_in_cycle[v] = false;
                }

                for (int e = 0; e < graph.number_of_edges; ++e) {
                    int edge_count = 0;
                    for (int part = 0; part < 3; ++part) {
                        if ((BIT(e) & u3_inv_pm[part]) > 0) {
                            ++edge_count;
                        }
                    }
                    if (edge_count != 2) {
                        cycle |= BIT(e);
                        for (int ii = 0; ii < 2; ++ii) {
                            vertex_in_cycle[graph.e2v[e][ii]] = true;
                        }
                        ++cycle_len;
                    }
                }
                if (cycle_len == graph.number_of_vertices - 4) {
                    v_minus_4_6c4c.insert(cycle);
                    Mask m = 0;
                    for (int v = 0; v < graph.number_of_vertices; ++v) {
                        if (!vertex_in_cycle[v]) {
                            m += BIT(v);
                        }
                        v_minus_4_unvertex_neib_masks.insert(m);
                    }
                    //return true;
                }
            }
        }
    }
    return false;
}

// bool find_planarity_from_6c4c(Graph& graph) {
//     has_33pp_from_3pm = false;
//     has_333pp_from_3pm = false;
//     u33pp_solutions = 0;
//     int min_edge_count = graph.number_of_edges;
//     set<vector<int>> triples;
//     for (int i = 0; i < 6; ++i) {
//         u3_inv_pm[0] = u6c4c_cycles[i];
//         for (int j = i + 1; j < 6; ++j) {
//             u3_inv_pm[1] = u6c4c_cycles[j];
//             for (int k = j + 1; k < 6; ++k) {
//                 u3_inv_pm[2] = u6c4c_cycles[k];
//                 using namespace boost;
//                 typedef adjacency_list<vecS,
//                                        vecS,
//                                        undirectedS,
//                                        property<vertex_index_t, int>
//                                        > boost_graph;
//                 boost_graph g1(graph.number_of_vertices);
//                 boost_graph g2(graph.number_of_vertices);
//                 for (int e = 0; e < graph.number_of_edges; ++e) {
//                     int edge_count = 0;
//                     for (int part = 0; part < 3; ++part) {
//                         if ((BIT(e) & u3_inv_pm[part]) > 0) {
//                             ++edge_count;
//                         }
//                     }
//                     if (edge_count != 1) {
//                         add_edge(graph.e2v[e][0], graph.e2v[e][1], g1);
//                     }
//                     if (edge_count != 3) {
//                         add_edge(graph.e2v[e][0], graph.e2v[e][1], g2);
//                     }
//                 }

//                 if (boyer_myrvold_planarity_test(g1) && boyer_myrvold_planarity_test(g2)) {
//                     return true;
//                 }
//             }
//         }
//     }
//     return false;
// }

bool find_33pp_from_6c4c(Graph& graph) {
    has_33pp_from_3pm = false;
    has_333pp_from_3pm = false;
    u33pp_solutions = 0;
    int min_edge_count = graph.number_of_edges;
    set<vector<int>> triples;
    for (int i = 0; i < 6; ++i) {
        u3_inv_pm[0] = u6c4c_cycles[i];
        for (int j = i + 1; j < 6; ++j) {
            u3_inv_pm[1] = u6c4c_cycles[j];
            for (int k = j + 1; k < 6; ++k) {
                u3_inv_pm[2] = u6c4c_cycles[k];
                int edge_count = 0;
                for (int e = 0; e < graph.number_of_edges; ++e) {
                    if ((BIT(e) & u3_inv_pm[0] & u3_inv_pm[1] & u3_inv_pm[2]) > 0) {
                        ++edge_count;
                    }
                }
                if (edge_count < min_edge_count) {
                    triples.clear();
                    min_edge_count = edge_count;
                }
                //if (edge_count == min_edge_count) {
                    vector<int> anti_triple;
                    for (int m = 0; m < 6; ++m) {
                        if (m != i && m != j && m != k) {
                            anti_triple.push_back(m);
                        }
                    }
                    vector<int> triple = {i, j, k};
                    triples.insert(anti_triple);
                    triples.insert(triple);
                //}
            }
        }
    }
    for (const auto& triple: triples) {
        u3_inv_pm[0] = u6c4c_cycles[triple[0]];
        u3_inv_pm[1] = u6c4c_cycles[triple[1]];
        u3_inv_pm[2] = u6c4c_cycles[triple[2]];
        find_33pp_from_3pm(graph);
        if (has_33pp_from_3pm) {
            has_33pp_from_3pm = false;
            all_33pp_triples.insert(triple);
            common_triples.insert(triple);
        }
        if (u33pp_solutions > 0) {
            return true;
        }
    }
    return false;
}

/*void analyze_edges_of_o6c4c_solution(Graph& graph) {
    set<pair<int, int>> counts[MAX_EDGE];
    for (int e = 0; e < graph.number_of_edges; ++e) {
        counts[e].clear();
    }

    for (int c = 0; c < all_circuits_in_6c4c.size(); ++c) {
        for (int vi = 0; vi < all_circuits_in_6c4c[c].size() - 1; ++vi) {
            int v0 = -1;
            if (vi > 0) {
                v0 = all_circuits_in_6c4c[c][vi - 1];
            } else {
                v0 = all_circuits_in_6c4c[c][all_circuits_in_6c4c[c].size() - 2];
            }
            int v1 = all_circuits_in_6c4c[c][vi];
            int v2 = all_circuits_in_6c4c[c][vi + 1];
            int v3 = -1;
            if (vi < all_circuits_in_6c4c[c].size() - 2) {
                v3 = all_circuits_in_6c4c[c][vi + 2];
            } else {
                v3 = all_circuits_in_6c4c[c][1];
            }
            if (v0 > v3) {
                swap(v0, v3);
            }
            int ei = graph.vv2e[v1][v2];
            counts[ei].insert(make_pair(v0, v3));
        }
    }
    int poor_count = 0;
    for (int e = 0; e < graph.number_of_edges; ++e) {
        if (counts[e].size() != 2 && counts[e].size() != 4) {
            cerr << "wut" << endl;
            continue;
        }
        if (counts[e].size() == 2) {
            ++poor_count;
            u6c4c_always_rich[e] = false;
        } else {
            u6c4c_always_poor[e] = false;
        }
    }
    u6c4c_min_poor = min(u6c4c_min_poor, poor_count);
    u6c4c_max_poor = max(u6c4c_max_poor, poor_count);
}*/

bool gen_o6c4c(Graph& graph, int cur_cycle_layer, int min_cycle_idx, bool only_find) {
    if (cur_cycle_layer == 6) {
        int edge_count[MAX_EDGE];
        for (int e = 0; e < graph.number_of_edges; ++e) {
            edge_count[e] = 0;
        }
        for (int c = 0; c < 6; ++c) {
            for (int e = 0; e < graph.number_of_edges; ++e) {
                if (BIT(e) & u6c4c_cycles[c]) {
                    ++edge_count[e];
                }
            }
        }
        for (int e = 0; e < graph.number_of_edges; ++e) {
            if (edge_count[e] != 4) {
                return false;
            }
        }

        ++all_6c4c_solutions;
        cur_6c4c.clear();
        for (int c = 0; c < 6; ++c) {
            cur_6c4c.insert(u6c4c_cycles[c]);
        }

        graph.all_6c4c.insert(cur_6c4c);


        set<vector<int>> cur_6c4c_triples;
        //set<vector<int>> cur_o6c4c_triples;

        for (int j = 1; j < 6; ++j) {
            for (int k = j + 1; k < 6; ++k) {
                vector<int> triple;
                triple.push_back(0);
                triple.push_back(j);
                triple.push_back(k);
                cur_6c4c_triples.insert(triple);

                // Mask cycle_mask = 0;
                // for (int e = 0; e < graph.number_of_edges; ++e) {
                //   int cnt = 0;
                //   for (const auto& c : triple) {
                //     if (BIT(e) & u6c4c_cycles[c]) {
                //       ++cnt;
                //     }
                //   }
                //   if (cnt != 2) {
                //     cycle_mask += BIT(e);
                //   }
                // }
                // if (graph.all_nonseparating_cycles.find(cycle_mask) == graph.all_nonseparating_cycles.end()) {
                //   cout << "nah" << endl;
                // }

                /*if (has_o6c4c) {
                    cur_o6c4c_triples.insert(triple);
                }*/
            }
        }

        o244_triples.clear();
        // TODO: slow
        // find_o244_flows_from_6c4c(cur_6c4c_triples, graph);

        // FIXME2024
        // if (o244_triples.size() != 9) {
        //   return false;
        // }

        if (check_orientability_6c4c(graph)) {
            // FIXME: commented out cerr
            // cerr << "found o6c4c" << endl;
            return true;
        }

        bool has_o6c4c = (same_cycles_different_orientations > 0);

        // FIXME: commented out cerr
        // cerr << endl << endl << endl;

        if (has_o6c4c) {
          /*if (!has_dominating_circuit) {
            cerr << "undom" << endl;
          }*/

          /* else {
            cerr << "dom!" << endl;
          }*/
        }

        // TODO2024: uncomment and research

        // for (int i = 0; i < 6; ++i) {
        //     for (int j = i + 1; j < 6; ++j) {
        //         for (int k = j + 1; k < 6; ++k) {
        //             set<Mask> triple;
        //             triple.insert(u6c4c_cycles[i]);
        //             triple.insert(u6c4c_cycles[j]);
        //             triple.insert(u6c4c_cycles[k]);
        //             all_6c4c_triples.insert(triple);
        //             if (has_o6c4c) {
        //                 all_o6c4c_triples.insert(triple);
        //             }
        //         }
        //     }
        // }

        // all_33pp_triples.clear();
        // find_33pp_from_6c4c(graph);
        // has_33pp_from_3pm = all_33pp_triples.size() > 0;
        // if (!has_33pp_from_3pm) {
        //     return false;
        // }
        
        // if (!has_o6c4c) {
        //     return false;
        // }
        
        // poor_mask = 0;
        // for (int e = 0; e < graph.number_of_edges; ++e) {
        //     if (u6c4c_edge_is_poor[e]) {
        //         poor_mask += BIT(e);
        //     }
        // }
        // //return false;
        
        // o244_triples.clear();
        // find_o244_flows_from_6c4c(all_33pp_triples, graph);
        // has_o244_flows = o244_triples.size() > 0;
        // if (!has_o244_flows) {
        //     return false;
        // }
        
        // bool have_same_triple = false;
        // common_triples.clear();
        // for (const auto& triple : all_33pp_triples) {
        //     if (o244_triples.find(triple) != o244_triples.end()) {
        //         have_same_triple = true;
        //         common_triples.insert(triple);
        //     }
        // }
        // if (!have_same_triple) {
        //     return false;
        // }
        // //has_o244_flows = true;
        // //bool have_same_triple = true;
        
        // find_333flows_from_6c4c(common_triples, graph);
        // if (!has_all_3flows) {
        //     return false;
        // }
        
        // if (has_o6c4c && has_33pp_from_3pm && has_all_3flows && has_o244_flows && have_same_triple) {// && has_dominating_circuit) {
        //     has_or_comb = true;
        // }
        // if (has_or_comb) {
        //     cerr << "has oriented combination" << endl;
        //     return true;
        // }
        // /*
        // if (has_33pp_from_3pm && has_333pp_from_3pm && has_all_3flows && has_o244_flows && have_same_triple) {
        //     has_un_comb = true;
        // }*/
        
        // /*if (has_or_comb && has_un_comb) {
        //     cerr << "has both" << endl;
        //     return true;
        // }*/
        
        // //if ((same_cycles_different_orientations > 0) && has_33pp_from_3pm && has_all_3flows) {
        // //cerr << "6c4c:\t" << (same_cycles_different_orientations > 0) << "\t" << has_33pp_from_3pm << "\t" <<
        // //        has_333pp_from_3pm << "\t" << has_all_3flows << "\t" << has_o244_flows << "\t" << have_same_triple << endl;
        //     //return true;
        // //}
        
        // if (same_cycles_different_orientations > 0) {
        //     all_33pp_solutions += u33pp_solutions;
        //     if (all_33pp_solutions > 0)
        //         return true;
        // }
        // return false;




        return false;

        // //     // if (only_find) {
        // //     //     return false;
        // //     // }
        // //     //
        // //     // if (find_v_minus_4_cycle_from_6c4c(graph)) {
        // //     //     cerr << "found v-4 cycle" << endl;
        // //     //     has_or_comb = true;
        // //     //     return true;
        // //     // }
        // //     // if (graph.all_even_v_minus_4_cycles.size() == v_minus_4_6c4c.size()) {
        // //     //     return true;
        // //     // }
        // //     //
        // //     // /*if (find_planarity_from_6c4c(graph)) {
        // //     //     cerr << "found planar decomposition" << endl;
        // //     //     has_or_comb = true;
        // //     //     return true;
        // //     // }*/
        // //     //
        // //     //
        // //     // for (int i = 0; i < 6; ++i) {
        // //     //     for (int j = i + 1; j < 6; ++j) {
        // //     //         for (int k = j + 1; k < 6; ++k) {
        // //     //             set<Mask> triple;
        // //     //             triple.insert(u6c4c_cycles[i]);
        // //     //             triple.insert(u6c4c_cycles[j]);
        // //     //             triple.insert(u6c4c_cycles[k]);
        // //     //             all_6c4c_triples.insert(triple);
        // //     //             if (has_o6c4c) {
        // //     //                 all_o6c4c_triples.insert(triple);
        // //     //             }
        // //     //         }
        // //     //     }
        // //     // }
        // //     //
        // //     // all_33pp_triples.clear();
        // //     // find_33pp_from_6c4c(graph);
        // //     // has_33pp_from_3pm = all_33pp_triples.size() > 0;
        // //     // if (!has_33pp_from_3pm) {
        // //     //     return false;
        // //     // }
        // //     //
        // //     // //all_oriented_vertices.clear();
        // //     // if (!has_o6c4c) {
        // //     //     return false;
        // //     // }
        // //     //
        // //     // poor_mask = 0;
        // //     // for (int e = 0; e < graph.number_of_edges; ++e) {
        // //     //     if (u6c4c_edge_is_poor[e]) {
        // //     //         poor_mask += BIT(e);
        // //     //     }
        // //     // }
        // //     // //return false;
        // //     //
        // //     // o244_triples.clear();
        // //     // find_o244_flows_from_6c4c(all_33pp_triples, graph);
        // //     // has_o244_flows = o244_triples.size() > 0;
        // //     // if (!has_o244_flows) {
        // //     //     return false;
        // //     // }
        // //     //
        // //     // bool have_same_triple = false;
        // //     // common_triples.clear();
        // //     // for (const auto& triple : all_33pp_triples) {
        // //     //     if (o244_triples.find(triple) != o244_triples.end()) {
        // //     //         have_same_triple = true;
        // //     //         common_triples.insert(triple);
        // //     //     }
        // //     // }
        // //     // if (!have_same_triple) {
        // //     //     return false;
        // //     // }
        // //     // //has_o244_flows = true;
        // //     // //bool have_same_triple = true;
        // //     //
        // //     // find_333flows_from_6c4c(common_triples, graph);
        // //     // if (!has_all_3flows) {
        // //     //     return false;
        // //     // }
        // //     //
        // //     // if (has_o6c4c && has_33pp_from_3pm && has_all_3flows && has_o244_flows && have_same_triple) {// && has_dominating_circuit) {
        // //     //     has_or_comb = true;
        // //     // }
        // //     // if (has_or_comb) {
        // //     //     cerr << "has oriented combination" << endl;
        // //     //     return true;
        // //     // }
        // //     // /*
        // //     // if (has_33pp_from_3pm && has_333pp_from_3pm && has_all_3flows && has_o244_flows && have_same_triple) {
        // //     //     has_un_comb = true;
        // //     // }*/
        // //     //
        // //     // /*if (has_or_comb && has_un_comb) {
        // //     //     cerr << "has both" << endl;
        // //     //     return true;
        // //     // }*/
        // //     //
        // //     // //if ((same_cycles_different_orientations > 0) && has_33pp_from_3pm && has_all_3flows) {
        // //     // //cerr << "6c4c:\t" << (same_cycles_different_orientations > 0) << "\t" << has_33pp_from_3pm << "\t" <<
        // //     // //        has_333pp_from_3pm << "\t" << has_all_3flows << "\t" << has_o244_flows << "\t" << have_same_triple << endl;
        // //     //     //return true;
        // //     // //}
        // //     //
        // //     // if (same_cycles_different_orientations > 0) {
        // //     //     all_33pp_solutions += u33pp_solutions;
        // //     //     if (all_33pp_solutions > 0)
        // //     //         return true;
        // //     // }
        // //     // return false;
        }
    for (int i = min_cycle_idx; i < bit_cycles.size(); ++i) {
        u6c4c_cycles[cur_cycle_layer] = bit_cycles[i];
        bool compat = true;
        for (int c1 = 0; c1 < cur_cycle_layer; ++c1) {
            for (int c2 = c1 + 1; c2 < cur_cycle_layer; ++c2) {
                if (inv(graph, u6c4c_cycles[c1]) & inv(graph, u6c4c_cycles[c2]) & inv(graph, u6c4c_cycles[cur_cycle_layer])) {
                    compat = false;
                    break;
                }
            }
            if (!compat) {
                break;
            }
        }
        if (compat) {
            int next_cycle_idx = i + 1;
            if (!graph.is_definitely_snark) {
              next_cycle_idx = i;
            }
            if (gen_o6c4c(graph, cur_cycle_layer + 1, next_cycle_idx, only_find)) {
                return true;
            }
        }
    }
    return false;
}

/*
void find_o6c4c_compatible_with_preimages(Graph& graph) {
    cerr << "overall full cycles: " << graph.all_2_factors.size() << endl;
    cerr << "full cycles from petersen: " << 2_factors_from_petersen.size() << endl;
    bit_cycles.clear();
    for (const auto& c : graph.all_2_factors) {
        if (2_factors_from_petersen.find(c) != 2_factors_from_petersen.end()) {
            bit_cycles.push_back(c);
        }
    }
    cerr << "left: " << bit_cycles.size() << endl;
    if (!gen_o6c4c(graph, 0, 0)) {
        cerr << "didn't find o6c4c" << endl;
    }
}
*/

void find_all_o6c4c(Graph& graph, bool only_find = false) {
  sol_count = 0;
  all_6c4c_triples.clear();
  all_o6c4c_triples.clear();
  o6c4c_aggregated_solutions = 0;
  all_o6c4c_solutions = 0;
  all_6c4c_solutions = 0;
  all_nz_mod5_from_o6c4c = 0;
  all_nz_mod6_from_o6c4c = 0;
  all_33pp_solutions = 0;
  two_factor_count.clear();
  bit_cycles.clear();
  u333pp_cycles_from_o6c4c.clear();
  for (const auto& c : graph.all_full_cycles) {
    bit_cycles.push_back(c);
    two_factor_count[c] = 0;
  }
  has_or_comb = false;
  has_un_comb = false;
  for (int e = 0; e < graph.number_of_edges; ++e) {
    u6c4c_always_poor[e] = true;
    u6c4c_always_rich[e] = true;

    o6c4c_always_1[e] = true;
    o6c4c_always_2[e] = true;
    o6c4c_always_3[e] = true;
    o6c4c_always_4[e] = true;

    o6c4c_never_1[e] = true;
    o6c4c_never_2[e] = true;
    o6c4c_never_3[e] = true;
    o6c4c_never_4[e] = true;
  }
  u6c4c_min_poor = graph.number_of_edges;
  u6c4c_max_poor = 0;

  v_minus_4_6c4c.clear();

  ch0l.clear();
  ch1l.clear();
  ch1chl.clear();
  ch2l.clear();

  oriented_vertices_to_s0_parity.clear();
  all_ve_types.clear();

  gen_o6c4c(graph, 0, 0, only_find);
  cerr << "all 6c4c solutions: " << all_6c4c_solutions << endl;
  cerr << "all aggregated o6c4c solutions: " << o6c4c_aggregated_solutions << endl;
  cerr << "all o6c4c solutions: " << all_o6c4c_solutions << endl;
  //cerr << "all nz-mod5 from o6c4c combinations: " << all_nz_mod5_from_o6c4c << endl;
  //cerr << "all nz-mod6 from o6c4c combinations: " << all_nz_mod6_from_o6c4c << endl;
  if (only_find) {
    return;
  }

  // bool has_unvertex = false;
  // for (const Mask& m : graph.all_vertex_neib_masks) {
  //     if (v_minus_4_unvertex_neib_masks.find(m) != v_minus_4_unvertex_neib_masks.end()) {
  //         has_unvertex = true;
  //         break;
  //     }
  // }
  // //if (graph.all_even_v_minus_4_cycles.size() == v_minus_4_6c4c.size()) {
  // if (has_unvertex) {
  //     //cerr << "all v-4 are good" << endl;
  //     cerr << "has unvertex" << endl;
  //     return;
  // } else {
  //     //cerr << "some v-4 are bad" << endl;
  //     cerr << "no unvertex found" << endl;
  //     cerr << graph.all_even_v_minus_4_cycles.size() << " " << v_minus_4_6c4c.size() << endl;
  //     for (const auto& c : graph.all_even_v_minus_4_cycles) {
  //         if (v_minus_4_6c4c.find(c) == v_minus_4_6c4c.end()) {
  //             graph.print();
  //             for (int e = 0; e < graph.number_of_edges; ++e) {
  //                 cerr << e;
  //                 if (BIT(e) & c) {
  //                     cerr << ": in cycle";
  //                 }
  //                 cerr << endl;
  //             }
  //             break;
  //         }
  //     }
  //     return;
  // }

  if (!only_find && !has_or_comb) {
      //cerr << "no v-4 cycle found" << endl;
      //cerr << "no planar decomposition found" << endl;
      //cerr << "no oriented solution found!" << endl;
  }

  //for (const auto& c : two_factor_count) {
  //    cerr << c.first << "\t" << c.second * 1.0 / o6c4c_aggregated_solutions << endl;
  //}
  //cerr << "ratio: " << graph.petersen_colourings.size() * 1.0 / o6c4c_aggregated_solutions << endl;
  //cerr << "ratio2: " << graph.profiles.size() * 1.0 / o6c4c_aggregated_solutions << endl;
  //cerr << "ratio3: " << o6c4c_aggregated_solutions * 1.0 / all_o6c4c_solutions << endl;

  //cerr << "all 6c4c solutions: " << all_6c4c_solutions << endl;
  /*cerr << "all aggregated o6c4c solutions: " << o6c4c_aggregated_solutions << endl;
  cerr << "all o6c4c solutions: " << all_o6c4c_solutions << endl;
  cerr << "all nz-mod5 from o6c4c combinations: " << all_nz_mod5_from_o6c4c << endl;
  cerr << "all 33pp solutions from 6c4c: " << all_33pp_solutions << endl;
  cerr << endl;*/
}

} // Exp6c4c
