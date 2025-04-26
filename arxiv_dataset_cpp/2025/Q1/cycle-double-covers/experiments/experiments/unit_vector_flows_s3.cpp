/*
 * File:   unit_vector_flows_s3.cpp
 * Author: Nikolay Ulyanov (ulyanick@gmail.com)
 *
 * TODO:
 *
 * Created on August 17, 2019
 *
 */

#include "unit_vector_flows_s3.h"

#include <cassert>
#include <math.h>
#include <map>
#include <random>

namespace ExpUnitVectorFlows {

const double EPS = 1e-8;
const vector<double> zero_vec{0.0, 0.0, 0.0, 0.0};

int edge_unit_vector_idx[MAX_EDGE];
vector<int> vertex_flowed_edges[MAX_EDGE];

vector<vector<double>> all_unit_vectors;
map<int, int> ops;
map<vector<int>, int> triples;
map<int, vector<vector<int>>> triples_for_vector;
vector<int> vector_flow;

/*********************************Methods*********************************/

bool same(double f1, double f2) {
  return abs(f1 - f2) < EPS;
}

double dist(const vector<double>& v1, const vector<double>& v2) {
  return sqrt(
      (v1[0] - v2[0]) * (v1[0] - v2[0]) +
      (v1[1] - v2[1]) * (v1[1] - v2[1]) +
      (v1[2] - v2[2]) * (v1[2] - v2[2]) +
      (v1[3] - v2[3]) * (v1[3] - v2[3]));
}

double norm(const vector<double>& v) {
  return dist(v, zero_vec);
}

vector<vector<double>> gen_even_permutations(const vector<vector<double>>& vectors) {
  vector<vector<double>> permutations;
  for (const auto& v : vectors) {
    permutations.push_back(v);
    permutations.push_back({v[0], v[2], v[3], v[1]});
    permutations.push_back({v[0], v[3], v[1], v[2]});
    permutations.push_back({v[1], v[2], v[0], v[3]});
    permutations.push_back({v[1], v[0], v[3], v[2]});
    permutations.push_back({v[1], v[3], v[2], v[0]});
    permutations.push_back({v[2], v[0], v[1], v[3]});
    permutations.push_back({v[2], v[1], v[3], v[0]});
    permutations.push_back({v[2], v[3], v[0], v[1]});
    permutations.push_back({v[3], v[0], v[2], v[1]});
    permutations.push_back({v[3], v[2], v[1], v[0]});
    permutations.push_back({v[3], v[1], v[0], v[2]});
  }
  return permutations;
}

vector<vector<double>> gen_all_permutations(const vector<vector<double>>& vectors) {
  vector<vector<double>> permutations;
  for (const auto& v : vectors) {
    vector<vector<double>> even = gen_even_permutations({v});
    vector<vector<double>> odd = gen_even_permutations({{v[1], v[0], v[2], v[3]}});
    for (const auto& v2 : even) {
      permutations.push_back(v2);
    }
    for (const auto& v2 : odd) {
      permutations.push_back(v2);
    }
  }
  return permutations;
}

vector<vector<double>> gen_plus_minus_variations(const vector<vector<double>>& vectors) {
  vector<vector<double>> permutations;
  for (const auto& v : vectors) {
    vector<vector<int>> multipliers;
    for (const auto& p : v) {
      if (same(p, 0)) {
        multipliers.push_back({1});
      } else {
        multipliers.push_back({1, -1});
      }
    }
    for (const auto& m0 : multipliers[0]) {
      for (const auto& m1 : multipliers[1]) {
        for (const auto& m2 : multipliers[2]) {
          for (const auto& m3 : multipliers[3]) {
            permutations.push_back({v[0] * m0, v[1] * m1, v[2] * m2, v[3] * m3});
          }
        }
      }
    }
  }
  return permutations;
}

vector<vector<double>> gen_all_plus_minus_permutations(const vector<vector<double>>& vectors) {
  return gen_plus_minus_variations(gen_all_permutations(vectors));
}

vector<vector<double>> gen_unit_vectors() {
  vector<vector<double>> nonunit_vectors;
  vector<vector<double>> new_vectors;

  new_vectors = gen_all_plus_minus_permutations({{0.0, 0.0, 0.0, 1.0}});
  for (const auto& v : new_vectors) {
    nonunit_vectors.push_back(v);
  }

  double phi = (1.0 + sqrt(5)) / 2;
  new_vectors = gen_plus_minus_variations(gen_even_permutations({{phi, 1.0, 1.0 / phi, 0.0}}));
  for (const auto& v : new_vectors) {
    nonunit_vectors.push_back(v);
  }

  vector<vector<double>> unit_vectors;
  for (const auto& v : nonunit_vectors) {
    double v_norm = norm(v);
    unit_vectors.push_back({v[0] / v_norm, v[1] / v_norm, v[2] / v_norm, v[3] / v_norm});
  }
  return unit_vectors;
}

void find_vector_relations(const vector<vector<double>>& unit_vectors) {
  vector<vector<double>> tmp_unit_vectors;
  for (size_t i = 0; i < unit_vectors.size(); ++i) {
    bool found_duplicate = false;
    for (size_t j = i + 1; j < unit_vectors.size(); ++j) {
      if (same(dist(unit_vectors[i], unit_vectors[j]), 0)) {
        found_duplicate = true;
        break;
      }
    }
    if (!found_duplicate) {
      tmp_unit_vectors.push_back(unit_vectors[i]);
    }
  }

  vector<bool> has_triple(tmp_unit_vectors.size(), false);
  for (size_t i = 0; i < tmp_unit_vectors.size(); ++i) {
    const auto v0 = tmp_unit_vectors[i];
    for (size_t j = i + 1; j < tmp_unit_vectors.size(); ++j) {
      const auto v1 = tmp_unit_vectors[j];
      for (size_t k = j + 1; k < tmp_unit_vectors.size(); ++k) {
        const auto v2 = tmp_unit_vectors[k];
        if (same(v0[0] + v1[0] + v2[0], 0) &&
            same(v0[1] + v1[1] + v2[1], 0) &&
            same(v0[2] + v1[2] + v2[2], 0) &&
            same(v0[3] + v1[3] + v2[3], 0)) {
          has_triple[i] = true;
          has_triple[j] = true;
          has_triple[k] = true;
        }
      }
    }
  }

  vector<vector<double>> too_much_unit_vectors;
  for (size_t i = 0; i < tmp_unit_vectors.size(); ++i) {
    if (has_triple[i]) {
      too_much_unit_vectors.push_back(tmp_unit_vectors[i]);
    }
  }
  all_unit_vectors.clear();
  vector<int> selection = {0, 1, 10, 13, 16, 17, 22, 23, 32, 39, 59, 60, 74, 75, 76, 77, 82, 83, 84, 85};
  for (const auto& idx : selection) {
    all_unit_vectors.push_back(too_much_unit_vectors[idx]);
  }

  cerr << "unit vectors count: " << all_unit_vectors.size() << endl;

  ops.clear();
  for (int i = 0; i < all_unit_vectors.size(); ++i) {
    if (ops.find(i) != ops.end()) {
      continue;
    }
    const auto v0 = all_unit_vectors[i];
    for (int j = i + 1; j < all_unit_vectors.size(); ++j) {
      const auto v1 = all_unit_vectors[j];
      if (same(v0[0] + v1[0], 0) &&
          same(v0[1] + v1[1], 0) &&
          same(v0[2] + v1[2], 0) &&
          same(v0[3] + v1[3], 0)) {
        ops[i] = j;
        ops[j] = i;
        cerr << "ops: " << i << " " << j << "; " <<
          v0[0] << " " << v0[1] << " " << v0[2] << " " << v0[3] << endl;
        break;
      }
    }
    assert(ops.find(i) != ops.end());
  }

  triples.clear();
  int triples_count = 0;
  for (int i = 0; i < all_unit_vectors.size(); ++i) {
    const auto v0 = all_unit_vectors[i];
    bool has_triple = false;
    for (int j = i + 1; j < all_unit_vectors.size(); ++j) {
      const auto v1 = all_unit_vectors[j];
      for (int k = j + 1; k < all_unit_vectors.size(); ++k) {
        const auto v2 = all_unit_vectors[k];
        if (same(v0[0] + v1[0] + v2[0], 0) &&
            same(v0[1] + v1[1] + v2[1], 0) &&
            same(v0[2] + v1[2] + v2[2], 0) &&
            same(v0[3] + v1[3] + v2[3], 0)) {
          triples_count++;
          triples[{i, j}] = k;
          triples[{j, i}] = k;
          triples[{i, k}] = j;
          triples[{k, i}] = j;
          triples[{j, k}] = i;
          triples[{k, j}] = i;
          triples_for_vector[i].push_back({j, k});
          triples_for_vector[j].push_back({i, k});
          triples_for_vector[k].push_back({i, j});
          cerr << "triples: " << i << " " << j << " " << k << endl;
        }
      }
    }
  }
  cerr << "unit vectors triples count: " << triples_count << endl;
}

bool gen_unit_vector_flows(Graph& graph, int cur_edge_idx) {
  if (cur_edge_idx == graph.number_of_edges) {
    vector<int> vectors;
    set<int> uniq_vectors;
    for (int e = 0; e < graph.number_of_edges; ++e) {
      vectors.push_back(edge_unit_vector_idx[e]);
      uniq_vectors.insert(edge_unit_vector_idx[e]);
      uniq_vectors.insert(ops[edge_unit_vector_idx[e]]);
    }
    cerr << "vector count: " << uniq_vectors.size() << endl;

    // set<vector<int>> graph_triples;
    // for (int v = 0; v < graph.number_of_vertices; ++v) {
    //   vector<int> v_triple;
    //   for (int j = 0; j < MAX_DEG; ++j) {
    //     int e_vector = edge_unit_vector_idx[graph.v2e[v][j]];
    //     e_vector = min(e_vector, ops[e_vector]);
    //     v_triple.push_back(e_vector);
    //   }
    //   sort(v_triple.begin(), v_triple.end());
    //   graph_triples.insert(v_triple);
    // }
    // cerr << "graph used triple count: " << graph_triples.size() << endl;

    graph.unit_vector_flows.push_back(vectors);
    return true;
    // return false;
  }

  int cur_edge = graph.faster_edge_order[cur_edge_idx];

  int v1 = graph.e2v[cur_edge][0];
  int v2 = graph.e2v[cur_edge][1];
  int flowed1 = vertex_flowed_edges[v1].size();
  int flowed2 = vertex_flowed_edges[v2].size();
  vector<int> flow_vector_indices;
  if (flowed1 == MAX_DEG - 1) {
    flow_vector_indices.push_back(triples[vertex_flowed_edges[v1]]);
  }
  if (flowed2 == MAX_DEG - 1) {
    int other_flow_idx = ops[triples[vertex_flowed_edges[v2]]];
    if (flowed1 == MAX_DEG - 1) {
      if (other_flow_idx != flow_vector_indices[0]) {
        return false;
      }
    } else {
      flow_vector_indices.push_back(other_flow_idx);
    }
  }
  if (flowed1 != MAX_DEG - 1 && flowed2 != MAX_DEG - 1) {
    for (size_t i = 0; i < all_unit_vectors.size(); ++i) {
      flow_vector_indices.push_back(i);
    }
  }

  for (const auto& v_idx : flow_vector_indices) {
    edge_unit_vector_idx[cur_edge] = v_idx;
    int op_idx = ops[v_idx];
    vertex_flowed_edges[v1].push_back(v_idx);
    vertex_flowed_edges[v2].push_back(op_idx);
    bool can_flow_v1 = (vertex_flowed_edges[v1].size() != 2) ||
        (triples.find(vertex_flowed_edges[v1]) != triples.end());
    bool can_flow_v2 = (vertex_flowed_edges[v2].size() != 2) ||
        (triples.find(vertex_flowed_edges[v2]) != triples.end());
    if (can_flow_v1 && can_flow_v2 && gen_unit_vector_flows(graph, cur_edge_idx + 1)) {
      return true;
    }
    vertex_flowed_edges[v1].pop_back();
    vertex_flowed_edges[v2].pop_back();
  }
  edge_unit_vector_idx[cur_edge] = NONE;
  return false;
}

bool gen_vector_flow(int cur_vec_idx, int max_flow, int vecs_count,
    const map<int, int>& vecs_ops,
    const map<int, vector<vector<int>>>& vecs_triples,
    const vector<int>& vector_order,
    vector<int>& vecs_flow) {
  if (cur_vec_idx == vecs_count) {
    return true;
  }
  int cur_vec = vector_order[cur_vec_idx];
  vector<int> possible_flows;
  const auto ops_it = vecs_ops.find(cur_vec);
  if (ops_it != vecs_ops.end() && vecs_flow[ops_it->second] != 0) {
    possible_flows = {-vecs_flow[ops_it->second]};
  }
  const auto triples_it = vecs_triples.find(cur_vec);
  for (const auto& triple : triples_it->second) {
    int v2 = triple[0];
    int v3 = triple[1];
    if (vecs_flow[v2] != 0 && vecs_flow[v3] != 0) {
      int f = 0 - vecs_flow[v2] - vecs_flow[v3];
      if (f == 0 || abs(f) > max_flow) {
        return false;
      } else {
        if (possible_flows.size() == 1 && possible_flows[0] != f) {
          return false;
        } else {
          possible_flows = {f};
        }
      }
    }
  }
  if (possible_flows.empty()) {
    for (int f = max_flow; f >= 1; --f) {
      possible_flows.push_back(f);
      possible_flows.push_back(-f);
    }
  }
  for (const auto& f : possible_flows) {
    vecs_flow[cur_vec] = f;
    if (gen_vector_flow(cur_vec_idx + 1, max_flow, vecs_count,
        vecs_ops, vecs_triples, vector_order, vecs_flow)) {
      return true;
    }
  }
  vecs_flow[cur_vec] = 0;
  /*if (cur_vec_idx > prev_max_idx) {
    cout << iteration << " " << cur_vec_idx << endl;
    prev_max_idx = cur_vec_idx;
  }*/
  return false;
}

void find_configuration_flow(const string& configuration_name, int vecs_count,
      const map<int, int>& vecs_ops,
      const map<int, vector<vector<int>>>& vecs_triples,
      vector<int>& vector_order) {
  if (vector_order.empty()) {
    for (int i = 0; i < vecs_count; ++i) {
      vector_order.push_back(i);
    }
  }
  int max_flow = 1;
  int max_checked_flow = 6;
  vector_flow.clear();
  for (int i = 0; i < vecs_count; ++i) {
    vector_flow.push_back(0);
  }
  while (!gen_vector_flow(0, max_flow, vecs_count,
      vecs_ops, vecs_triples, vector_order, vector_flow)) {
    ++max_flow;
    if (max_flow >= max_checked_flow) {
      cerr << "didn't find any nz-flow for " << configuration_name << endl;
      return;
    }
    cerr << "increasing to nz" << max_flow + 1 << "-flow" << endl;
    for (int i = 0; i < vecs_count; ++i) {
      vector_flow[i] = 0;
    }
  }
  cerr << "found nz" << max_flow + 1 << "-flow for " << configuration_name << " configuration" << endl;
  for (int i = 0; i < vecs_count; ++i) {
    cerr << i << ": " << vector_flow[i] << endl;
  }
}

void find_all_unit_vector_flows(Graph& graph) {
  if (triples.size() == 0) {
    vector<vector<double>> unit_vectors = gen_unit_vectors();
    find_vector_relations(unit_vectors);
  }

  for (int e = 0; e < graph.number_of_edges; ++e) {
    edge_unit_vector_idx[e] = NONE;
  }

  for (int v = 0; v < graph.number_of_vertices; ++v) {
    vertex_flowed_edges[v].clear();
  }

  gen_unit_vector_flows(graph, 0);
  cerr << "number of unit vector flows: " << graph.unit_vector_flows.size() << endl;
}

} // ExpUnitVectorFlows
