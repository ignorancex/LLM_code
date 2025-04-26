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

vector<vector<double>> gen_unit_vectors1() {
  vector<vector<double>> unit_vectors;
  double phi = (1.0 + sqrt(5)) / 2;
  int w = 1;
  for (int a1 = -w; a1 <= w; ++a1) {
    for (int b1 = -w; b1 <= w; ++b1) {
      for (int a2 = a1; a2 <= w; ++a2) {
        for (int b2 = -w; b2 <= w; ++b2) {
          if (a2 == a1 && b2 < b1) {
            continue;
          }
          for (int a3 = a2; a3 <= w; ++a3) {
            for (int b3 = -w; b3 <= w; ++b3) {
              if (a3 == a2 && b3 < b2) {
                continue;
              }
              for (int a4 = a3; a4 <= w; ++a4) {
                for (int b4 = -w; b4 <= w; ++b4) {
                  if (a4 == a3 && b4 < b3) {
                    continue;
                  }

                  int s1 = a1 * a1 + b1 * b1 + a2 * a2 + b2 * b2 + a3 * a3 + b3 * b3 + a4 * a4 + b4 * b4;
                  int s2 = b1 * (2 * a1 + b1) + b2 * (2 * a2 + b2) + b3 * (2 * a3 + b3) + b4 * (2 * a4 + b4);
                  if (s1 != s2) {
                    continue;
                  }
                  int root = static_cast<int>(round(sqrt(s1)));
                  if (root == 0 || (root * root != s1)) {
                    continue;
                  }
                  double p1 = (static_cast<double>(a1) + b1 * phi) / root;
                  double p2 = (static_cast<double>(a2) + b2 * phi) / root;
                  double p3 = (static_cast<double>(a3) + b3 * phi) / root;
                  double p4 = (static_cast<double>(a4) + b4 * phi) / root;
                  vector<vector<double>> new_vectors = gen_all_plus_minus_permutations({{p1, p2, p3, p4}});
                  for (const auto& v : new_vectors) {
                    double v_norm = norm(v);
                    unit_vectors.push_back({v[0] / v_norm, v[1] / v_norm, v[2] / v_norm, v[3] / v_norm});
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return unit_vectors;
}

vector<vector<double>> gen_unit_vectors2() {
  vector<vector<double>> nonunit_vectors;
  vector<vector<double>> new_vectors;

  // new_vectors = gen_plus_minus_variations({{1.0, 1.0, 1.0, 1.0}});
  // for (const auto& v : new_vectors) {
  //   nonunit_vectors.push_back(v);
  // }

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

vector<vector<double>> gen_unit_vectors() {
  return gen_unit_vectors2();
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
        cerr << "ops: " << i << " " << j << endl;
        break;
      }
    }
    assert(ops.find(i) != ops.end());
  }

  triples.clear();
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
  cerr << "unit vectors triples count: " << triples.size() / 6 << endl;
}

//int iteration;
//int prev_max_idx;

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
    // for (const auto& idx : uniq_vectors) {
    //   cerr << idx << " ";
    // }
    // cerr << endl;
    graph.unit_vector_flows.push_back(vectors);
    return true;
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
  //++iteration;
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
  //iteration = 0;
  //prev_max_idx = 0;
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

void find_nonoriented_flow(const string& configuration_name,
    const vector<vector<double>>& or_unit_vectors,
    const map<int, int>& or_ops,
    const map<int, vector<vector<int>>>& or_triples_for_vector) {
  vector<vector<double>> nonor_unit_vectors;
  map<int, int> nonor_ops; // empty
  vector<int> vector_order; // empty
  map<int, vector<vector<int>>> nonor_triples_for_vector;
  map<int, int> reorder;
  map<int, int> original;
  for (int v = 0; v < or_unit_vectors.size(); ++v) {
    if (ops[v] < v) {
      continue;
    }
    reorder[v] = nonor_unit_vectors.size();
    original[nonor_unit_vectors.size()] = v;
    nonor_unit_vectors.push_back(or_unit_vectors[v]);
  }
  for (int v = 0; v < nonor_unit_vectors.size(); ++v) {
    int orig = original[v];
    const auto orig_it = or_triples_for_vector.find(orig);
    for (const auto& triple : orig_it->second) {
      int or1 = triple[0];
      or1 = min(or1, ops[or1]);
      assert(reorder.find(or1) != reorder.end());
      int or2 = triple[1];
      or2 = min(or2, ops[or2]);
      assert(reorder.find(or2) != reorder.end());
      nonor_triples_for_vector[v].push_back({reorder[or1], reorder[or2]});
    }
  }
  find_configuration_flow(configuration_name, nonor_unit_vectors.size(),
    nonor_ops, nonor_triples_for_vector, vector_order);
}

void find_oriented_desargues_flow() {
  vector<int> vector_order; // empty
  find_configuration_flow("oriented desargues", all_unit_vectors.size(),
    ops, triples_for_vector, vector_order);
}

void find_nonoriented_desargues_flow() {
  find_nonoriented_flow("nonoriented desargues", all_unit_vectors,
    ops, triples_for_vector);
}

int find_fast_vector_order(int vector_count,
    const map<int, vector<vector<int>>>& triples, int start_vector,
    vector<int>& vector_order) {
  vector_order.clear();
  vector_order.push_back(start_vector);
  unordered_set<int> visited_vectors;
  visited_vectors.insert(start_vector);
  vector<int> width;
  width.push_back(0);
  for (int i = 0; i < vector_count; ++i) {
    int vo = vector_order[i];
    const auto triple_it = triples.find(vo);
    for (const auto& triple : triple_it->second) {
      for (int vi = 0; vi < 2; ++vi) {
        int v2 = triple[vi];
        if (visited_vectors.find(v2) == visited_vectors.end()) {
          vector_order.push_back(v2);
          width.push_back(width[i] + 1);
          visited_vectors.insert(v2);
        }
      }
    }
  }
  return width.back();
}

vector<int> find_fast_vector_order(int vector_count,
    const map<int, vector<vector<int>>>& triples) {
  vector<int> vector_order;
  int best_vector_ans = 0;
  int best_vector_idx = 0;
  for (int vi = 0; vi < vector_count; ++vi) {
    int cur_ans = find_fast_vector_order(vector_count, triples, vi, vector_order);
    if (cur_ans > best_vector_ans) {
      best_vector_ans = cur_ans;
      best_vector_idx = vi;
    }
  }
  find_fast_vector_order(vector_count, triples, best_vector_idx, vector_order);
  return vector_order;
}

void find_oriented_cremona_richmond_flow() {
  map<int, int> cr_ops;
  map<vector<int>, int> vec_idx;
  vector<vector<int>> vecs;
  for (int i1 = 0; i1 < 6; ++i1) {
    for (int i2 = i1 + 1; i2 < 6; ++i2) {
      for (int i3 = i2 + 1; i3 < 6; ++i3) {
        for (int i4 = i3 + 1; i4 < 6; ++i4) {
          for (int p1 = 0; p1 < 4; ++p1) {
            for (int p2 = p1 + 1; p2 < 4; ++p2) {
              vector<int> d = {i1, i2, i3, i4, p1, p2};
              vec_idx[d] = vecs.size();
              vecs.push_back(d);
              vector<int> signs = {1, 1, 1, 1};
              signs[p1] = -1;
              signs[p2] = -1;
              int p1_op = -1;
              int p2_op = -1;
              for (int s_idx = 0; s_idx < signs.size(); ++s_idx) {
                if (signs[s_idx] < 0) {
                  continue;
                }
                if (p1_op == -1) {
                  p1_op = s_idx;
                } else {
                  p2_op = s_idx;
                }
              }
              assert(p1_op != -1);
              assert(p2_op != -1);
              assert(p1 + p2 + p1_op + p2_op == 6);
              vector<int> d_op = {i1, i2, i3, i4, p1_op, p2_op};
              if (vec_idx.find(d_op) != vec_idx.end()) {
                int op_idx = vec_idx[d_op];
                cr_ops[vec_idx[d]] = op_idx;
                cr_ops[op_idx] = vec_idx[d];
              }
            }
          }
        }
      }
    }
  }

  map<int, vector<vector<int>>> cr_triples_for_vector;
  int triple_count = 0;
  for (int i1 = 0; i1 < vecs.size(); ++i1) {
    for (int i2 = i1 + 1; i2 < vecs.size(); ++i2) {
      for (int i3 = i2 + 1; i3 < vecs.size(); ++i3) {
        vector<int> ors = {0, 0, 0, 0, 0, 0};
        vector<int> counts = {0, 0, 0, 0, 0, 0};
        vector<int> cands = {i1, i2, i3};
        for (int i = 0; i < 3; ++i) {
          int cand = cands[i];
          vector<int> d = vecs[cand];
          vector<int> signs = {0, 0, 0, 0, 0, 0};
          for (int j = 0; j < 4; ++j) {
            ++counts[d[j]];
            signs[d[j]] = 1;
          }
          signs[d[d[4]]] = -1;
          signs[d[d[5]]] = -1;
          for (int k = 0; k < 6; ++k) {
            ors[k] += signs[k];
          }
        }
        bool is_triple = true;
        for (int k = 0; k < 6; ++k) {
          if (counts[k] != 2 || ors[k] != 0) {
            is_triple = false;
            break;
          }
        }
        if (is_triple) {
          cr_triples_for_vector[i1].push_back({i2, i3});
          cr_triples_for_vector[i2].push_back({i1, i3});
          cr_triples_for_vector[i3].push_back({i1, i2});
          ++triple_count;
        }
      }
    }
  }
  vector<int> vector_order = find_fast_vector_order(vecs.size(), cr_triples_for_vector);
  find_configuration_flow("oriented cremona-richmond", vecs.size(),
    cr_ops, cr_triples_for_vector, vector_order);
}

void find_nonoriented_cremona_richmond_flow() {
  int cr_unit_vectors_count = 15;
  map<int, int> cr_ops; // empty
  vector<int> vector_order; // empty
  map<int, vector<vector<int>>> cr_triples_for_vector;
  map<string, int> p;
  p["12"] =  0;
  p["13"] =  1;
  p["14"] =  2;
  p["15"] =  3;
  p["16"] =  4;
  p["23"] =  5;
  p["24"] =  6;
  p["25"] =  7;
  p["26"] =  8;
  p["34"] =  9;
  p["35"] = 10;
  p["36"] = 11;
  p["45"] = 12;
  p["46"] = 13;
  p["56"] = 14;

  cr_triples_for_vector[p["12"]].push_back({p["34"], p["56"]});
  cr_triples_for_vector[p["12"]].push_back({p["35"], p["46"]});
  cr_triples_for_vector[p["12"]].push_back({p["36"], p["45"]});

  cr_triples_for_vector[p["13"]].push_back({p["24"], p["56"]});
  cr_triples_for_vector[p["13"]].push_back({p["25"], p["46"]});
  cr_triples_for_vector[p["13"]].push_back({p["26"], p["45"]});

  cr_triples_for_vector[p["14"]].push_back({p["23"], p["56"]});
  cr_triples_for_vector[p["14"]].push_back({p["25"], p["36"]});
  cr_triples_for_vector[p["14"]].push_back({p["26"], p["35"]});

  cr_triples_for_vector[p["15"]].push_back({p["23"], p["46"]});
  cr_triples_for_vector[p["15"]].push_back({p["24"], p["36"]});
  cr_triples_for_vector[p["15"]].push_back({p["26"], p["34"]});

  cr_triples_for_vector[p["16"]].push_back({p["23"], p["45"]});
  cr_triples_for_vector[p["16"]].push_back({p["24"], p["35"]});
  cr_triples_for_vector[p["16"]].push_back({p["25"], p["34"]});

  cr_triples_for_vector[p["23"]].push_back({p["14"], p["56"]});
  cr_triples_for_vector[p["23"]].push_back({p["15"], p["46"]});
  cr_triples_for_vector[p["23"]].push_back({p["16"], p["45"]});

  cr_triples_for_vector[p["24"]].push_back({p["13"], p["56"]});
  cr_triples_for_vector[p["24"]].push_back({p["15"], p["36"]});
  cr_triples_for_vector[p["24"]].push_back({p["16"], p["35"]});

  cr_triples_for_vector[p["25"]].push_back({p["13"], p["46"]});
  cr_triples_for_vector[p["25"]].push_back({p["14"], p["36"]});
  cr_triples_for_vector[p["25"]].push_back({p["16"], p["34"]});

  cr_triples_for_vector[p["26"]].push_back({p["13"], p["45"]});
  cr_triples_for_vector[p["26"]].push_back({p["14"], p["35"]});
  cr_triples_for_vector[p["26"]].push_back({p["15"], p["34"]});

  cr_triples_for_vector[p["34"]].push_back({p["12"], p["56"]});
  cr_triples_for_vector[p["34"]].push_back({p["15"], p["26"]});
  cr_triples_for_vector[p["34"]].push_back({p["16"], p["25"]});

  cr_triples_for_vector[p["35"]].push_back({p["12"], p["46"]});
  cr_triples_for_vector[p["35"]].push_back({p["14"], p["26"]});
  cr_triples_for_vector[p["35"]].push_back({p["16"], p["24"]});

  cr_triples_for_vector[p["36"]].push_back({p["12"], p["45"]});
  cr_triples_for_vector[p["36"]].push_back({p["14"], p["25"]});
  cr_triples_for_vector[p["36"]].push_back({p["15"], p["24"]});

  cr_triples_for_vector[p["45"]].push_back({p["12"], p["36"]});
  cr_triples_for_vector[p["45"]].push_back({p["13"], p["26"]});
  cr_triples_for_vector[p["45"]].push_back({p["16"], p["23"]});

  cr_triples_for_vector[p["46"]].push_back({p["12"], p["35"]});
  cr_triples_for_vector[p["46"]].push_back({p["13"], p["25"]});
  cr_triples_for_vector[p["46"]].push_back({p["15"], p["23"]});

  cr_triples_for_vector[p["56"]].push_back({p["12"], p["34"]});
  cr_triples_for_vector[p["56"]].push_back({p["13"], p["24"]});
  cr_triples_for_vector[p["56"]].push_back({p["14"], p["23"]});

  find_configuration_flow("nonoriented cremona-richmond", cr_unit_vectors_count,
    cr_ops, cr_triples_for_vector, vector_order);
}

void find_configuration_flows() {
  find_oriented_desargues_flow(); // nz5
  // find_nonoriented_desargues_flow(); // no flow
  // find_oriented_cremona_richmond_flow(); // nz7
  // find_nonoriented_cremona_richmond_flow(); // nz2
}

void find_all_unit_vector_flows(Graph& graph) {
  if (triples.size() == 0) {
    vector<vector<double>> unit_vectors = gen_unit_vectors();
    find_vector_relations(unit_vectors);
    // find_configuration_flows();
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
