/*
 * File:   unit_vector_flows.cpp
 * Author: Nikolay Ulyanov (ulyanick@gmail.com)
 *
 * TODO:
 *
 * Created on August 17, 2019
 *
 */

#include "unit_vector_flows.h"

#include <cassert>
#include <math.h>
#include <map>


namespace ExpUnitVectorFlows {

const double EPS = 1e-8;
const vector<double> zero_vec{0.0, 0.0, 0.0};

int edge_unit_vector_idx[MAX_EDGE];
vector<int> vertex_flowed_edges[MAX_EDGE];

vector<vector<double>> all_unit_vectors;
vector<bool> is_petersen_vector;
map<int, int> ops;
map<vector<int>, int> triples;

/*********************************Methods*********************************/

bool same(double f1, double f2) {
  return abs(f1 - f2) < EPS;
}

double dist(const vector<double>& v1, const vector<double>& v2) {
  return sqrt(
      (v1[0] - v2[0]) * (v1[0] - v2[0]) +
      (v1[1] - v2[1]) * (v1[1] - v2[1]) +
      (v1[2] - v2[2]) * (v1[2] - v2[2]));
}

double norm(const vector<double>& v) {
  return dist(v, zero_vec);
}

vector<vector<double>> gen_even_permutations(const vector<vector<double>>& vectors) {
  vector<vector<double>> permutations;
  for (const auto& v : vectors) {
    permutations.push_back(v);
    permutations.push_back({v[1], v[2], v[0]});
    permutations.push_back({v[2], v[0], v[1]});
  }
  return permutations;
}

vector<vector<double>> gen_all_permutations(const vector<vector<double>>& vectors) {
  vector<vector<double>> permutations;
  for (const auto& v : vectors) {
    vector<vector<double>> even = gen_even_permutations({v});
    vector<vector<double>> odd = gen_even_permutations({{v[1], v[0], v[2]}});
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
          permutations.push_back({v[0] * m0, v[1] * m1, v[2] * m2});
        }
      }
    }
  }
  return permutations;
}

vector<vector<double>> gen_all_plus_minus_permutations(const vector<vector<double>>& vectors) {
  return gen_plus_minus_variations(gen_all_permutations(vectors));
}

vector<vector<double>> gen_even_plus_minus_permutations(const vector<vector<double>>& vectors) {
  return gen_plus_minus_variations(gen_even_permutations(vectors));
}

vector<vector<double>> gen_unit_vectors() {
  // double phi = (1.0 + sqrt(5)) / 2;
  // double x = 2.0 / sqrt(sqrt(5));
  // double y = x * phi;

  vector<vector<double>> nonunit_vectors;
  // vector<vector<double>> new_vectors;

  // new_vectors = gen_even_plus_minus_permutations({{0.0, 0.0, 1.0}});
  // for (const auto& v : new_vectors) {
  //   nonunit_vectors.push_back(v);
  // }
  // new_vectors = gen_even_plus_minus_permutations({{1.0, phi, phi + 1.0}});
  // for (const auto& v : new_vectors) {
  //   nonunit_vectors.push_back(v);
  // }

  // works for Tietze's snark, and more! e. g. 20.05: g1, g4, g5
  nonunit_vectors = {
    {-1.0, -9.030971600656904e-18, 9.549708297316796e-17},
    {-0.9687598666735441, 1.8036648461847414e-17, -0.24800064661741741},
    // {-0.9077435189160142, -0.09872652454106681, 0.4077435189160141},
    // {-0.9077435189160142, -0.09872652454106676, -0.407743518916014},
    // {-0.9077435189160142, 0.09872652454106672, -0.4077435189160141},
    // {-0.9077435189160142, 0.09872652454106676, 0.407743518916014},
    // {-0.865754201881895, -0.5, 0.02167168484089834},
    // {-0.865754201881895, 0.4999999999999999, 0.02167168484089809},
    {-0.8440825170409966, -0.5350655226660493, -0.03506552266604854},
    {-0.8440825170409966, -0.5350655226660493, 0.03506552266604854},
    {-0.8440825170409966, 0.5350655226660493, -0.03506552266604854},
    {-0.8440825170409966, 0.5350655226660493, 0.03506552266604854},
    {-0.8306886792158457, 0.5567372075069474, -6.223285321735374e-19},
    {-0.8090169943749471, 0.3090169943749477, -0.5000000000000006},
    {-0.8090169943749471, 0.3090169943749477, 0.5000000000000006},
    {-0.8090169943749468, -0.3090169943749473, -0.5000000000000011},
    {-0.8090169943749468, -0.3090169943749473, 0.5000000000000011},
    {-0.6597428722985966, 0.5, 0.5610163477575298},
    {-0.6597428722985965, -0.49999999999999983, 0.5610163477575301},
    {-0.56101634775753, 0.6597428722985966, -0.5},
    {-0.56101634775753, 0.6597428722985966, 0.49999999999999994},
    {-0.5567372075069474, 6.223285321735375e-19, 0.8306886792158457},
    {-0.5350655226660491, -0.035065522666048496, -0.8440825170409967},
    {-0.5350655226660491, -0.035065522666048496, 0.8440825170409967},
    {-0.5350655226660491, 0.035065522666048496, -0.8440825170409967},
    {-0.5350655226660491, 0.035065522666048496, 0.8440825170409967},
    // {-0.5000000000000001, -0.021671684840898278, 0.8657542018818948},
    // {-0.5000000000000001, 0.021671684840898264, -0.8657542018818948},
    {-0.5, -0.56101634775753, 0.6597428722985965},
    {-0.5, 0.56101634775753, -0.6597428722985966},
    {-0.4999999999999999, -0.8090169943749478, 0.30901699437494734},
    {-0.4999999999999999, -0.56101634775753, -0.6597428722985966},
    {-0.49999999999999983, 0.56101634775753, 0.6597428722985966},
    {-0.4999999999999998, -0.8090169943749476, -0.3090169943749473},
    {-0.49999999999999933, 0.8090169943749478, 0.30901699437494756},
    {-0.499999999999999, 0.8090169943749479, -0.3090169943749478},
    // {-0.4077435189160141, -0.9077435189160142, -0.09872652454106666},
    // {-0.4077435189160141, -0.9077435189160142, 0.09872652454106666},
    // {-0.4077435189160141, 0.9077435189160142, -0.09872652454106666},
    // {-0.4077435189160141, 0.9077435189160142, 0.09872652454106666},
    {-0.309016994374947, -0.4999999999999998, 0.8090169943749479},
    {-0.30901699437494695, -0.49999999999999967, -0.8090169943749477},
    {-0.30901699437494695, 0.49999999999999967, -0.8090169943749477},
    {-0.30901699437494695, 0.49999999999999967, 0.8090169943749477},
    {-0.24800064661741753, -0.9687598666735441, 7.912695146636231e-17},
    // {-0.0987265245410667, -0.4077435189160143, -0.9077435189160141},
    // {-0.0987265245410667, -0.4077435189160143, 0.9077435189160141},
    // {-0.0987265245410667, 0.4077435189160143, -0.9077435189160141},
    // {-0.0987265245410667, 0.4077435189160143, 0.9077435189160141},
    {-0.03506552266604833, -0.844082517040997, -0.5350655226660488},
    {-0.03506552266604833, -0.844082517040997, 0.5350655226660488},
    {-0.03506552266604833, 0.844082517040997, -0.5350655226660488},
    {-0.03506552266604833, 0.844082517040997, 0.5350655226660488},
    // {-0.02167168484089829, 0.8657542018818949, 0.5},
    // {-0.02167168484089828, 0.8657542018818949, -0.5},
    {-2.444157480957888e-16, 0.2480006466174177, -0.9687598666735441},
    {-2.361580023510447e-17, -0.8306886792158458, 0.5567372075069474},
    {-2.2259851428115202e-17, 1.0, -9.030971600656901e-18},
    {-2.1836291966026995e-18, 1.7344678888130534e-17, 1.0},
    {-6.223285321735376e-19, -0.24800064661741755, -0.9687598666735441},
    {6.223285321735376e-19, 0.24800064661741755, 0.9687598666735441},
    {2.1836291966026995e-18, -1.7344678888130534e-17, -1.0},
    {2.2259851428115202e-17, -1.0, 9.030971600656901e-18},
    {2.361580023510447e-17, 0.8306886792158458, -0.5567372075069474},
    {2.444157480957888e-16, -0.2480006466174177, 0.9687598666735441},
    // {0.02167168484089828, -0.8657542018818949, 0.5},
    // {0.02167168484089829, -0.8657542018818949, -0.5},
    {0.03506552266604833, -0.844082517040997, -0.5350655226660488},
    {0.03506552266604833, -0.844082517040997, 0.5350655226660488},
    {0.03506552266604833, 0.844082517040997, -0.5350655226660488},
    {0.03506552266604833, 0.844082517040997, 0.5350655226660488},
    // {0.0987265245410667, -0.4077435189160143, -0.9077435189160141},
    // {0.0987265245410667, -0.4077435189160143, 0.9077435189160141},
    // {0.0987265245410667, 0.4077435189160143, -0.9077435189160141},
    // {0.0987265245410667, 0.4077435189160143, 0.9077435189160141},
    {0.24800064661741753, 0.9687598666735441, -7.912695146636231e-17},
    {0.30901699437494695, -0.49999999999999967, -0.8090169943749477},
    {0.30901699437494695, -0.49999999999999967, 0.8090169943749477},
    {0.30901699437494695, 0.49999999999999967, 0.8090169943749477},
    {0.309016994374947, 0.4999999999999998, -0.8090169943749479},
    // {0.4077435189160141, -0.9077435189160142, -0.09872652454106666},
    // {0.4077435189160141, -0.9077435189160142, 0.09872652454106666},
    // {0.4077435189160141, 0.9077435189160142, -0.09872652454106666},
    // {0.4077435189160141, 0.9077435189160142, 0.09872652454106666},
    {0.499999999999999, -0.8090169943749479, 0.3090169943749478},
    {0.49999999999999933, -0.8090169943749478, -0.30901699437494756},
    {0.4999999999999998, 0.8090169943749476, 0.3090169943749473},
    {0.49999999999999983, -0.56101634775753, -0.6597428722985966},
    {0.4999999999999999, 0.56101634775753, 0.6597428722985966},
    {0.4999999999999999, 0.8090169943749478, -0.30901699437494734},
    {0.5, -0.56101634775753, 0.6597428722985966},
    {0.5, 0.56101634775753, -0.6597428722985965},
    // {0.5000000000000001, -0.021671684840898264, 0.8657542018818948},
    // {0.5000000000000001, 0.021671684840898278, -0.8657542018818948},
    {0.5350655226660491, -0.035065522666048496, -0.8440825170409967},
    {0.5350655226660491, -0.035065522666048496, 0.8440825170409967},
    {0.5350655226660491, 0.035065522666048496, -0.8440825170409967},
    {0.5350655226660491, 0.035065522666048496, 0.8440825170409967},
    {0.5567372075069474, -6.223285321735375e-19, -0.8306886792158457},
    {0.56101634775753, -0.6597428722985966, -0.49999999999999994},
    {0.56101634775753, -0.6597428722985966, 0.5},
    {0.6597428722985965, 0.49999999999999983, -0.5610163477575301},
    {0.6597428722985966, -0.5, -0.5610163477575298},
    {0.8090169943749468, 0.3090169943749473, -0.5000000000000011},
    {0.8090169943749468, 0.3090169943749473, 0.5000000000000011},
    {0.8090169943749471, -0.3090169943749477, -0.5000000000000006},
    {0.8090169943749471, -0.3090169943749477, 0.5000000000000006},
    {0.8306886792158457, -0.5567372075069474, 6.223285321735374e-19},
    {0.8440825170409966, -0.5350655226660493, -0.03506552266604854},
    {0.8440825170409966, -0.5350655226660493, 0.03506552266604854},
    {0.8440825170409966, 0.5350655226660493, -0.03506552266604854},
    {0.8440825170409966, 0.5350655226660493, 0.03506552266604854},
    // {0.865754201881895, -0.4999999999999999, -0.02167168484089809},
    // {0.865754201881895, 0.5, -0.02167168484089834},
    // {0.9077435189160142, -0.09872652454106676, -0.407743518916014},
    // {0.9077435189160142, -0.09872652454106672, 0.4077435189160141},
    // {0.9077435189160142, 0.09872652454106676, 0.407743518916014},
    // {0.9077435189160142, 0.09872652454106681, -0.4077435189160141},
    {0.9687598666735441, -1.8036648461847414e-17, 0.24800064661741741},
    {1.0, 9.030971600656904e-18, -9.549708297316796e-17}
  };

  // nonunit_vectors = {
  // };


  // didn't help
  // although could help with girth 4
  // new_vectors = gen_even_plus_minus_permutations({{y, x, 2.0}});
  // for (const auto& v : new_vectors) {
  //   nonunit_vectors.push_back(v);
  // }
  // new_vectors = gen_even_plus_minus_permutations({{y-1, x+phi, phi-1}});
  // for (const auto& v : new_vectors) {
  //   nonunit_vectors.push_back(v);
  // }
  // new_vectors = gen_even_plus_minus_permutations({{y+1,     x-phi, phi-1}});
  // for (const auto& v : new_vectors) {
  //   nonunit_vectors.push_back(v);
  // }
  // new_vectors = gen_even_plus_minus_permutations({{y+1/phi, x-1,   phi}});
  // for (const auto& v : new_vectors) {
  //   nonunit_vectors.push_back(v);
  // }
  // new_vectors = gen_even_plus_minus_permutations({{y-1/phi, x+1,   phi}});
  // for (const auto& v : new_vectors) {
  //   nonunit_vectors.push_back(v);
  // }

  vector<vector<double>> unit_vectors;
  for (const auto& v : nonunit_vectors) {
    double v_norm = norm(v);
    unit_vectors.push_back({v[0] / v_norm, v[1] / v_norm, v[2] / v_norm});
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
            same(v0[2] + v1[2] + v2[2], 0)) {
          has_triple[i] = true;
          has_triple[j] = true;
          has_triple[k] = true;
        }
      }
    }
  }
  all_unit_vectors.clear();
  for (size_t i = 0; i < tmp_unit_vectors.size(); ++i) {
    if (has_triple[i]) {
      all_unit_vectors.push_back(tmp_unit_vectors[i]);
    }
  }
  is_petersen_vector.clear();
  double phi = (1.0 + sqrt(5)) / 2;
  vector<double> petersen_coords = {0, 0.5, 1, phi/2, (phi-1)/2};
  for (size_t i = 0; i < all_unit_vectors.size(); ++i) {
    bool coords_are_all_petersen = true;
    for (const auto& x : all_unit_vectors[i]) {
      bool is_petersen_coord = false;
      for (const auto& candidate : petersen_coords) {
        if (same(abs(x), candidate)) {
          is_petersen_coord = true;
          break;
        }
      }
      if (!is_petersen_coord) {
        coords_are_all_petersen = false;
        break;
      }
    }
    is_petersen_vector.push_back(coords_are_all_petersen);
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
      if (same(v0[0] + v1[0], 0) && same(v0[1] + v1[1], 0) && same(v0[2] + v1[2], 0)) {
        ops[i] = j;
        ops[j] = i;
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
            same(v0[2] + v1[2] + v2[2], 0)) {
          triples_count += 1;
          triples[{i, j}] = k;
          triples[{j, i}] = k;
          triples[{i, k}] = j;
          triples[{k, i}] = j;
          triples[{j, k}] = i;
          triples[{k, j}] = i;
        }
      }
    }
  }
  cerr << "unit vectors triples count: " << triples_count << endl;
}

bool gen_unit_vector_flows(Graph& graph, int cur_edge_idx) {
  if (cur_edge_idx == graph.number_of_edges) {
    graph.print();

    vector<int> vectors;
    set<int> uniq_vectors;
    for (int e = 0; e < graph.number_of_edges; ++e) {
      vectors.push_back(edge_unit_vector_idx[e]);
      uniq_vectors.insert(edge_unit_vector_idx[e]);
      uniq_vectors.insert(ops[edge_unit_vector_idx[e]]);
      cerr << "e: " << e << "; " << edge_unit_vector_idx[e] << endl;
    }
    cerr << "compatible Petersen profiles: ";
    for (const auto& profile : graph.profiles) {
      bool is_compatible = true;
      for (int e = 0; e < graph.number_of_edges; ++e) {
        if (!is_petersen_vector[edge_unit_vector_idx[e]] && profile.first[e] == 'r') {
          is_compatible = false;
          break;
        }
      }
      if (is_compatible) {
        cerr << profile.first << " ";
      }
    }
    cerr << ";" << endl;

    cerr << "compatible o6c4c profiles: ";
    for (const auto& profile : graph.o6c4c_2xcdcs_profiles) {
      bool is_compatible = true;
      for (int e = 0; e < graph.number_of_edges; ++e) {
        if (!is_petersen_vector[edge_unit_vector_idx[e]] && profile[e] == 'r') {
          is_compatible = false;
          break;
        }
      }
      if (is_compatible) {
        cerr << profile << " ";
      }
    }
    cerr << ";" << endl;

    cerr << "compatible both profiles: ";
    for (const auto& profile : graph.o6c4c_2xcdcs_profiles) {
      if (graph.profiles.find(profile) == graph.profiles.end()) {
        continue;
      }
      bool is_compatible = true;
      for (int e = 0; e < graph.number_of_edges; ++e) {
        if (!is_petersen_vector[edge_unit_vector_idx[e]] && profile[e] == 'r') {
          is_compatible = false;
          break;
        }
      }
      if (is_compatible) {
        cerr << profile << " ";
      }
    }
    cerr << ";" << endl;

    graph.unit_vector_flows.push_back(vectors);
    cerr << "vector count: " << uniq_vectors.size() << endl;

    set<vector<int>> graph_triples;
    for (int v = 0; v < graph.number_of_vertices; ++v) {
      cerr << "v: " << v << "; ";
      for (int j = 0; j < MAX_DEG; ++j) {
        cerr << graph.v2v[v][j] << " ";
      }
      cerr << "; triple: ";
      vector<int> v_triple;
      for (int j = 0; j < MAX_DEG; ++j) {
        int e_vector = edge_unit_vector_idx[graph.v2e[v][j]];
        cerr << e_vector << " ";
        // e_vector = min(e_vector, ops[e_vector]);
        v_triple.push_back(e_vector);
      }
      cerr << endl;
      sort(v_triple.begin(), v_triple.end());
      graph_triples.insert(v_triple);
    }
    cerr << "graph used triple count: " << graph_triples.size() << endl;

    for (const auto& v : uniq_vectors) {
      cerr << "vector v: " << v << "; ";
      for (const auto& x : all_unit_vectors[v]) {
        cerr << x << " ";
      }
      cerr << endl;
    }
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
