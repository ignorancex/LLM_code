/*
 * File:   graph.cpp
 * Author: Nikolay Ulyanov (ulyanick@gmail.com)
 *
 */

#include "graph.h"
#include <iostream>
#include <cassert>

using namespace std;


int edge_count_from_vertex(int vertex_count) {
  return vertex_count * MAX_DEG / 2;
}

Graph::Graph(unsigned int n, bool is_definitely_snark) {
  number_of_vertices = n;
  number_of_edges = 0;
  is_definitely_snark = is_definitely_snark;
  for (int v = 0; v < number_of_vertices; ++v) {
    deg[v] = 0;
    lexi_deg[v] = 0;
    for (int u = 0; u < number_of_vertices; ++u) {
      vv2e[v][u] = NONE;
    }
  }

  int max_number_of_edges = edge_count_from_vertex(number_of_vertices);
  for (int e1 = 0; e1 < max_number_of_edges; ++e1) {
    for (int e2 = 0; e2 < max_number_of_edges; ++e2) {
      edges_are_neibs[e1][e2] = false;
    }
  }
}

bool Graph::has_edge(unsigned int v, unsigned int w) const {
  return (vv2e[v][w] != NONE);
}

void Graph::add_edge(unsigned int v, unsigned int w) {
  assert(!has_edge(v, w));
  assert(v < number_of_vertices);
  assert(w < number_of_vertices);
  if (v > w) {
    swap(v, w);
  }
  int e = number_of_edges;
  v2v[v][deg[v]] = w;
  v2v[w][deg[w]] = v;
  v2e[v][deg[v]] = e;
  v2e[w][deg[w]] = e;
  ++deg[v];
  ++deg[w];

  lexi_v2v[v][lexi_deg[v]] = w;
  ++lexi_deg[v];

  e2v[e][0] = v;
  e2v[e][1] = w;
  vv2e[v][w] = e;
  vv2e[w][v] = e;
  ++number_of_edges;
  if (number_of_edges == edge_count_from_vertex(number_of_vertices)) {
    finish_init();
  }
}

void Graph::finish_init() {
  for (int v = 0; v < number_of_vertices; ++v) {
    if (deg[v] != MAX_DEG) {
      if (MAX_DEG == 3) {
        cerr << "Error: graph is not cubic" << endl;
      } else {
        cerr << "Error: graph is not regular (where vertex degree = " << MAX_DEG << ")" << endl;
      }
      exit(1);
    }
  }

  for (int e = 0; e < number_of_edges; ++e) {
    int neib_edge_count = 0;
    for (int vi = 0; vi < 2; ++vi) {
      int v = e2v[e][vi];
      for (int e2i = 0; e2i < MAX_DEG; ++e2i) {
        int e2 = v2e[v][e2i];
        if (e2 != e) {
          e2e[e][neib_edge_count] = e2;
          edges_are_neibs[e][e2] = true;
          ++neib_edge_count;
        }
      }
    }
  }

  if (number_of_edges != edge_count_from_vertex(number_of_vertices)) {
    cerr << "Error: invalid number of edges" << endl;
    exit(1);
  }

  find_faster_edge_order();
}

bool read_graph(const string& filetype, Graph& graph) {
  if (filetype == "mc") {
    return decode_multicode(stdin, graph);
  } else if (filetype == "adj") {
    return decode_adjacency(cin, graph);
  } else if (filetype == "bghm") {
    return decode_bghm(cin, graph);
  }
  assert(filetype == "g6");
  return decode_graph6(cin, graph);
}

bool decode_multicode(FILE* input, Graph& graph) {
  unsigned char code_for_number_of_vertices[1];
  if (!fread(code_for_number_of_vertices, sizeof(unsigned char), 1, input)) {
    return false;
  }
  assert(code_for_number_of_vertices[0] % 2 == 0);
  graph = Graph(code_for_number_of_vertices[0]);
  const int edge_count = edge_count_from_vertex(graph.number_of_vertices);
  const int code_length = edge_count + graph.number_of_vertices - 1;
  unsigned char code[code_length];
  if (!fread(code, sizeof(unsigned char), code_length, input)) {
    return false;
  }

  int cur_v = 1;
  // parsing code_length
  for (int i = 0; i < code_length; ++i) {
    if (code[i] == 0) {
      ++cur_v;
    } else {
      graph.add_edge(cur_v - 1, code[i] - 1);
    }
  }

  graph.finish_init();
  return true;
}

bool decode_adjacency(istream& input, Graph& graph) {
  int number_of_vertices = 0;
  input >> number_of_vertices;
  if (number_of_vertices == 0) {
    return false;
  }
  assert(number_of_vertices % 2 == 0);

  graph = Graph(number_of_vertices);
  const int edge_count = edge_count_from_vertex(graph.number_of_vertices);
  for (int i = 0; i < edge_count; ++i) {
    int u, v;
    input >> u >> v;
    graph.add_edge(u, v);
  }

  graph.finish_init();
  return true;
}

bool decode_bghm(istream& input, Graph& graph) {
  // FIXME
  return false;
}

// Convert graph6 character sequence to 6-bit integers
vector<int> string_to_graph6(const string& s) {
  vector<int> data;
  for (const auto& c : s) {
    data.push_back(static_cast<int>(c) - 63);
  }
  int min_v = 64;
  int max_v = 0;
  for (const auto& v : data) {
    min_v = min(min_v, v);
    max_v = max(max_v, v);
  }
  if (data.size() > 0 && (min_v < 0 or max_v > 63)) {
    data.clear();
    return data;
  }
  return data;
}

// read initial one-, four- or eight-unit value from graph6 integer sequence
// return value and rest of sequence
int graph6_to_vertex_count(const vector<int>& data) {
  if (data[0] <= 62) {
    return data[0];
  } else {
    throw std::invalid_argument("expected small vertex count");
  }
  return 0;
}

vector<bool> graph6_to_edge_pairs(const vector<int>& data) {
  vector<bool> edge_pairs;
  for (int i = 1; i < data.size(); ++i) {
    for (int bit_index = 5; bit_index >= 0; --bit_index) {
      int bit = (data[i] >> bit_index);
      if ((bit & 1) == 1) {
        edge_pairs.push_back(true);
      } else {
        edge_pairs.push_back(false);
      }
    }
  }
  return edge_pairs;
}

// graph6 specification:
// https://cs.anu.edu.au/~bdm/data/formats.txt
bool decode_graph6(istream& input, Graph& graph) {
  string s;
  getline(input, s);
  if (s.empty()) {
    return false;
  }
  vector<int> data = string_to_graph6(s);
  int vertex_count = graph6_to_vertex_count(data);
  int edge_pair_count = (vertex_count * (vertex_count - 1) / 2 + 5) / 6;
  if (data.size() - 1 != edge_pair_count) {
    throw std::invalid_argument("wrong edge pair count");
  }
  graph = Graph(vertex_count);
  vector<bool> edge_pairs = graph6_to_edge_pairs(data);
  int v1 = 0;
  int v2 = 1;
  for (const auto& ep : edge_pairs) {
    if (ep) {
      graph.add_edge(v1, v2);
    }
    if (v2 - v1 > 1) {
      v1 += 1;
    } else {
      v1 = 0;
      v2 += 1;
    }
  }
  graph.finish_init();
  return true;
}

void Graph::print() const {
  cerr << "Printing graph:" << endl;
  for (int i = 0; i < number_of_vertices; ++i) {
    cerr << i << ":\t";
    for (int j = 0; j < MAX_DEG; ++j) {
      cerr << v2v[i][j] << "(e" << v2e[i][j] << ")\t";
    }
    cerr << endl;
  }

  if (!all_dominating_circuits.empty()) {
    cerr << "dominating circuit: ";
    map<int, set<int>> neibs;
    int first_v = MAX_VERTEX;
    for (const auto& c : all_dominating_circuits) {
      for (int e = 0; e < number_of_edges; ++e) {
        if (BIT(e) & c) {
          int v1 = e2v[e][0];
          int v2 = e2v[e][1];
          neibs[v1].insert(v2);
          neibs[v2].insert(v1);
          first_v = min(first_v, min(v1, v2));
        }
      }
      break;
    }
    set<int> visited;
    cerr << first_v << " ";
    visited.insert(first_v);
    int cur_v = first_v;
    do {
      bool found = false;
      for (const auto& v : neibs[cur_v]) {
        if (visited.find(v) == visited.end()) {
          cerr << v << " ";
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
    cerr << first_v << endl;
  }
}

void Graph::print(const Mask& dominating_circuit) const {
  cerr << "Printing graph:" << endl;
  for (int i = 0; i < number_of_vertices; ++i) {
    cerr << i << ":\t";
    for (int j = 0; j < MAX_DEG; ++j) {
      cerr << v2v[i][j] << "(e" << v2e[i][j] << ")\t";
    }
    cerr << endl;
  }

  cerr << "dominating circuit: ";
  map<int, set<int>> neibs;
  int first_v = MAX_VERTEX;
  for (int e = 0; e < number_of_edges; ++e) {
    if (BIT(e) & dominating_circuit) {
      int v1 = e2v[e][0];
      int v2 = e2v[e][1];
      neibs[v1].insert(v2);
      neibs[v2].insert(v1);
      first_v = min(first_v, min(v1, v2));
    }
  }
  set<int> visited;
  cerr << first_v << " ";
  visited.insert(first_v);
  int cur_v = first_v;
  do {
    bool found = false;
    for (const auto& v : neibs[cur_v]) {
      if (visited.find(v) == visited.end()) {
        cerr << v << " ";
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
  cerr << first_v << endl;
}

// TODO:
// void Graph::print(const Mask& dominating_circuit, const set<Mask>& u5cdc) const {
// }

void Graph::print(set<int>& oriented_vertices, set<int>& poor_edges) const {
  cerr << "Printing graph:" << endl;
  for (int i = 0; i < number_of_vertices; ++i) {
    cerr << i << ":\t";
    for (int j = 0; j < MAX_DEG; ++j) {
      cerr << v2v[i][j] << "(e" << v2e[i][j] << ")\t";
    }
    cerr << endl;
  }

  if (!all_dominating_circuits.empty()) {
    cerr << "dominating circuit: ";

    map<int, set<int>> neibs;
    int first_v = MAX_VERTEX;
    bool found_circuit = false;
    for (const auto& c : all_dominating_circuits) {
      neibs.clear();
      first_v = MAX_VERTEX;
      int poor_edge_count = 0;
      set<int> dominating_vertices;
      for (int e = 0; e < number_of_edges; ++e) {
        if (BIT(e) & c) {
          if (poor_edges.find(e) != poor_edges.end()) {
            ++poor_edge_count;
          }
          int v1 = e2v[e][0];
          int v2 = e2v[e][1];
          dominating_vertices.insert(v1);
          dominating_vertices.insert(v2);
          neibs[v1].insert(v2);
          neibs[v2].insert(v1);
          first_v = min(first_v, min(v1, v2));
        }
      }
      if ((number_of_edges - poor_edge_count) % 2 != 0) {
        // TODO: why do i check this?
        continue;
      }
      bool has_all_oriented_vertices = true;
      for (const auto& v : oriented_vertices) {
        if (dominating_vertices.find(v) == dominating_vertices.end()) {
          has_all_oriented_vertices = false;
          break;
        }
      }
      if (!has_all_oriented_vertices) {
        continue;
      }
      found_circuit = true;
      break;
    }
    if (!found_circuit) {
      cerr << "FAIL" << endl;
      return;
    }

    set<int> visited;
    cerr << first_v << " ";
    visited.insert(first_v);
    int cur_v = first_v;
    do {
      bool found = false;
      for (const auto& v : neibs[cur_v]) {
        if (visited.find(v) == visited.end()) {
          cerr << v << " ";
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
    cerr << first_v << endl;
  }
}

int find_faster_edge_order(int start_edge, Graph& graph) {
  // TODO: combine visited_edges, faster_edge_order and width into some combined structure
  // width works as a heuristic
  graph.faster_edge_order.clear();
  graph.faster_edge_order.push_back(start_edge);
  unordered_set<int> visited_edges;
  visited_edges.insert(start_edge);
  int width[MAX_EDGE];
  width[0] = 0;
  for (int i = 0; i < graph.number_of_edges; ++i) {
    int eo = graph.faster_edge_order[i];
    for (int vi = 0; vi < 2; ++vi) {
      int v = graph.e2v[eo][vi];
      for (int j = 0; j < MAX_DEG; ++j) {
        int e = graph.v2e[v][j];
        if (visited_edges.find(e) == visited_edges.end()) {
          graph.faster_edge_order.push_back(e);
          width[visited_edges.size()] = width[i] + 1;
          visited_edges.insert(e);
        }
      }
    }
  }
  return width[graph.number_of_edges - 1];
}

void Graph::find_faster_edge_order() {
  int best_edge_ans = 0;
  int best_edge_idx = 0;
  for (int ei = 0; ei < number_of_edges; ++ei) {
    int cur_ans = ::find_faster_edge_order(ei, *this);
    if (cur_ans > best_edge_ans) {
      best_edge_ans = cur_ans;
      best_edge_idx = ei;
    }
  }
  ::find_faster_edge_order(best_edge_idx, *this);
}

void build_petersen_graph(Graph& graph) {
  assert(graph.number_of_vertices == 10);
  graph.add_edge(0, 6);
  graph.add_edge(0, 4);
  graph.add_edge(0, 8);
  graph.add_edge(1, 9);
  graph.add_edge(1, 5);
  graph.add_edge(1, 6);
  graph.add_edge(2, 7);
  graph.add_edge(2, 4);
  graph.add_edge(2, 9);
  graph.add_edge(3, 8);
  graph.add_edge(3, 5);
  graph.add_edge(3, 7);
  graph.add_edge(4, 5);
  graph.add_edge(6, 7);
  graph.add_edge(8, 9);
}
