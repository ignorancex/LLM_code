/*
 * File:   33pp_from_nz5.cpp
 * Author: Nikolay Ulyanov
 *
 * Created on 17 December 2016
 * First version finished on 10 January 2017
 */

#include "util.h"

#include <unordered_set>
#include <unordered_map>
#include <set>
#include <climits>
#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <string>

using namespace std;

/*************************Defines and global variables*************************/

bool PRINT_SOLUTIONS = true;

unsigned int edge_index[MAXN][MAXN];
unsigned int oriented_edge_count[MAXN];
int vertices[REG * MAXN / 2][2];

GRAPH graph, oriented_graph;

int flow[REG * MAXN / 2];
int flow_sum[MAXN];

const int MAXF = 6000;
int all_flows[MAXF][MAXN];
int number_of_flows = 0;
set<string> all_flows_str;

int start_vertex[MAXN];
bool in_circuit[MAXN];
int circuit_length = 0;
int separate_circuit_lengths[MAXN];
int separate_circuits[MAXN][2 * MAXN];
int degree[MAXN];
int circuit[MAXN];
bool edge_in_circuit[REG * MAXN / 2];
int number_of_circuits = 0;

string cur_mod_flow_str;
int cur_mod_flow[REG * MAXN / 2];
int cur_flow[REG * MAXN / 2];
bool know_flow[REG * MAXN / 2];

int flow_values[8] = {-4, -3, -2, -1, 1, 2, 3, 4};
int flow_permutations[40320][8];
//int perm_idx[REG * MAXN / 2];

int cur_3flow[REG * MAXN / 2];
bool know_3flow[REG * MAXN / 2];

int other_3flow[REG * MAXN / 2];

int number_of_sols;

int mod_types[MAXN];

int neib_types[MAXN];

/*********************************Methods**************************************/
bool build_edge_flows() {
    /*for (int i = 0; i < number_of_vertices; ++i) {
        if (!in_circuit[i] && neib_types[i] == 4)
            return false;
    }*/
    for (int e = 0; e < number_of_edges; ++e) {
        know_flow[e] = false;
        if (edge_in_circuit[e]) {
            know_flow[e] = true;
            cur_flow[e] = cur_mod_flow[e];
            if (cur_flow[e] > 2) {
                cur_flow[e] -= 5;
            }
        }
    }
    bool found_new = true;
    while (found_new) {
        found_new = false;
        for (int v = 0; v < number_of_vertices; ++v) {
            int known_flows_count = 0;
            int f = 0;
            for (int j = 0; j < REG; ++j) {
                int next_vertex = graph[v][j];
                int ei = edge_index[v][next_vertex];
                if (know_flow[ei]) {
                    ++known_flows_count;
                    if (v < next_vertex)
                        f += cur_flow[ei];
                    else
                        f -= cur_flow[ei];
                }
            }
            if (known_flows_count == REG - 1) {
                found_new = true;
                for (int j = 0; j < REG; ++j) {
                    int next_vertex = graph[v][j];
                    int ei = edge_index[v][next_vertex];
                    if (!know_flow[ei]) {
                        know_flow[ei] = true;
                        if (v < next_vertex) {
                            cur_flow[ei] = -f;
                        } else {
                            cur_flow[ei] = f;
                        }
                    }
                }
            }
            if (known_flows_count == REG && f != 0) {
                //cerr << "wat" << endl; // it happens
                return false;
            }
        }
    }
    for (int e = 0; e < number_of_edges; ++e) {
        if (!know_flow[e]) {
            //cerr << "sometimes happens" << endl;
            return false;
        }
    }


    // build 3-flow
    for (int e = 0; e < number_of_edges; ++e) {
        know_3flow[e] = false;
        if (edge_in_circuit[e]) {
            know_3flow[e] = true;
            cur_3flow[e] = cur_flow[e];
            if (cur_3flow[e] % 2 == 0) {
                cur_3flow[e] /= 2;
            }
        }
    }
    bool found_3new = true;
    while (found_3new) {
        found_3new = false;
        for (int v = 0; v < number_of_vertices; ++v) {
            int known_flows_count = 0;
            int f = 0;
            for (int j = 0; j < REG; ++j) {
                int next_vertex = graph[v][j];
                int ei = edge_index[v][next_vertex];
                if (know_3flow[ei]) {
                    ++known_flows_count;
                    if (v < next_vertex)
                        f += cur_3flow[ei];
                    else
                        f -= cur_3flow[ei];
                }
            }
            if (known_flows_count == REG - 1) {
                found_3new = true;
                for (int j = 0; j < REG; ++j) {
                    int next_vertex = graph[v][j];
                    int ei = edge_index[v][next_vertex];
                    if (!know_3flow[ei]) {
                        know_3flow[ei] = true;
                        if (v < next_vertex) {
                            cur_3flow[ei] = -f;
                        } else {
                            cur_3flow[ei] = f;
                        }
                        if (abs(f) > 2) {
                            // cerr << "more than 2 also happens" << endl;
                            return false;
                        }
                    }
                }
            }
            if (known_flows_count == REG && f != 0) {
                //TODO: cerr << "why  or when this even happens? wat3" << endl;
                return false;
            }
        }
    }
    for (int e = 0; e < number_of_edges; ++e) {
        if (!know_3flow[e]) {
            cerr << "never happens" << endl;
            return false;
        }
    }

    for (int e = 0; e < number_of_edges; ++e) {
        int sec_flow = 2 * cur_flow[e] - 3 * cur_3flow[e];
        other_3flow[e] = sec_flow;
        if (abs(sec_flow) > 2) {
            //cerr << "wtf1" << endl; //TODO: why this happens?
            return false;
        }
        if (edge_in_circuit[e] && abs(sec_flow) != 1) {
            cerr << "wtf2" << endl;
            return false;
        }
        if (!edge_in_circuit[e] && abs(sec_flow) == 1) {
            cerr << "wtf3" << endl;
            return false;
        }
    }

    if (PRINT_SOLUTIONS) {
        //if (number_of_circuits == 0) {
            for (int j = 0; j <= number_of_circuits; ++j) {
                for (int i = 0; i < separate_circuit_lengths[j]; ++i) {
                    cerr << separate_circuits[j][i] << " ";
                }
                cerr << "; ";
            }
            cerr << "mod_types: ";
            for (int j = 0; j <= number_of_circuits; ++j) {
                for (int i = 0; i < separate_circuit_lengths[j]; ++i) {
                    cerr << mod_types[separate_circuits[j][i]] << " ";
                }
                cerr << "; ";
            }
            cerr << endl;
        //}
    }
    ++number_of_sols;
    return false;//true;
}

bool start_or_continue_build_middle_cycle(int min_possible_edge);

bool build_middle_cycle(int cur_vertex, int min_possible_edge) {
    if (cur_vertex == start_vertex[number_of_circuits]) {
        
        /*//has_dominating_circuit
        bool is_ignored[MAXN];
        for (int i = 0; i < number_of_vertices; ++i)
            is_ignored[i] = true;
        for (int e = 0; e < number_of_edges; ++e) {
            if (edge_in_circuit[e]) {
                is_ignored[vertices[e][0]] = false;
                is_ignored[vertices[e][1]] = false;
            }
        }
        for (int i = 0; i < number_of_edges; ++i) {
            if (is_ignored[vertices[i][0]] && is_ignored[vertices[i][1]]) {
                return false;
            }
        }*/

        if (build_edge_flows()) {
            return true;
        }
        //return false;
    }
    if (cur_vertex == start_vertex[number_of_circuits] && separate_circuit_lengths[number_of_circuits] > 0) {
        return start_or_continue_build_middle_cycle(min_possible_edge);
    }

    for (int j = 0; j < REG; ++j) {
        int next_vertex = graph[cur_vertex][j];
        int ei = edge_index[cur_vertex][next_vertex];

        if (ei < min_possible_edge)
            continue;
        if (edge_in_circuit[ei])
            continue;
        if (in_circuit[next_vertex])
            continue;

        circuit[circuit_length] = next_vertex;
        separate_circuits[number_of_circuits][separate_circuit_lengths[number_of_circuits]] = next_vertex;
        ++circuit_length;
        ++separate_circuit_lengths[number_of_circuits];
        in_circuit[next_vertex] = true;
        edge_in_circuit[ei] = true;
        circuit[circuit_length] = next_vertex;
        if (build_middle_cycle(next_vertex, min_possible_edge))
            return true;

        // undo
        --circuit_length;
        --separate_circuit_lengths[number_of_circuits];
        in_circuit[next_vertex] = false;
        edge_in_circuit[ei] = false;
    }
    return false;
}

bool start_or_continue_build_middle_cycle(int min_possible_edge) {
    if (min_possible_edge >= number_of_edges)
        return false;
    int v1 = vertices[min_possible_edge][0];
    int v2 = vertices[min_possible_edge][1];
    if (!edge_in_circuit[min_possible_edge] && !in_circuit[v1] && !in_circuit[v2]) {
        ++number_of_circuits;
        circuit[circuit_length] = v2;
        ++circuit_length;
        separate_circuits[number_of_circuits][0] = v2;
        separate_circuit_lengths[number_of_circuits] = 1;
        start_vertex[number_of_circuits] = v1;
        in_circuit[v2] = true;
        edge_in_circuit[min_possible_edge] = true;

        if (build_middle_cycle(v2, min_possible_edge))
            return true;

        // undo
        --number_of_circuits;
        --circuit_length;
        in_circuit[vertices[min_possible_edge][1]] = false;
        edge_in_circuit[min_possible_edge] = false;
    }
    return start_or_continue_build_middle_cycle(min_possible_edge + 1);
}

bool prepare_build_middle_cycle() {
    circuit_length = 0;
    separate_circuit_lengths[0] = 0;
    for (int i = 0; i < number_of_vertices; ++i) {
        in_circuit[i] = false;
    }
    for (int i = 0; i < number_of_edges; ++i) {
        edge_in_circuit[i] = false;
    }
    number_of_circuits = -1;

    if (start_or_continue_build_middle_cycle(0))
        return true;
    return false;
}


bool gen_all_nz5_flows(int edge_idx) {
    if (edge_idx == number_of_edges) {
        for (int i = 0; i < number_of_edges; ++i)
            all_flows[number_of_flows][i] = flow[i];
        ++number_of_flows;
        return false;
    }

    int cur_perm_idx = rand() % 40320;
    for (int iv = 0; iv < 8; ++iv) {
        int v = flow_permutations[cur_perm_idx][iv];

        flow[edge_idx] = v;
        int v1 = vertices[edge_idx][0];
        int v2 = vertices[edge_idx][1];
        bool all_flowed = true;
        for (int j = 0; j < REG; ++j) {
            if (flow[edge_index[v1][graph[v1][j]]] == 0) {
                all_flowed = false;
                break;
            }
        }
        if (all_flowed && flow_sum[v1] != v)
            continue;

        all_flowed = true;
        for (int j = 0; j < REG; ++j) {
            if (flow[edge_index[v2][graph[v2][j]]] == 0) {
                all_flowed = false;
                break;
            }
        }
        if (all_flowed && flow_sum[v2] != -v)
            continue;

        flow_sum[v1] -= v;
        flow_sum[v2] += v;

        if (gen_all_nz5_flows(edge_idx + 1))
            return true;
        if (number_of_flows == MAXF)
            return false;

        flow_sum[v1] += v;
        flow_sum[v2] -= v;

    }
    flow[edge_idx] = 0;
    return false;
}

void get_neib_types() {
    for (int i = 0; i < number_of_vertices; ++i) {
        int sum = 0;
        int same = 0;
        for (int j = 0; j < REG; ++j) {
            sum += mod_types[graph[i][j]];
            if (mod_types[graph[i][j]] == mod_types[graph[i][(j + 1) % REG]]) {
                same += 1;
            }
        }
        if (same == 0) {
            neib_types[i] = 1;
        }
        if (sum % 5 == 0) {
            neib_types[i] = 2;
        }
        if (same == 1 && (sum % 5 != 0)) {
            neib_types[i] = 3;
        }
        if (same == 3) {
            neib_types[i] = 4;
        }
    }
}

void printtypes() {
    cerr << "Printing types" << endl;
    int types[4];
    for (int i = 0; i < 4; ++i)
        types[i] = 0;
    for (int i = 0; i < number_of_vertices; ++i) {
        cerr << i << " (" << mod_types[i] << ") :";
        for (int j = 0; j < REG; ++j) {
            cerr << " " << mod_types[graph[i][j]];
        }
        cerr << " type " << neib_types[i];
        ++types[neib_types[i] - 1];
        cerr << endl;
    }
    cerr << "types stats: ";
    for (int i = 0; i < 4; ++i) {
        cerr << types[i] * 1.0 / number_of_vertices << " ";
    }
    cerr << endl;
}

bool has_33pp() {
    //Label the edges of the graph
    int edge_label = 0;
    for (int i = 0; i < number_of_vertices; ++i) {
        oriented_edge_count[i] = 0;
        for (int j = 0; j < REG; ++j) {
            if (i < graph[i][j]) {
                oriented_graph[i][oriented_edge_count[i]] = graph[i][j];
                ++oriented_edge_count[i];
                edge_index[i][graph[i][j]] = edge_label;
                edge_index[graph[i][j]][i] = edge_label;
                vertices[edge_label][0] = i;
                vertices[edge_label][1] = graph[i][j];
                ++edge_label;
            }
        }
    }
    if (edge_label != number_of_edges) {
        cerr << "Error: invalid number of edges" << endl;
        exit(1);
    }

    number_of_flows = 0;
    for (int i = 0; i < number_of_edges; ++i)
        flow[i] = 0;
    for (int i = 0; i < number_of_vertices; ++i)
        flow_sum[i] = 0;

    int perm_num = 0;
    do {
        for (int i = 0; i < 8; ++i)
            flow_permutations[perm_num][i] = flow_values[i];
        ++perm_num;
    } while (next_permutation(flow_values, flow_values + 8));

    gen_all_nz5_flows(0);
    all_flows_str.clear();
    //printgraph(graph);
    int f = 100;
    for (int v = 0; v < number_of_vertices; ++v) {
        cerr << v << ": ";
        for (int j = 0; j < REG; ++j) {
            cerr << graph[v][j] << "(";
            if (graph[v][j] < v)
                cerr << -all_flows[f][edge_index[v][graph[v][j]]];
            else
                cerr << all_flows[f][edge_index[v][graph[v][j]]];
            cerr << ") ";
        }
        cerr << endl;
    }
    cerr << endl;
    //return true;

    /*static const vector<vector<int>> proto_types = { // buggy
        {-2, 1, 1}, {-2, -2, 4},
        {-3, 1, 2}, {-3, -1, 4},
        {-4, 1, 3}, {-2, -1, 3},
        {-4, 2, 2}, {-1, -1, 2}
    };*/
    static const vector<vector<int>> proto_types = {
        {-2, 1, 1}, {-4, 1, 3},
        {-3, 1, 2}, {-4, 2, 2},
        {-2, -1, 3}, {-2, -2, 4},
        {-1, -1, 2}, {-3, -1, 4}
    };

    for (int f = 0; f < number_of_flows; ++f) {
        int counts[8];
        int sum = 0;
        for (int i = 0; i < 8; ++i) {
            counts[i] = 0;
        }
        for (int v = 0; v < number_of_vertices; ++v) {
            vector<int> fs;
            for (int j = 0; j < REG; ++j) {
                if (v < graph[v][j]) {
                    fs.push_back(all_flows[f][edge_index[v][graph[v][j]]]);
                } else {
                    fs.push_back(-all_flows[f][edge_index[v][graph[v][j]]]);
                }
            }
            sort(fs.begin(), fs.end());
            for (int i = 0; i < 8; ++i) {
                if (proto_types[i] == fs) {
                    ++counts[i];
                    ++sum;
                    break;
                }
            }
        }
        cerr << sum << " ";
        cerr << "counts: ";
        for (int i = 0; i < 8; ++i) {
            cerr << counts[i] << " ";
        }
        cerr << endl;
        continue;
        set<vector<int>> stypes;
        for (int v = 0; v < number_of_vertices; ++v) {
            vector<int> fs;
            for (int j = 0; j < REG; ++j) {
                fs.push_back(abs(all_flows[f][edge_index[v][graph[v][j]]]));
            }
            sort(fs.begin(), fs.end());
            stypes.insert(fs);
        }
        /*if (stypes.size() <= 3) {
            for (const auto& t : stypes) {
                for (int i = 0; i < REG; ++i)
                    cerr << t[i];
                cerr << " ";
            }
            cerr << "\t";
            for (int e = 0; e < number_of_edges; ++e) {
                //cerr << all_flows[f][e] << " ";
            }
            cerr << endl;
        }*/
        //continue;
        string profile;
        bool good_flow = true;
        for (int i = 0; i < number_of_edges; ++i) {
            int flow_val = all_flows[f][i];
            if (flow_val < 0)
                flow_val = 5 + flow_val;
            if (i == 0 && flow_val != 1) {
                good_flow = false;
                break;
            }
            profile += to_string(flow_val);
        }
        if (good_flow)
            all_flows_str.insert(profile);
    }

    int before = all_flows_str.size();
    cerr << "before: " << before << endl;
    PRINT_SOLUTIONS = true;
    //if (PRINT_SOLUTIONS) {
    //    printgraph(graph);
    //}
    int after = 0;
    for (auto& f : all_flows_str) {
        cur_mod_flow_str = f;
        for (int i = 0; i < number_of_edges; ++i) {
            cur_mod_flow[i] = cur_mod_flow_str[i] - '0';
        }

        for (int v = 0; v < number_of_vertices; ++v) {
            int type = 0;
            for (int ni = 0; ni < REG; ++ni) {
                int u1 = graph[v][ni];
                int f1 = cur_mod_flow[edge_index[v][u1]];
                if (u1 < v)
                    f1 = 5 - f1;

                int u2 = graph[v][(ni + 1) % REG];
                int f2 = cur_mod_flow[edge_index[v][u2]];
                if (u2 < v)
                    f2 = 5 - f2;

                if (f1 == f2)
                    type = f1;
            }
            mod_types[v] = type;
        }

        get_neib_types();
        number_of_sols = 0;
        PRINT_SOLUTIONS = false;
        prepare_build_middle_cycle();
        if (number_of_sols == 0) {
            ++after;
            cerr << "no 33pp for: " << cur_mod_flow_str << " ";
            for (int i = 0; i < number_of_edges; ++i) {
                cerr << "[" << to_string(vertices[i][0]) << " > " << to_string(vertices[i][1]) << "] " << f[i] << ";";
            }
            cerr << endl;
        } else {
            //if (number_of_sols < 6) {
            /*    printtypes();
                cerr << "solutions for: " << cur_mod_flow_str << endl;
                for (int i = 0; i < number_of_edges; ++i) {
                    cerr << "[" << to_string(vertices[i][0]) << " > " << to_string(vertices[i][1]) << "] " << cur_mod_flow[i] << ";";
                }
                cerr << endl;
                number_of_sols = 0;
                PRINT_SOLUTIONS = true;
                prepare_build_middle_cycle();*/
            //}
        }
        if ((number_of_sols > 0) && PRINT_SOLUTIONS) {
            cerr << "number of solutions: " << number_of_sols << endl;

            /*cerr << "solution for: " << cur_mod_flow_str << " ";
            for (int i = 0; i < number_of_edges; ++i) {
                cerr << "[" << to_string(vertices[i][0]) << " > " << to_string(vertices[i][1]) << "] " << f[i] << ";";
            }
            cerr << " |||||||||| ";

            for (int i = 0; i < number_of_edges; ++i) {
                cerr << "[" << to_string(vertices[i][0]) << " > " << to_string(vertices[i][1]) << "] " << cur_flow[i] << ";";
            }

            cerr << " |||||||||| ";

            for (int i = 0; i < number_of_edges; ++i) {
                cerr << "[" << to_string(vertices[i][0]) << " > " << to_string(vertices[i][1]) << "] " << cur_3flow[i] << ";";
            }
            cerr << endl;

            cerr << "mid-cycle:\t";
            for (int i = 0; i < number_of_edges; ++i) {
                cerr << " ";
                if (edge_in_circuit[i])
                    cerr << "1";
                else
                    cerr << "0";
                cerr << "\t";
            }
            cerr << endl;

            cerr << "nz5 flow:\t";
            for (int i = 0; i < number_of_edges; ++i) {
                if (cur_flow[i] > 0)
                    cerr << " ";
                cerr << cur_flow[i] << "\t";
            }
            cerr << endl;

            cerr << "F1 flow:\t";
            for (int i = 0; i < number_of_edges; ++i) {
                if (cur_3flow[i] >= 0)
                    cerr << " ";
                cerr << cur_3flow[i] << "\t";
            }
            cerr << endl;

            cerr << "F2 flow:\t";
            for (int i = 0; i < number_of_edges; ++i) {
                if (other_3flow[i] >= 0)
                    cerr << " ";
                cerr << other_3flow[i] << "\t";
            }
            cerr << endl;

            cerr << "(F1+F2)/2:\t";
            for (int i = 0; i < number_of_edges; ++i) {
                int f = (cur_3flow[i] + other_3flow[i]) / 2;
                if (f >= 0)
                    cerr << " ";
                cerr << f << "\t";
            }
            cerr << endl;

            cerr << "(F1-F2)/2:\t";
            for (int i = 0; i < number_of_edges; ++i) {
                int f = (cur_3flow[i] - other_3flow[i]) / 2;
                if (f >= 0)
                    cerr << " ";
                cerr << f << "\t";
            }
            cerr << endl;*/
        }
    }
    cerr << "after: " << after << endl;
    if (after > 0) {
        printgraph(graph);
    }
    return after == 0;
}


int main(int argc, char** argv) {
    srand(time(NULL));
    int to_skip = 0;
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Error: invalid number of arguments.\n");
        fprintf(stderr, "Usage: %s <number_of_vertices>\n", argv[0]);
        exit(1);
    } else {
        number_of_vertices = atoi(argv[1]);
        number_of_edges = REG * number_of_vertices / 2;
        if (number_of_vertices > MAXN) {
            fprintf(stderr, "Number of vertices is too big (limit is %d) \n", MAXN);
            exit(0);
        }
        if (argc >= 3) {
            to_skip = atoi(argv[2]);
        }
    }

    int codelength = number_of_vertices * REG / 2 + number_of_vertices;
    unsigned char code[codelength];
    unsigned long long int number_of_graphs_read = 0;
    unsigned long long int number_of_graphs_with_problems = 0;


    while (fread(code, sizeof(unsigned char), codelength, stdin)) {
        decode_multicode(code, codelength, graph);
        ++number_of_graphs_read;

        if (to_skip >= number_of_graphs_read)
            continue;

        //cerr << "g" << number_of_graphs_read << "\t";

        if (!has_33pp()) {
            cerr << "didn't find a solution for this graph!" << endl;
            ++number_of_graphs_with_problems;
        }
        //break;
    }

    cerr << "Read " << number_of_graphs_read << " graphs" << endl;
    cerr << "Found " << number_of_graphs_with_problems << " graphs which have problems" << endl;

    cout << "Read " << number_of_graphs_read << " graphs" << endl;
    cout << "Found " << number_of_graphs_with_problems << " graphs which have problems" << endl;

    return(EXIT_SUCCESS);
}
