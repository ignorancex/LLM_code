/*
 * File:   33pp_from_o6c4c.cpp
 * Author: Nikolay Ulyanov
 *
 * Created on 10 May 2017
 */


// probably has some kind of clash of number_of_sols variables

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
#include <armadillo>

using namespace std;
using namespace arma;

/*************************Defines and global variables*************************/

unsigned long long int number_of_graphs_read = 0;

bool PRINT_SOLUTIONS = true;

//Number of colours allowed for the colouring (should always be 6)
#define NUMBER_OF_COLOURS 6

//Marks for edges
unsigned int marks[REG * MAXN / 2];
#define RESETMARKS {for (int mki = 0; mki < REG * MAXN / 2; ++mki) marks[mki] = 0;}
#define MARK(v) ++marks[v]
#define UNMARK(v) --marks[v]
#define ISMARKED_HIGHEST(v) (marks[v] == 4)

#define MAXVAL INT_MAX - 2

unsigned int marks1[REG * MAXN / 2];
static int markvalue1 = MAXVAL;
#define RESETMARKS1 {if ((markvalue1 += 1) > MAXVAL) \
      {markvalue1 = 1; for (int mki = 0; mki < REG * MAXN / 2; ++mki) marks1[mki] = 0;}}
#define MARK1(v) marks1[v] = markvalue1
#define UNMARK1(v) marks1[v] = markvalue1 - 1
#define ISMARKED1(v) (marks1[v] == markvalue1)


unsigned int marks2[REG * MAXN / 2];
static int markvalue2 = MAXVAL;
#define RESETMARKS2 {if ((markvalue2 += 1) > MAXVAL) \
      {markvalue2 = 1; for (int mki = 0; mki < REG * MAXN / 2; ++mki) marks2[mki] = 0;}}
#define MARK2(v) marks2[v] = markvalue2
#define UNMARK2(v) marks2[v] = markvalue2 - 1
#define ISMARKED2(v) (marks2[v] == markvalue2)


unsigned int marks3[REG * MAXN / 2];
static int markvalue3 = MAXVAL;
#define RESETMARKS3 {if ((markvalue3 += 1) > MAXVAL) \
      {markvalue3 = 1; for (int mki = 0; mki < REG * MAXN / 2; ++mki) marks3[mki] = 0;}}
#define MARK3(v) marks3[v] = markvalue3
#define UNMARK3(v) marks3[v] = markvalue3 - 1
#define ISMARKED3(v) (marks3[v] == markvalue3)


unsigned int marks4[REG * MAXN / 2];
static int markvalue4 = MAXVAL;
#define RESETMARKS4 {if ((markvalue4 += 1) > MAXVAL) \
      {markvalue4 = 1; for (int mki = 0; mki < REG * MAXN / 2; ++mki) marks4[mki] = 0;}}
#define MARK4(v) marks4[v] = markvalue4
#define UNMARK4(v) marks4[v] = markvalue4 - 1
#define ISMARKED4(v) (marks4[v] == markvalue4)

unsigned int edge_index[MAXN][MAXN];
unsigned int oriented_edge_count[MAXN];
int vertices[REG * MAXN / 2][2];

bool nz5_found = false;

//Temp for debugging
unsigned long long int cycles_edge_bitvectors[MAXN];
unsigned long long int cycles_vertex_bitvectors[MAXN];
int cycles_size = 0;

bool orientation_found = false;

//index is edge_label
int colours_edges[REG * MAXN / 2];

int cycle_colour[REG * MAXN / 2];
int orientations[REG * MAXN / 2][REG * MAXN / 2];

bool k_colouring_found = false;

int vertex_marks[MAXN];
int current_colour = 0;
int current_vertex_count = 0;

int cycle_lengths[REG * MAXN / 2];
int cycles[REG * MAXN / 2][MAXN];

GRAPH graph, oriented_graph;

int vertex_permutations[NUMBER_OF_COLOURS * MAXN][REG];

unordered_map<string, int> triads;
bool is_oriented_vertex[MAXN];

bool admisible_length[MAXN];
int max_admisible_length = 0;
int start_vertex[MAXN];
bool in_circuit[MAXN];
int circuit_length = 0;
int degree[MAXN];
int circuit[MAXN];
bool edge_in_circuit[REG * MAXN / 2];
int separate_circuits[MAXN][2 * MAXN];
int separate_circuit_lengths[MAXN];
int number_of_circuits = 0;

string cur_mod_flow_str;
int cur_mod_flow[REG * MAXN / 2];
int cur_flow[REG * MAXN / 2];
bool know_flow[REG * MAXN / 2];

int flow_values[8] = {-4, -3, -2, -1, 1, 2, 3, 4};
int flow_permutations[40320][8];

int cur_3flow[REG * MAXN / 2];
bool know_3flow[REG * MAXN / 2];

int other_3flow[REG * MAXN / 2];

int number_of_sols;

int mod_types[MAXN];

const int not_num[REG][2] = {{1, 2}, {2, 0}, {0, 1}};
const int permutations[6][REG] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};

int number_of_leftout_edges = 0;
int leftout_edges[MAXN];
int leftout_partition[MAXN];
int max_partition_size = 0;
int partition_size[REG];

int number_of_pairs[REG];
bool is_in[MAXN][REG];
int pairs[REG][MAXN][2];
int bipartition[MAXN][REG];

bool is_forbidden_triad[MAXN][MAXN][MAXN];

bool quadruples[MAXN][MAXN][MAXN][MAXN];
bool is_poor_edge[REG * MAXN / 2];

int number_of_solutions = 0;
int number_of_oriented_vertices = 0;

bool found_solution = false;
bool please_break = false;

int total_number_of_good_cycles = 0;
int number_of_good_cycles[MAXN];

int circuit_state[MAXN];
int circuit_number[REG * MAXN / 2];
int is_forward[MAXN][MAXN];

int edge_profiles[REG * MAXN / 2][NUMBER_OF_COLOURS];

unordered_set<string> abs_profiles;
unordered_set<string> profiles;
unordered_map<string, unordered_set<int>> same_profiles;
string two_profile[REG * MAXN / 2][2];

int flow[REG * MAXN / 2];
int flow_sum[MAXN];

int weight_permutations[720][NUMBER_OF_COLOURS];
//const int set_count = 2;
//const int weights[2][NUMBER_OF_COLOURS] = {{0, 0, 1, 2, 4, 8}, {1, 1, 3, 5, 9, 17}};
//const int denoms[2] = {3, 6};

const int MAXF = 1000;
int all_flows[MAXF][MAXN];
int number_of_flows = 0;
vector<string> all_flows_str;

bool filtered = false;
unordered_set<string> o6c4c_sols;

/*********************************Methods**************************************/

bool build_edge_flows() {
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
    
    int low = -4;
    int high = 4;
    string prof = two_profile[edge_idx][0];
    if (same_profiles[prof].size() > 1) {
        for (const int e : same_profiles[prof]) {
            if (flow[e] != 0) {
                int v = flow[e];
                if (two_profile[e][0] != prof)
                    v = -v;
                if (low > v || high < v)
                    return false;
                low = v;
                high = v;
            }
        }
    }

    for (int v = low; v <= high; ++v) {
        if (v == 0)
            continue;
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

bool has_nz5s() {

    filtered = false;
    vector<string> sol_parts;
    int cur_colour = -1;
    string part;
    vector<int> lens;
    for (int i = 0; i < cycles_size; ++i) {
        if (cycle_colour[i] == cur_colour) {
            //part += "+";
        } else if (cycle_colour[i] > 0) {
            sort(lens.begin(), lens.end());
            for (int j = 0; j < lens.size(); ++j) {
                if (j > 0)
                    part += "+";
                part += to_string(lens[j]);
            }
            part += ";";
            sol_parts.push_back(part);
            part = "";
            lens.clear();
        }
        lens.push_back(cycle_lengths[i]);
        //part += to_string(cycle_lengths[i]);
        cur_colour = cycle_colour[i];
    }
    sort(lens.begin(), lens.end());
    for (int j = 0; j < lens.size(); ++j) {
        if (j > 0)
            part += "+";
        part += to_string(lens[j]);
    }
    part += ";";
    sol_parts.push_back(part);
    sort(sol_parts.begin(), sol_parts.end());
    string sol;
    for (auto& sol_part : sol_parts)
        sol += sol_part;
    if (o6c4c_sols.find(sol) != o6c4c_sols.end()) {
        filtered = true;
        return false;
    }
    o6c4c_sols.insert(sol);

    number_of_solutions = 0;

    unsigned long long int mask = 0;
    //for (unsigned long long int mask = 0; mask < BIT(cycles_size - 6); ++mask) {

        abs_profiles.clear();
        profiles.clear();
        same_profiles.clear();

        for (int i = 0; i < number_of_edges; ++i) {
            for (int j = 0; j < NUMBER_OF_COLOURS; ++j) {
                edge_profiles[i][j] = 0;
            }
        }


        bool had_colour[NUMBER_OF_COLOURS];
        for (int i = 0; i < NUMBER_OF_COLOURS; ++i)
            had_colour[i] = false;
        int shift = 0;

        for (int i = 0; i < cycles_size; ++i) {
            cycles[i][cycle_lengths[i]] = cycles[i][0];
            int mult = 1;
            if (had_colour[cycle_colour[i]]) {
                //cerr << "bit: " << mask << " " << shift << " " << (mask & BIT(shift)) << endl;
                if ((mask & BIT(shift)) != 0)
                    mult = -1;
                ++shift;
            }
            for (int j = 0; j < cycle_lengths[i]; ++j) {
                if (cycles[i][j] < cycles[i][j + 1]) {
                    edge_profiles[edge_index[cycles[i][j]][cycles[i][j + 1]]][cycle_colour[i]] = mult;
                } else {
                    edge_profiles[edge_index[cycles[i][j]][cycles[i][j + 1]]][cycle_colour[i]] = -mult;
                }
            }
            had_colour[cycle_colour[i]] = true;
        }

        /*int cycle_counts[NUMBER_OF_COLOURS];
        for (int i = 0; i < NUMBER_OF_COLOURS; ++i)
            cycle_counts[i] = 0;
        for (int i = 0; i < cycles_size; ++i) {
            ++cycle_counts[cycle_colour[i]];
        }
        //too_many_cycles = false;
        for (int i = 0; i < NUMBER_OF_COLOURS; ++i)
            if (cycle_counts[i] != 2) {
                too_many_cycles = true;
                return false;
            }
        bool had_colour[NUMBER_OF_COLOURS];
        for (int i = 0; i < NUMBER_OF_COLOURS; ++i)
            had_colour[i] = false;
        for (int i = 0; i < cycles_size; ++i) {
            cycles[i][cycle_lengths[i]] = cycles[i][0];
            int mult = 1;
            if (had_colour[cycle_colour[i]])
                mult = -1;
            for (int j = 0; j < cycle_lengths[i]; ++j) {
                if (cycles[i][j] < cycles[i][j + 1]) {
                    edge_profiles[edge_index[cycles[i][j]][cycles[i][j + 1]]][cycle_colour[i]] = mult;
                } else {
                    edge_profiles[edge_index[cycles[i][j]][cycles[i][j + 1]]][cycle_colour[i]] = -mult;
                }
            }
            had_colour[cycle_colour[i]] = true;
        }*/



        for (int i = 0; i < number_of_edges; ++i) {
            string abs_profile, profile, inverse_profile;
            for (int j = 0; j < NUMBER_OF_COLOURS; ++j) {
                if (edge_profiles[i][j] == 0) {
                    abs_profile += "0";
                    profile += "0";
                    inverse_profile += "0";
                } else {
                    abs_profile += "1";
                    profile += to_string(edge_profiles[i][j]);
                    inverse_profile += to_string(-edge_profiles[i][j]);
                }
            }
            abs_profiles.insert(abs_profile);
            profiles.insert(profile);
            profiles.insert(inverse_profile);
            same_profiles[profile].insert(i);
            same_profiles[inverse_profile].insert(i);
            two_profile[i][0] = profile;
            two_profile[i][1] = inverse_profile;
        }

        double vec_edge_profiles[number_of_edges * NUMBER_OF_COLOURS];

        for (int i = 0; i < number_of_edges; ++i) {
            for (int j = 0; j < NUMBER_OF_COLOURS; ++j) {
                vec_edge_profiles[number_of_edges * j + i] = edge_profiles[i][j];
            }
        }
        mat four_cover_matrix(&vec_edge_profiles[0], number_of_edges, NUMBER_OF_COLOURS);
        mat inv_four_cover_matrix = pinv(four_cover_matrix);

        const double EPS = 1e-5;
        for (int f = 0; f < number_of_flows; ++f) {
            double vec_flow[number_of_edges];
            for (int i = 0; i < number_of_edges; ++i) {
                vec_flow[i] = all_flows[f][i];
            }
            mat flows_vec(&vec_flow[0], number_of_edges, 1);
            vec weights = inv_four_cover_matrix * flows_vec;
            if (norm(four_cover_matrix * weights - flows_vec) < EPS) {
                // check for 33-pp
                cur_mod_flow_str = all_flows_str[f];

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

                number_of_sols = 0;
                PRINT_SOLUTIONS = false;
                prepare_build_middle_cycle();
                if (number_of_sols == 0) {
                    cerr << "no 33pp for: " << cur_mod_flow_str << " ";
                    for (int i = 0; i < number_of_edges; ++i) {
                        cerr << "[" << to_string(vertices[i][0]) << " > " << to_string(vertices[i][1]) << "] " << all_flows_str[f][i] << ";";
                    }
                    cerr << endl;
                } else {
                    //if (number_of_sols < 6) {
                        cerr << endl << endl;
                        cerr << "solutions for: " << cur_mod_flow_str << endl;
                        for (int i = 0; i < number_of_edges; ++i) {
                            cerr << "[" << to_string(vertices[i][0]) << " > " << to_string(vertices[i][1]) << "] " << cur_mod_flow[i] << ";";
                        }
                        cerr << endl;
                        number_of_sols = 0;
                        PRINT_SOLUTIONS = true;
                        prepare_build_middle_cycle();
                    //}
                }
                if ((number_of_sols > 0) && PRINT_SOLUTIONS) {
                    cerr << "number of solutions: " << number_of_sols << endl;

                    string profile;
                    for (int i = 0; i < number_of_edges; ++i)
                        profile += to_string(abs(all_flows[f][i])) + ";";
                    printgraph(graph);
                    ++number_of_solutions;
                    cerr << "mask: " << mask << endl;
                    cerr << "profile: " << profile << endl;
                    cerr << "weights: " << weights.col(0) << endl;
                    //weights.print("weights:");
                    for (int i = 0; i < number_of_vertices; ++i) {
                        if (is_oriented_vertex[i]) {
                            cerr << i << " (";
                            for (int j = 0; j < REG; ++j) {
                                cerr << all_flows[f][edge_index[i][graph[i][j]]];
                                if (j < REG - 1)
                                    cerr << ", ";
                            }
                            cerr << ");  ";
                        }
                    }
                    cerr << endl;
                }
            }
            //if (number_of_solutions > 0)
              //  return true;
        }
    return false;
    //}
}

void determine_all_cycles(unsigned int current_vertex, unsigned int previous_vertex,
        unsigned int first_cycle_vertex, unsigned long long int *current_vertex_cycle_bitvector,
        unsigned long long int *current_edge_cycle_bitvector, int iteration);

void orient_cycles(int cur_cycle);

void print_cycle(int current_cycle_index) {
    for (int i = 0; i < cycle_lengths[current_cycle_index]; ++i) {
        cerr << cycles[current_cycle_index][i] << " ";
    }
    cerr << endl;
}

void orient_other_edges_of_cycle(unsigned int current_vertex,
        unsigned int previous_vertex, unsigned int first_vertex, int current_cycle_index,
        int locally_marked1[], int& locally_marked1_size, int locally_marked2[], int& locally_marked2_size,
        int locally_marked3[], int& locally_marked3_size, int locally_marked4[], int& locally_marked4_size, int vertex_count)
{
    cycles[current_cycle_index][vertex_count] = current_vertex;
    int ei = edge_index[previous_vertex][current_vertex];

    if (previous_vertex < current_vertex) {
        //fprintf(stderr, "%d\n", markvalue1);
        if (!ISMARKED1(ei)) {
            //mark1
            MARK1(ei);
            locally_marked1[locally_marked1_size] = ei;
            orientations[current_cycle_index][ei] = 1;
            ++locally_marked1_size;
        } else if (!ISMARKED3(ei)) {
            //mark3
            MARK3(ei);
            locally_marked3[locally_marked3_size] = ei;
            orientations[current_cycle_index][ei] = 3;
            ++locally_marked3_size;
        } else {
            //Conflict detected
            return;
        }
    } else {
        //mark2
        if (!ISMARKED2(ei)) {
            MARK2(ei);
            locally_marked2[locally_marked2_size] = ei;
            orientations[current_cycle_index][ei] = 2;
            ++locally_marked2_size;
        } else if (!ISMARKED4(ei)) {
            //mark4
            MARK4(ei);
            locally_marked4[locally_marked4_size] = ei;
            orientations[current_cycle_index][ei] = 4;
            ++locally_marked4_size;
        } else {
            //Conflict detected
            return;
        }
    }

    if (current_vertex == first_vertex) {
        //i.e. cycle was fully marked without conflicts!
        orient_cycles(current_cycle_index + 1);

        return;
    }

    for (int i = 0; i < REG; ++i) {
        if (graph[current_vertex][i] != previous_vertex) {
            if ((BIT(edge_index[current_vertex][graph[current_vertex][i]]) & cycles_edge_bitvectors[current_cycle_index]) != 0) {
                    orient_other_edges_of_cycle(graph[current_vertex][i], current_vertex, first_vertex, current_cycle_index,
                            locally_marked1, locally_marked1_size, locally_marked2, locally_marked2_size,
                            locally_marked3, locally_marked3_size, locally_marked4, locally_marked4_size, vertex_count + 1);
                    return;
            }
        }
    }
    fprintf(stderr, "Error: no second neighbour found\n");
    exit(1);

}

void unmark_locally_marked(int locally_marked1[], int& locally_marked1_size, int locally_marked2[], int& locally_marked2_size,
        int locally_marked3[], int& locally_marked3_size, int locally_marked4[], int& locally_marked4_size)
{
    for (int i = 0; i < locally_marked1_size; ++i) {
        DEBUGASSERT(ISMARKED1(locally_marked1[i]));
        UNMARK1(locally_marked1[i]);
    }
    locally_marked1_size = 0;

    for (int i = 0; i < locally_marked2_size; ++i) {
        DEBUGASSERT(ISMARKED2(locally_marked2[i]));
        UNMARK2(locally_marked2[i]);
    }
    locally_marked2_size = 0;

    for (int i = 0; i < locally_marked3_size; ++i) {
        DEBUGASSERT(ISMARKED3(locally_marked3[i]));
        UNMARK3(locally_marked3[i]);
    }
    locally_marked3_size = 0;

    for (int i = 0; i < locally_marked4_size; ++i) {
        DEBUGASSERT(ISMARKED4(locally_marked4[i]));
        UNMARK4(locally_marked4[i]);
    }
    locally_marked4_size = 0;
}

void orient_cycles(int cur_cycle) {
    if (cur_cycle >= cycles_size) {
        orientation_found = true;
        return;
    }

    for (int i = 0; i < number_of_vertices; ++i) {
        //TODO: best van reeds gemarkeerde edge vertrekken?
        if ((BIT(i) & cycles_vertex_bitvectors[cur_cycle]) != 0) {
            for (int j = 0; j < oriented_edge_count[i]; ++j) {
                if ((BIT(edge_index[i][oriented_graph[i][j]]) & cycles_edge_bitvectors[cur_cycle]) != 0) {
                    int locally_marked1[REG * MAXN / 2];
                    int locally_marked1_size = 0;

                    int locally_marked2[REG * MAXN / 2];
                    int locally_marked2_size = 0;

                    int locally_marked3[REG * MAXN / 2];
                    int locally_marked3_size = 0;

                    int locally_marked4[REG * MAXN / 2];
                    int locally_marked4_size = 0;

                    orient_other_edges_of_cycle(oriented_graph[i][j], i, i, cur_cycle,
                            locally_marked1, locally_marked1_size, locally_marked2, locally_marked2_size,
                            locally_marked3, locally_marked3_size, locally_marked4, locally_marked4_size, 0);

                    /**
                     * Opm: met cur_cycle > 0 slechts een heel kleine fractie sneller
                     * want meestal vrij snel een orientation gevonden?)
                     */
                    if (!orientation_found) {
                        //Now try other orientation
                        //First undo previous orientation
                        unmark_locally_marked(locally_marked1, locally_marked1_size, locally_marked2, locally_marked2_size,
                                locally_marked3, locally_marked3_size, locally_marked4, locally_marked4_size);

                        orient_other_edges_of_cycle(i, oriented_graph[i][j], oriented_graph[i][j], cur_cycle,
                                locally_marked1, locally_marked1_size, locally_marked2, locally_marked2_size,
                                locally_marked3, locally_marked3_size, locally_marked4, locally_marked4_size, 0);

                        if (!orientation_found) {
                            unmark_locally_marked(locally_marked1, locally_marked1_size, locally_marked2, locally_marked2_size,
                                    locally_marked3, locally_marked3_size, locally_marked4, locally_marked4_size);
                        }

                        if (cur_cycle == 0 && orientation_found) {
                            fprintf(stderr, "INFO: orientation found by changing orientation of first cycle (never happens?)\n");
                        }
                    } 
                    return;
                }
            }
            fprintf(stderr, "Error: no cycle neighbours found\n");
            exit(1);
        }
    }
}

bool is_orientable() {
    RESETMARKS1;
    RESETMARKS2;
    RESETMARKS3;
    RESETMARKS4;

    //Orientation of first cycle does not matter? (But is hardly any difference in timing)
    DEBUGASSERT(cycles_size > 1);

    orientation_found = false;
    orient_cycles(0);

    return orientation_found;
}

//Is waarschijnlijk heel duur?
bool would_be_conflicting_colouring(int colour, unsigned long long int cycle_edge_bitvector) {
    unsigned long long int mask = BIT(colour);
    for (int i = 0; i < REG * number_of_vertices / 2; ++i) {
        if (((BIT(i) & cycle_edge_bitvector) != 0) && (colours_edges[i] & mask) != 0) {
            return true;
        }
    }
    //TODO: kan eventueel indices van edges opslaan in array!
    //Of zelfs al op voorhand doen (bij bouwen van cykels..)
    //Of op moment dat orientable c4c gevonden (en > NUMBER_OF_COLOURS cykels)
    //Maar kost eigenlijk niets volgens profiler!
    return false;
}

//Important: assumes that there won't be any conflicts
void colour_edges_of_cycle(int colour, unsigned long long int cycle_edge_bitvector) {
    unsigned long long int mask = BIT(colour);
    for (int i = 0; i < REG * number_of_vertices / 2; ++i) {
        if ((BIT(i) & cycle_edge_bitvector) != 0) {
            colours_edges[i] |= mask;
        }
    }
}

void undo_colouring_of_edges(int colour, unsigned long long int cycle_edge_bitvector) {
    unsigned long long int mask = ~BIT(colour);
    for (int i = 0; i < REG * number_of_vertices / 2; ++i) {
        if ((BIT(i) & cycle_edge_bitvector) != 0) {
            colours_edges[i] &= mask;
        }
    }
}


void colour_cycles(int cur_cycle) {
    if (cur_cycle >= cycles_size) {
        //TODO: eventueel controleren of popcount van alle edges == 2 is..
        k_colouring_found = true;
        return;
    }

    for (int i = 0; i < NUMBER_OF_COLOURS; ++i) {
        cycle_colour[cur_cycle] = i;
        //Probeer cyckel met kleur i te kleuren
        //Overloop eerst alle edge indices van cycles_edge_bitvectors[cur_cycle]
        //En kijk of er nog geen edge met kleur i tussen zit
        //Als geen conflict
        //Kleur dan al die bogen met kleur i
        //Recursie
        //abort als colouring found
        if (!would_be_conflicting_colouring(i, cycles_edge_bitvectors[cur_cycle])) {
            colour_edges_of_cycle(i, cycles_edge_bitvectors[cur_cycle]);

            //recursie
            colour_cycles(cur_cycle + 1);

            if (k_colouring_found)
                return;

            undo_colouring_of_edges(i, cycles_edge_bitvectors[cur_cycle]);
        }
    }
}

bool is_k_c4c() {
    if (current_colour < NUMBER_OF_COLOURS)
        return false;
    if (cycles_size <= NUMBER_OF_COLOURS) {
        for (int i = 0; i < cycles_size; ++i) {
            cycle_colour[i] = i;
        }
        return true;
    }

    //Probeer cykels op alle mogelijke manieren te kleuren
    //Als conflict, laatste kleuring ongedaan maken
    //Stop als kleuring gevonden
    //Is zeer dom en inefficient algoritme, maar zal wellicht niet veel opgeroepen
    //worden omdat er meestal <= number_of_colours cykels zullen zijn?
    //En meestal toch snel een kleuring gevonden?

    //Eventueel later optimalisaties
    //Overloop eerst ganse cykel en kijk welke kleuren nog niet gebruikt zijn..

    // TODO: do i need this?

    for (int i = 0; i < REG * number_of_vertices / 2; ++i)
        colours_edges[i] = 0;

    k_colouring_found = false;

    colour_cycles(0);

    return k_colouring_found;
}

void form_next_cycle(int iteration) {
    int edge_count = 0;

    //Continue to make a new cycle from the first edge which wasn't already covered 4 times
    //TODO: eventueel optimalisatie: beter vertrekken van edges die al reeds 1x gemarkeerd waren?
    for (int i = 0; i < number_of_vertices; ++i) {
        if (vertex_marks[i] != current_colour)
            continue;
        int edge_count = 0;
        int next_vertex = -1;
        for (int j = 0; j < REG; ++j) {
            if (!ISMARKED_HIGHEST(edge_index[i][graph[i][j]]) && vertex_marks[graph[i][j]] == current_colour) {
                next_vertex = graph[i][j];
                ++edge_count;
            }
        }
        if (edge_count == 0)
            continue;
        if (edge_count == 1)
            return;
        if (current_vertex_count != 0 && edge_count == REG)
            continue;

        MARK(edge_index[i][next_vertex]);
        //cycles[cycles_size][cycle_lengths[cycles_size]] = next_vertex;
        ++vertex_marks[next_vertex];
        ++current_vertex_count;
        ++cycle_lengths[cycles_size];

        unsigned long long int current_vertex_cycle_bitvector = BIT(i);
        unsigned long long int current_edge_cycle_bitvector = 0;
        determine_all_cycles(next_vertex, i, i,
                &current_vertex_cycle_bitvector, &current_edge_cycle_bitvector, iteration + 1);

        --vertex_marks[next_vertex];
        //cycles[cycles_size][cycle_lengths[cycles_size] - 1] = i;
        //++vertex_marks[i];
        //determine_all_cycles(i, next_vertex, next_vertex,
        //        &current_vertex_cycle_bitvector, &current_edge_cycle_bitvector, iteration + 1);
        //--vertex_marks[i];

        UNMARK(edge_index[i][next_vertex]);
        --current_vertex_count;
        --cycle_lengths[cycles_size];
        return;
    }

    //TODO: eerst kijken of orientable is? Of beter eerst kijken of NUMBER_OF_COLOURS-c4c is?
    //Meest restrectieve eerst doen (statistieken!)
    // UPD: comment here is old, i've changed the order of checks here

    //Check if it's a NUMBER_OF_COLOURS-c4c
    if (is_k_c4c()) {

        //Now check if it's orientable
        if (is_orientable()) {

            for (int i = 0; i < number_of_vertices; ++i) {
                for (int j = 0; j < number_of_vertices; ++j) {
                    for (int k = 0; k < number_of_vertices; ++k) {
                        for (int m = 0; m < number_of_vertices; ++m) {
                            quadruples[i][j][k][m] = false;
                        }
                    }
                }
            }

            for (int i = 0; i < number_of_edges; ++i) {
                is_poor_edge[i] = false;
            }

            for (int i = 0; i < number_of_vertices; ++i) {
                is_oriented_vertex[i] = false;
            }

            triads.clear();
            number_of_oriented_vertices = 0;
            for (int c = 0; c < cycles_size; ++c) {
                cycles[c][cycle_lengths[c]] = cycles[c][0];
                cycles[c][cycle_lengths[c] + 1] = cycles[c][1];
                cycles[c][cycle_lengths[c] + 2] = cycles[c][2];
                for (int i = 0; i < cycle_lengths[c]; ++i) {
                    int v1 = cycles[c][i];
                    int v2 = cycles[c][i + 1];
                    int v3 = cycles[c][i + 2];
                    int v4 = cycles[c][i + 3];
                    if (quadruples[v1][v2][v3][v4]) {
                        is_poor_edge[edge_index[v2][v3]] = true;
                    }
                    quadruples[v1][v2][v3][v4] = true;
                    quadruples[v4][v3][v2][v1] = true;

                    string cur_triad = to_string(v1) + "," + to_string(v2) + "," + to_string(v3);
                    if (triads.find(cur_triad) == triads.end()) {
                        triads[cur_triad] = 0;
                    }
                    bool is_oriented_current_vertex = is_oriented_vertex[v2];
                    if (!is_oriented_current_vertex && triads[cur_triad] == 1) {
                        is_oriented_vertex[v2] = true;
                        ++number_of_oriented_vertices;
                    }
                    ++triads[cur_triad];
                }
            }

            for (int i = 0; i < number_of_vertices; ++i) {
                for (int j = 0; j < number_of_vertices; ++j) {
                    for (int k = 0; k < number_of_vertices; ++k) {
                        is_forbidden_triad[i][j][k] = false;
                    }
                }
            }

            for (int i = 0; i < number_of_vertices; ++i) {
                if (is_oriented_vertex[i]) {
                    for (int j1 = 0; j1 < REG; ++j1) {
                        for (int j2 = 0; j2 < REG; ++j2) {
                            if (j1 != j2) {
                                string cur_triad = to_string(graph[i][j1]) + "," + to_string(i) + "," + to_string(graph[i][j2]);
                                if (triads.find(cur_triad) == triads.end()) {
                                    triads[cur_triad] = 0;
                                }
                                if (triads[cur_triad] == 0) {
                                    is_forbidden_triad[graph[i][j1]][i][graph[i][j2]] = true;
                                }
                            }
                        }
                    }
                }
            }

            int number_of_poor_edges = 0;
            for (int i = 0; i < number_of_edges; ++i) {
                if (is_poor_edge[i]) {
                    ++number_of_poor_edges;
                }
            }

            has_nz5s();
            if (!filtered) {
            //if (number_of_solutions > 0) {
                if (PRINT_SOLUTIONS) {
                    cerr << "g" << number_of_graphs_read << ": ";
                }
                cerr << "number of solutions: " << number_of_solutions << "; oriented: " << number_of_oriented_vertices << "; poor edges: " << number_of_poor_edges << "; ";
                //if (PRINT_SOLUTIONS) {
                    cerr << endl;
                //}
                cout << "number of solutions: " << number_of_solutions << "; oriented: " << number_of_oriented_vertices << "; poor edges: " << number_of_poor_edges << "; ";

                //if (please_break) {
                    cout << endl;
                  //  return;
                //}
                //nz5_found = true;

                if (PRINT_SOLUTIONS) {
                    for (int i = 0; i < cycles_size; ++i) {
                        fprintf(stderr, "cycle %d (colour %d): ", i, cycle_colour[i]);
                        print_cycle(i);
                    }
                    cerr << "oriented vertices: ";
                    for (int i = 0; i < number_of_vertices; ++i) {
                        if (is_oriented_vertex[i]) {
                            cerr << i << " ";
                        }
                    }
                    cerr << endl;
                    cerr << "success!\tcycles: ";
                    int cur_colour = -1;
                    for (int i = 0; i < cycles_size; ++i) {
                        if (cycle_colour[i] == cur_colour)
                            cerr << "+";
                        else if (cycle_colour[i] > 0)
                            cerr << "; ";
                        cerr << cycle_lengths[i];
                        cur_colour = cycle_colour[i];
                    }
                    cerr << endl << endl;
                }
            }
            //}
        }
    } else {
        //is not orientable
    }

}

void determine_all_cycles(unsigned int current_vertex, unsigned int previous_vertex,
        unsigned int first_cycle_vertex, unsigned long long int *current_vertex_cycle_bitvector,
        unsigned long long int *current_edge_cycle_bitvector, int iteration) {

    //TODO: current_edge_cycle_bitvector waarschijnlijk niet nodig?!
    //Eventueel nog behouden voor debug!
    //Remark: it is assumed that {current_vertex, previous_vertex} was already marked
    //and is valid (i.e. at most marked 4 times)

    if (nz5_found)
        return;

    if (current_vertex == first_cycle_vertex) { //I.e. cycle was formed
        if (cycle_lengths[cycles_size] == 4) // XXX small optimization
            return;
 
        // i'm not convinced that i need to separate oriented vertices from each other
        /*string cur_triad = to_string(previous_vertex) + "," + to_string(current_vertex) + "," + to_string(cycles[cycles_size][1]);
        if (triads.find(cur_triad) == triads.end()) { // FIXME make this check global
            triads[cur_triad] = 0;
        }
        bool is_oriented_current_vertex = is_oriented_vertex[current_vertex];
        if (!is_oriented_current_vertex && triads[cur_triad] == 1) {
            bool has_oriented_neib = false;
            for (int j = 0; j < REG; ++j) {
                if (is_oriented_vertex[graph[current_vertex][j]]) {
                    has_oriented_neib = true;
                }
            }
            if (has_oriented_neib)
                return;
            is_oriented_vertex[current_vertex] = true;
        }
        ++triads[cur_triad];*/


        *current_edge_cycle_bitvector |= BIT(edge_index[previous_vertex][current_vertex]);

        //Temp for debugging
        cycles_edge_bitvectors[cycles_size] = *current_edge_cycle_bitvector;
        cycles_vertex_bitvectors[cycles_size] = *current_vertex_cycle_bitvector;
        ++cycles_size;

        bool increased = (current_vertex_count == number_of_vertices);
        if (increased) {
            current_vertex_count = 0;
            ++current_colour;
        }
        form_next_cycle(iteration);

        if (increased) {
            current_vertex_count = number_of_vertices;
            --current_colour;
        }
        --cycles_size;

        *current_edge_cycle_bitvector &= ~BIT(edge_index[previous_vertex][current_vertex]);

        //--triads[cur_triad];
        //is_oriented_vertex[current_vertex] = is_oriented_current_vertex;
    } else {
        //Mark
        *current_vertex_cycle_bitvector |= BIT(current_vertex);
        *current_edge_cycle_bitvector |= BIT(edge_index[previous_vertex][current_vertex]);

        //i.e. vertex not visited before
        for (int idx = 0; idx < REG; ++idx) {
            int i = vertex_permutations[iteration][idx];
            int next_vertex = graph[current_vertex][i];
            if (vertex_marks[next_vertex] == current_colour && next_vertex != previous_vertex) {
                if (!ISMARKED_HIGHEST(edge_index[current_vertex][next_vertex])) {
                    /*string cur_triad = to_string(previous_vertex) + "," + to_string(current_vertex) + "," + to_string(next_vertex);

                    if (triads.find(cur_triad) == triads.end()) {
                        triads[cur_triad] = 0;
                    }

                    bool is_oriented_current_vertex = is_oriented_vertex[current_vertex];
                    if (!is_oriented_current_vertex && triads[cur_triad] == 1) {
                        bool has_oriented_neib = false;
                        for (int j = 0; j < REG; ++j) {
                            if (is_oriented_vertex[graph[current_vertex][j]]) {
                                has_oriented_neib = true;
                            }
                        }
                        if (has_oriented_neib)
                            continue;
                        is_oriented_vertex[current_vertex] = true;
                    }

                    ++triads[cur_triad];*/

                    MARK(edge_index[current_vertex][next_vertex]);
                    //cycles[cycles_size][cycle_lengths[cycles_size]] = next_vertex;
                    ++vertex_marks[next_vertex];
                    ++current_vertex_count;
                    ++cycle_lengths[cycles_size];

                    determine_all_cycles(next_vertex, current_vertex, first_cycle_vertex,
                            current_vertex_cycle_bitvector, current_edge_cycle_bitvector, iteration + 1);

                    UNMARK(edge_index[current_vertex][next_vertex]);
                    --vertex_marks[next_vertex];
                    --current_vertex_count;
                    --cycle_lengths[cycles_size];

                    //--triads[cur_triad];
                    //is_oriented_vertex[current_vertex] = is_oriented_current_vertex;
                }
            }
        }

        //Unmark
        *current_vertex_cycle_bitvector &= ~BIT(current_vertex);
        *current_edge_cycle_bitvector &= ~BIT(edge_index[previous_vertex][current_vertex]);
    }
}

bool has_solution() {
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
    if (edge_label != REG * number_of_vertices / 2) {
        fprintf(stderr, "Error: invalid number of edges \n");
        exit(1);
    }

    number_of_flows = 0;
    for (int i = 0; i < number_of_edges; ++i)
        flow[i] = 0;
    for (int i = 0; i < number_of_vertices; ++i)
        flow_sum[i] = 0;
    gen_all_nz5_flows(0);
    cerr << "number of flows: " << number_of_flows << "; ";
    cout << "number of flows: " << number_of_flows << "; ";

    all_flows_str.clear();
    for (int f = 0; f < number_of_flows; ++f) {
        string profile;
        for (int i = 0; i < number_of_edges; ++i) {
            int flow_val = all_flows[f][i];
            if (flow_val < 0)
                flow_val = 5 + flow_val;
            profile += to_string(flow_val);
        }
        all_flows_str.push_back(profile);
    }

    /*for (int i = 0; i < number_of_flows; ++i) {
        for (int j = 0; j < number_of_edges; ++j) {
            cerr << all_flows[i][j] << ", ";
        }
        cerr << endl;
    }*/

    RESETMARKS;
    for (int i = 0; i < number_of_vertices; ++i) {
        vertex_marks[i] = 0;
    }

    MARK(edge_index[0][graph[0][0]]);
    //cycles[0][0] = graph[0][0];
    current_colour = 0;
    vertex_marks[graph[0][0]] = 1;
    current_vertex_count = 1;
    cycle_lengths[0] = 1;

    nz5_found = false;
    cycles_size = 0;

    //triads.clear();

    unsigned long long int current_vertex_cycle_bitvector = BIT(0);
    unsigned long long int current_edge_cycle_bitvector = 0;

    //Start making all possible cycles which contain edge {0, graph[0][0]}
    determine_all_cycles(graph[0][0], 0, 0, &current_vertex_cycle_bitvector,
            &current_edge_cycle_bitvector, 0);

    return nz5_found;
}



int main(int argc, char** argv) {
    int idx[NUMBER_OF_COLOURS] = {0, 1, 2, 3, 4, 5};
    int perm_num = 0;
    do {
        for (int i = 0; i < NUMBER_OF_COLOURS; ++i)
            weight_permutations[perm_num][i] = idx[i];
        ++perm_num;
    } while (next_permutation(idx, idx + NUMBER_OF_COLOURS));

    cout << "perm count: " << perm_num << endl;
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

    for (int i = 0; i < number_of_vertices; ++i) {
        admisible_length[i] = (i >= number_of_edges / 2);// && i % 3 == 0); // last condition doesn't work
        if (admisible_length[i]) {
            max_admisible_length = i;
        }
    }

    vector<int> indices;
    for (int i = 0; i < REG; ++i)
        indices.push_back(i);

    for (int i = 0; i < 6 * MAXN; ++i) {
        random_shuffle(indices.begin(), indices.end());
        for (int j = 0; j < REG; ++j)
            vertex_permutations[i][j] = indices[j];
    }

    int codelength = number_of_vertices * REG / 2 + number_of_vertices;
    unsigned char code[codelength];
    unsigned long long int number_of_graphs_without_solution = 0;

    // hard graphs:
    // 20: 4
    // 22: 10, 14, 15, 19?
    // 24: ...
    //unordered_map<int, int> skippers;
    //skippers[20] = 3; // actually all 6 checked
    //skippers[22] = 31;
    //skippers[24] = 2;

    cerr << endl;
    while (fread(code, sizeof(unsigned char), codelength, stdin)) {
        decode_multicode(code, codelength, graph);
        ++number_of_graphs_read;
        //printgraph(graph);
        //if (skippers.find(number_of_vertices) != skippers.end() && skippers[number_of_vertices] >= number_of_graphs_read)
          //  continue;
        if (to_skip >= number_of_graphs_read)
            continue;
        cerr << "g" << number_of_graphs_read << "\t";
        cout << "g" << number_of_graphs_read << "\t";

        if (!has_solution()) {
            cerr << "didn't find a solution for this graph!" << endl;
            cout << "didn't find a solution for this graph!" << endl;
            ++number_of_graphs_without_solution;
        } else {
            cerr << "has combined solution" << endl;
            cout << "has combined solution" << endl;
        }
        break;
    }
    cerr << "fin" << endl;

    cerr << "Read " << number_of_graphs_read << " graphs" << endl;
    cerr << "Found " << number_of_graphs_without_solution << " graphs which do not have a combined solution" << endl;

    cout << "Read " << number_of_graphs_read << " graphs" << endl;
    cout << "Found " << number_of_graphs_without_solution << " graphs which do not have a combined solution" << endl;

    return(EXIT_SUCCESS);
}
