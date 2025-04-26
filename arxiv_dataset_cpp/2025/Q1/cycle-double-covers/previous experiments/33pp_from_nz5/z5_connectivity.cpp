/*
 * File:   z5_connectivity.cpp
 * Author: Nikolay Ulyanov
 *
 * Created on 27 January 2017
 * 
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

const int MAXF = 100000;
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

int flow_values[4] = {1, 2, 3, 4};
int flow_permutations[24][4];

int cur_3flow[REG * MAXN / 2];
bool know_3flow[REG * MAXN / 2];

int other_3flow[REG * MAXN / 2];

int number_of_sols;

int mod_types[MAXN];

int neib_types[MAXN];

/*********************************Methods**************************************/
bool gen_one_solution(int edge_idx) {
    if (edge_idx == number_of_edges) {
        for (int i = 0; i < number_of_edges; ++i)
            all_flows[number_of_flows][i] = flow[i];
        ++number_of_flows;
        return false;
    }

    int cur_perm_idx = rand() % 24;
    for (int iv = 0; iv < 4; ++iv) {
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
        if (all_flowed && flow_sum[v2] != 5 - v)
            continue;

        flow_sum[v1] -= v; if (flow_sum[v1] < 0) flow_sum[v1] += 5;
        flow_sum[v2] += v; if (flow_sum[v2] > 4) flow_sum[v2] -= 5;

        if (gen_one_solution(edge_idx + 1))
            return true;
        if (number_of_flows == MAXF)
            return false;

        flow_sum[v1] += v; if (flow_sum[v1] > 4) flow_sum[v1] -= 5;
        flow_sum[v2] -= v; if (flow_sum[v2] < 0) flow_sum[v2] += 5;

    }
    flow[edge_idx] = 0;
    return false;
}

bool has_z5_connectivity() {
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
        for (int i = 0; i < 4; ++i)
            flow_permutations[perm_num][i] = flow_values[i];
        ++perm_num;
    } while (next_permutation(flow_values, flow_values + 4));

    gen_one_solution(0);
    cerr << "number of flows: " << number_of_flows << endl;
    return number_of_flows > 0;
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

        cerr << "g" << number_of_graphs_read << "\t";

        if (!has_z5_connectivity()) {
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
