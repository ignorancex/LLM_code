/*
 * File:   util.cpp
 * Contains some useful defines and auxiliary methods.
 * Modified by: Nikolay Ulyanov (ulyanick@gmail.com)
 */

#include "util.h"
#include <iostream>
#include <cstdio>
#include <functional>

using namespace std;

/*************************** Useful auxiliary methods ***************************/

void AddEdge(GRAPH graph, unsigned int adj[], unsigned int v, unsigned int w) {
    graph[v][adj[v]] = w;
    graph[w][adj[w]] = v;
    ++adj[v];
    ++adj[w];
}

bool DecodeNextGraphInMulticode(FILE* input, int& number_of_vertices, int& number_of_edges, GRAPH graph) {
    unsigned char code_for_number_of_vertices[1];
    if (!fread(code_for_number_of_vertices, sizeof(unsigned char), 1, input)) {
        return false;
    }

    number_of_vertices = code_for_number_of_vertices[0];
    if (number_of_vertices > MAXN) {
        cerr << "Number of vertices is too big (limit is " << MAXN << ")" << endl;
        exit(1);
    }
    number_of_edges = REG * number_of_vertices / 2;

    int code_length = number_of_vertices * REG / 2 + number_of_vertices - 1;
    unsigned char code[code_length];

    if (!fread(code, sizeof(unsigned char), code_length, input)) {
        return false;
    }

    unsigned int adj[number_of_vertices];
    for (int i = 0; i < number_of_vertices; ++i) {
        adj[i] = 0;
    }

    int current_vertex = 0;
    for (int i = 0; i < code_length; ++i) {
        if (code[i] == 0) {
            ++current_vertex;
        } else {
            AddEdge(graph, adj, current_vertex, code[i] - 1);
        }
    }

    for (int i = 0; i < number_of_vertices; ++i) {
        if (adj[i] != REG) {
            cerr << "Error: graph is not cubic" << endl;
            exit(1);
        }
    }
    return true;
}

bool DecodeGraph6(FILE* input, int& number_of_vertices, int& number_of_edges, GRAPH graph) {
    // TODO: implement
    return false;
}

void ReadGraphs(const function<bool()>& experiment, int number_of_graphs_to_skip, int& number_of_vertices, int& number_of_edges, GRAPH graph) {
    unsigned long long int number_of_graphs_read = 0;
    unsigned long long int number_of_graphs_with_failed_experiment = 0;
    while (DecodeNextGraphInMulticode(stdin, number_of_vertices, number_of_edges, graph)) {
        ++number_of_graphs_read;
        if (number_of_graphs_to_skip >= number_of_graphs_read) {
            continue;
        }
        cerr << "g" << number_of_graphs_read << "\t";

        if (!experiment()) {
            ++number_of_graphs_with_failed_experiment;
        }
    }

    cerr << "Fin" << endl;
    cerr << "Read " << number_of_graphs_read << " graphs" << endl;
    cerr << "Found " << number_of_graphs_with_failed_experiment << " graphs with failed experiment" << endl;
}

void ParseArgs(int argc, char** argv, int& number_of_graphs_to_skip) {
    if (argc > 2) {
        cerr << "Error: invalid number of arguments" << endl;
        cerr << "Usage: " << argv[0] << " [optional: <number_of_graphs_to_skip>]" << endl;
        exit(1);
    } else {
        if (argc >= 2) {
            number_of_graphs_to_skip = atoi(argv[1]);
        }
    }
}

void PrintGraph(GRAPH graph, int number_of_vertices) {
    cerr << "Printing graph:" << endl;
    for (int i = 0; i < number_of_vertices; ++i) {
        cerr << i << " :";
        for (int j = 0; j < REG; ++j) {
            cerr << " " << graph[i][j];
        }
        cerr << endl;
    }
}
