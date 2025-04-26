/*
 * File:   three_4_flows.cpp
 * Main result: the conjecture of 233-flows is wrong for Petersen graph (and no 234-flows; but it has 333-flows and 244-flows)
 * Program checks existence of 233-, 333-, 234-, 334- and 244- flows for snarks
 * Actually it looks like 333-flows and 244-flows always exist
 *
 * Author: Nikolay Ulyanov (ulyanick@gmail.com)
 *
 * Created on 10 March 2016, 15:30
 */

#include "util.h"

#include <climits>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <array>

using namespace std;

/********************************** Variables *************************************/

GRAPH graph;
int number_of_vertices;
int number_of_edges;
int number_of_graphs_to_skip = 0;
unsigned int edge_index[MAXN][MAXN];

/* The code is too slow to find any solutions here, but still, here are the main candidates for decompositions into abcd- flows */
// const int number_of_profiles = 1;
// const int number_of_partitions = 4;
// int profiles[number_of_profiles][number_of_partitions] = {{2, 2, 3, 3}};

const int number_of_profiles = 6;
const int number_of_partitions = 3;
int profiles[number_of_profiles][number_of_partitions] = {{2, 3, 3}, {3, 3, 3}, {2, 3, 4}, {3, 3, 4}, {2, 3, 5}, {2, 4, 4}};

int vertices[MAXN * REG / 2][2];
int default_trackback_to = REG * MAXN;
int trackback_to;
int vertex_last_edge_index[MAXN];
int partition_num[MAXN * REG / 2][2];
int partition_degs[MAXN][number_of_partitions];
int number_of_flowed_edges_at_vertex[MAXN];
int edge_flows[MAXN * REG / 2];
int vertex_flows[MAXN];
int max_flow_values[number_of_partitions];

bool has_profile[number_of_profiles];


/********************************** Functions *************************************/

bool BuildFlow(int partition, int edge_index) {
    if (edge_index >= number_of_edges) {
        return true;
    }
    if (partition_num[edge_index][0] == partition) {
        return BuildFlow(partition, edge_index + 1);
    }
    if (number_of_partitions == 4) {
        if (partition_num[edge_index][1] == partition) {
            return BuildFlow(partition, edge_index + 1);
        }
    }

    int right_bound = max_flow_values[partition] - 1;
    int left_bound = -right_bound;

    int v1 = vertices[edge_index][0];
    int v2 = vertices[edge_index][1];

    if (number_of_flowed_edges_at_vertex[v1] == partition_degs[v1][partition] - 1) {
        right_bound = left_bound = vertex_flows[v1];
    }
    if (number_of_flowed_edges_at_vertex[v2] == partition_degs[v2][partition] - 1) {
        right_bound = left_bound = -vertex_flows[v2];
    }

    for (int flow = left_bound; flow <= right_bound; ++flow) {
        if (flow == 0)
            continue;
        edge_flows[edge_index] = flow;

        vertex_flows[v1] -= flow;
        vertex_flows[v2] += flow;
        ++number_of_flowed_edges_at_vertex[v1];
        ++number_of_flowed_edges_at_vertex[v2];

        bool all_is_good = (
            (vertex_flows[v1] == 0 || number_of_flowed_edges_at_vertex[v1] != partition_degs[v1][partition]) &&
            (vertex_flows[v2] == 0 || number_of_flowed_edges_at_vertex[v2] != partition_degs[v2][partition]) &&
            (abs(vertex_flows[v1]) < max_flow_values[partition] || number_of_flowed_edges_at_vertex[v1] != partition_degs[v1][partition] - 1) &&
            (abs(vertex_flows[v2]) < max_flow_values[partition] || number_of_flowed_edges_at_vertex[v2] != partition_degs[v2][partition] - 1)
        );
        if (all_is_good && BuildFlow(partition, edge_index + 1)) {
            return true;
        }

        // undoing
        vertex_flows[v1] += flow;
        vertex_flows[v2] -= flow;
        --number_of_flowed_edges_at_vertex[v1];
        --number_of_flowed_edges_at_vertex[v2];
    }
    return false;
}

bool CheckNowhereZeroness(int partition) {
    for (int e = 0; e < number_of_edges; ++e) {
        edge_flows[e] = 0;
    }
    for (int v = 0; v < number_of_vertices; ++v) {
        vertex_flows[v] = 0;
        number_of_flowed_edges_at_vertex[v] = 0;
    }
    return BuildFlow(partition, 0);
}

bool CheckPartitions() {
    for (int partition = 0; partition < number_of_partitions; ++partition) {
        for (int v = 0; v < number_of_vertices; ++v) {
            partition_degs[v][partition] = REG;
        }
    }
    for (int e = 0; e < number_of_edges; ++e) {
        int partition = partition_num[e][0];
        --partition_degs[vertices[e][0]][partition];
        --partition_degs[vertices[e][1]][partition];
        if (number_of_partitions == 4) {
            int partition2 = partition_num[e][1];
            --partition_degs[vertices[e][0]][partition2];
            --partition_degs[vertices[e][1]][partition2];
        }
    }
    for (int partition = 0; partition < number_of_partitions; ++partition) {
        for (int v = 0; v < number_of_vertices; ++v) {
            if (partition_degs[v][partition] == 1) {
                trackback_to = min(trackback_to, vertex_last_edge_index[v]);
            }
        }
    }

    if (max_flow_values[0] == 2) {
        for (int v = 0; v < number_of_vertices; ++v) {
            if (partition_degs[v][0] == REG) {
                trackback_to = min(trackback_to, vertex_last_edge_index[v]);
            }
        }
    }

    if (trackback_to <= number_of_edges) {
        return false;
    }

    for (int partition = 0; partition < number_of_partitions; ++partition) {
        if (!CheckNowhereZeroness(partition)) {
            return false;
        }
    }
    return true;
}

bool BuildPartitions(int edge_index) {
    if (edge_index >= number_of_edges)
        return CheckPartitions();
    for (int partition = 0; partition < number_of_partitions; ++partition) {
        partition_num[edge_index][0] = partition;
        if (number_of_partitions == 4) {
            for (int partition2 = partition + 1; partition2 < number_of_partitions; ++partition2) {
                partition_num[edge_index][1] = partition2;

                if (BuildPartitions(edge_index + 1)) {
                    return true;
                }
                if (trackback_to < edge_index)
                    return false;
                trackback_to = default_trackback_to;

            }
        } else {
            if (BuildPartitions(edge_index + 1)) {
                return true;
            }
            if (trackback_to < edge_index)
                return false;
            trackback_to = default_trackback_to;
        }
    }
    return false;
}

bool CompareProfiles(int i, int j) {
    for (int partition = 0; partition < number_of_partitions; ++partition) {
        if (profiles[i][partition] < profiles[j][partition]) {
            return false;
        }
    }
    return true;
}

bool HasPartitionedFlows() {
    //Label the edges of the graph
    int edge_label = 0;
    for (int i = 0; i < number_of_vertices; ++i) {
        for (int j = 0; j < REG; ++j) {
            if (i < graph[i][j]) {
                edge_index[i][graph[i][j]] = edge_label;
                edge_index[graph[i][j]][i] = edge_label;
                vertices[edge_label][0] = i;
                vertices[edge_label][1] = graph[i][j];
                vertex_last_edge_index[i] = edge_label;
                vertex_last_edge_index[graph[i][j]] = edge_label;
                ++edge_label;
            }
        }
    }
    if (edge_label != number_of_edges) {
        cerr << "Error: invalid number of edges" << endl;
        exit(1);
    }

    for (int i = 0; i < number_of_profiles; ++i) {
        has_profile[i] = false;
    }

    for (int profile = 0; profile < number_of_profiles; ++profile) {
        for (int i = 0; i < profile; ++i) {
            if (has_profile[i] && CompareProfiles(profile, i)) {
                has_profile[profile] = true;
                break;
            }
        }
        if (has_profile[profile]) {
            for (int partition = 0; partition < number_of_partitions; ++partition) {
                cerr << profiles[profile][partition];
            }
            cerr << ":     ; ";
            continue;
        }

        for (int partition = 0; partition < number_of_partitions; ++partition) {
            max_flow_values[partition] = profiles[profile][partition];
        }
        trackback_to = default_trackback_to;

        for (int partition = 0; partition < number_of_partitions; ++partition) {
            for (int v = 0; v < number_of_vertices; ++v) {
                partition_degs[v][partition] = REG;
            }
        }

        for (int v = 0; v < number_of_vertices; ++v) {
            vertex_flows[v] = 0;
            number_of_flowed_edges_at_vertex[v] = 0;
        }
        for (int e = 0; e < number_of_edges; ++e) {
            edge_flows[e] = 0;
            partition_num[e][0] = -1;
            partition_num[e][1] = -1;
        }

        has_profile[profile] = BuildPartitions(0);
        for (int partition = 0; partition < number_of_partitions; ++partition) {
            cerr << profiles[profile][partition];
        }
        cerr << ": ";
        if (has_profile[profile])
            cerr << "YES!; ";
        else
            cerr << "nope; ";
    }
    cerr << endl;
    return false;
}

int main(int argc, char** argv) {
    ParseArgs(argc, argv, number_of_graphs_to_skip);
    ReadGraphs(HasPartitionedFlows, number_of_graphs_to_skip, number_of_vertices, number_of_edges, graph);
    return(EXIT_SUCCESS);
}
