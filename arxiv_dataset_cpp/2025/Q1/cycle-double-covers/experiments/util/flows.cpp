/*
 * File:   flows.cpp
 * Author: Nikolay Ulyanov (ulyanick@gmail.com)
 *
 * TODO:
 *
 */

#include "flows.h"
#include "graph.h"

#include <vector>

using namespace std;


namespace UtilFlows {

/*********************************Methods*********************************/

Mask build_full_cycle_from_nz5_flow(Graph& graph, const vector<int>& f) {
    bool edge_in_nz5_cycle[MAX_EDGE];
    for (int e = 0; e < graph.number_of_edges; ++e) {
        edge_in_nz5_cycle[e] = false;
    }
    for (int v = 0; v < graph.number_of_vertices; ++v) {
        int max_flow = 0;
        int edge_with_max_flow = 0;
        for (int j = 0; j < MAX_DEG; ++j) {
            if (abs(f[graph.v2e[v][j]]) > max_flow) {
                max_flow = abs(f[graph.v2e[v][j]]);
                edge_with_max_flow = graph.v2e[v][j];
            }
        }
        edge_in_nz5_cycle[edge_with_max_flow] = true;
    }

    for (int v = 0; v < graph.number_of_vertices; ++v) {
        bool all_max = true;
        for (int j = 0; j < MAX_DEG; ++j) {
            if (!edge_in_nz5_cycle[graph.v2e[v][j]]) {
                all_max = false;
                break;
            }
        }
        if (all_max) {
            return 0;
        }
    }

    for (int v = 0; v < graph.number_of_vertices; ++v) {
        bool all_max = true;
        for (int j = 0; j < MAX_DEG; ++j) {
            if (!edge_in_nz5_cycle[graph.v2e[v][j]]) {
                all_max = false;
                break;
            }
        }
        if (all_max) {
            return 0;
        }
    }

    bool edge_in_pm[MAX_EDGE];
    for (int e = 0; e < graph.number_of_edges; ++e) {
        edge_in_pm[e] = false;
    }
    bool found_new_edge_info = true;
    while (found_new_edge_info) {
        found_new_edge_info = false;
        for (int v = 0; v < graph.number_of_vertices; ++v) {
            int cycle_edges = 0;
            int pm_edges = 0;
            for (int j = 0; j < MAX_DEG; ++j) {
                if (edge_in_nz5_cycle[graph.v2e[v][j]]) {
                    ++cycle_edges;
                }
                if (edge_in_pm[graph.v2e[v][j]]) {
                    ++pm_edges;
                }
            }
            if (cycle_edges == 2) {
                for (int j = 0; j < MAX_DEG; ++j) {
                    if (!edge_in_nz5_cycle[graph.v2e[v][j]] && !edge_in_pm[graph.v2e[v][j]]) {
                        found_new_edge_info = true;
                        edge_in_pm[graph.v2e[v][j]] = true;
                    }
                }
            }
            if (cycle_edges == 1 && pm_edges == 1) {
                for (int j = 0; j < MAX_DEG; ++j) {
                    if (!edge_in_nz5_cycle[graph.v2e[v][j]] && !edge_in_pm[graph.v2e[v][j]]) {
                        found_new_edge_info = true;
                        edge_in_nz5_cycle[graph.v2e[v][j]] = true;
                    }
                }
            }
            if (pm_edges == 2) {
                return 0;
            }
        }
    }

    int cycle_edges = 0;
    Mask cycle_mask = 0;
    for (int e = 0; e < graph.number_of_edges; ++e) {
        if (edge_in_nz5_cycle[e]) {
            ++cycle_edges;
            cycle_mask += BIT(e);
        }
    }
    return cycle_mask;
}

bool check_3flow(Graph& graph, const Mask c) {
    int deg[MAX_VERTEX];
    for (int v = 0; v < graph.number_of_vertices; ++v) {
        deg[v] = 0;
    }
    for (int e = 0; e < graph.number_of_edges; ++e) {
        if ((BIT(e) & c) > 0) {
            for (int j = 0; j < 2; ++j) {
                ++deg[graph.e2v[e][j]];
            }
        }
    }

    bool vertex_in_cur_part[MAX_VERTEX];
    bool edge_in_cur_part[MAX_EDGE];
    int vertex_colour[MAX_VERTEX];
    for (int v = 0; v < graph.number_of_vertices; ++v) {
        vertex_colour[v] = -1;
        vertex_in_cur_part[v] = false;
        if (deg[v] == 3) {
            vertex_in_cur_part[v] = true;
        }
    }

    for (int e = 0; e < graph.number_of_edges; ++e) {
        edge_in_cur_part[e] = ((BIT(e) & c) > 0);
    }

    bool edge_visited[MAX_EDGE];
    for (int e = 0; e < graph.number_of_edges; ++e) {
        edge_visited[e] = !edge_in_cur_part[e];
    }

    int queue[MAX_VERTEX];
    int queue_size;
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
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

} // UtilFlows
