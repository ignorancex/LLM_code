/*
 * File:   petersen_colouring.cpp
 * Author: Nikolay Ulyanov (ulyanick@gmail.com)
 *
 */

#include "graph.h"
#include "petersen_colouring.h"

#include <cassert>
#include <unordered_set>
#include <unordered_map>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>

using namespace std;


namespace ExpPetersenColouring {

bool edge_is_poor[MAX_EDGE];
// TODO: add edge_is_rich
bool petersen_vertex_is_poor[10]; // vertex is poor when at least one of images of petersen vertices is connected to itself
int edge_colour[MAX_EDGE];
int uncoloured_edges_count_near_vertex[MAX_VERTEX];
bool had_four = false;

// cc-mapping
unordered_map<int, int> ve_to_petersen_edge;
unordered_map<int, int> mask_to_petersen_vertex;

bool petersen_always_poor[MAX_EDGE];
bool petersen_always_rich[MAX_EDGE];
int petersen_min_poor;
int petersen_max_poor;

int strong_vertex_partition[MAX_VERTEX];


/*********************************Methods*********************************/

bool petersen_colouring_check_edge(int e, const Graph& graph) {
    int v1 = graph.e2v[e][0];
    int v2 = graph.e2v[e][1];
    if (uncoloured_edges_count_near_vertex[v1] != 0 || uncoloured_edges_count_near_vertex[v2] != 0) {
        return true;
    }
    unordered_set<int> colours;
    for (int j = 0; j < MAX_DEG; ++j) {
        int e2 = graph.v2e[v1][j];
        colours.insert(edge_colour[e2]);
    }
    for (int j = 0; j < MAX_DEG; ++j) {
        int e2 = graph.v2e[v2][j];
        colours.insert(edge_colour[e2]);
    }
    if (colours.size() == 3) {
        edge_is_poor[e] = true;
        return true;
    } else if (colours.size() == 5) {
        edge_is_poor[e] = false;
        return true;
    }
    return false;
}

bool normal_colour_edges_last_step(Graph& graph) {
    vector<int> colours;
    for (int e = 0; e < graph.number_of_edges; ++e) {
        colours.push_back(edge_colour[e]);
    }
    // TODO: add sanity check,
    // that poor and rich edges have exactly the properties we ask of them

    string prof;
    int poor_count = 0;
    unordered_map<int, pair<int, int>> poor_masks;
    vector<int> petersen_edges;
    for (int v = 0; v < 10; ++v) {
        petersen_vertex_is_poor[v] = false;
    }
    // TODO: put under debug file, or remove cerr code
    //cerr << "petersen edges: ";
    for (int e = 0; e < graph.number_of_edges; ++e) {
        if (edge_is_poor[e]) {
            ++poor_count;
            //cerr << "POOR: ";
            //cerr << "0 ";
            prof += "p";
            petersen_always_rich[e] = false;
        } else {
            //cerr << "rich: ";
            //cerr << "1 ";
            prof += "r";
            petersen_always_poor[e] = false;
        }

        //cerr << graph.e2v[e][0] << " (";
        int mask1 = 0;
        for (int j = 0; j < MAX_DEG; ++j) {
            //cerr << edge_colour[graph.v2e[graph.e2v[e][0]][j]];
            mask1 += BIT(edge_colour[graph.v2e[graph.e2v[e][0]][j]]);
        }
        //cerr << ") -" << edge_colour[e] << "> " << graph.e2v[e][1] << " (";
        int mask2 = 0;
        for (int j = 0; j < MAX_DEG; ++j) {
            //cerr << edge_colour[graph.v2e[graph.e2v[e][1]][j]];
            mask2 += BIT(edge_colour[graph.v2e[graph.e2v[e][1]][j]]);
        }
        if (mask1 == mask2) {
            petersen_vertex_is_poor[mask_to_petersen_vertex[mask1]] = true;
        }
        //cerr << "); ";
        int petersen_edge = ve_to_petersen_edge[combine_hash_from_mask_and_edge_colour(mask1, edge_colour[e])];
        if (edge_is_poor[e]) {
            poor_masks[petersen_edge] = make_pair(mask1, edge_colour[e]);
        }
        petersen_edges.push_back(petersen_edge);
    }
    //cerr << endl;
    petersen_min_poor = min(petersen_min_poor, poor_count);
    petersen_max_poor = max(petersen_max_poor, poor_count);

    // TODO: recheck the experiment, or remove this code
    // bool dominating = false;
    // for (int v = 0; v < 10; ++v) {
    //     if (!petersen_vertex_is_poor[v]) {
    //         //cerr << "can dominate with " << v << endl;
    //         dominating = true;
    //         break;
    //     }
    // }
    // if (!dominating) {
    //     //cerr << "no domination!" << endl;
    // }

    // check that colouring is strong
    // aka, poor edges form a cutset
    bool is_strong = true;
    for (int v = 0; v < graph.number_of_vertices; ++v) {
        strong_vertex_partition[v] = 0;
    }
    {
        int e = graph.faster_edge_order[0];
        int v = min(graph.e2v[e][0], graph.e2v[e][1]);
        strong_vertex_partition[v] = 1;
    }
    for (int i = 0; i < graph.number_of_edges; ++i) {
        int e = graph.faster_edge_order[i];
        int v1 = graph.e2v[e][0];
        int v2 = graph.e2v[e][1];
        if (strong_vertex_partition[v1] == 0) {
        swap(v1, v2);
        }
        assert(strong_vertex_partition[v1] != 0);
        int v2_partition = edge_is_poor[e] ? (3 - strong_vertex_partition[v1]) : strong_vertex_partition[v1];
        if ((strong_vertex_partition[v2] != 0) && (strong_vertex_partition[v2] != v2_partition)) {
        is_strong = false;
        break;
        }
        strong_vertex_partition[v2] = v2_partition;
    }

    graph.normal5_colourings.push_back(colours);
    graph.petersen_colourings.push_back(petersen_edges);
    graph.petersen_profiles.push_back(prof);
    graph.petersen_poor_counts.push_back(poor_count);
    graph.petersen_is_strong.push_back(is_strong);
    ++graph.profiles[prof];
    graph.petersen_colourings_by_profiles[prof].push_back(graph.petersen_colourings.size() - 1);
    if (is_strong) {
        graph.has_strong_petersen_colouring = true;
        cerr << "strong petersen colouring poor count: " << poor_count << endl;
    }
    //cerr << endl;
    //cerr << "#poor: " << poor_count << "; " << "unique poor: " << poor_masks.size() << endl;
    // for (const auto& pm_pair : poor_masks) {
    //     //cerr << "(";
    //     for (int i = 1; i <= 5; ++i) {
    //         if (BIT(i) & pm_pair.second.first) {
    //             //cerr << i;
    //         }
    //     }
    //     //cerr << " " << pm_pair.second.second << "), ";
    // }
    //cerr << endl;
    //cerr << endl;
    return false;
}

bool normal_colour_edges(int cur_edge_idx, Graph& graph) {
    if (cur_edge_idx == graph.number_of_edges) {
        return normal_colour_edges_last_step(graph);
    }

    int cur_edge = graph.faster_edge_order[cur_edge_idx];
    bool had_colour = (edge_colour[cur_edge] != 0);
    int low_colour = 1;
    int high_colour = 5;
    if (had_colour) {
        low_colour = high_colour = edge_colour[cur_edge];
    }

    // initialization
    if (!had_colour) {
        // we add a new coloured edge, so we need to decrease uncoloured counts
        for (const auto& v : graph.e2v[cur_edge]) {
            --uncoloured_edges_count_near_vertex[v];
        }
    }
    for (int i = low_colour; i <= high_colour; ++i) {
        // more initialization
        bool had_four_backup = had_four;
        if (i == 4) {
            had_four = true;
        }

        edge_colour[cur_edge] = i;

        // conditions
        bool checks_passed = true;
        for (const auto& v : graph.e2v[cur_edge]) {
            for (int j = 0; j < MAX_DEG; ++j) {
                int ei = graph.v2e[v][j];
                if ((ei != cur_edge && edge_colour[ei] == edge_colour[cur_edge]) ||
                        (uncoloured_edges_count_near_vertex[v] == 0 && !petersen_colouring_check_edge(ei, graph))) {
                    checks_passed = false;
                    break;
                }
            }
            if (!checks_passed) {
                break;
            }
        }

        // recursion
        if (checks_passed && normal_colour_edges(cur_edge_idx + 1, graph)) {
            return true;
        }

        // undo after recursion
        had_four = had_four_backup;
        if (!had_four_backup && i == 4) {
            break;
            // NOTE the idea here
            // is that we can have following situation:
            // had_four_backup = False, i == 4, we are still in the loop
            // we are going to try now colour 5
            // but this is redundant, because we get same results when replacing colour 4 with 5
        }
    }
    // more undo after brute-forcing
    if (!had_colour) {
        for (const auto& v : graph.e2v[cur_edge]) {
            ++uncoloured_edges_count_near_vertex[v];
        }
        edge_colour[cur_edge] = 0;
    }
    return false;
}

void find_all_petersen_colourings(Graph& graph) {
    graph.profiles.clear();
    graph.has_strong_petersen_colouring = false;

    for (int e = 0; e < graph.number_of_edges; ++e) {
        edge_colour[e] = 0;
        edge_is_poor[e] = false;
        petersen_always_poor[e] = true;
        petersen_always_rich[e] = true;
    }
    petersen_min_poor = graph.number_of_edges;
    petersen_max_poor = 0;

    for (int v = 0; v < graph.number_of_vertices; ++v) {
        uncoloured_edges_count_near_vertex[v] = MAX_DEG;
    }

    // colour edges around vertex 0
    for (int j = 0; j < MAX_DEG; ++j) {
        edge_colour[graph.v2e[0][j]] = j + 1;
        --uncoloured_edges_count_near_vertex[0];
        --uncoloured_edges_count_near_vertex[graph.v2v[0][j]];
    }
    had_four = false;
    normal_colour_edges(0, graph);

    graph.print();
    cerr << "number of petersen colourings: " << graph.petersen_colourings.size() << "; number of profiles: " << graph.profiles.size() << endl;
    cerr << "colourings grouped by profiles:" << endl;
    for (const auto& prof : graph.petersen_colourings_by_profiles) {
      cerr << prof.first << "; ";
      int rich_count = 0;
      int poor_count = 0;
      for (int e = 0; e < graph.number_of_edges; ++e) {
        if (prof.first[e] == 'r') {
            rich_count++;
        } else {
            poor_count++;
        }
      }
      cerr << "rich: " << rich_count << "; ";
      cerr << "poor: " << poor_count << "; ";
      for (const auto& colouring_idx : prof.second) {
        cerr << colouring_idx << " ";
      }
      cerr << endl;
    }
    for (int col_idx = 0; col_idx < graph.petersen_colourings.size(); ++col_idx) {
        cerr << "g" << graph.number << ": ";
        for (int e = 0; e < graph.number_of_edges; ++e) {
            cerr << e << ": " << graph.petersen_colourings[col_idx][e] << "; ";
        }
        // TODO/FIXME: currently this is (almost) useless
        // bool is_circular = true;
        // for (int v = 0; v < graph.number_of_vertices; ++v) {
        //     int rich_count = 0;
        //     for (int j = 0; j < MAX_DEG; ++j) {
        //         int e = graph.v2e[v][j];
        //         if (graph.petersen_profiles[col_idx][e] == 'r') {
        //             rich_count++;
        //         }
        //     }
        //     if (rich_count == 2) {
        //         is_circular = false;
        //         break;
        //     }
        // }
        // if (is_circular) {
        //     cerr << "circular!";
        // }
        cerr << endl;
    }
}

// FIXME: fix number of stars here
/********************************cc-mapping********************************/

void create_cc_mapping(const Graph& petersen_graph) {
    for (int v = 0; v < petersen_graph.number_of_vertices; ++v) {
        int mask = 0;
        for (int j = 0; j < MAX_DEG; ++j) {
            int e = petersen_graph.v2e[v][j];
            mask += BIT(petersen_graph.normal5_colourings[0][e]);
        }
        mask_to_petersen_vertex[mask] = v;
        for (int j = 0; j < MAX_DEG; ++j) {
            int e = petersen_graph.v2e[v][j];
            ve_to_petersen_edge[combine_hash_from_mask_and_edge_colour(mask, petersen_graph.normal5_colourings[0][e])] = e;
        }
    }
    cerr << "number of mask/edge_colour pairs: " << ve_to_petersen_edge.size() << endl;
}

} // ExpPetersenColouring
