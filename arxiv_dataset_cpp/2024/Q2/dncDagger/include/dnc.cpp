#include <cassert>
#include <chrono>
#include <iostream>
#include <cmath>
#include <RInside.h>
#include <Rcpp.h>
#include "OrderScoring.h"
#include "opruner_right.h"
#include "auxiliary.h"
#include "dnc.h"

using namespace std;
using namespace boost;

namespace boost
{
    void renumber_vertex_indices(DiGraph const &) {}
}

struct Vis
{
public:
    vector<vector<int>> &cycles;

    Vis(vector<vector<int>> &cycles) : cycles(cycles) {}

    void cycle(auto const &path, DiGraph const &g)
    {
        auto indices = get(vertex_index, g);
        // store the cycle
        vector<int> cycle;
        for (auto v : path)
        {
            cycle.push_back(get(indices, v));
        }
        cycles.push_back(cycle);

        // for (auto v : path)
        //     cout << get(indices, v) << " ";

        // cout << "\n";
    };
};

/**
 * Function that converts a matrix to a boost graph
 */
template <typename U, typename T>
U matrix_to_graph(const vector<vector<T>> &matrix)
{
    size_t p = matrix.size();
    U G(p);
    for (size_t i = 0; i < matrix.size(); i++)
    {
        for (size_t j = 0; j < matrix[i].size(); j++)
        {
            if (matrix[i][j])
            {
                add_edge(i, j, G);
            }
        }
    }
    return (G);
}

/*  Divide and conquer algorithm.
    Returns the optimal DAG.
*/
std::tuple<vector<int>, double, size_t, size_t> dnc(OrderScoring &scoring)
{

    size_t p = scoring.numparents.size();
    vector<vector<bool>> h_max_adj(p, vector<bool>(p));
    vector<vector<bool>> h_min_adj(p, vector<bool>(p));

    // print H_min_adj and H_max_adj
    tie(h_min_adj, h_max_adj) = get_diff_matrices(scoring);

    Graph G_H_min = matrix_to_graph<Graph, bool>(h_min_adj);
    Graph G_H_max = matrix_to_graph<Graph, bool>(h_max_adj);

    IsoComps iso_comps = structure_components(G_H_min, G_H_max);

    // Go through the isocomps and update the subcomponents
    // accoring to their dependence.
    // iterate over isolated components
    for (auto &iso_comp : iso_comps.iso_comps)
    {
        if (iso_comp.connected)
            continue; // Just one subcomponent

        int round = 1;
        bool new_cubcomponents_created = true;
        while (new_cubcomponents_created)
        {
            // cout << "*************** Round: " << round++ << endl;
            subcomponents_update(iso_comp, scoring);
            new_cubcomponents_created = restructure_components(iso_comp, scoring);
        }
    }

    // Create the final DAG and order by, for each isocomp,
    // joining the subcomponents accorinng to a topological order of their dependencies
    // size_t p = scoring.numparents.size();
    vector<vector<bool>> adjmat_full(p, vector<bool>(p, 0));
    vector<int> order_full;
    vector<int> sub_order;
    for (auto &iso_comp : iso_comps.iso_comps)
    {
        sub_order = concatenate_subcomponents(iso_comp, scoring);
        order_full.insert(order_full.end(), sub_order.begin(), sub_order.end());
    }

    // copy the columns opt_adjmat for each subcomponent, corresponding to
    // the suborder, to the full_adjmat

    for (auto &iso_comp : iso_comps.iso_comps)
    {
        for (auto &subcomp : iso_comp.subcomps)
        {
            for (size_t i = 0; i < subcomp.opt_adjmat.size(); i++)
            {
                for (auto &node : subcomp.nodes)
                {
                    adjmat_full[i][node] = subcomp.opt_adjmat[i][node];
                }
            }
        }
    }

    // Sum the scores of the subcomponents and get the particle counts
    // for the isolated components

    double score = 0;
    size_t max_n_particles = 0;
    size_t tot_n_particles = 0;
    for (auto &iso_comp : iso_comps.iso_comps)
    {
        max_n_particles = max(max_n_particles, iso_comp.max_n_particles);
        tot_n_particles += iso_comp.tot_n_particles;
        for (auto &subcomp : iso_comp.subcomps)
        {
            score += subcomp.score;
        }
    }

    return (make_tuple(order_full, score, max_n_particles, tot_n_particles));
}

/**
 * Concatenate the subcomponents of an isloated component
 * accorinng to a topological order of their dependencies.
 */
vector<int> concatenate_subcomponents(const IsoComp &iso_comp, OrderScoring &scoring)
{
    size_t p = iso_comp.nodes.size();
    vector<int> order;
    vector<vector<bool>> comp_dep = subcomponents_dependence(iso_comp, scoring);
    DiGraph D = matrix_to_graph<DiGraph, bool>(comp_dep);

    typedef graph_traits<DiGraph>::vertex_descriptor Vertex;
    typedef vector<Vertex> container;
    container c;

    topological_sort(D, back_inserter(c));
    // print the topological order
    // cout << "Topological comp order: " << endl;

    // for ( container::reverse_iterator ii=c.rbegin(); ii!=c.rend(); ++ii){
    //     cout << *ii << " ";
    //     // print the suborder in the subcomponent
    //     cout << "Suborder: ";
    //     for (size_t j = 0; j < iso_comp.subcomps[*ii].nodes.size(); j++)
    //     {
    //         cout << iso_comp.subcomps[*ii].suborder[j] << " ";
    //     }
    //     cout << endl;
    // }

    // cout << endl;

    // Join the nodes of the suborders in the topological order.
    for (auto &comp_id : c)
    {
        const SubComp &subcomp = iso_comp.subcomps[comp_id];
        // insert the nodes in the suborder in the order vector
        order.insert(order.end(), subcomp.suborder.begin(), subcomp.suborder.end());
    }

    // // print the order
    // cout << "Suborder: " << endl;
    // for (size_t i = 0; i < order.size(); i++)
    // {
    //     cout << order[i] << " ";
    // }
    // cout << endl;

    return (order);
}

/**
 * Merge components that are dependent on each other in a cycle and neigboring cycles.
 *
 */
vector<size_t> merged_neig_cycles(const vector<vector<bool>> &compdep)
{
    // Turn the compdep into a diected Boost graph
    size_t p = compdep.size();
    DiGraph D(p);
    for (size_t i = 0; i < compdep.size(); i++)
    {
        for (size_t j = 0; j < compdep[i].size(); j++)
        {
            if (compdep[i][j])
            {
                add_edge(i, j, D);
            }
        }
    }

    // Allocate new vector of cycles on heap
    vector<vector<int>> cycles;

    Vis vis(cycles); // BUG: could give segfault?

    // Find the cycles in D
    tiernan_all_cycles(D, vis);

    // ///print vis cycles
    // cout << "Cycles: " << endl;
    // for (size_t i = 0; i < vis.cycles.size(); i++)
    // {
    //     for (size_t j = 0; j < vis.cycles[i].size(); j++)
    //     {
    //         cout << vis.cycles[i][j] << " ";
    //     }
    //     cout << endl;
    // }

    // Create undirected graph with edges between all nodes in each of the cycles
    Graph G(p);
    for (size_t i = 0; i < vis.cycles.size(); i++)
    {
        for (size_t j = 0; j < vis.cycles[i].size(); j++)
        {
            for (size_t k = 0; k < vis.cycles[i].size(); k++)
            {
                if (j != k)
                {
                    add_edge(vis.cycles[i][j], vis.cycles[i][k], G);
                }
            }
        }
    }

    // Find the connected components of G
    vector<size_t> component(num_vertices(G));
    int num = connected_components(G, &component[0]);
    // print component vector
    // cout << "Number of components: " << num << endl;
    // cout << "Component vector: ";
    // for (size_t i = 0; i < component.size(); i++)
    // {
    //     cout << component[i] << " ";
    // }
    // cout << endl;

    return (component);
}

/**
 * Restructure the components of an isolated component.
 */
bool restructure_components(IsoComp &iso_comp, OrderScoring &scoring)
{
    // 1. Get the component dependence graph
    // subcomponents_update(iso_comp, scoring);

    // Get the component dependence matrix
    vector<vector<bool>> comp_dep = subcomponents_dependence(iso_comp, scoring);
    // print_matrix(comp_dep);
    //  2.
    //  Merge components that are dependent on each other in a cycle and neigboring cycles.
    //  This is for the graph of components. So its the new components membership after merging the cycles.
    vector<size_t> membership_comp = merged_neig_cycles(comp_dep);
    size_t n_comp = *max_element(membership_comp.begin(), membership_comp.end()) + 1;

    // if (n_comp == iso_comp.subcomps.size()) {
    if (n_comp == membership_comp.size())
    {
        // No cycles, just return
        // cout << "No cycles found." << endl;
        return false;
    }

    vector<SubComp> new_subcomps;
    // Go through and create the new components
    // merge some of them and keep some of them.

    for (size_t i = 0; i < n_comp; i++)
    {
        // find all the components that are merged into this component
        vector<int> comps_to_merge;
        for (size_t j = 0; j < membership_comp.size(); j++)
        {
            if (membership_comp[j] == i)
            {
                comps_to_merge.push_back(j);
            }
        }

        //  If only one, dont merge, use the same as before (a copy)
        if (comps_to_merge.size() == 1)
        {
            new_subcomps.push_back(iso_comp.subcomps[comps_to_merge[0]]);
        }
        else
        {
            // Merge them if more then one

            // Create a new subcomponent
            SubComp new_subcomp;

            // Add all the nodes from the subcomponents to the new subcomponent
            for (size_t j = 0; j < comps_to_merge.size(); j++)
            {
                for (size_t k = 0; k < iso_comp.subcomps[comps_to_merge[j]].nodes.size(); k++)
                {
                    new_subcomp.nodes.push_back(iso_comp.subcomps[comps_to_merge[j]].nodes[k]);
                }
            }

            // Add the new subcomponent to the new subcomponents
            new_subcomps.push_back(new_subcomp);
        }
    }

    // 3.
    // Update the isolate component

    iso_comp.subcomps = new_subcomps;

    return (true);
}

/**
 * Translates the membership vector of the original graph to
 * the new merged components.
 */

vector<int> merged_components_membership(const vector<int> &membership, const vector<int> &membership_comp)
{
    size_t p = membership.size();
    vector<int> ultimate_membership(p, 0);
    for (size_t i = 0; i < p; i++)
    {
        int original_component = membership[i];
        int merged_component = membership_comp[original_component];
        ultimate_membership[i] = merged_component;
    }
    return (ultimate_membership);
}

/**
 * Update the subcomponents of an isolated component.
 */
void subcomponents_update(IsoComp &iso_comp, OrderScoring &scoring)
{

    size_t p = scoring.numparents.size();
    // create a matrix, comp_dep, of component dependencies

    // iterate over the subcomponents
    for (size_t i = 0; i < iso_comp.subcomps.size(); i++)
    {
        SubComp &subcomp = iso_comp.subcomps[i];

        // If this subcomponent hasnt been touched before
        if (subcomp.score == 0)
        {

            // find the rest of the nodes, not in subcomp.nodes
            vector<int> restnodes;
            for (size_t j = 0; j < p; j++)
            {
                // add the node to restnodes if it is not in subcomp.nodes
                if (find(subcomp.nodes.begin(), subcomp.nodes.end(), j) == subcomp.nodes.end())
                {
                    restnodes.push_back(j);
                }
            }

            // run opruner_right conditioned on restnodes (initial nodes)
            const auto &[order, log_score, suborder, suborder_log_score, node_scores, max_n_particles, tot_n_particles] = opruner_right(scoring, restnodes);
            // find the correspodning DAG
            vector<vector<bool>> adjmat = order_to_dag(order, scoring);

            // updated the isocomp and subcomp with the new information
            subcomp.suborder = suborder; // is this the suborder?
            subcomp.score = suborder_log_score;
            // subcomp.tot_n_particles = tot_n_particles;
            // subcomp.max_n_particles = max_n_particles;
            subcomp.opt_adjmat = adjmat;

            // update the isolated component
            iso_comp.tot_n_particles += tot_n_particles;
            iso_comp.max_n_particles = max(iso_comp.max_n_particles, max_n_particles);
        }
    }
}

/**
 * Evaluates the subcomponens dependence matrix for an isolated compnent.
 */
vector<vector<bool>> subcomponents_dependence(const IsoComp &iso_comp, OrderScoring &scoring)
{
    size_t n_subcomps = iso_comp.subcomps.size();
    vector<vector<bool>> comp_dep = vector<vector<bool>>(n_subcomps, vector<bool>(n_subcomps, 0));

    // iterate over the subcomponents
    for (size_t i = 0; i < iso_comp.subcomps.size(); i++)
    {
        const SubComp &subcomp = iso_comp.subcomps[i];
        // iterate over the subcomponents
        for (size_t j = 0; j < iso_comp.subcomps.size(); j++)
        {
            const SubComp &subcomp2 = iso_comp.subcomps[j];

            if (i == j)
                continue; // same subcomponent

            // check if the are any edges between nodes in the subcomponents
            // with id i and j in subcomp.opt_adjmat

            for (size_t k = 0; k < subcomp.nodes.size(); k++)
            {
                for (size_t l = 0; l < subcomp2.nodes.size(); l++)
                {
                    if (subcomp.opt_adjmat[subcomp2.nodes[l]][subcomp.nodes[k]] == 1)
                    {
                        comp_dep[j][i] = 1;
                        continue;
                    }
                }
            }
        }
    }
    return (comp_dep);
}

void printIsoComps(IsoComps &iso_comps)
{
    // iterate over isolated components and print their content
    for (size_t i = 0; i < iso_comps.iso_comps.size(); i++)
    {
        IsoComp &iso_comp = iso_comps.iso_comps[i];
        cout << "Isolated component " << i << " has " << iso_comp.nodes.size() << " nodes and " << iso_comp.subcomps.size() << " subcomponents." << endl;
        for (size_t j = 0; j < iso_comp.subcomps.size(); j++)
        {
            SubComp &subcomp = iso_comp.subcomps[j];
            cout << "Subcomponent " << j << " has " << subcomp.nodes.size() << " nodes." << endl;
            // pring the nodes
            cout << "Nodes: ";
            for (size_t k = 0; k < subcomp.nodes.size(); k++)
            {
                cout << subcomp.nodes[k] << " ";
            }
            cout << endl;
            // print all the other info of the subcomponent
            cout << "Suborder: ";
            for (size_t k = 0; k < subcomp.suborder.size(); k++)
            {
                cout << subcomp.suborder[k] << " ";
            }
            cout << endl;
            cout << "Score: " << subcomp.score << endl;
            cout << endl;
        }
    }
}

IsoComps structure_components(Graph &G_H_min, Graph &G_H_max)
{
    // get components of G_H_min and G_H_max
    vector<int> component_sub(num_vertices(G_H_min));
    int num_sub = connected_components(G_H_min, &component_sub[0]);
    // print the compnent vector
    // cout << "Number of subcomponents: " << num_sub << endl; // "Number of components:
    // cout << "SubComponent vector: ";
    // for (size_t i = 0; i < component_sub.size(); i++)
    // {
    //     cout << component_sub[i] << " ";
    // }
    // cout << endl;

    vector<int> component_max(num_vertices(G_H_max));
    int num_max = connected_components(G_H_max, &component_max[0]);
    // get isolated components
    IsoComps iso_comps;
    iso_comps.iso_comps = vector<IsoComp>(num_max);

    for (int i = 0; i < num_max; i++)
    {
        // Find all the nodes in this isoloated component
        vector<int> subnodes;
        for (size_t j = 0; j < num_vertices(G_H_max); j++)
        {
            if (component_max[j] == i)
            {
                subnodes.push_back(j);
            }
        }

        IsoComp iso_comp;
        iso_comp.nodes = subnodes;
        iso_comp.subcompids = vector<int>();
        iso_comp.scores = vector<double>();
        iso_comp.subcomps = vector<SubComp>();

        iso_comps.iso_comps[i] = iso_comp;
    }

    // add sub components to the right isolated components
    // iterate over subcomp ids
    for (int comp_id = 0; comp_id < num_sub; comp_id++)
    {
        // store all the nodes in sub component i in a vector called subnodes
        vector<int> subnodes;
        for (size_t j = 0; j < num_vertices(G_H_min); j++)
        {
            if (component_sub[j] == comp_id)
            {
                subnodes.push_back(j);
            }
        }

        SubComp subcomp;
        subcomp.nodes = subnodes;

        // Now get the Isolated component that this is a subcomponent of
        // Just take the first node in the subcomponent and get the isolated component
        IsoComp &iso_comp = iso_comps.iso_comps[component_max[subnodes[0]]];

        // // print the nodes in the subcomponent
        // cout << "Subcomponent " << comp_id << " has " << subnodes.size() << " nodes." << endl;
        // cout << "Nodes: ";
        // for (size_t j = 0; j < subnodes.size(); j++)
        // {
        //     cout << subnodes[j] << " ";
        // }
        // cout << endl;
        // // print nodes in isocomp
        // cout << "Nodes in isocomp: ";
        // for (size_t j = 0; j < iso_comp.nodes.size(); j++)
        // {
        //     cout << iso_comp.nodes[j] << " ";
        // }
        // cout << endl;

        // Add the subcomponent to the isolated component
        iso_comp.subcomps.push_back(subcomp);
        iso_comp.subcompids.push_back(comp_id);
    }

    return (iso_comps);
}

/**
 * Merged components dependencies.
 */
vector<vector<bool>> merged_component_dependencies(const vector<vector<bool>> &compdep, const vector<int> &membership_comp)
{
    // create zero matrix called merged_components_adjmat
    size_t p = compdep.size();
    vector<vector<bool>> merged_components_adjmat(p, vector<bool>(p, 0));
    // go through the compdep and update the merged_components_adjmat
    // i.e. create a matrix of dependencies for the new, merged comopnents

    for (size_t i = 0; i < compdep.size(); i++)
    {
        for (size_t j = 0; j < compdep.size(); j++)
        {
            if (compdep[i][j] == 0 && compdep[j][i] == 0)
                continue;
            if (i == j)
                continue;
            int merged_i = membership_comp[i];
            int merged_j = membership_comp[j];
            if (merged_i == merged_j)
                continue; // same merged component, ie no edge

            if (compdep[i][j] == 1 && compdep[j][i] == 0)
            {
                merged_components_adjmat[merged_i][merged_j] = 1;
            }

            if (compdep[i][j] == 0 && compdep[j][i] == 1)
            {
                merged_components_adjmat[merged_j][merged_i] = 1;
            }
        }
    }

    return (merged_components_adjmat);
}

pair<vector<vector<bool>>, vector<vector<bool>>> get_diff_matrices(OrderScoring &scores)
{
    // just call the other function
    return get_diff_matrices(scores.rowmaps_backwards, scores.rowmaps_forward, scores.scoretable, scores.potential_parents);
}

pair<vector<vector<bool>>, vector<vector<bool>>> get_diff_matrices(vector<Rcpp::IntegerVector> rowmaps_backwards, vector<Rcpp::IntegerVector> rowmaps_forward, vector<vector<vector<double>>> scoretable, vector<vector<int>> potential_parents)
{

    size_t nvars = rowmaps_backwards.size();

    vector<vector<bool>> H_min_isset(nvars, vector<bool>(nvars, false));
    vector<vector<bool>> H_max_isset(nvars, vector<bool>(nvars, false));

    vector<vector<double>> H_min(nvars, vector<double>(nvars, 0));
    vector<vector<double>> H_max(nvars, vector<double>(nvars, 0));

    for (size_t i = 0; i < nvars; i++)
    {
        // total number of possible parents
        size_t n_pos_parents = potential_parents[i].size();

        // For the possible parents, i.e. not plus1 parents
        // We exclude each parent in turn and see how the score changes
        for (size_t parent_ind = 0; parent_ind < n_pos_parents; parent_ind++)
        {
            // if no possible parents, skip
            if (n_pos_parents == 0)
                continue;                                  // since seq is weird
            int parent = potential_parents[i][parent_ind]; // parent to exclude
            for (auto &hash : rowmaps_forward[i])
            {
                // Check if the parent is in the hash
                int check = hash % (1 << (parent_ind + 1));

                if (check < (1 << parent_ind))
                {
                    // Compute the hash with the parent excluded
                    int hash_with_parent = hash + (1 << (parent_ind));

                    // Using the no plus1 score table, i.e. index 1
                    double score_diff = scoretable[i][0][rowmaps_backwards[i][hash_with_parent]] - scoretable[i][0][rowmaps_backwards[i][hash]];

                    // Update the H matrices
                    if (!H_max_isset[i][parent])
                    {
                        H_max[i][parent] = score_diff;
                        H_max_isset[i][parent] = true;
                    }
                    else
                    {
                        H_max[i][parent] = max(H_max[i][parent], score_diff);
                    }

                    if (!H_min_isset[i][parent])
                    {
                        H_min[i][parent] = score_diff;
                        H_min_isset[i][parent] = true;
                    }
                    else
                    {
                        H_min[i][parent] = min(H_min[i][parent], score_diff);
                    }
                }
            }
        }

        // For the plus1 parents
        // plus1parents are those parents that are not in the aliases
        vector<int> plus1parents;

        for (size_t j = 0; j < nvars; j++)
        {
            if (j == i)
            {
                continue;
            }
            if (find(potential_parents[i].begin(), potential_parents[i].end(), j) == potential_parents[i].end())
            {
                // label not in aliases so it is a plus1 parent
                plus1parents.push_back(j);
            }
        }

        for (size_t j = 0; j < plus1parents.size(); j++)
        {
            vector<double> score_diffs;

            for (size_t k = 0; k < scoretable[i][0].size(); k++)
            {
                score_diffs.push_back(scoretable[i][j + 1][k] - scoretable[i][0][k]);
            }

            H_max[i][plus1parents[j]] = *max_element(score_diffs.begin(), score_diffs.end());
            H_min[i][plus1parents[j]] = *min_element(score_diffs.begin(), score_diffs.end());
        }
    }
    // Threshold H_min and H_max at 0 and save in new matrices
    // H_min_mat and H_max_mat
    vector<vector<bool>> H_min_mat(nvars, vector<bool>(nvars, false));
    vector<vector<bool>> H_max_mat(nvars, vector<bool>(nvars, false));
    for (size_t i = 0; i < nvars; i++)
    {
        for (size_t j = 0; j < nvars; j++)
        {
            if (i == j)
                continue;
            // should maybe take care of float value precision
            H_min_mat[i][j] = H_min[i][j] > 0;
            H_max_mat[i][j] = H_max[i][j] > 0;
        }
    }

    return make_pair(H_min_mat, H_max_mat);
}

// [[Rcpp::plugins(cpp17)]]

// [[Rcpp::export]]

Rcpp::List r_dnc(Rcpp::List ret)
{
    OrderScoring scoring = get_score(ret);
    const auto &[dnc_order, dnc_score, dnc_max_n_particles, dnc_tot_n_particles] = dnc(scoring);

    vector<int> order(dnc_order.begin(), dnc_order.end());
    // increase order values by 1
    for (size_t i = 0; i < order.size(); i++)
    {
        order[i] += 1;
    }

    Rcpp::List L = Rcpp::List::create(Rcpp::Named("order") = order,
                                      Rcpp::Named("log_score") = dnc_score,
                                      Rcpp::Named("max_n_particles") = dnc_max_n_particles,
                                      Rcpp::Named("tot_n_particles") = dnc_tot_n_particles);
    return(L);
}