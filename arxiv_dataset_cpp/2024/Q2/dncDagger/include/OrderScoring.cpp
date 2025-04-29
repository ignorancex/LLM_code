#include <cassert>
#include <chrono>
#include <iostream>
#include <cmath>
#include <RInside.h>
#include <Rcpp.h>
#include "OrderScoring.h"
#include "auxiliary.h"

using namespace std;

OrderScoring::OrderScoring(
    vector<vector<int>> potential_parents,
    vector<int> numparents,
    vector<Rcpp::IntegerVector> rowmaps_backwards,
    vector<Rcpp::IntegerVector> rowmaps_forward,
    vector<vector<int>> potential_plus1_parents,
    vector<vector<vector<double>>> scoretable,
    vector<vector<vector<double>>> scoresmatrices,
    vector<vector<vector<double>>> maxmatrix,
    vector<vector<vector<int>>> maxrow,
    vector<vector<vector<int>>> parenttable,
    bool MAP) : 
                potential_plus1_parents(potential_plus1_parents),
                MAP(MAP),
                potential_parents(potential_parents),
                rowmaps_backwards(rowmaps_backwards),
                rowmaps_forward(rowmaps_forward),                
                numparents(numparents),
                scoretable(scoretable), // this is called bannedscore in R
                scoresmatrices(scoresmatrices),
                maxmatrix(maxmatrix),
                maxrow(maxrow),
                parenttable(parenttable)
{
}

/**
 * Re-calculation scores after swapping up node_a so that (a, b) -> (b, a).
 *
 * Order of node1 is lower than node2.
 *
 * Returns: (nodea_score, nodeb_score)
 */
tuple<double, double> OrderScoring::swap_nodes(int nodea_index, int nodeb_index,
                                               vector<int> &ordering,
                                               const vector<double> &node_scores)
{
    int node_a = ordering[nodea_index];
    int node_b = ordering[nodeb_index];
    double nodea_score = 0.0; // just to initialize
    double nodeb_score = 0.0; // just to initialize

    if (MAP == true)
    {

        // Computing score for nodea, which is moved up (what is up and down? I guess up is right.., i.e. less possible parents.)
        // If b is (was) a potential parent for a, we have to recompute the scores since b is now banned.
        if (find(potential_parents[node_a].begin(), potential_parents[node_a].end(), node_b) != potential_parents[node_a].end())
        {
            myswap(nodea_index, nodeb_index, ordering);
            nodea_score = score_pos(ordering, nodeb_index);
            myswap(nodea_index, nodeb_index, ordering);
        }
        else
        { // Since b is not a potential parent of a, f_bar_z is not altered.
            // But this means that before the swap it was a potential plus1 parent.
            // After the swap, i.e. (b, a) it is not however since its to the left in the ordering.
            // This we can check if the current maximal score of a is the plus1 score for b.
            // If it is, it means that we might have used that score table, so we need to re-calculate (no shortcut possible).
            // If not (it is presumably is lower) we should use the old score.

            // Check if b was (is) a plus1 parent for a. This should always be the true.
            vector<int>::iterator itr = find(potential_plus1_parents[node_a].begin(), potential_plus1_parents[node_a].end(), node_b); // O(p)
            if (itr != potential_plus1_parents[node_a].end())
            {
                // Subtract the bs plus1 score contibution.

                myswap(nodea_index, nodeb_index, ordering);
                int f_bar_z = get_f_bar_z(nodeb_index, ordering); // ok? (is this for node_a then?) f_bar_z is an indexing of a parent subset (probably for b's parents.)
                // find the correct index j and take it.

                // get the index (of b?) in the potential_plus1_parents[nodea], since this is used with scorematrices[nodea].
                int plus1_parents_index = distance(potential_plus1_parents[node_a].begin(), itr);
                // Why is it + 1 here? - since the potential_plus1_parents does not have the empty set included at index 0
                // as scorematrices has.

                // the problem here migth be that I dont check if b is to the right in the ordering.
                //  Thts why wie subratct in the sum case! Since it cant be a plus1 parent!
                //  Since we maximize we cant sum now, though.. Do we have to re-calculate?

                // recaclulate if this score is equal to the current, since that meand we used it..?
                // it could be used if it was NOT a potential prent befor the sap, as then it was a potential plus1 parent.

                double nodeb_as_plus1_score = scoresmatrices[node_a][f_bar_z][plus1_parents_index + 1];
                myswap(nodea_index, nodeb_index, ordering);

                // cout << node_scores[node_a] << " is to compare with " << nodeb_as_plus1_score << endl;

                if (node_scores[node_a] == nodeb_as_plus1_score)
                {
                    // recalculate sinc it means b was (couldalso be another one with same score) the plus1 parent wich maximal score.
                    // cout << "recalcylate node score from scratch" << endl;
                    myswap(nodea_index, nodeb_index, ordering);
                    nodea_score = score_pos(ordering, nodeb_index);
                    myswap(nodea_index, nodeb_index, ordering);
                }
                else
                {
                    // cout << "use old score" << endl;
                    nodea_score = node_scores[node_a];
                }
            }
        }

        /**
         * Computing score for node_b.
         *
         * which is moved down (to the left? I.e. more possible parents)
         *  (a, b) -> (b, a)
         *
         */

        // If a is a potential parent of b.
        if (find(potential_parents[node_b].begin(), potential_parents[node_b].end(), node_a) != potential_parents[node_b].end())
        {
            myswap(nodea_index, nodeb_index, ordering);
            nodeb_score = score_pos(ordering, nodea_index); // Recalculate b score
            myswap(nodea_index, nodeb_index, ordering);
        }
        else // If a is NOT a potential parent of b. Then a can be a plus1 parent which should be taken into account.
        {
            vector<int>::iterator itr = find(potential_plus1_parents[node_b].begin(),
                                             potential_plus1_parents[node_b].end(),
                                             node_a);
            // Find the score where a is a plus1 parent of b. This should always exist?
            // Since a is to the right and not a possible normal parent.
            if (itr != potential_plus1_parents[node_b].cend()) // Should never reach cend?
            {
                int plus1_parents_index = distance(potential_plus1_parents[node_b].begin(), itr); // since "no parents" is the first
                myswap(nodea_index, nodeb_index, ordering);
                int f_bar_z = get_f_bar_z(nodea_index, ordering);
                double nodea_as_plus1_score = scoresmatrices[node_b][f_bar_z][plus1_parents_index + 1]; // +1
                myswap(nodea_index, nodeb_index, ordering);

                // Add the score to the sum using the log max trick.
                double maxsc = max(node_scores[node_b], nodea_as_plus1_score);
                // nodeb_score = log(exp(node_scores[node_b] - max) + exp(nodea_as_plus1_score - max)) + max;

                nodeb_score = maxsc;
            }
        }

        // swap(ordering[nodea_index], ordering[nodeb_index]); // last swap
        myswap(nodea_index, nodeb_index, ordering);

        return (make_tuple(nodea_score, nodeb_score));
    }

    else // MAP=False
    {
        // Computing score for nodea, which is moved up (what is up and down? I guess up is right.., i.e. less possible parents.)
        // If b is (was) a potential parent for a, we have to recompute the scores since b is now banned.
        if (find(potential_parents[node_a].begin(), potential_parents[node_a].end(), node_b) != potential_parents[node_a].end())
        {
            myswap(nodea_index, nodeb_index, ordering);
            nodea_score = score_pos(ordering, nodeb_index);
            myswap(nodea_index, nodeb_index, ordering);
        }
        else
        { // Since b is not a potential parent of a, f_bar_z is not altered.
            // Check if b was a plus1 parent for a
            vector<int>::iterator itr = find(potential_plus1_parents[node_a].begin(), potential_plus1_parents[node_a].end(), node_b);
            if (itr != potential_plus1_parents[node_a].end())
            {
                // Subtract the bs plus1 score contibution.

                myswap(nodea_index, nodeb_index, ordering);
                int f_bar_z = get_f_bar_z(nodeb_index, ordering); // ok?
                // find the correct index j and take it.
                int plus1_parents_index = distance(potential_plus1_parents[node_a].begin(), itr);
                double nodeb_as_plus1_score = scoresmatrices[node_a][f_bar_z][plus1_parents_index + 1];
                myswap(nodea_index, nodeb_index, ordering);

                // Sice there new additin is so small I get precision error, sice we get log(1-0.99999999999999) = -inf
                double maxsc = max(node_scores[node_a], nodeb_as_plus1_score); // I should use order_scores not node scores here. Or something. This is not right at least.
                // cout << nodeb_as_plus1_score - node_scores[node_a] << " " << endl;
                if (abs(nodeb_as_plus1_score - node_scores[node_a]) > 0.000001)
                { // 0.000001 is arbitrary
                    // OK
                    // cout << "NO RECOMPUTE order score for node" << endl;
                    nodea_score = log(exp(node_scores[node_a] - maxsc) - exp(nodeb_as_plus1_score - maxsc)) + maxsc; // gets inf... 0 at node_scores[node_a] but something at node_scores[node_b]
                }
                else
                {
                    // round off error. Recompute.
                    // cout << "RECOMPUTE order score for node" << endl;
                    myswap(nodea_index, nodeb_index, ordering);
                    nodea_score = score_pos(ordering, nodeb_index);
                    myswap(nodea_index, nodeb_index, ordering);
                }

                // cout << "true score " << true_score << " calcuated score " << node_scores[node_a] << endl;
                // assert(abs(nodea_score-true_score) < 0.001);
            }
        }

        /**
         * Computing score for node_b.
         *
         * which is moved down (to the left? I.e. more possible parents)
         *  (a, b) -> (b, a)
         *
         */

        // If a is a potential parent of b.
        if (find(potential_parents[node_b].begin(), potential_parents[node_b].end(), node_a) != potential_parents[node_b].end())
        {
            myswap(nodea_index, nodeb_index, ordering);
            nodeb_score = score_pos(ordering, nodea_index); // Recalculate b score
            myswap(nodea_index, nodeb_index, ordering);
        }
        else // If a is not a potential parent of a. Then a can be a plus1 parent which should be taken into account.
        {
            vector<int>::iterator itr = find(potential_plus1_parents[node_b].begin(),
                                             potential_plus1_parents[node_b].end(),
                                             node_a);
            // Find the score where a is a plus1 parent of b. This should always exist?
            // Since a is to the right and not a possible normal parent.
            if (itr != potential_plus1_parents[node_b].cend()) // Should never reach cend?
            {
                int plus1_parents_index = distance(potential_plus1_parents[node_b].begin(), itr); // since "no parents" is the first
                myswap(nodea_index, nodeb_index, ordering);
                int f_bar_z = get_f_bar_z(nodea_index, ordering); // some numbering of the parent setting?
                double nodea_as_plus1_score = scoresmatrices[node_b][f_bar_z][plus1_parents_index + 1];
                myswap(nodea_index, nodeb_index, ordering);

                // Add the score to the sum using the log max trick.
                double maxsc = max(node_scores[node_b], nodea_as_plus1_score);
                nodeb_score = log(exp(node_scores[node_b] - maxsc) + exp(nodea_as_plus1_score - maxsc)) + maxsc;
            }
        }

        // swap(ordering[nodea_index], ordering[nodeb_index]); // last swap
        myswap(nodea_index, nodeb_index, ordering);

        return (make_tuple(nodea_score, nodeb_score));
    }
}

vector<double> OrderScoring::score(const vector<int> &ordering, const size_t &from_orderpos, const size_t &n_elements) const
{
    size_t n = ordering.size();                          // n is p :)
    vector<double> orderscores = vector<double>(n, 0.0); // O(p)           // orderscores <- vector("double", n)
    vector<int> active_plus1_parents_indices;            // active_plus1_parents_indices < -vector("list", n)
    int f_bar_z;

    // Go through the node not banned by ordering.
    for (size_t position = from_orderpos; position < from_orderpos + n_elements; ++position)
    {
        int node = ordering[position];
        if (position == n - 1)
        {
            // no parents allowed, i.e. only first row, only first list
            orderscores[node] = scoretable[node][0][0]; // orderscores[node] <- scoretable [[node]][[1]][1, 1]
        }
        else
        {
            // This should get the right row, based on the allowed/not allowed parents.
            f_bar_z = get_f_bar_z(position, ordering);
            // Get the plus1 nodes, i.e., not allowed nodes, but that are still allowed by ordering.
            active_plus1_parents_indices = get_plus1_indices(position, ordering); // index 0, for no plus 1 parent si always included here!
            // Fin the max/sum score among the orderingallowed plus1 parent, or no plus1 parent.
            vector<double> plus1_parents_scores(active_plus1_parents_indices.size());
            for (size_t j = 0; j < plus1_parents_scores.size(); j++)
            {
                // j=0 is for no plus1 parents.
                plus1_parents_scores[j] = scoresmatrices[node][f_bar_z][active_plus1_parents_indices[j]]; // allowedscorelist is in numerical order
            }

            if (MAP == true)
            {
                auto max_el = max_element(plus1_parents_scores.begin(), plus1_parents_scores.end());
                orderscores[node] = *max_el; // max_element(plus1_parents_scores.begin(), plus1_parents_scores.end());
                // get the active plus1parent
                int plus1_parent_ind = std::distance(plus1_parents_scores.begin(), max_el);
                // handle if the index is 0, since then there is no plus1 parent.
                size_t plus1_parent = potential_plus1_parents[node][active_plus1_parents_indices[plus1_parent_ind]];
                vector<size_t> plus1_parents;
                if (plus1_parent != 0)
                { // 0 represents no plus1parent
                    plus1_parents.push_back(plus1_parent);
                }
            }
            else
            {
                orderscores[node] = sum_log_probs(plus1_parents_scores);
            }
        }
    }
    return (orderscores);
}

double OrderScoring::score_order(const vector<int> &ordering, const size_t &from_orderpos, const size_t &n_elements) const
{
    vector<double> node_scores = score(ordering, from_orderpos, n_elements); // O(p^2)?
    return accumulate(node_scores.begin(), node_scores.end(), 0.0);
}

/**
 * position is the index in the ordering of the node.
 * Maybe we cab in the MAP=True case make a similar function to get the optimal DAG.
 */
double OrderScoring::score_pos(const vector<int> &ordering, const size_t &position) const
{
    double orderscore(0.0); // O(p)           // orderscores <- vector("double", n)
    int f_bar_z;
    int node = ordering[position];

    if (position == ordering.size() - 1)
    {
        // no parents allowed, i.e. only first row, only first list.
        // Shouldnt we consider all plus 1 parents.?????

        orderscore = scoretable[node][0][0]; // orderscores[node] <- scoretable [[node]][[1]][1, 1]. What is in scoretable[node][0][1]?
                                             // scoretable[node][plus1 index][the numbering of the edge setting? 2^(#permitted parents)]
                                             // so if no parent are permitted parents, there will only be one element in the last list.

        // Whats the difference between scoretable and scorematrices?
        // scorematrices (called bannedscore in R) has the Maximal scores over all parent combinations for each plus1 node. (and for each parent setting/eligable parents it is the max over al its subsets?)
        // scorematrices[node][parent_conf][plus1parent], so e.g. last (the indexing comes from the get_f_bar_z function) row for each node is for no parents, and all the different plus1 parents in each column.
        // is the first column for no plus1 parent?
        // scoretable has all the scores for evey edge setting for each plus1 node.? In R, each matrix is for one plus1 parent?
        // How do we get the edge setting and why do we care about them, I thought we summed over all of them?
    }
    else
    {
        f_bar_z = get_f_bar_z(position, ordering);
        vector<int> active_plus1_parents_indices = get_plus1_indices(position, ordering); // O(p). This seems to add
        vector<double> plus1_parents_scores(active_plus1_parents_indices.size());

        // cout <<  plus1_parents_scores.size() << endl;
        for (size_t j = 0; j < plus1_parents_scores.size(); j++) // O(p)
        {
            plus1_parents_scores[j] = scoresmatrices[node][f_bar_z][active_plus1_parents_indices[j]]; // allowedscorelist is in numerical order
        }

        if (MAP == true)
        {
            orderscore = *max_element(plus1_parents_scores.begin(), plus1_parents_scores.end()); // O(p) ~ #plus1 parents
        }
        else
        {
            orderscore = sum_log_probs(plus1_parents_scores);
        }
    }

    return orderscore;
}

/**
 * position is the index in the ordering of the node.
 * Maybe we cab in the MAP=True case make a similar function to get the optimal DAG.
 */
vector<int> OrderScoring::get_opt_parents(const vector<int> &ordering, const size_t &position) const
{

    vector<int> parents;
    int f_bar_z;
    int node = ordering[position];

    if (position == ordering.size() - 1)
    {        
        // no parents allowed.
    }
    else
    {
        f_bar_z = get_f_bar_z(position, ordering);
        vector<int> active_plus1_parents_indices = get_plus1_indices(position, ordering); // O(p). This seems to add
        vector<double> plus1_parents_scores(active_plus1_parents_indices.size());

        // cout <<  plus1_parents_scores.size() << endl;
        for (size_t j = 0; j < plus1_parents_scores.size(); j++) // O(p)
        {
            plus1_parents_scores[j] = scoresmatrices[node][f_bar_z][active_plus1_parents_indices[j]]; // allowedscorelist is in numerical order
        }
        
        auto maxscore_itr = max_element(plus1_parents_scores.begin(), plus1_parents_scores.end()); // O(p) ~ #plus1 parents 
        double maxscore = *maxscore_itr;
        // get the index of the max score
        // get maxscore plus1 node parent
        int maxscore_plus1_index = distance(plus1_parents_scores.begin(), maxscore_itr);

        int row_in_parent_table = maxrow[node][f_bar_z][maxscore_plus1_index];
        
        // Add the parents in a vector.
        // Parenttable for one node looks like. These elements are indices 
        // in potential_parents, so the have to be mapped to the nodes.
        //       [,1] [,2]
        // [1,]   -1   -1
        // [2,]    0   -1
        // [3,]    1   -1
        // [4,]    0    1
        for (size_t i = 0; i < parenttable[node][row_in_parent_table].size(); i++)
        {
            // print the parent 

            if (parenttable[node][row_in_parent_table][i] == -1) break;
            //cout << "parent index" << parenttable[node][row_in_parent_table][i] << endl;
            parents.push_back(potential_parents[node][parenttable[node][row_in_parent_table][i]]);
        }
        
    }

    return (parents);
}

double OrderScoring::sum_log_probs(const vector<double> &log_probs) const
{
    double max_log_prob = *max_element(log_probs.begin(), log_probs.end());
    double score(0.0);
    for (auto &val : log_probs)
    {
        score += exp(val - max_log_prob);
    }
    return (max_log_prob + log(score));
}

vector<int> OrderScoring::get_plus1_indices(const int &position, const vector<int> &ordering) const
{

    // Find the possible parents after in the ordering (allowedscorelist)
    // potential_plus1_parents is a list of possible parent (sets?) (set with extra +1 parent) for node (node ocnsidering the ordering).
    // allowedscorelist is a filetered version of plius1listparents, removing the node before in the ordering
    const int node = ordering[position];
    vector<int> active_plus1_parents_indices;
    active_plus1_parents_indices.push_back(0); // f(null)=0, no parents is always a possibility.

    // O(p)
    // cout << potential_plus1_parents[node].size() << endl;
    for (size_t j = 0; j < potential_plus1_parents[node].size(); j++) // Is j the plus1node? -No, but potential_plus1_parents[node][j] is.
    {
        if (find(ordering.begin() + position + 1, ordering.end(), potential_plus1_parents[node][j]) != ordering.end())
        {
            active_plus1_parents_indices.push_back(j + 1); // +1 since the 0 is no "+1"-parent at all. So the indices are shifted.
        }
    }
    return (active_plus1_parents_indices);
}

/*
This returns
1. the active parents, i.e. the nodes in potentialparents that are not banned by the order
2. the indices of the benned parents, where the indices are indices in the potential_parents vector.
*/
tuple<vector<int>, vector<int>> OrderScoring::get_active_and_banned(const int &position, const vector<int> &ordering) const
{
    // TODO: node should be a parameter and the position should be found.
    const int node = ordering[position];
    // Find the banned scores before in the ordering. (Double banned?)
    // potential_parents has the banned nodes for node.
    // bannednodes is a filtered version of potential_parents, removing the nodes after in the ordering.
    // it only purose is to compute f_bar_z[node], i.e. f(Z)

    // When I write banned, I mean banned, by the ordering, from the potential ones.(?)
    vector<int> parent_indices_banned_by_ordering; // Index in the banned_parents[node] vector.
    vector<int> active_parents;                    // A vector containing the parents we are talking about.
    // See for each potential parent, if it is banned by the ordering or not.
    for (size_t j = 0; j < potential_parents[node].size(); j++)
    {
        // check if potential_parents[node][j] is banned by the ordering.
        if (find(ordering.begin(), ordering.begin() + position + 1, potential_parents[node][j]) != ordering.begin() + position + 1)
        {
            parent_indices_banned_by_ordering.push_back(j); // This has inly ints. It for computing f(Z)
        }
        else
        {
            active_parents.push_back(potential_parents[node][j]); // add the node as parent.
        }
    }

    return make_tuple(active_parents, parent_indices_banned_by_ordering);
}

/**
 * The position is the index in the scoretable of the node.
 * It should be a number between 0 and 2^(#permitted parents) -1
 * Its like a binary number, but it is not in numerica order.
 * for a set {a,b,c}, f({c}) = 001, f({b}) = 010, f({a})=100. fbar is like the inverse, treating 0 as ones?.
 *
 */

int OrderScoring::get_f_bar_z(const int &position, const vector<int> &ordering) const
{
    // int OrderScoring::get_f_bar_z(const int &node, const vector<int> & parent_indices_banned_by_ordering) const{
    int f_bar_z;
    const int node = ordering[position];
    // Find the banned scores before in the ordering. (Double banned?)
    // potential_parents has the banned nodes for node.
    // bannednodes is a filtered version of potential_parents, removing the nodes after in the ordering.
    // it only purose is to compute f_bar_z[node], i.e. f(Z)

    //   // When I write banned, I mean banned, by the ordering, from the potential ones.(?)
    vector<int> parent_indices_banned_by_ordering; // Index in the banned_parents[node] vector.
    vector<int> active_parents;                    // A vector containing the parents we are talking about.
    tie(active_parents, parent_indices_banned_by_ordering) = get_active_and_banned(position, ordering);

    // Compute f(Z) (the labelling), where Z is the (banned?) parents of node, accoring to the paper.
    // I.e. f_bar_z[node] = f(Pa(node)). Shouldnt it be the banned ones?
    if (potential_parents[node].size() == 0 || parent_indices_banned_by_ordering.size() == 0)
    {
        f_bar_z = 0; // all parents allowed f(null) = 0
    }
    else
    {
        int indextmp = 0;
        for (auto &item : parent_indices_banned_by_ordering)
        {
            indextmp += pow(2, item); // compute f(Z). add 1 since nodes are labeled from 0.
        }
        f_bar_z = rowmaps_backwards[node][indextmp]; // I think indextmp is the actual f_bar_z.
    }
    return (f_bar_z);
}

// /**
//  * The position is the index in the scoretable of the node.
//  * It should be a number between 0 and 2^(#permitted parents) -1
//  * Its like a binary number, but it is not in numerica order.
//  * for a set {a,b,c}, f({c}) = 001, f({b}) = 010, f({a})=100. fbar is like the inverse, treating 0 as ones?.
//  *
//  */

// vector<int> OrderScoring::get_opt_edges(const int &position, const vector<int> &ordering) const
// {
//   int f_bar_z;
//   const int node = ordering[position];
//   // Find the banned scores before in the ordering. (Double banned?)
//   // potential_parents has the banned nodes for node.
//   // bannednodes is a filtered version of potential_parents, removing the nodes after in the ordering.
//   // it only purose is to compute f_bar_z[node], i.e. f(Z)

//   // When I write banned, I mean banned, by the ordering, from the potential ones.(?)
//   vector<int> parent_indices_banned_by_ordering; // Index in the banned_parents[node] vector.
//   vector<int> active_parents; // A vector containing the parents we are talking about.
//   // See for each potential parent, if it is banned by the ordering or not.
//   for (size_t j = 0; j < potential_parents[node].size(); j++)
//   {
//     // check if potential_parents[node][j] is banned by the ordering.
//     if (find(ordering.begin(), ordering.begin() + position + 1, potential_parents[node][j]) != ordering.begin() + position + 1)
//     {
//       parent_indices_banned_by_ordering.push_back(j); // This has inly ints. It for computing f(Z)
//     } else {
//         active_parents.push_back(potential_parents[node][j]); // add the node as parent.
//     }
//   }

//   return (active_parents);
// }

OrderScoring get_score(Rcpp::List ret)
{

    // Read MAP flag
    bool MAP = Rcpp::as<int>(ret["MAP"]);
    // cout << "MAP:" << MAP << endl;

    // Read numparents
    vector<int> numparents = Rcpp::as<vector<int>>(ret["numparents"]);
    size_t p = numparents.size();

    // Read scoretable. How are these interpreted?
    vector<vector<vector<double>>> scoretable = Rcpp::as<vector<vector<vector<double>>>>(ret["scoretable"]);

    // Read parent table
    // Rcpp::List parenttableR = Rcpp::as<Rcpp::List>(ret["parenttable"]);
    //
    // vector<Rcpp::IntegerMatrix> parenttable;
    // for (size_t i = 0; i < p; i++)
    // {
    //     Rcpp::IntegerMatrix m = Rcpp::as<Rcpp::IntegerMatrix>(parenttableR[i]);
    //     parenttable.push_back(m);
    // }

    Rcpp::List parenttableR = Rcpp::as<Rcpp::List>(ret["parenttable"]);
    vector<vector<vector<int>>> parenttable;
    for (size_t i = 0; i < p; i++)
    {
        Rcpp::NumericMatrix m = Rcpp::as<Rcpp::NumericMatrix>(parenttableR[i]);
        vector<vector<int>> mat(m.rows(), vector<int>(m.cols()));
        for (int j = 0; j < m.rows(); j++)
        {
            for (int k = 0; k < m.cols(); k++)
            {
                mat[j][k] = m(j, k);
            }
        }
        parenttable.push_back(mat);
    }

    // Read max matrices.
    // For each node, the columns should be determined by the alowed/banned parents
    // through fbarz or fz.
    // The columns are for all the plus1 parents. First column is for no plus1 parent.
    // Would be nice if this also contained the actual parents in some way!
    Rcpp::List maxmatrixR = Rcpp::as<Rcpp::List>(ret["maxmatrix"]);
    vector<vector<vector<double>>> maxmatrix;
    for (size_t i = 0; i < p; i++)
    {
        Rcpp::NumericMatrix m = Rcpp::as<Rcpp::NumericMatrix>(maxmatrixR[i]);
        vector<vector<double>> mat(m.rows(), vector<double>(m.cols()));
        for (int j = 0; j < m.rows(); j++)
        {
            for (int k = 0; k < m.cols(); k++)
            {
                mat[j][k] = m(j, k);
            }
        }
        maxmatrix.push_back(mat);
    }

    // Read maxrows.

    Rcpp::List maxrowR = Rcpp::as<Rcpp::List>(ret["maxrow"]);
    vector<vector<vector<int>>> maxrow;
    for (size_t i = 0; i < p; i++)
    {
        Rcpp::NumericMatrix m = Rcpp::as<Rcpp::NumericMatrix>(maxrowR[i]);
        vector<vector<int>> mat(m.rows(), vector<int>(m.cols()));
        for (int j = 0; j < m.rows(); j++)
        {
            for (int k = 0; k < m.cols(); k++)
            {
                mat[j][k] = m(j, k);
            }
        }
        maxrow.push_back(mat);
    }

    // Read banned score, or max matrices. How are these interpreted?
    // For each node, the columns should be determined by the alowed/banned parents
    // through fbarz or fz.
    // The columns are for all the plus1 parents. First column is for no plus1 parent.
    // Would be nice if this also contained the actual parents in some way!
    Rcpp::List bannedscoreR = Rcpp::as<Rcpp::List>(ret["bannedscore"]);
    vector<vector<vector<double>>> bannedscore;
    for (size_t i = 0; i < p; i++)
    {
        Rcpp::NumericMatrix m = Rcpp::as<Rcpp::NumericMatrix>(bannedscoreR[i]);
        vector<vector<double>> mat(m.rows(), vector<double>(m.cols()));
        for (int j = 0; j < m.rows(); j++)
        {
            for (int k = 0; k < m.cols(); k++)
            {
                mat[j][k] = m(j, k);
            }
        }
        bannedscore.push_back(mat);
    }

    // Read rowmaps_backwards
    Rcpp::List rowmaps_backwardsR = Rcpp::as<Rcpp::List>(ret["rowmaps_backwards"]);
    vector<Rcpp::IntegerVector> rowmaps_backwards;
    for (size_t i = 0; i < p; i++)
    {
        Rcpp::IntegerVector m = Rcpp::as<Rcpp::IntegerVector>(rowmaps_backwardsR[i]);
        rowmaps_backwards.push_back(m);
    }

    // Read rowmaps_forward
    Rcpp::List rowmaps_forwardR = Rcpp::as<Rcpp::List>(ret["rowmaps_forward"]);
    vector<Rcpp::IntegerVector> rowmaps_forward;
    for (size_t i = 0; i < p; i++)
    {
        Rcpp::IntegerVector m = Rcpp::as<Rcpp::IntegerVector>(rowmaps_forwardR[i]);
        rowmaps_forward.push_back(m);
    }

    // Read aliases
    Rcpp::List aliasesR = Rcpp::as<Rcpp::List>(ret["aliases"]);
    vector<vector<int>> aliases;
    for (size_t i = 0; i < p; i++)
    {
        vector<int> m = Rcpp::as<vector<int>>(aliasesR[i]);
        aliases.push_back(m);
    }

    // Read plus1listsparents
    Rcpp::List plus1listsparentsR = Rcpp::as<Rcpp::List>(ret["plus1listsparents"]);
    vector<vector<int>> plus1listsparents;
    for (size_t i = 0; i < p; i++)
    {
        Rcpp::IntegerVector m = Rcpp::as<Rcpp::IntegerVector>(plus1listsparentsR[i]);
        vector<int> tmp;
        for (auto e : m)
        {
            tmp.push_back(e);
        }
        plus1listsparents.push_back(tmp);
    }

    vector<int> scorepositions(p);
    for (size_t i = 0; i < p; ++i)
    {
        scorepositions[i] = i;
    }

    OrderScoring scoring(aliases,
                         numparents,
                         rowmaps_backwards,
                         rowmaps_forward,
                         plus1listsparents,
                         scoretable,
                         bannedscore,
                         maxmatrix,
                         maxrow,
                         parenttable,
                         MAP);

    return (scoring);
}
