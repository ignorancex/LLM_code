#include <cassert>
#include <chrono>
#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>


using namespace std;
using std::vector;


bool approximatelyEqual(float a, float b, float epsilon)
{
    return fabs(a - b) <= ((fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}

bool essentiallyEqual(float a, float b, float epsilon)
{
    return fabs(a - b) <= ((fabs(a) > fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}

bool definitelyGreaterThan(float a, float b, float epsilon)
{
    return (a - b) > ((fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}

bool definitelyLessThan(float a, float b, float epsilon)
{
    return (b - a) > ((fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}

int rand_int_in_range(const size_t &from, const size_t &to)
{
    return (from + (std::rand() % (to - from + 1)));
}

// void print_matrix(vector<vector<double>> &M)
// {
//     //cout << setprecision(2);
//     for (auto &row : M)
//     {
//         for (auto &el : row)
//         {
//             cout << el << "\t";
//         }
        
//         cout << endl;
//     }
//     //cout << setprecision(5);
// }


/**
 * Row matrix of bool vectors, representing sets of orderings, and selects the
 * indices with highest scores.
 *
 * It does it column by columns and adding the indices corresponding to zeros
 * and ones respectively as arrays. in the next itetration, column n+1 these indices are used.
 *
 */
vector<int> unique_sets(const vector<vector<bool>> &mats,
                        const vector<double> &order_scores, 
                        double EPSILON)
{
    size_t N = mats.size();
    size_t p = mats[0].size();
    vector<int> samesets;
    vector<vector<int>> q;
    // The starting indecis are all indices. The first element indicates which column of
    // mats these indices should be used. To start, this is column 0.
    vector<int> start(N + 1, 0);
    for (size_t i = 0; i < N; i++)
    {
        start[i + 1] = i;
    }
    q.push_back(move(start));

    while (q.size() > 0)
    {
        // take a set of indeces from the queue
        vector<int> inds = q.back();
        size_t n = inds[0]; // ge the column to look at
        q.pop_back();       // remove from the queue

        vector<int> ones(1, n + 1);  // get the indices correspoing to ones. And indicate that thes should be used at column 1.
        vector<int> zeros(1, n + 1); // same for zeros.
        vector<int>::iterator i;
        for (i = inds.begin() + 1; i != inds.end(); ++i)
        {
            // For each index, check if 0 or 1 and add to corresponding vector.
            if (mats[*i][n] == 0)
            {
                zeros.push_back(*i); // Isnt this alreadu n+1 long?????
            }
            else
            {
                ones.push_back(*i);
            }
        }
        if (n == p - 1)
        { // Add to the unique elements
            if (zeros.size() > 1)
            {
                // PrintVector(zeros);
                //  Get the max scoring index
                double maxscore = -INFINITY;
                int max_ind = -1;
                vector<int>::iterator i;
                for (i = zeros.begin() + 1; i != zeros.end(); ++i)
                {
                    // cout << "row " << *i << " score " << order_scores[*i] << endl;
                    if (order_scores[*i] - maxscore < EPSILON)
                    {
                        // cout << order_scores[*i] - maxscore << endl;
                    }
                    // if (abs(order_scores[*i] - maxscore) > EPSILON)
                    if (order_scores[*i] > maxscore)
                    {
                        maxscore = order_scores[*i];
                        max_ind = *i;
                    }
                }
                samesets.push_back(max_ind);
            }

            if (ones.size() > 1)
            {
                // PrintVector(ones);
                //  Get the max scoring index
                double maxscore = -INFINITY;
                int max_ind = -1;
                vector<int>::iterator i;
                for (i = ones.begin() + 1; i != ones.end(); ++i)
                {
                    // if(definitelyGreaterThan(order_scores[*i], maxscore, EPSILON))
                    if (order_scores[*i] > maxscore)
                    {
                        maxscore = order_scores[*i];
                        max_ind = *i;
                    }
                }

                samesets.push_back(max_ind);
            }
        }
        else
        { // Add elements to the queue
            if (zeros.size() > 1)
                q.push_back(move(zeros));
            if (ones.size() > 1)
                q.push_back(move(ones));
        }
    }

    return (samesets);
}