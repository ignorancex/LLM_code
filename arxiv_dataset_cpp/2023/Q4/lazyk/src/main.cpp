#include "lazyk.cpp"

int main()
{
    // vector<vector<double>> state({{0.5, 0.3, 0.2}, {0.2, 0.7, 0.1}, {0.5, 0.3, 0.2}, {0.2, 0.7, 0.1}, {0.5, 0.3, 0.2}, {0.2, 0.7, 0.1}, {0.5, 0.3, 0.2}, {0.2, 0.7, 0.1}});
    vector<vector<double>> state({{0.5, 0.3, 0.2}, {0.5, 0.3, 0.2}});
    // vector<vector<double>> state({{0.5, 0.3, 0.2}});
    // vector<vector<double>> state({{1.0}, {1.0}});
    Lazyk lazyk(state);

    int i = 0;
    double last_log_prob = -INFINITY;

    // Time the total executiont time
    clock_t begin = clock();

    // Keep track of all the states to check if we don't have duplicates
    unordered_set<vector<int>, vector_hash> states;
    while (!lazyk.end())
    {
        vector<int> assignment = lazyk.getAssignment();
        if (states.find(assignment) != states.end())
        {
            throw "Duplicate state";
        }
        states.insert(assignment);

        // double log_prob = state.cost(assignment);
        // if (log_prob < last_log_prob)
        // {
        //     throw "Log prob should not decrease";
        // }
        // last_log_prob = log_prob;
        for (int i = 0; i < assignment.size(); i++)
        {
            cout << assignment[i] << " ";
        }
        cout << endl;
        ++lazyk;
        i++;
    }
    std::cout << endl  << "Total: " << i << std::endl;

    // print the total execution time
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << elapsed_secs << endl;
    return 0;
}