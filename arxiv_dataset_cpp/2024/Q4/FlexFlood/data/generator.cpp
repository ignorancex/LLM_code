#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <iomanip>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
#include <ext/pb_ds/tag_and_trait.hpp>
#include "../src/RandomNumberGenerator.hpp"
int main(int argc, char** argv) {
    using namespace std;
    using namespace __gnu_pbds;
    using Tree = tree<vector<double>, null_type, less<vector<double>>, rb_tree_tag, tree_order_statistics_node_update>;

    assert(argc == 8);
    const int dimension = stoi(argv[1]), num_data = stoi(argv[2]), num_sample_query = stoi(argv[3]), num_query = stoi(argv[4]);
    const string data_filename = argv[5], sample_query_filename = argv[6], query_filename = argv[7];
    assert(dimension >= 0);
    assert(num_data >= 0);
    assert(num_sample_query >= 0);
    assert(num_query >= 0);

    vector<vector<double>> data(num_data, vector<double>(dimension));
    Tree tree;
    for(int i = 0; i < num_data; ++i) {
        for(int d = 0; d < dimension; ++d) {
            data[i][d] = normal(3e8, 1e8);
        }
        tree.insert(data[i]);
    }

    vector<string> sample_query_type(num_sample_query);
    vector<vector<double>> sample_search_lower(num_sample_query, vector<double>(dimension)), sample_search_upper(num_sample_query, vector<double>(dimension));
    for(int i = 0; i < num_sample_query; ++i) {
        sample_query_type[i] = "Search";
        for(int d = 0; d < dimension; ++d) {
            double length = rrng(0, 3e8);
            sample_search_lower[i][d] = rrng(0, 1e9 - length);
            sample_search_upper[i][d] = sample_search_lower[i][d] + length;
        }
    }

    vector<string> query_type(num_query);
    vector<vector<double>> search_lower(num_query, vector<double>(dimension)), search_upper(num_query, vector<double>(dimension));
    vector<vector<double>> update(num_query, vector<double>(dimension));
    int update_count = 0;
    int update_total = num_query / 2;
    for(int i = 0; i < num_query; ++i) {
        if((i / 10000) % 2 == 0) {
            ++update_count;
            int type = rng(0, 2);
            if(type == 0) {
                query_type[i] = "Insert";
                for(int d = 0; d < dimension; ++d) {
                    update[i][d] = normal(3e8 + 4e8 * update_count / update_total, 1e8);
                }
                tree.insert(update[i]);
            } else if(type == 1) {
                query_type[i] = "Erase";
                if(tree.empty()) {
                    vector<double> dat(dimension);
                    for(int d = 0; d < dimension; ++d) {
                        update[i][d] = rrng(0, 1e9);
                    }
                } else {
                    update[i] = *tree.find_by_order(rng(0, (int)tree.size()));
                    tree.erase(update[i]);
                }
            } else {
                assert(false);
            }
        } else {
            query_type[i] = "Search";
            for(int d = 0; d < dimension; ++d) {
                double length = rrng(0, 3e8);
                search_lower[i][d] = rrng(0, 1e9 - length);
                search_upper[i][d] = search_lower[i][d] + length;
            }
        }
    }

    ofstream dataFile(data_filename);
    ofstream sampleFile(sample_query_filename);
    ofstream queryFile(query_filename);
    dataFile << fixed << setprecision(10);
    sampleFile << fixed << setprecision(10);
    queryFile << fixed << setprecision(10);

    for(int i = 0; i < num_data; ++i) {
        for(int d = 0; d < dimension; ++d) {
            dataFile << data[i][d] << " \n"[d + 1 == dimension];
        }
    }
    dataFile.close();

    for(int i = 0; i < num_sample_query; ++i) {
        sampleFile << sample_query_type[i] << ' ';
        for(int d = 0; d < dimension; ++d) {
            sampleFile << sample_search_lower[i][d] << " ";
        }
        for(int d = 0; d < dimension; ++d) {
            sampleFile << sample_search_upper[i][d] << " \n"[d + 1 == dimension];
        }
    }
    sampleFile.close();

    for(int i = 0; i < num_query; ++i) {
        queryFile << query_type[i] << ' ';
        if(query_type[i] == "Search") {
            for(int d = 0; d < dimension; ++d) {
                queryFile << search_lower[i][d] << ' ';
            }
            for(int d = 0; d < dimension; ++d) {
                queryFile << search_upper[i][d] << " \n"[d + 1 == dimension];
            }
        } else if(query_type[i] == "Insert") {
            for(int d = 0; d < dimension; ++d) {
                queryFile << update[i][d] << " \n"[d + 1 == dimension];
            }
        } else if(query_type[i] == "Erase") {
            for(int d = 0; d < dimension; ++d) {
                queryFile << update[i][d] << " \n"[d + 1 == dimension];
            }
        } else {
            assert(false);
        }
    }
    queryFile.close();
}