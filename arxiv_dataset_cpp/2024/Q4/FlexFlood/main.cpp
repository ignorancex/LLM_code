#include <bits/stdc++.h>
#include "./src/FlexFlood.hpp"
int main(int argc, char** argv) {
    using namespace std;
    assert(argc == 4);
    string data_filename = argv[1], sample_query_filename = argv[2], query_filename = argv[3];
    ifstream dataFile(data_filename), sampleFile(sample_query_filename), queryFile(query_filename);
    if(!dataFile) {
        cerr << "Can't Open " + data_filename << endl;
        return 0;
    }
    if(!sampleFile) {
        cerr << "Can't Open " + sample_query_filename << endl;
        return 0;
    }
    if(!queryFile) {
        cerr << "Can't Open " + query_filename << endl;
        return 0;
    }

    constexpr int dimension = 3;

    constexpr int num_data = 100000;
    vector<array<double, dimension>> data(num_data);
    for(int i = 0; i < num_data; ++i) {
        for(int d = 0; d < dimension; ++d) {
            dataFile >> data[i][d];
        }
    }
    dataFile.close();

    constexpr int num_sample = 1000;
    vector<string> sample_type(num_sample);
    vector<array<double, dimension>> sample_lowers(num_sample), sample_uppers(num_sample);
    for(int i = 0; i < num_sample; ++i) {
        sampleFile >> sample_type[i];
        assert(sample_type[i] == "Search");
        for(int d = 0; d < dimension; ++d) {
            sampleFile >> sample_lowers[i][d];
        }
        for(int d = 0; d < dimension; ++d) {
            sampleFile >> sample_uppers[i][d];
        }
    }
    sampleFile.close();

    FlexFlood<double, dimension> index(data.begin(), data.end(), sample_lowers, sample_uppers);

    constexpr int num_query = 2000000;

    long long sum = 0;

    chrono::system_clock::time_point start, end;
    start = chrono::system_clock::now();

    for(int q = 0; q < num_query; ++q) {
        string query_type;
        queryFile >> query_type;
        if(query_type == "Search") {
            array<double, dimension> search_lower, search_upper;
            for(int d = 0; d < dimension; ++d) {
                queryFile >> search_lower[d];
            }
            for(int d = 0; d < dimension; ++d) {
                queryFile >> search_upper[d];
            }
            vector<array<double, dimension>> result = index.enumerate(search_lower, search_upper);
            sum += result.size();
        } else if(query_type == "Insert") {
            array<double, dimension> update;
            for(int d = 0; d < dimension; ++d) {
                queryFile >> update[d];
            }
            index.insert(update);
        } else if(query_type == "Erase") {
            array<double, dimension> update;
            for(int d = 0; d < dimension; ++d) {
                queryFile >> update[d];
            }
            index.erase(update);
        } else {
            assert(false);
        }
    }

    end = chrono::system_clock::now();
    auto time = end - start;
    auto msec = chrono::duration_cast<chrono::milliseconds>(time).count();
    cerr << msec << " (ms)" << endl;

    cout << sum << '\n';

    queryFile.close();

    index.clear();

    return 0;
}