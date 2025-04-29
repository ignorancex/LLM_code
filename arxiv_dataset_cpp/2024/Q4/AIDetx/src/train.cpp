#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
#include <set>
#include <unordered_map>

#include "argparse.hpp"
#include "json.hpp"

using namespace std;
using json = nlohmann::json;

struct MyArgs : public argparse::Args {
    string &human_data = kwarg("h,human", "The human data file");
    string &ai_data = kwarg("g,gpt", "The AI generated data file");
    string &output = kwarg("o,output", "The output folder to save the models and configs");
    string &alphabet = kwarg("a,alphabet", "The alphabet file if empty, will use the alphabet from the data (converts to lower case)").set_default("");
    int &k = kwarg("k", "k-order Markov model").set_default(5);
};


MyArgs args;
set<char> alphabet;
int n_chars = 0;
int mapping[256];


void sanitize(ifstream &human_data, ifstream &ai_data) {
    if (!human_data.is_open()) {
        printf("Error: Could not open human data file\n");
        exit(1);
    }
    if (!ai_data.is_open()) {
        printf("Error: Could not open AI data file\n");
        exit(1);
    }
    if (args.k < 1 || args.k > 10) {
        printf("Error: k must be between 1 and 10\n");
        exit(1);
    }
}


void append_to_alphabet(ifstream &data) {
    char c;
    while ((c = data.get()) != EOF) {
        if (c < 32) {
            continue;
        }
        alphabet.insert(tolower(c));
    }
}


void process_alphabet(ifstream &human_data, ifstream &ai_data) {
    for (int i = 0; i < 256; i++) {
        mapping[i] = -1;
    }
    if (!args.alphabet.empty()){
        ifstream alphabet_file(args.alphabet);
        if (!alphabet_file.is_open()) {
            printf("Error: Could not open alphabet file\n");
            exit(1);
        }
        append_to_alphabet(alphabet_file);
    } else {
        append_to_alphabet(human_data);
        append_to_alphabet(ai_data);
    }

    for (char c : alphabet) {
        mapping[static_cast<unsigned char>(c)] = n_chars++;
    }
}


void save_config(){
    filesystem::create_directories(args.output);
    ofstream config_out(args.output + "/config.json");
    if (!config_out.is_open()) {
        printf("Error: Could not create config file\n");
        exit(1);
    }
    string alphabet_str = "";
    for (char c : alphabet) {
        alphabet_str += c;
    }
    json config = {
        {"k", args.k},
        {"n_chars", n_chars},
        {"alphabet", alphabet_str}
    };
    cout << config.dump(4) << endl;
    config_out << config.dump(4);
    config_out.close();
}


void create_model(ifstream &data, string model_name) {
    unordered_map<string, int*> model;
    string key = "";
    char c;
    int *values;
    
    int i = 0;
    while (i < args.k) {
        c = tolower(data.get());
        if (mapping[static_cast<unsigned char>(c)] == -1) {
            continue;
        }
        key += c;
        i++;
    }

    while ((c = data.get()) != EOF) {
        c = tolower(c);
        if (mapping[static_cast<unsigned char>(c)] == -1) {
            continue;
        }
        if (model.find(key) == model.end()) {
            values = new int[n_chars];
            for (int i = 0; i < n_chars; i++) {
                values[i] = 0;
            }
            values[mapping[static_cast<unsigned char>(c)]] = 1;
            model[key] = values;
        }
        else {
            model[key][mapping[static_cast<unsigned char>(c)]]++;
        }
        key.erase(0, 1);
        key += c;
    }

    ofstream model_out(args.output + "/" + model_name + ".txt");
    if (!model_out.is_open()) {
        printf("Error: Could not create model file\n");
        exit(1);
    }
    int total;
    for (auto const& [key, value] : model) {
        total = 0;
        model_out << key << " ";
        for (int i = 0; i < n_chars; i++) {
            model_out << value[i] << " ";
            total += value[i];
        }
        model_out << total << endl;
        delete[] value;
    }
    model_out.close();
    model.clear();

    printf("Model %s created\n", model_name.c_str());
}


int main(int argc, char* argv[]) {

    args.parse(argc, argv, false);
    ifstream human_data(args.human_data);
    ifstream ai_data(args.ai_data);
    
    sanitize(human_data, ai_data);
    process_alphabet(human_data, ai_data);
    save_config();
    create_model(human_data, "human");
    create_model(ai_data, "ai");

    return 0;
}