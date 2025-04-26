/*
 * File:   main.cpp
 * Author: Nikolay Ulyanov (ulyanick@gmail.com)
 *
 * Created on 14 August 2017
 * Study of properties and relations
 * between various constructions relevant to snarks:
 * - normal (Petersen) colouring
 * - o6c4c
 * - o5cdc
 * - nz5, nz-mod5
 * - (3, 3)-flow parity-pair-covers (aka 33-pp)
 * - dominating circuits
 * - (m, n, k)-flow double covers
 * - unit vector flows on S3
 * - unit vector flows on S2
 *
 */

#include "main.h"
#include "args.h"
#include "constants.h"
#include "graph.h"

// place for including all needed experiments
#include "experiments/cycles.h"
#include "experiments/petersen_colouring.h"
#include "experiments/preimages.h"
#include "experiments/o5cdc.h"
#include "experiments/o6c4c.h"
// #include "experiments/unit_vector_flows_s3.h"
#include "experiments/unit_vector_flows.h"

#include <iostream>
#include <cstdlib>
#include <algorithm>

using namespace std;


int main(int argc, char** argv) {
  init();
  Args args(argc, argv);

  // initialize petersen graph
  Graph petersen_graph(10);
  process_petersen_graph(petersen_graph);

  size_t read_graphs_count = 0;
  while (true) {
    // initialize graph, which we experiment with
    Graph graph;
    if (!read_graph(args.filetype, graph)) {
      break;
    }

    ++read_graphs_count; // enumerating from 1, not 0
    // filtering graphs
    if (args.start_idx != NONE && read_graphs_count < args.start_idx) {
      continue;
    }
    if (args.finish_idx != NONE && args.finish_idx < read_graphs_count) {
      break;
    }
    if (args.idxs.size() > 0 && (args.idxs.find(read_graphs_count) == args.idxs.end())) {
      continue;
    }
    graph.number = read_graphs_count;
    cerr << "g" << read_graphs_count << "\t" << endl << flush;
    cout << "g" << read_graphs_count << "\t" << endl << flush;

    // build cycles
    ExpCycles::find_all_cycles(graph);

    run_experiments(petersen_graph, graph);
  }

  cerr << "fin" << endl;
  return(EXIT_SUCCESS);
}

void init() {
  cerr.precision(17);
  srand(time(NULL));
}

void process_petersen_graph(Graph& petersen_graph) {
  build_petersen_graph(petersen_graph);
  petersen_graph.print();
  ExpPetersenColouring::find_all_petersen_colourings(petersen_graph);
  ExpPetersenColouring::create_cc_mapping(petersen_graph);
  ExpCycles::find_all_cycles(petersen_graph);
  cerr << "petersen cycles: " << petersen_graph.all_cycles.size() << endl;
  cerr << "petersen perfect matchings: " << petersen_graph.all_full_cycles.size() << endl;

  Exp5cdc::find_all_o5cdc(petersen_graph, true);
}

void run_experiments(const Graph& petersen_graph, Graph& graph) {
  // Exp5cdc::find_all_o5cdc(graph); // NOTE: this is very slow, and probably BROKEN
  Exp6c4c::find_all_o6c4c(graph);
  ExpPetersenColouring::find_all_petersen_colourings(graph);
  ExpUnitVectorFlows::find_all_unit_vector_flows(graph);
}
