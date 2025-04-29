#include <sys/time.h>
#include <sys/resource.h>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <limits.h>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <libgen.h>
#include <llama.h>
#include <benchmark.h>
#include <inttypes.h>

#include "clock.hpp"

std::string input_file = "/home/pfernando/deploy-cyclone/benchmarks/data/twitter/twitter_rv_15066953.net";
std::string database_directory = "llama_db";
#define MAX_GRAPH_NODES 10000

//stats
unsigned long tx_begin_time = 0UL;
unsigned long tx_cnt = 0UL;

typedef ll_mlcsr_ro_graph benchmarkable_graph_t;
#define benchmarkable_graph(g)  ((g)->ro_graph())

template <class Graph>
class ll_t_outdegree : public ll_benchmark<Graph> {
  FILE* _out;
  public:

  ll_t_outdegree() : ll_benchmark<Graph>("") {
	_out = stdout;
  }

  virtual ~ll_t_outdegree(void) {
  }

  unsigned long outdegree(node_t nodeId){
	Graph& G = *this->_graph;
	return G.out_degree(nodeId);
  }

  virtual double run(void){
	// making compiler happy
	return NAN;
  }

};


/**
 * Load a part of the graph using the writable representation
 *
 * @param graph the writable graph
 * @param data_source the data source
 * @param loader_config the loader configuration
 * @param batch_size the batch size (the max number of edges to load)
 * @return true if the graph was loaded
 */
bool load_batch_via_writable_graph(ll_writable_graph& graph,
	ll_data_source& data_source, const ll_loader_config& loader_config,
	size_t batch_size) {


  bool loaded = data_source.pull(&graph, batch_size);
  if (!loaded) return false;
  graph.checkpoint(&loader_config);
  return true;
}
  
ll_loader_config loader_config;
ll_database *database;
ll_writable_graph *graph;
ll_file_loaders loaders;
ll_file_loader *loader;
ll_concat_data_source combined_data_source;
ll_data_source *d; 
benchmarkable_graph_t *G; 
ll_t_outdegree<benchmarkable_graph_t> *benchmark;

const double frac_read = 0.5;
const int preload_batch= 100*1000;
const int num_threads = 4;
const int run_batch = 10;

void open_llama(){
  const char* first_input = input_file.c_str();
  const char* file_type = ll_file_extension(first_input);

  if (strcmp(ll_file_extension(input_file.c_str()), file_type)!=0) {
	fprintf(stderr, "Error: All imput files must have the same "
		"file extension.\n");
	exit(-1);
  }


  // Open the database
  database = new ll_database(database_directory.c_str());
  database->set_num_threads(num_threads);
  graph = database->graph();

  // Load the graph
  loader = loaders.loader_for(first_input);
  if (loader == NULL) {
	fprintf(stderr, "Error: Unsupported input file type\n");
	exit(-1);
  }

  d = loader->create_data_source(input_file.c_str());
  combined_data_source.add(d);

  G = &(benchmarkable_graph(graph));
  benchmark = new ll_t_outdegree<benchmarkable_graph_t>();
  benchmark->initialize(G);


}

void preload_llama(){
    if (!load_batch_via_writable_graph(*graph, combined_data_source,
                loader_config, preload_batch)){
        printf("error pre-loading\n");
        exit(-1);
    }
}


void close_llama(){
  benchmark->finalize();
}


int main(int argc, char** argv)
{


  double coin;

  open_llama();
  preload_llama();

  printf("Starting real-run\n");
  for(; ;){
	coin = ((double)rand())/RAND_MAX;
	if(coin > frac_read){ // update
	  if (!load_batch_via_writable_graph(*graph, combined_data_source,
			loader_config, run_batch)) break;
	tx_cnt+=run_batch;
	}else{
	  node_t node_id = rand() % MAX_GRAPH_NODES;
	  unsigned long od = benchmark->outdegree(node_id);
	  //printf("outdegree of node, %" PRId64 ": %ld \n",node_id, od);
	  //return_d = run_benchmark(G, benchmark);
	  tx_cnt++;
	}
	if (tx_cnt >= 5000)
	{
	  unsigned long total_elapsed_time = (rtc_clock::current_time() - tx_begin_time);
	  std::cout << "LOAD = "
		<< ((double)1000000 * (tx_cnt)) / total_elapsed_time
		<< " tx/sec " <<std::endl;
	  tx_cnt   = 0;
	  tx_begin_time = rtc_clock::current_time();
	}
  }
	
  close_llama();
  return 0;
}
