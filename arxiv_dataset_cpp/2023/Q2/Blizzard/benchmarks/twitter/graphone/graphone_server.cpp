
#include <assert.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <omp.h>
#include <libgen.h>
// #include <benchmark.h>
#include <inttypes.h>
#include <time.h>

#include "../core/libcyclone.hpp"
#include "../core/logging.hpp"
#include "../core/clock.hpp"

#include "graphone.hpp"
#include "graph_interface.h"

// typedef ll_mlcsr_ro_graph benchmarkable_graph_t;
// #define benchmarkable_graph(g)  ((g)->ro_graph())

std::string input_file = "/home/pfernando/deploy-cyclone/benchmarks/data/twitter/twitter_rv_15066953.net";
std::string database_directory = "/mnt/pmem0/llama_db";
const int preload_batch= 100*1000;
const int write_batch = 1 << 12;

/* batching related structure */
typedef struct b_edge_st{
	unsigned long from;
	unsigned long to;
}b_edge_t;


// template <class Graph>
// class ll_t_outdegree : public ll_benchmark<Graph> {
// 	FILE* _out;
// 	public:

// 	ll_t_outdegree() : ll_benchmark<Graph>("") {
// 		_out = stdout;
// 	}

// 	virtual ~ll_t_outdegree(void) {
// 	}

// 	unsigned long outdegree(node_t nodeId){
// 		Graph& G = *this->_graph;
// 		return G.out_degree(nodeId);
// 	}

// 	virtual double run(void){
// 		// making compiler happy
// 		return NAN;
// 	}

// };


// ll_loader_config loader_config;
// ll_database *database;
// ll_writable_graph *graph;
// ll_file_loaders loaders;
// ll_file_loader *loader;
// ll_concat_data_source combined_data_source;
// ll_data_source *d;
// benchmarkable_graph_t *G;
// ll_t_outdegree<benchmarkable_graph_t> *benchmark;

// int num_threads = 4;
// // edge_t edge_buffer[LLAMA_INGEST_BATCH_SIZE];
int batch_counter = 0;


// bool load_batch_via_writable_graph(ll_writable_graph *graph,
// 		ll_data_source& data_source, const ll_loader_config& loader_config,
// 		size_t batch_size) {


// 	bool loaded = data_source.pull(graph, batch_size);
// 	if (!loaded) return false;
// 	graph->checkpoint(&loader_config);
// 	return true;
// }


void callback(const unsigned char *data,
		const int len,
		rpc_cookie_t *cookie, unsigned long *pmdk_state)
{
	cookie->ret_value  = malloc(sizeof(llama_res_t));
	cookie->ret_size   = sizeof(llama_res_t);
	llama_res_t *res = (llama_res_t *) cookie->ret_value;

	llama_req_t *req = (llama_req_t *)data;

	if(req->op == OP_ADD_EDGE){
		batch_counter++;
		if(batch_counter > write_batch){
                    g_waitfor_archive();
                    batch_counter = 0;
                    // g_dump();
		}
                g_add_edge(req->data1, req->data2);
                // BOOST_LOG_TRIVIAL(info) << "edge added";
		res->state = 1UL;
	}else if(req->op == OP_OUTDEGREE){
                            // BOOST_LOG_TRIVIAL(info) << "Q1";
            res->outdegree = g_query_outdegree(req->data1);//benchmark->outdegree(req->data1);
                            // BOOST_LOG_TRIVIAL(info) << "QUERY";
		res->state = 1UL; // we hard code the return status for now
	}
}

void gc(rpc_cookie_t *cookie)
{
	free(cookie->ret_value);
}

rpc_callbacks_t rpc_callbacks =  {
	callback,
	gc,
	NULL
};

/* opening and pre-loading the db */
// void open_llama(){
// 	const char* first_input = input_file.c_str();
// 	const char* file_type = ll_file_extension(first_input);

// 	if (strcmp(ll_file_extension(input_file.c_str()), file_type)!=0) {
// 		fprintf(stderr, "Error: All imput files must have the same "
// 				"file extension.\n");
// 		exit(-1);
// 	}

// 	// Open the database
// 	database = new  ll_database(database_directory.c_str());
// 	database->set_num_threads(num_threads);
// 	graph = database->graph();

// 	// Load the graph
// 	loader = loaders.loader_for(first_input);
// 	if (loader == NULL) {
// 		fprintf(stderr, "Error: Unsupported input file type\n");
// 		exit(-1);
// 	}

// 	d = loader->create_data_source(input_file.c_str());
// 	combined_data_source.add(d);
// 	G = &(benchmarkable_graph(graph));
// 	benchmark = new ll_t_outdegree<benchmarkable_graph_t>();
// 	benchmark->initialize(G);
// }


// void preload_llama(){
// 	if (!load_batch_via_writable_graph(graph, combined_data_source,
// 				loader_config, preload_batch)){
// 		printf("error pre-loading\n");
// 		exit(-1);
// 	}
// 	BOOST_LOG_TRIVIAL(info) << "pre-loaded data in to twitter graph..";
// }
// void close_llama(){
// 	benchmark->finalize();
// 	delete benchmark;
// 	delete database;
// }

void preload(){
    // for (int i = 0; i < 100000; ++i){
    //     unsigned long fromnode_id = rand() % max_llama_nodes;
    //     unsigned long tonode_id = rand() % max_llama_nodes;
    //     add_edge(fromnode_id,tonode_id);
    // }
    // BOOST_LOG_TRIVIAL(info) << "pre-loaded data";

    FILE *fp;
    fp = fopen("/mnt/pmem0/twitter/twitter_rv_15066953.net", "r");
    if(fp == NULL){
        BOOST_LOG_TRIVIAL(info) << "could not open twitter data file";
        exit(-1);
    }

    unsigned long max = 0, i = 0;
    unsigned long tonode_id, fromnode_id;
    while(EOF != fscanf(fp, "%lu %lu", &tonode_id, &fromnode_id)){
        // max = std::max(max, std::max(tonode_id, fromnode_id));

        g_add_edge(fromnode_id,tonode_id);
        ++i;
        if (i % (1<<15) == 0){
            g_waitfor_archive();
        }
        if (i % 1000000 == 0){
            BOOST_LOG_TRIVIAL(info) << "Loading... " << i;
        }

        if (i > preload_batch){
            break;
        }
    }

    BOOST_LOG_TRIVIAL(info) << "done preloading " << max;
}

int main(int argc, char *argv[])
{
	if(argc != 7) {
		printf("Usage1: %s replica_id replica_mc clients cluster_config quorum_config ports\n", argv[0]);
		exit(-1);
	}

	int server_id = atoi(argv[1]);
	cyclone_network_init(argv[4],
			atoi(argv[6]),
			atoi(argv[2]),
			atoi(argv[6]) + num_queues*num_quorums + executor_threads);

        srand(rtc_clock::current_time());
	// open_llama();
	// preload_llama();
        // uint64_t num_vertex = max_llama_nodes;
        int s = init_and_create_graph();
        preload();
        // g_dump();
        // std::cout << "should be 2: " << g_query_outdegree(2) << std::endl;
	dispatcher_start(argv[4],
			argv[5],
			&rpc_callbacks,
			server_id,
			atoi(argv[2]),
			atoi(argv[3]));
}


