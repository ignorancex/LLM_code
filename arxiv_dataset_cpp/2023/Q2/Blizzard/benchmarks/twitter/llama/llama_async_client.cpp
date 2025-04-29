#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <assert.h>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <rte_launch.h>
#include <rte_malloc.h>

#include "clock.hpp"
#include "logging.hpp"
#include "libcyclone.hpp"

#include "llama.hpp"



unsigned long tx_cnt = 0UL;
unsigned long total_latency = 0UL;
unsigned long tx_begin_time = 0UL;

double frac_read = 0.99;

int driver(void *arg);

/* callback context */
typedef struct cb_st
{
	uint8_t request_type;
	unsigned long request_id;
} cb_t;


void async_callback(void *args, int code, unsigned long msg_latency)
{
    struct cb_st *cb_ctxt = (struct cb_st *)args;
    if(code == REP_SUCCESS){
        //BOOST_LOG_TRIVIAL(info) << "received message " << cb_ctxt->request_id;
		tx_cnt++;
        rte_free(cb_ctxt);
    }else{
        BOOST_LOG_TRIVIAL(info) << "timed-out message " << cb_ctxt->request_id;
		exit(-1);
    }

    total_latency += msg_latency;
    if (tx_cnt >= 10000)
    {
        unsigned long total_elapsed_time = (rtc_clock::current_time() - tx_begin_time);
		std::cout << "LOAD = "
            << ((double)1000000 * tx_cnt) / total_elapsed_time
            << " tx/sec "
            << "LATENCY = "
            << ((double)total_latency) / tx_cnt
            << " us "
			<<std::endl;
        tx_cnt   = 0;
        total_latency  = 0;
        tx_begin_time = rtc_clock::current_time();
    }
}


typedef struct driver_args_st {
	int leader;
	int me;
	int mc;
	int replicas;
	int clients;
	int partitions;
	void **handles;
	void operator() ()
	{
		(void)driver((void *)this);
	}
} driver_args_t;

int driver(void *arg)
{
	unsigned long request_id = 0UL; // wrap around at MAX
	driver_args_t *dargs = (driver_args_t *)arg;
	int me = dargs->me;
	int mc = dargs->mc;
	int replicas = dargs->replicas;
	int clients = dargs->clients;
	int partitions = dargs->partitions;
	void **handles = dargs->handles;
	char *buffer = new char[DISP_MAX_MSGSIZE];
	
	srand(time(NULL));
	int ret;
	int rpc_flags;
	int my_core;

	llama_req_t *req = (llama_req_t *)buffer;

	double coin;
	
	BOOST_LOG_TRIVIAL(info) << "MAX_NODES = " << max_llama_nodes;
	BOOST_LOG_TRIVIAL(info) << "FRAC_READ = " << frac_read;

    unsigned long fromnode_id, tonode_id,node_id;
    /* FILE *fp;
    fp = fopen(TWITTER_DATA_FILE, "r");
    if(fp == NULL){
        BOOST_LOG_TRIVIAL(info) << "could not open twitter data file";
        exit(-1);
    } */
	
	struct cb_st *cb_ctxt;
	srand(rtc_clock::current_time());
	for( ; ; ){
			double coin = ((double)rand())/RAND_MAX;
			if(coin > frac_read) {
				/* if(fscanf(fp,"%lu %lu", &tonode_id, &fromnode_id) != 2){
					BOOST_LOG_TRIVIAL(info) << "error reading file values, exiting";
					exit(-1);
				} */

				rpc_flags = 0;
				req->op    = OP_ADD_EDGE;
				req->data1 = tonode_id;
				req->data2 = fromnode_id;
			}
			else {
				unsigned long node_id = rand() % max_llama_nodes; // TODO: use adaptive max value
				req->op    = OP_OUTDEGREE;
				rpc_flags = RPC_FLAG_RO;
				req->data1 = node_id;
			}

			my_core = executor_threads-1;
			cb_ctxt = (struct cb_st *)rte_malloc("callback_ctxt", sizeof(struct cb_st), 0);
			cb_ctxt->request_type = rpc_flags;
		
			cb_ctxt->request_id = request_id++;
		do{
            ret = make_rpc_async(handles[0],
                    buffer,
                    sizeof(llama_req_t),
                    async_callback,
                    (void *)cb_ctxt,
                    1UL << my_core,
                    rpc_flags);
            if(ret == EMAX_INFLIGHT){
                continue;
            }
        }while(ret);
	}
	return 0;
}

int main(int argc, const char *argv[]) {
	if(argc != 11) {
		printf("Usage: %s client_id_start client_id_stop mc replicas" 
				"clients partitions cluster_config quorum_config_prefix" 
				"server_ports inflight_cap\n", argv[0]);
		exit(-1);
	}

	int client_id_start = atoi(argv[1]);
	int client_id_stop  = atoi(argv[2]);
	driver_args_t *dargs;
	void **prev_handles;
	cyclone_network_init(argv[7], 1, atoi(argv[3]), 
			2 + client_id_stop - client_id_start); // 2 - sync and async queues
	driver_args_t ** dargs_array =
		(driver_args_t **)malloc((client_id_stop - client_id_start)*sizeof(driver_args_t *));
	for(int me = client_id_start; me < client_id_stop; me++) {
		dargs = (driver_args_t *) malloc(sizeof(driver_args_t));
		dargs_array[me - client_id_start] = dargs;
		if(me == client_id_start) {
			dargs->leader = 1;
		}
		else {
			dargs->leader = 0;
		}
		dargs->me = me;
		dargs->mc = atoi(argv[3]);
		dargs->replicas = atoi(argv[4]);
		dargs->clients  = atoi(argv[5]);
		dargs->partitions = atoi(argv[6]);
		dargs->handles = new void *[dargs->partitions];
		char fname_server[50];
		char fname_client[50];
		for(int i=0;i<dargs->partitions;i++) {
			sprintf(fname_server, "%s", argv[7]);
			sprintf(fname_client, "%s%d.ini", argv[8], i);
			dargs->handles[i] = cyclone_client_init(dargs->me,
					dargs->mc,
					1 + me - client_id_start,
					fname_server,
					atoi(argv[9]),
					fname_client,
					CLIENT_ASYNC,
					atoi(argv[10]));
		}
	}
	for (int me = client_id_start; me < client_id_stop; me++){
        cyclone_launch_clients(dargs_array[me-client_id_start]->handles[0],
				driver, dargs_array[me-client_id_start], 1+  me-client_id_start);
    } 
	rte_eal_mp_wait_lcore();
}
