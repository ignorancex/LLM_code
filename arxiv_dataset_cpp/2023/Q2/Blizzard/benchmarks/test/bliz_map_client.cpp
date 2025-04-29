
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

#include <vector>

#include "../core/logging.hpp"
#include "../core/clock.hpp"
#include "../core/libcyclone.hpp"
#include "bliz_map.hpp"


unsigned long tx_wr_block_cnt = 0UL;
unsigned long tx_ro_block_cnt = 0UL;
unsigned long tx_failed_cnt = 0UL;
unsigned long total_latency = 0UL;
unsigned long tx_begin_time = 0UL;
std::vector<struct cb_st *> *timeout_vector;
pthread_mutex_t vlock = PTHREAD_MUTEX_INITIALIZER;
volatile int64_t timedout_msgs = 0;
int driver(void *arg);

#define TIMEOUT_VECTOR_THRESHOLD 32
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
		cb_ctxt->request_type == OP_PUT ? tx_wr_block_cnt++ : tx_ro_block_cnt++;
		rte_free(cb_ctxt);
	}else if (timedout_msgs < TIMEOUT_VECTOR_THRESHOLD){
		__sync_synchronize();
		//BOOST_LOG_TRIVIAL(info) << "timed-out message " << cb_ctxt->request_id;
		tx_failed_cnt++;
		pthread_mutex_lock(&vlock);
		timeout_vector->push_back(cb_ctxt);
		timedout_msgs++;
		//__sync_fetch_and_add(&timedout_msgs, 1);
		pthread_mutex_unlock(&vlock);
	}

	total_latency += msg_latency;
	if ((tx_wr_block_cnt + tx_ro_block_cnt) >= 5000)
	{
		unsigned long total_elapsed_time = (rtc_clock::current_time() - tx_begin_time);
		BOOST_LOG_TRIVIAL(info) << "LOAD = "
			<< ((double)1000000 * (tx_wr_block_cnt + tx_ro_block_cnt)) / total_elapsed_time
			<< " tx/sec "
			<< "LATENCY = "
			<< ((double)total_latency) / (tx_wr_block_cnt + tx_ro_block_cnt)
			<< " us "
			<< "wr/rd = "
			<<((double)tx_wr_block_cnt/tx_ro_block_cnt);
		tx_wr_block_cnt   = 0;
		tx_ro_block_cnt   = 0;
		tx_failed_cnt  = 0;
		total_latency  = 0;
		tx_begin_time = rtc_clock::current_time();
	}else if(tx_failed_cnt >= 5000){
		unsigned long total_elapsed_time = (rtc_clock::current_time() - tx_begin_time);
		BOOST_LOG_TRIVIAL(info) << "Ecsessive message timing out "
			<< " timedout count "
			<< tx_failed_cnt
			<< " success count "
			<< tx_wr_block_cnt + tx_ro_block_cnt
			<< " elpsd time "
			<< total_elapsed_time;
		tx_wr_block_cnt   = 0;
		tx_ro_block_cnt   = 0;
		tx_failed_cnt  = 0;
		total_latency  = 0;
		tx_begin_time = rtc_clock::current_time();
	}
}

typedef struct driver_args_st
{
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
	//struct proposal *prop = (struct proposal *)buffer;
	srand(time(NULL));
	int ret;
	int rpc_flags;
	int my_core;

	blizop_t *blizop = (blizop_t *)buffer;

	double frac_read = 0.5;
	const char *frac_read_env = getenv("KV_FRAC_READ");
	if (frac_read_env != NULL)
	{
		frac_read = atof(frac_read_env);
	}
	BOOST_LOG_TRIVIAL(info) << "FRAC_READ = " << frac_read;


	unsigned long keys = blizmap_keys;
	const char *keys_env = getenv("KV_KEYS");
	if (keys_env != NULL)
	{
		keys = atol(keys_env);
	}
	BOOST_LOG_TRIVIAL(info) << "KEYS = " << keys;

	srand(rtc_clock::current_time());
	struct cb_st *cb_ctxt;
	for( ; ; ){
		if(timedout_msgs > 0){ // re-try timedout messages
			__sync_synchronize();
			pthread_mutex_lock(&vlock);
			if(timeout_vector->size() > 0){
				cb_ctxt = timeout_vector->at(0);
				timeout_vector->erase(timeout_vector->begin());
				//__sync_fetch_and_sub(&timedout_msgs, 1);
				timedout_msgs--;
				//TODO: populate new content for timeout messages?
			}
			pthread_mutex_unlock(&vlock);

		}else{
			blizop->key   = rand() % keys;
			my_core = blizop->key % executor_threads;
			double coin = ((double)rand()) / RAND_MAX;
			if (coin > frac_read){
				rpc_flags = 0;
				blizop->op    = OP_PUT;
				snprintf(blizop->value,value_sz,"v%ld\n",blizop->key);
			}
			else{
				rpc_flags = RPC_FLAG_RO;
				blizop->op    = OP_GET;
			}


			cb_ctxt = (struct cb_st *)rte_malloc("callback_ctxt", sizeof(struct cb_st), 0);
			cb_ctxt->request_type = rpc_flags;
		}
		cb_ctxt->request_id = request_id++;
		// BOOST_LOG_TRIVIAL(info) << "sent message id  " << cb_ctxt->request_id;
		do{
			ret = make_rpc_async(handles[0],
					buffer,
					sizeof(blizop_t),
					async_callback,
					(void *)cb_ctxt,
					1UL << my_core,
					rpc_flags);
			if(ret == EMAX_INFLIGHT){
				//sleep a bit
				continue;
			}
		}while(ret);
	}
	return 0;
}

int main(int argc, const char *argv[])
{
	if (argc != 11)
	{
		printf("Usage: %s client_id_start client_id_stop mc replicas clients partitions cluster_config quorum_config_prefix server_ports inflight_cap\n", argv[0]);
		exit(-1);
	}

	int client_id_start = atoi(argv[1]);
	int client_id_stop  = atoi(argv[2]);
	driver_args_t *dargs;
	void **prev_handles;
	timeout_vector = new std::vector<struct cb_st *>();
	cyclone_network_init(argv[7], 1, atoi(argv[3]), 2 + client_id_stop - client_id_start);
	driver_args_t ** dargs_array =
		(driver_args_t **)malloc((client_id_stop - client_id_start) * sizeof(driver_args_t *));
	for (int me = client_id_start; me < client_id_stop; me++)
	{
		dargs = (driver_args_t *) malloc(sizeof(driver_args_t));
		dargs_array[me - client_id_start] = dargs;
		if (me == client_id_start)
		{
			dargs->leader = 1;
		}
		else
		{
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
		for (int i = 0; i < dargs->partitions; i++){
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
		cyclone_launch_clients(dargs_array[me-client_id_start]->handles[0],driver, dargs_array[me-client_id_start], 1+  me-client_id_start);
	}
	rte_eal_mp_wait_lcore();
}
