#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <assert.h>
#include <vector>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <rte_launch.h>
#include <rte_malloc.h>


#include "logging.hpp"
#include "clock.hpp"
#include "libcyclone.hpp"
#include "rocksdb.hpp"

#include "../../common/genzip.hpp"

const double alpha = 1.08;
const int nreqs = 20;   /// TBD: make them commandline params using a common option parser


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
typedef struct cb_st {
	uint8_t request_type;
	unsigned long request_id;
} cb_t;


void async_callback(void *args, int code, unsigned long msg_latency) {
	struct cb_st *cb_ctxt = (struct cb_st *)args;
	if (code == REP_SUCCESS) {
		//BOOST_LOG_TRIVIAL(info) << "received message " << cb_ctxt->request_id;
		cb_ctxt->request_type == OP_INCR ? tx_wr_block_cnt++ : tx_ro_block_cnt++;
		rte_free(cb_ctxt);
	}
	else if (timedout_msgs < TIMEOUT_VECTOR_THRESHOLD) {
		__sync_synchronize();
		//BOOST_LOG_TRIVIAL(info) << "timed-out message " << cb_ctxt->request_id;
		tx_failed_cnt++;
		pthread_mutex_lock(&vlock);
		timeout_vector->push_back(cb_ctxt);
		timedout_msgs++;
		pthread_mutex_unlock(&vlock);
	}

	total_latency += msg_latency;
	if ((tx_wr_block_cnt + tx_ro_block_cnt) >= 5000) {
		unsigned long total_elapsed_time = (rtc_clock::current_time() - tx_begin_time);
		std::cout << "LOAD = "
		          << ((double)1000000 * (tx_wr_block_cnt + tx_ro_block_cnt)) / total_elapsed_time
		          << " tx/sec "
		          << "LATENCY = "
		          << ((double)total_latency) / (tx_wr_block_cnt + tx_ro_block_cnt)
		          << " us "
		          << "wr/rd = "
		          << ((double)tx_wr_block_cnt / tx_ro_block_cnt)
		          << std::endl;
		tx_wr_block_cnt   = 0;
		tx_ro_block_cnt   = 0;
		tx_failed_cnt  = 0;
		total_latency  = 0;
		tx_begin_time = rtc_clock::current_time();
	}
	else if (tx_failed_cnt >= 5000) {
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

typedef struct driver_args_st {
	int leader;
	int me;
	int mc;
	int replicas;
	int clients;
	int partitions;
	int buf_cap;
	void **handles;
	void operator() ()
	{
		(void)driver((void *)this);
	}
} driver_args_t;

int driver(void *arg) {
	unsigned long request_id = 0UL; // wrap around at MAX
	driver_args_t *dargs = (driver_args_t *)arg;
	int me = dargs->me;
	int mc = dargs->mc;
	int replicas = dargs->replicas;
	int clients = dargs->clients;
	int partitions = dargs->partitions;
	void **handles = dargs->handles;
	char *buffer = new char[DISP_MAX_MSGSIZE];

	int ret;
	int rpc_flags;
	int my_core;

	char article_name[value_sz]; // noria uses varchar[16] for its benchmark
	unsigned long keys = rocks_keys;

	double frac_read = ((double)(nreqs - 1)) / nreqs;
	BOOST_LOG_TRIVIAL(info) << "KEYS = " << keys;
	BOOST_LOG_TRIVIAL(info) << "ZIPFIAN (ALPHA) = " << alpha;

	rockskv_t *kv = (rockskv_t *)buffer;
	struct cb_st *cb_ctxt;
	// we populate articles in rocksdb loader
	BOOST_LOG_TRIVIAL(info) << "run bench";
	rand_val(1234);

	for ( int rcount = 0 ; ; rcount = ++rcount % nreqs){
		if (timedout_msgs > 0){
			__sync_synchronize();
			pthread_mutex_lock(&vlock);
			if (timeout_vector->size() > 0){
				cb_ctxt = timeout_vector->at(0);
				timeout_vector->erase(timeout_vector->begin());
				timedout_msgs--;
			}
			pthread_mutex_unlock(&vlock);
		}
		else {
			kv->key.art_id = zipf(alpha, keys);
			if (rcount){ // read request
				rpc_flags = RPC_FLAG_RO;
				kv->op    = OP_GET;
				kv->key.prefix = ART;
			}
			else {   // update request
				rpc_flags = 0;
				kv->op    = OP_INCR;
				kv->key.prefix = VOTE;
				kv->prio = 1;
			}
			my_core  = kv->key.art_id % executor_threads;
			cb_ctxt = (struct cb_st *)rte_malloc("callback_ctxt", sizeof(struct cb_st), 0);
			cb_ctxt->request_type = rpc_flags;
		}

		cb_ctxt->request_id = request_id++;
		do {
			ret = make_rpc_async(handles[0],
			                     buffer,
			                     sizeof(rockskv_t),
			                     async_callback,
			                     (void *)cb_ctxt,
			                     1UL << my_core,
			                     rpc_flags);
			if (ret == EMAX_INFLIGHT){
				//sleep a bit
				continue;
			}
		} while (ret);
	}
		return 0;
	}

	int main(int argc, const char *argv[])
	{
		if (argc != 11) {
			printf("Usage: %s client_id_start client_id_stop mc replicas clients partitions cluster_config quorum_config_prefix server_ports inflight_cap\n", argv[0]);
			exit(-1);
		}

		int client_id_start = atoi(argv[1]);
		int client_id_stop  = atoi(argv[2]);
		driver_args_t *dargs;
		void **prev_handles;
		cyclone_network_init(argv[7], 1, atoi(argv[3]), 2 + client_id_stop - client_id_start);
		driver_args_t ** dargs_array =
		    (driver_args_t **)malloc((client_id_stop - client_id_start) * sizeof(driver_args_t *));
		for (int me = client_id_start; me < client_id_stop; me++) {
			dargs = (driver_args_t *) malloc(sizeof(driver_args_t));
			dargs_array[me - client_id_start] = dargs;
			if (me == client_id_start) {
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
			dargs->buf_cap = atoi(argv[10]);
			dargs->handles = new void *[dargs->partitions];
			BOOST_LOG_TRIVIAL(info) << "no. of partitions: " << dargs->partitions;
			char fname_server[50];
			char fname_client[50];
			for (int i = 0; i < dargs->partitions; i++) {
				sprintf(fname_server, "%s", argv[7]);
				sprintf(fname_client, "%s%d.ini", argv[8], i); // only one raft instance running
				dargs->handles[i] = cyclone_client_init(dargs->me,
				                                        dargs->mc,
				                                        1 + me - client_id_start,
				                                        fname_server,
				                                        atoi(argv[9]),
				                                        fname_client,
				                                        CLIENT_ASYNC,
				                                        dargs->buf_cap);
			}
		}
		for (int me = client_id_start; me < client_id_stop; me++) {
			cyclone_launch_clients(dargs_array[me - client_id_start]->handles[0], driver, dargs_array[me - client_id_start], 1 +  me - client_id_start);
		}
		rte_eal_mp_wait_lcore();
	}
