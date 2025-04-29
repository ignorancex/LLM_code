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

#include "logging.hpp"
#include "clock.hpp"
#include "libcyclone.hpp"
#include "dpdk_client.hpp"
#include "vector/adjvector-client.h"



#define TWITTER_DATA_FILE "/home/pfernando/cyclone/cyclone.git/benchmarks/data/twitter/twitter_rv_15066953.net"

/* right now we statically allocate node list */
#define MAX_GRAPH_NODES 10000

/* pmem structure names */
const uint16_t adjvector_st = 0;


int driver(void *arg);

typedef struct driver_args_st
{
	int leader;
	int me;
	int mc;
	int replicas;
	int clients;
	int partitions;
	int buf_cap;
	pmemdsclient::DPDKPMClient *dpdkClient;
	pmemdsclient::AdjVectorEngine *adjVector;
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
	pmemdsclient::DPDKPMClient *pmlib = dargs->dpdkClient;
	pmemdsclient::AdjVectorEngine *adjv = dargs->adjVector;
	srand(time(NULL));
	int ret;
	int rpc_flags;
	int my_core;

	double coin;
	double frac_read = 0.5;
	//twitter bench specific
	unsigned long fromnode_id, tonode_id,node_id;
	FILE *fp;
	fp = fopen(TWITTER_DATA_FILE, "r");
	if(fp == NULL){
		BOOST_LOG_TRIVIAL(info) << "could not open twitter data file";
		exit(-1);
	}
	
	unsigned  long key;
	char value_buffer[64];
	srand(rtc_clock::current_time());

	pmlib->open("twitterApp");
	uint8_t creation_flag = 0;
	adjv->create(creation_flag);

	for(int i=0 ;i<100000 ;i++ ){ // pre-loading some values
		if(fscanf(fp,"%lu %lu", &tonode_id, &fromnode_id) != 2){
			BOOST_LOG_TRIVIAL(info) << "error reading file values, exiting";
			exit(-1);
		}			
		//BOOST_LOG_TRIVIAL(info) << "preload, add_edge from -> to : " << fromnode_id  << " -> "<< tonode_id;
		adjv->add_edge(fromnode_id,tonode_id);
	}
	for( ; ; ){
		coin = ((double)rand())/RAND_MAX;
		if(coin > frac_read){ // update
			if(fscanf(fp,"%lu %lu", &tonode_id, &fromnode_id) != 2){			
				BOOST_LOG_TRIVIAL(info) << "error reading file values, exiting";
				exit(-1);
			}
			//BOOST_LOG_TRIVIAL(info) << "add_edge from -> to : " << fromnode_id << " -> " << tonode_id;
			adjv->add_edge(fromnode_id,tonode_id, nullptr);
		}else {
			unsigned long node_id = rand() % MAX_GRAPH_NODES;
			//BOOST_LOG_TRIVIAL(info) << "vertex_outdegree of :" << node_id;
			adjv->vertex_outdegree(node_id, nullptr);
		}
	}
	adjv->close(nullptr);
	pmlib->close(nullptr);
	fclose(fp);
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
	cyclone_network_init(argv[7], 1, atoi(argv[3]), 2 + client_id_stop - client_id_start);
	driver_args_t ** dargs_array =
		(driver_args_t **)malloc((client_id_stop - client_id_start) * sizeof(driver_args_t *));
	//BOOST_LOG_TRIVIAL(info) << "no. clients: " << (client_id_stop - client_id_start);
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
		dargs->buf_cap = atoi(argv[10]);
		dargs->handles = new void *[dargs->partitions];
		BOOST_LOG_TRIVIAL(info) << "no. of partitions: " << dargs->partitions;
		char fname_server[50]/* IMPORTANT - set to large enough value */;
		char fname_client[50];
		for (int i = 0; i < dargs->partitions; i++){
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
			BOOST_LOG_TRIVIAL(info) << "init dpdkclient and adjacency-vector";
			dargs->dpdkClient = new pmemdsclient::DPDKPMClient(dargs->handles[i]);
			dargs->adjVector = new pmemdsclient::AdjVectorEngine(dargs->dpdkClient,adjvector_st,1000,1UL);
		}
	}
	for (int me = client_id_start; me < client_id_stop; me++){
		cyclone_launch_clients(dargs_array[me-client_id_start]->handles[0],driver, dargs_array[me-client_id_start], 1+  me-client_id_start);
	}
	rte_eal_mp_wait_lcore();
}
