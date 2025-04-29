#include <assert.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>

#include "../../../core/libcyclone.hpp"
#include "../../../core/logging.hpp"
#include "../../../core/clock.hpp"
#include "libpmemobj.h"


#include "pmemds.h"

static const std::string pmem_path = "/mnt/pmem0/pmemds";
pmemds::PMLib *pmlib;



/*
 * Figure out the proper partitions to operate on
 */
int partition(pm_rpc_t *op){
	int partition;
	unsigned int op_id = OP_ID(op->meta);
	if(op_id == PUT || op_id == INSERT || op_id == INCREASE_PRIO || op_id == DECREASE_PRIO){
		partition = op->key%executor_threads;	
		return partition;
	}
	return INT_MAX; // not being used 
}




void callback(const unsigned char *data,
              const int len,
              rpc_cookie_t *cookie, unsigned long *pmdk_state)
{
	pm_rpc_t *request, *response;
	/* set mbuf/wal commit state as a thread local */
	//TX_SET_BLIZZARD_MBUF_COMMIT_ADDR(pmdk_state);
    request = (pm_rpc_t *) data;
#ifdef __COMMUTE
    pmlib->exec(partition(request), request,&response,&(cookie->ret_size));
#else
    pmlib->exec(0, request,&response,&(cookie->ret_size)); // single partition
#endif
    cookie->ret_value = (void *)response;
}
/*commute rules
 * 1. update operations going in to different partitions commute
 * 2. Read operations commute with each other irrespective of the partition
 * 3. Others do not
 */
int commute_callback(void  *arg1, void *arg2)
{
	pm_rpc_t *op1 = (pm_rpc_t *) arg1;
    pm_rpc_t *op2 = (pm_rpc_t *) arg2;

	unsigned int op1_id = OP_ID(op1->meta);
	unsigned int op2_id = OP_ID(op2->meta);
	// hashmap inserts and queue inserts
	if((op1_id == PUT || op1_id == INSERT || op1_id == INCREASE_PRIO || op1_id == DECREASE_PRIO) &&
			(op2_id == PUT || op2_id == INSERT || op2_id == INCREASE_PRIO || op2_id == DECREASE_PRIO) &&
				(partition(op1) != partition(op2))){
		return 1;
	}
	if(op1_id == GET_TOPK && op2_id == GET_TOPK){
		return 1;
	}
	return 0;
}

void gc(rpc_cookie_t *cookie)
{
    free(cookie->ret_value);
}

rpc_callbacks_t rpc_callbacks = { callback,
								  gc,
								  commute_callback
								};

int main(int argc, char *argv[])
{
    if (argc != 7)
    {
        printf("Usage1: %s replica_id replica_mc clients cluster_config quorum_config ports\n", argv[0]);
        exit(-1);
    }
    int server_id = atoi(argv[1]);
    cyclone_network_init(argv[4],
                         atoi(argv[6]),
                         atoi(argv[2]),
                         atoi(argv[6]) + num_queues * num_quorums + executor_threads);

    assert(pmlib == NULL);
    pmlib = new pmemds::PMLib(pmem_path,executor_threads);
    if (pmlib == nullptr)
    {
        BOOST_LOG_TRIVIAL(fatal) << "cannot open pmemds";
        exit(-1);
    }

    dispatcher_start(argv[4],
                     argv[5],
                     &rpc_callbacks,
                     server_id,
                     atoi(argv[2]),
                     atoi(argv[3]));
}

