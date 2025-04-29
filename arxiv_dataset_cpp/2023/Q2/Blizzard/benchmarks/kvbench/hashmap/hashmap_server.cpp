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

void callback(const unsigned char *data,
              const int len,
              rpc_cookie_t *cookie, unsigned long *pmdk_state)
{
	pm_rpc_t *request, *response;
    assert((sizeof(pm_rpc_t) == len) && "wrong payload length");
	/* set mbuf/wal commit state as a thread local */
	//TX_SET_BLIZZARD_MBUF_COMMIT_ADDR(pmdk_state);
    request = (pm_rpc_t *) data;
    pmlib->exec(0,request,&response,&(cookie->ret_size));
	cookie->ret_value = (void *)response;
}

int commute_callback(void *arg1, void *arg2)
{
	pm_rpc_t *op1 = (pm_rpc_t *) arg1;
	pm_rpc_t *op2 = (pm_rpc_t *) arg2;

    /* this is the signle partition data-structure. So we ignore partitions */
	unsigned int op1_id = OP_ID(op1->meta);
	unsigned int op2_id = OP_ID(op2->meta);
	/* the key inserts and reads can operate in parallel as long as they do
	 * not operate on the same key.
	 */ 
	if( (op1_id == PUT || op1_id == GET) &&
			(op2_id == PUT || op2_id == GET) ){
		if(op1->key != op2->key){
			return 1;
		}
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
    pmlib = new pmemds::PMLib(pmem_path);
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

