
#include<assert.h>
#include<errno.h>
#include<string.h>
#include<stdlib.h>
#include<stdio.h>
#include <time.h>
#include<unistd.h>


#include <libpmemobj++/make_persistent.hpp>
#include <libpmemobj++/transaction.hpp>
#include <libpmemobj++/persistent_ptr.hpp>
#include <libpmemobj++/pool.hpp>
#include <libpmemobj++/experimental/concurrent_hash_map.hpp>

#include "../core/libcyclone.hpp"
#include "../core/logging.hpp"
#include "../core/clock.hpp"
#include "bliz_map.hpp"



using map_type = pmem::obj::experimental::concurrent_hash_map<unsigned long, int>;
map_type::accessor a;

struct root {
	pmem::obj::persistent_ptr<map_type> map_p;
};

pmem::obj::persistent_ptr<struct root> rootp;

pmem::obj::pool<root> pop;

void callback(const unsigned char *data,
		const int len,
		rpc_cookie_t *cookie)
{
	cookie->ret_value = malloc(len);
	cookie->ret_size = len;
	blizop_t *blizop = (blizop_t *)data;	

	if(blizop->op == OP_PUT){
		rootp->map_p->insert(a,blizop->key); 	
		a->second = 4;
		a.release();
	}else if(blizop->op == OP_GET){

	}else{
		BOOST_LOG_TRIVIAL(fatal) << "wrong key-op : "<< blizop->op
			<< "key : "<<blizop->key;
	}
}

int wal_callback(const unsigned char *data,
		const int len,
		rpc_cookie_t *cookie)
{
	return cookie->log_idx;
}


void gc(rpc_cookie_t *cookie)
{
	free(cookie->ret_value);
}

rpc_callbacks_t rpc_callbacks =
{
	callback,
	gc,
	wal_callback
};

#define LAYOUT "hashmap"
#define POOLSIZE 1024*1024*64
#define CREATE_MODE_RW (S_IWUSR | S_IRUSR)
//const char* file = "/dev/shm/concurrent_hash_map";
const char* file = "/mnt/pmem1/blizmap";

void create_map(){
	if (file_exists(file) != 0)
        pop = pmem::obj::pool<root>::create(file, LAYOUT, POOLSIZE,CREATE_MODE_RW);
    else
        pop = pmem::obj::pool<root>::open(file, LAYOUT);

	//pmem::obj::persistent_ptr<struct root> rootp = pmemobj_root(pop, sizeof (struct root));
	rootp = pop.root();
	
	pmem::obj::transaction::run(pop, [&] {
			rootp->map_p = pmem::obj::make_persistent<map_type>(/* optional constructor arguments */);
			});
		
}

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

	create_map();

	dispatcher_start(argv[4],
			argv[5],
			&rpc_callbacks,
			server_id,
			atoi(argv[2]),
			atoi(argv[3]));

}

