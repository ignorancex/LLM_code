#include <assert.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>

#include <leveldb/db.h>
#include <leveldb/options.h>
#include <leveldb/write_batch.h>

#include "libcyclone.hpp"
#include "logging.hpp"
#include "clock.hpp"
#include "rocksdb.hpp"

leveldb::DB* db = NULL;
void callback(const unsigned char *data, const int len,
										rpc_cookie_t *cookie, unsigned long *pmdk_state)
{
  leveldb::Status s;
  cookie->ret_size   = sizeof(rockskv_t);
  cookie->ret_value  = malloc(cookie->ret_size);

  rockskv_t *rock = (rockskv_t *)data;
  
	if(rock->op == OP_INCR) {
    leveldb::WriteOptions write_options;
    if(use_rocksdbwal) {
      write_options.sync       = true;
      // write_options.disableWAL = false;
    }
    else {
      write_options.sync       = false;
      // write_options.disableWAL = true;
    }
    assert(len == sizeof(rockskv_t) && "unexpected payload size");
		/*
		 * 1. lookup the vote count for given article id
		 * 2. increase the vote count and write back
		 */ 
	votekey_t vkey;
	vkey.prefix = VOTE;
	vkey.art_id = rock->key.art_id;
    leveldb::Slice key((const char *)&vkey, sizeof(votekey_t));
	std::string vcvalue_str;
		leveldb::Status	st = db->Get(leveldb::ReadOptions(), key, &vcvalue_str);
    if(st.IsNotFound()) {
			BOOST_LOG_TRIVIAL(fatal) << __LINE__ << s.ToString() << "rocksdb get : " 
				<< vkey.prefix << vkey.art_id << " slice : " << key.ToString();
			exit(-1);
    }
		unsigned long *vc_value	= (unsigned long *)vcvalue_str.c_str();
		//BOOST_LOG_TRIVIAL(info) << "vote value : " << vkey.art_id << " " << *vc_value;
		*vc_value = *vc_value + rock->prio;
    leveldb::Slice value((const char *)vc_value, sizeof(unsigned long));
    s = db->Put(write_options, key, value);
    if (!s.ok()){
			BOOST_LOG_TRIVIAL(fatal) << __LINE__ << s.ToString() << "rocksdb put : " 
				<< vkey.prefix << vkey.art_id;
			exit(-1);
    }
    memcpy(cookie->ret_value, data, len); // do we really need this echo

  }else if(rock->op == OP_GET){
		/*
		 * 1. get the vote count for ariticle id
		 * 2. get the article name
		 * 3. combine the results and send back
		 */ 
		votekey_t vkey;
		vkey.prefix = VOTE;
		vkey.art_id    = (unsigned long)rock->key.art_id;
		//BOOST_LOG_TRIVIAL(info) << "get request : " << vkey.prefix << vkey.art_id;
    leveldb::Slice count_key((const char *)&vkey, sizeof(votekey_t));
		std::string vote_count;
		leveldb::Status st = db->Get(leveldb::ReadOptions(), count_key, &vote_count);
    if(st.IsNotFound()) {
			BOOST_LOG_TRIVIAL(fatal) << __LINE__ << s.ToString() << "rocksdb get : " 
				<< vkey.prefix << vkey.art_id;
			exit(-1);
    }
		unsigned long *vc_value	= (unsigned long *)vote_count.c_str();

		vkey.prefix = ART;
    leveldb::Slice art_key((const char *)&vkey, sizeof(votekey_t));
		std::string article_name;
    s = db->Get(leveldb::ReadOptions(), art_key, &article_name);
    if(s.IsNotFound()) {
			BOOST_LOG_TRIVIAL(fatal) << __LINE__ << s.ToString() << "rocksdb get : " 
				<< vkey.prefix << vkey.art_id;
			exit(-1);
    }
		//BOOST_LOG_TRIVIAL(info) << article_name;
		//BOOST_LOG_TRIVIAL(info) << *vc_value;
		/// TBD - append vc in to return string
		rockskv_t *rock_back = (rockskv_t *)cookie->ret_value;
    rock_back->key = rock->key;
    memcpy(rock_back->value, article_name.c_str(), value_sz);
  }
 }

int commute_callback(unsigned long cmask1, void *op1, unsigned long cmask2, void *op2)
{
  return 0; // not used 
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



void opendb(){
  leveldb::Options options;
  int num_threads=rocksdb_num_threads;
  options.create_if_missing = true;
  options.write_buffer_size = 1024 * 1024 * 256;
  // options.target_file_size_base = 1024 * 1024 * 512;
  // options.IncreaseParallelism(num_threads);
  // options.max_background_compactions = num_threads;
  // options.max_background_flushes = num_threads;
  // options.max_write_buffer_number = num_threads;
  // options.wal_dir = log_dir;
  //options.env->set_affinity(num_quorums + executor_threads, 
  //			    num_quorums + executor_threads + num_threads);
  leveldb::Status s = leveldb::DB::Open(options, data_dir, data_dir, &db);
  if (!s.ok()){
    BOOST_LOG_TRIVIAL(fatal) << s.ToString().c_str();
    exit(-1);
  }
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
  opendb();
  
  
  dispatcher_start(argv[4], 
		   argv[5], 
		   &rpc_callbacks,
		   server_id, 
		   atoi(argv[2]), 
		   atoi(argv[3]));
}


