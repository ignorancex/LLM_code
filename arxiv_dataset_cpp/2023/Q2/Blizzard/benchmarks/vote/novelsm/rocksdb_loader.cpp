/*
 * Copyright (c) 2015, Intel Corporation
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in
 *       the documentation and/or other materials provided with the
 *       distribution.
 *
 *     * Neither the name of Intel Corporation nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include <assert.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#include <leveldb/db.h>
#include <leveldb/options.h>
#include <leveldb/write_batch.h>
#include "rocksdb.hpp"

#include "logging.hpp"
#include "clock.hpp"

// Rate measurement stuff
leveldb::DB* db = NULL;
const int BATCH_SIZE = 100;
const unsigned long vote_value = 0; // we initialize every article vote to 0

void  print_rocksdb_content(){
 BOOST_LOG_TRIVIAL(info) << "Printing content.";
 leveldb::Iterator* it = db->NewIterator(leveldb::ReadOptions());
  for (it->SeekToFirst(); it->Valid(); it->Next()) {
	  //votekey_t *vkey = (votekey_t *)it->key().ToString().c_str();
	  votekey_t *vkey = (votekey_t *)it->key().data();
	  if(vkey->prefix == ART){
		std::cout << "key : " << vkey->prefix << vkey->art_id  <<
					 " value : " << it->value().ToString()<< std::endl; 
      }else if(vkey->prefix == VOTE){
		unsigned long *val = (unsigned long *) it->value().data();
		std::cout << "key : " << vkey->prefix << vkey->art_id << 
					 " value : " << *val << std::endl; 
	  }else{
		std::cout << "unknown data type" << std::endl;
	  }
  }
  assert(it->status().ok()); // Check for any errors found during the scan
  delete it;
}



void load(unsigned long keys)
{
	// article id -> name key
	votekey_t art_key;
	art_key.prefix = ART;
	art_key.art_id = 0UL;
	// ariticle id -> nvotes
	votekey_t vc_key;
	vc_key.prefix = VOTE;
	vc_key.art_id = 0UL;

  unsigned char value_base[value_sz];
  BOOST_LOG_TRIVIAL(info) << "Start loading.";
  std::cout <<"votekey_t size : " << sizeof(votekey_t) << std::endl;
  unsigned long ts_heartbeat = rtc_clock::current_time();
  for(unsigned long i=0UL;i<keys;i+=BATCH_SIZE) {
    leveldb::WriteBatch batch;
    for(unsigned long j=i;j<i + BATCH_SIZE && j < keys;j++) {
			art_key.art_id = j;				
			leveldb::Slice art_rockskey((const char *)&art_key, sizeof(votekey_t));
			snprintf((char *)&value_base[0],value_sz,"Article #%lu",j);	
			leveldb::Slice art_rocksval((const char *)&value_base[0], value_sz);
			batch.Put(art_rockskey, art_rocksval);

			vc_key.art_id  = j;
			leveldb::Slice vc_rockskey((const char *)&vc_key, sizeof(votekey_t));
			leveldb::Slice vc_rocksval((const char *)&vote_value, sizeof(unsigned long));
			batch.Put(vc_rockskey, vc_rocksval);
    }
    
	leveldb::WriteOptions write_options;
    write_options.sync = false;
    // write_options.disableWAL = true;
    leveldb::Status s = db->Write(write_options, &batch);
    if (!s.ok()){
      BOOST_LOG_TRIVIAL(fatal) << s.ToString();
      exit(-1);
    }
    if(rtc_clock::current_time() > (ts_heartbeat + 60*1000000)) {
      BOOST_LOG_TRIVIAL(info) << "Heartbeat: " << i;
      ts_heartbeat = rtc_clock::current_time();
    }
  }
  BOOST_LOG_TRIVIAL(info) << "Completed loading.";
  //print_rocksdb_content();
}


void opendb(){
  leveldb::Options options;
  options.create_if_missing = true;
  options.error_if_exists   = true;
  // auto env = leveldb::Env::Default();
  //env->set_affinity(0, 10); 
  // env->SetBackgroundThreads(2, leveldb::Env::LOW);
  // env->SetBackgroundThreads(1, leveldb::Env::HIGH);
  // options.env = env;
  // options.PrepareForBulkLoad();
  options.write_buffer_size = 1024 * 1024 * 256;
  // options.target_file_size_base = 1024 * 1024 * 512;
  // options.max_background_compactions = 2;
  // options.max_background_flushes = 1;
  // options.max_write_buffer_number = 3;
  //options.min_write_buffer_number_to_merge = max(num_threads/2, 1);
  // options.compaction_style = leveldb::kCompactionStyleNone;
  //options.memtable_factory.reset(new leveldb::VectorRepFactory(1000));
  leveldb::Status s = leveldb::DB::Open(options, preload_dir, preload_dir, &db);
  if (!s.ok()){
    BOOST_LOG_TRIVIAL(fatal) << s.ToString().c_str();
    exit(-1);
  }
}

void closedb()
{
  delete db;
}

int main(int argc, char *argv[])
{
  opendb();
  load(rocks_keys);
  closedb();
}

