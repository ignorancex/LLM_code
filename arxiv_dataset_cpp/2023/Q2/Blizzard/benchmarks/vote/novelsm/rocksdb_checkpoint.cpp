
#include<assert.h>
#include<errno.h>
#include<string.h>
#include<stdlib.h>
#include<stdio.h>
#include <time.h>
#include<unistd.h>
#include <leveldb/db.h>
#include <leveldb/options.h>
#include <leveldb/write_batch.h>
#include <leveldb/utilities/checkpoint.h>

#include "rocksdb.hpp"
#include "logging.hpp"
#include "clock.hpp"

// Rate measurement stuff
leveldb::DB* db = NULL;

void opendb(){
  leveldb::Options options;
  options.create_if_missing = true;
  //options.error_if_exists   = true;
  auto env = leveldb::Env::Default();
  //env->set_affinity(0, 10); 
  env->SetBackgroundThreads(2, leveldb::Env::LOW);
  env->SetBackgroundThreads(1, leveldb::Env::HIGH);
  options.env = env;
  options.write_buffer_size = 1024 * 1024 * 256;
  options.target_file_size_base = 1024 * 1024 * 512;
  options.max_background_compactions = 2;
  options.max_background_flushes = 1;
  options.max_write_buffer_number = 3;

  leveldb::Status s = leveldb::DB::Open(options, preload_dir, &db);
  if (!s.ok()){
    BOOST_LOG_TRIVIAL(fatal) << s.ToString().c_str();
    exit(-1);
  }
  // Create checkpoint
  leveldb::Checkpoint * checkpoint_ptr;
  s = leveldb::Checkpoint::Create(db, &checkpoint_ptr);
  if (!s.ok()){
    BOOST_LOG_TRIVIAL(fatal) << s.ToString().c_str();
    exit(-1);
  }
  s = checkpoint_ptr->CreateCheckpoint(data_dir);
  if (!s.ok()){
    BOOST_LOG_TRIVIAL(fatal) << s.ToString().c_str();
    exit(-1);
  }
  delete checkpoint_ptr;
}

void closedb()
{
    delete db;
}

int main(int argc, char *argv[])
{
  opendb();
  closedb();
}

