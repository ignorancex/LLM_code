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


#include<assert.h>
#include<errno.h>
#include "../../../core/libcyclone.hpp"
#include<string.h>
#include<stdlib.h>
#include "../../../core/logging.hpp"
#include "../../../core/clock.hpp"
#include<stdio.h>
#include <time.h>
#include<unistd.h>

// Rate measurement stuff
static unsigned long *marks;
static unsigned long *completions;


void callback(const unsigned char *data,
	      const int len,
	      rpc_cookie_t *cookie, unsigned long *pmdk_state)
{
  cookie->ret_value  = NULL;
  cookie->ret_size   = 0;
  cookie->ret_value  = malloc(len);
  cookie->ret_size   = len;
  memcpy(cookie->ret_value, data, len);
}

// we explicitly set serial processing. All the commute
// scheduling techniques will be engaged. But the request
// processing will be done serially. This will give us the 
// worst case overhead of commute scheduling optimizaiton.
int commute_callback(void *arg1, void *arg2)
{
  return 0; 
  // return 1; 
}

void gc(rpc_cookie_t *cookie)
{
  free(cookie->ret_value);
}

rpc_callbacks_t rpc_callbacks =  {
  callback,
  gc,
  commute_callback
};

int main(int argc, char *argv[])
{
  if(argc != 7) {
    printf("Usage1: %s replica_id replica_mc clients cluster_config quorum_config ports\n", argv[0]);
    exit(-1);
  }
  marks       = (unsigned long *)malloc(executor_threads*sizeof(unsigned long));
  completions = (unsigned long *)malloc(executor_threads*sizeof(unsigned long));
  memset(marks, 0, executor_threads*sizeof(unsigned long));
  memset(completions, 0, executor_threads*sizeof(unsigned long));
  int server_id = atoi(argv[1]);
/*

//gdb attach debug
{
    int i = 0;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    fflush(stdout);
    while (0 == i)
        sleep(5);
}
*/
  cyclone_network_init(argv[4],
		       atoi(argv[6]),
		       atoi(argv[2]),
		       atoi(argv[6]) + num_queues*num_quorums + executor_threads);
  dispatcher_start(argv[4], 
		   argv[5], 
		   &rpc_callbacks,
		   server_id, 
		   atoi(argv[2]), 
		   atoi(argv[3]));
}


