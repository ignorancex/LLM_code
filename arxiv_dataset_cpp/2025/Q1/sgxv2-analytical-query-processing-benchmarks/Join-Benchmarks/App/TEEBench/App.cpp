/*
 * Copyright (C) 2011-2020 Intel Corporation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the
 *     distribution.
 *   * Neither the name of Intel Corporation nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
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
 *
 */


#include <cstring>
#include <ctime>
#include <pwd.h>
#include <pthread.h>

#include "ErrorSupport.h"
#include "Logger.hpp"
#include "commons.h"
#include "data-types.h"
#include "generator.h"
#include "sgx_urts.h"
#include "Enclave_u.h"

constexpr uint64_t CPMS = CYCLES_PER_MICROSECOND;

/* Global EID shared by multiple threads */
sgx_enclave_id_t global_eid = 0;

struct timespec ts_start;
char algorithm_name[128];

/* Initialize the enclave:
 *   Call sgx_create_enclave to initialize an enclave instance
 */
int initialize_enclave(void) {
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;

    /* Call sgx_create_enclave to initialize an enclave instance */
    /* Debug Support: set 2nd parameter to 1 */
    logger(INFO, "Creating Enclave...");
    ret = sgx_create_enclave("joinenclave.signed.so", 1, nullptr, nullptr, &global_eid, nullptr);
    if (ret != SGX_SUCCESS) {
        ret_error_support(ret);
        return -1;
    }
    logger(INFO, "Done. Enclave id = %d", global_eid);
    return 0;
}

/* Application entry */
int SGX_CDECL main(int argc, char *argv[]) {
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    struct table_t tableR;
    struct table_t tableS;
    result_t *results;
    sgx_status_t ret;
    struct timespec tw1, tw2;

    /* Cmd line parameters */
    args_t params;

    /* Set default values for cmd line params */
//    params.algorithm    = &algorithms[0]; /* NPO_st */
    params.r_size = 2097152; /* 2*2^20 */
    params.s_size = 2097152; /* 2*2^20 */
    params.r_seed = 11111;
    params.s_seed = 22222;
    params.nthreads = 2;
    params.selectivity = 100;
    params.skew = 0;
    params.alloc_core = 0;
    params.sort_r = 0;
    params.sort_s = 0;
    params.r_from_path = 0;
    params.s_from_path = 0;
    params.materialize = 0;
    strcpy(params.algorithm_name, "RHO");

    initLogger();
    parse_args(argc, argv, &params, nullptr);

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(params.alloc_core, &cpuset);
    auto rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        logger(ERROR, "Error calling pthread_setaffinity_np: %d", rc);
    }

    logger(INFO, "Welcome from TEEBench Enclave!");
    logger(INFO, "Number of threads = %d (N/A for every algorithm)", params.nthreads);
    logger(INFO, "One micro second is %d cycles", CPMS);

    seed_generator(params.r_seed);

    if (params.r_from_path) {
        logger(INFO, "Build relation R from file %s", params.r_path);
        create_relation_from_file(&tableR, params.r_path, params.sort_r);
        params.r_size = tableR.num_tuples;
    } else {
        logger(INFO, "Build relation R with size = %.2lf MB (%u tuples)",
               B_TO_MB(sizeof(struct row_t) * params.r_size),
               params.r_size);
        create_relation_pk(&tableR, params.r_size, params.sort_r);
//        if (params.selectivity != 100)
//        {
//            create_relation_pk_selectivity(&tableR, params.r_size, params.sort_r, params.selectivity);
//        }
//        else
//        {
//            create_relation_pk(&tableR, params.r_size, params.sort_r);
//        }
    }
    logger(DBG, "DONE");

    seed_generator(params.s_seed);
    if (params.s_from_path) {
        logger(INFO, "Build relation S from file %s", params.s_path);
        create_relation_from_file(&tableS, params.s_path, params.sort_s);
        params.s_size = tableS.num_tuples;
    } else {
        logger(INFO, "Build relation S with size = %.2lf MB (%u tuples)",
               B_TO_MB(sizeof(struct row_t) * params.s_size),
               params.s_size);
        if (params.skew > 0) {
            logger(INFO, "Skew relation: %.2lf", params.skew);
            create_relation_zipf(&tableS, params.s_size, params.r_size, params.skew, params.sort_s);
        } else if (params.selectivity != 100) {
            logger(INFO, "Table S selectivity = %d", params.selectivity);
            uint32_t maxid = params.selectivity != 0 ? (100 * params.r_size / params.selectivity) : 0;
            create_relation_fk_sel(&tableS, params.s_size, maxid, params.sort_s);
        } else {
            create_relation_fk(&tableS, params.s_size, params.r_size, params.sort_s);
        }
    }

    logger(DBG, "DONE");

    initialize_enclave();
    ecall_init_logger(global_eid, logger_get_start());

    logger(INFO, "Running algorithm %s", params.algorithm_name);

    clock_gettime(CLOCK_MONOTONIC, &tw1); // POSIX; use timespec_get in C11
    uint64_t cpu_counter = 0;
    uint64_t mutex_cpu_cntr = 0;
    ecall_preload_relations(global_eid,
                            &tableR,
                            &tableS);
    strcpy(algorithm_name, params.algorithm_name);

    // Allow as many cores as necessary
    for (size_t i = 0; i < params.nthreads; ++i) {
        CPU_SET(i, &cpuset);
    }
    rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        logger(ERROR, "Error calling pthread_setaffinity_np: %d", rc);
    }

    auto config = joinconfig_t{};
    config.NTHREADS = params.nthreads;
    config.MATERIALIZE = params.materialize;
    config.ALLOC_CORE = params.alloc_core;

    ret = ecall_join_preload(global_eid, params.algorithm_name, &config);

    clock_gettime(CLOCK_MONOTONIC, &tw2);
    double time_s = (((double) cpu_counter / CPMS) / 1000000.0);
    logger(INFO, "Total join runtime: %.2fs", time_s);
    logger(INFO, "throughput = %.2lf [M rec / s]",
           (double) (params.r_size + params.s_size) / (time_s));
    if (ret != SGX_SUCCESS) {
        ret_error_support(ret);
    }
    sgx_destroy_enclave(global_eid);
    delete_relation(&tableR);
    delete_relation(&tableS);
    return 0;
}
