#include "cht/CHTJoin.hpp"
#include "cht/CHTJoinWrapper.hpp"

#ifndef ENCLAVE
#include "ocalls.hpp"
#include <malloc.h>
#else
#include "ocalls_t.h"
#endif

#include <pthread.h>

struct thr_arg_t
{
    int tid;
    CHTJoin *chtJoin;
};

static void * run(void * args)
{
    thr_arg_t * arg = reinterpret_cast<thr_arg_t*>(args);
    arg->chtJoin->join(arg->tid);
    return NULL;
}

constexpr uint64_t CPMS = CYCLES_PER_MICROSECOND;

static void print_result (join_result_t res, uint64_t numtuplesR, uint64_t numtuplesS,
                          struct timers_t timers)
{
    uint64_t numtuples = numtuplesR + numtuplesS;
    double cyclestuple = (double) timers.total / (double)numtuples;
    uint64_t time_usec_tsc = timers.total / CPMS;
    double throughput = (double) numtuples / (double) time_usec_tsc;
    logger(INFO, "Total input tuples : %lu", numtuples);
    logger(INFO, "Result tuples : %lu", res.matches);
    logger(INFO, "Total Join Time (cycles)    : %lu", timers.total);
    logger(INFO, "Partition Overall (cycles)  : %lu", timers.partition);
    logger(INFO, "Build+Join Overall (cycles) : %lu", (timers.build + timers.probe));
    logger(INFO, "Build (cycles)              : %lu", timers.build);
    logger(INFO, "Join (cycles)               : %lu", timers.probe);
    logger(INFO, "Cycles-per-tuple            : %.4lf", cyclestuple);
    logger(INFO, "Cycles-per-Rtuple-partition : %.4lf", (double)timers.partition / numtuplesR);
    logger(INFO, "Cycles-per-tuple-join       : %.4lf", (double) (timers.build + timers.probe) / numtuples);
    logger(INFO, "Cycles-per-Rtuple-build     : %.4lf", (double) (timers.build) / numtuplesR);
    logger(INFO, "Cycles-per-Stuple-probe     : %.4lf", (double) (timers.probe) / numtuplesS);
    logger(INFO, "Total Runtime (us) : %lu ", res.time_usec);
    logger(INFO, "Total RT from TSC (us) : %lu ", time_usec_tsc);
    logger(INFO, "Throughput (M rec/sec) : %.2lf", throughput);
}

template<int numbits>
join_result_t CHTJ(const table_t *relR, const table_t *relS, const joinconfig_t *config)
{
    auto nthreads = config->NTHREADS;
    auto output = (tuple_t *) aligned_alloc(64, sizeof(tuple_t) * relR->num_tuples);

//	numa_localize(output, relR->size, nthreads);

    CHTJoin chtJoin {nthreads, 1<<numbits, relR, relS, output};
    std::vector<pthread_t> threads(nthreads);
//    pthread_attr_t attr;
//    cpu_set_t set;

    std::vector<thr_arg_t> args(nthreads);
//    pthread_attr_init(&attr);

    for (int i = 0; i < nthreads; ++i) {
        int cpu_idx = i % CORES;
//
//        CPU_ZERO(&set);
//        CPU_SET(cpu, &set);
//        pthread_attr_setaffinity_np(&attr, sizeof(set), &set);

        args[i].tid = i;
        args[i].chtJoin = &chtJoin;

        (void) (cpu_idx);
        int rv = pthread_create(&threads[i], NULL, run, (void*)&args[i]);
        if (rv){
//            printf("[ERROR] return code from pthread_create() is %d\n", rv);
//            exit(-1);
            logger(INFO,"return code = %d", rv);
            ocall_throw("Return code from pthread_create() failed");
        }
    }

    for (int i = 0; i < nthreads; ++i) {
        pthread_join(threads[i], NULL);
    }

    join_result_t res = chtJoin.get_join_result();
    struct timers_t timers = chtJoin.get_timers();
    print_result(res, relR->num_tuples, relS->num_tuples, timers);
    free(output);
    return res;
}
template join_result_t CHTJ<7>(const table_t *relR, const table_t *relS, const joinconfig_t *config);
