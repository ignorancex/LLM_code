#include <cmath>
#include "joins.hpp"

#include "PerfEvent.hpp"
#include "Logger.hpp"
#include "commons.h"
#include "data-types.h"
#include "generator.h"

#include "rdtscpWrapper.h"

#include <sys/prctl.h>

using namespace std;

struct timespec ts_start;

constexpr uint64_t CPMS = CYCLES_PER_MICROSECOND;

int main(int argc, char *argv[]) {
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    initLogger();
    logger(INFO, "Welcome from native!");

    struct table_t tableR;
    struct table_t tableS;
    int64_t results;

    /* Cmd line parameters */
    args_t params;

    /* Set default values for cmd line params */
    params.r_size          = 2097152; /* 2*2^20 */
    params.s_size          = 2097152; /* 2*2^20 */
    params.r_seed          = 11111;
    params.s_seed          = 22222;
    params.nthreads        = 2;
    params.selectivity     = 100;
    params.skew            = 0;
    params.alloc_core      = 0;
    params.sort_r          = 0;
    params.sort_s          = 0;
    params.r_from_path     = 0;
    params.s_from_path     = 0;
    params.materialize     = 0;
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

    logger(DBG, "Number of threads = %d (N/A for every algorithm)", params.nthreads);
    logger(INFO, "One micro second is %d cycles", CPMS);

    seed_generator(params.r_seed);
    if (params.r_from_path)
    {
        logger(INFO, "Build relation R from file %s", params.r_path);
        create_relation_from_file(&tableR, params.r_path, params.sort_r);
        params.r_size = tableR.num_tuples;
    }
    else
    {
        logger(INFO, "Build relation R with size = %.2lf MB (%d tuples)",
               (double) sizeof(struct row_t) * params.r_size/pow(2,20),
               params.r_size);
        create_relation_pk(&tableR, params.r_size, params.sort_r);
    }
    logger(DBG, "DONE");

    seed_generator(params.s_seed);
    if (params.s_from_path)
    {
        logger(INFO, "Build relation S from file %s", params.s_path);
        create_relation_from_file(&tableS, params.s_path, params.sort_s);
        params.s_size = tableS.num_tuples;
    }
    else
    {
        logger(INFO, "Build relation S with size = %.2lf MB (%d tuples)",
               (double) sizeof(struct row_t) * params.s_size/pow(2,20),
               params.s_size);
        if (params.skew > 0) {
            create_relation_zipf(&tableS, params.s_size, params.r_size, params.skew, params.sort_s);
        }
        else if (params.selectivity != 100)
        {
            logger(INFO, "Table S selectivity = %d", params.selectivity);
            uint32_t maxid = params.selectivity != 0 ? (100 * params.r_size / params.selectivity) : 0;
            create_relation_fk_sel(&tableS, params.s_size, maxid, params.sort_s);
        }
        else {
            create_relation_fk(&tableS, params.s_size, params.r_size, params.sort_s);
        }
    }
    logger(DBG, "DONE");

    logger(INFO, "Running algorithm %s", params.algorithm->name);

    // Allow as many cores as necessary
    for (size_t i = 0; i < params.nthreads; ++i) {
        CPU_SET(i, &cpuset);
    }
    rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        logger(ERROR, "Error calling pthread_setaffinity_np: %d", rc);
    }

    if (params.mitigation) {
        // Disable speculative loads after stores (enable speculative store bypass mitigation)
        const int result = prctl(PR_SET_SPECULATION_CTRL, PR_SPEC_STORE_BYPASS, PR_SPEC_DISABLE, 0, 0);

        if (result == 0) {
            logger(INFO, "Successfully enabled speculative store bypass mitigation");
        } else {
            logger(ERROR, "Enabling speculative store bypass mitigation failed with error code %d", result);
        }
    } else {
        logger(INFO, "The speculative store bypass mitigation is not enabled.");
    }

    uint64_t cpu_cntr = 0;
    result_t join_result{};
    auto config = joinconfig_t{};
    config.NTHREADS = params.nthreads;
    config.MATERIALIZE = params.materialize;

    {
        rdtscpWrapper rdtscpWrapper(&cpu_cntr);
        run_join(&join_result, &tableR, &tableS, params.algorithm_name, &config);
    }

    double time_s = (((double) cpu_cntr / CPMS) / 1000000.0);
    logger(INFO, "Total join runtime: %.2fs", time_s);
    logger(INFO, "throughput = %.2lf [M rec / s]",
           (double) (params.r_size + params.s_size) / (time_s));
    logger(INFO, "Matches = %lu", join_result.totalresults);
    delete_relation(&tableR);
    delete_relation(&tableS);
}
