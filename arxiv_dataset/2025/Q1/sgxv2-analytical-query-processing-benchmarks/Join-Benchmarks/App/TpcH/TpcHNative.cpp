#include "Logger.hpp"
#include "TpcHCommons.hpp"
#include "tpch.hpp"
#include "Barrier.hpp"
#include <thread>
#include <vector>
#include <malloc.h>

char experiment_filename[512];
int write_to_file = 0;

int
main(int argc, char *argv[]) {
    initLogger();
    joinconfig_t joinconfig;
    logger(INFO, "************* TPC-H APP *************");
    // 1. Parse args
    tcph_args_t params;
    tpch_parse_args(argc, argv, &params);
    uint8_t query = params.query;
    joinconfig.NTHREADS = (int) params.nthreads;
    joinconfig.RADIXBITS = -1;
    logger(INFO, "Run Q%d (scale %d) with join algorithm %s (%d threads)", query, params.scale, params.algorithm_name,
           joinconfig.NTHREADS);

    // 2. load required TPC-H tables
    LineItemTable l {0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    OrdersTable o {0, nullptr, nullptr, nullptr};
    CustomerTable c {0, nullptr, nullptr, nullptr};
    PartTable p {0, nullptr, nullptr, nullptr, nullptr};
    NationTable n {0, nullptr};

    constexpr uint64_t num_threads = 16;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (auto cid = 0; cid < 16; ++cid) {
        CPU_SET(cid, &cpuset);
    }
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        logger(ERROR, "Error calling pthread_setaffinity_np: %d", rc);
    }

    logger(INFO, "Hogging memory");
    mallopt(M_TRIM_THRESHOLD, -1);
    mallopt(M_MMAP_MAX, 0);
    mallopt(M_TOP_PAD, INT32_MAX);
    mallopt(M_ARENA_MAX, 1);
    Barrier b {num_threads};
    auto memory_hog = [&b]() {
        b.wait();
        std::vector<char> huge_area(1ULL << 32);
        std::fill(huge_area.begin(), huge_area.end(), 42);
    };

    std::vector<std::thread> memory_hog_threads {};
    for (uint64_t i = 0; i < num_threads - 1; ++i) {
        memory_hog_threads.emplace_back(memory_hog);
    }
    memory_hog();

    for (std::thread &thread : memory_hog_threads) {
        thread.join();
    }
    logger(INFO, "Done.");

    logger(INFO, "Loading tables from storage.");
    load_orders_from_binary(&o, query, params.scale);
    load_customers_from_binary(&c, query, params.scale);
    load_parts_from_binary(&p, query, params.scale);
    load_nations_from_binary(&n, query, params.scale);
    load_lineitems_from_binary(&l, query, params.scale);
    logger(INFO, "Done.");

    // 4. execute specified query
    result_t result{};
    switch (query) {
        case 3:
            tpch_q3(&result, &c, &o, &l, params.algorithm_name, &joinconfig);
            break;
        case 10:
            tpch_q10(&result, &c, &o, &l, &n, params.algorithm_name, &joinconfig);
            break;
        case 12:
            tpch_q12(&result, &l, &o, params.algorithm_name, &joinconfig);
            break;
        case 19:
            tpch_q19(&result, &l, &p, params.algorithm_name, &joinconfig);
            break;
        default:
            logger(ERROR, "TPC-H Q%d is not supported", query);
    }
    // 4. report and clean up
    logger(INFO, "Query completed");

    free_orders(&o);
    free_part(&p);
    free_customer(&c);
    free_lineitem(&l);
    free_nation(&n);
}
