#include "multithreadscan_utils.hpp"

#include <algorithm>
#include <numeric>
#include <optional>

#include "Enclave_u.h"
#include "Logger.hpp"
#include "sgxerrors.h"
#include "../shared/ResultAllocators.hpp"

template <class T, class U = std::allocator<T>>
size_t cum_sum_vector_size(std::vector<std::vector<T, U>> vec_of_vecs, std::optional<int> thread_id = std::nullopt) {
    auto end = thread_id.has_value() ? vec_of_vecs.begin() + thread_id.value() : vec_of_vecs.end();

    return std::accumulate(vec_of_vecs.begin(),
                           end,
                           0UL,
                           [](auto acc, auto vec) {return acc + std::size(vec);});
}

/*
 * Joins all thread results in parallel
 */
bool join_results(const Configuration &config, ScanResults &results, int i) {
    if (config.mode == Mode::bitvector) {
        return true;
    }
    else if (config.mode == Mode::noIndex) {
        size_t start_idx = cum_sum_vector_size(results.index_results_per_thread, i);

        std::copy_n(results.index_results_per_thread[i].begin(),
                    results.index_results_per_thread[i].size(),
                    results.index_results.data() + start_idx);
    } else if (config.mode == Mode::dict) {
        size_t start_idx = cum_sum_vector_size(results.dict_results_per_thread, i);
        std::copy_n(results.dict_results_per_thread[i].begin(),
                    results.dict_results_per_thread[i].size(),
                    results.dict_results.data() + start_idx);
    }
    return true;
}

bool reserve_unified_result_vector(const Configuration &config, ScanResults &results) {
    if (config.mode == Mode::noIndex) {
        size_t total_size = cum_sum_vector_size(results.index_results_per_thread);
        results.index_results.resize(total_size);
    } else if (config.mode == Mode::dict) {
        size_t total_size = cum_sum_vector_size(results.dict_results_per_thread);
        results.dict_results.resize(total_size);
    }
    return true;
}

/**
 * Joins the TSC counter of each thread, by calculating the average and adding it to the overall counter. Resets the
 * thread counters to 0.
 */
bool join_counter_values(uint64_t &cycle_cntr, std::vector<uint64_t> &cpu_cntr_per_thread) {
    uint64_t cycle_cntr_sum = 0;
    for (unsigned long &i : cpu_cntr_per_thread) {
        cycle_cntr_sum += i;
        i = 0;
    }
    cycle_cntr += cycle_cntr_sum / cpu_cntr_per_thread.size();
    return true;
}

/**
 * sets the affinity of all threads, except the main thread(is set at the start)
 */
bool setThreadAff(std::vector<std::thread> &threads, const bool numa, uint32_t phys_cores_per_socket) {

    const auto hardware_threads = std::thread::hardware_concurrency();

    if (phys_cores_per_socket == 0) {
        phys_cores_per_socket = hardware_threads / 2;
    }

    std::vector<uint32_t> core_ids {};

    if (!numa) {
        for (int cid = 1; cid < hardware_threads; ++cid) {
            core_ids.push_back(cid);
        }
    } else {
        for (uint cid = phys_cores_per_socket; cid < hardware_threads; ++cid) {
            core_ids.push_back(cid);
        }
        for (int cid = 0; cid < phys_cores_per_socket; ++cid) {
            core_ids.push_back(cid);
        }
    }

    for (size_t t = 0; t < threads.size(); ++t) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_ids[t], &cpuset);
        int rc = pthread_setaffinity_np(threads[t].native_handle(),
                                            sizeof(cpu_set_t), &cpuset);
        if (rc != 0) {
            error("Error calling pthread_setaffinity_np: " + std::to_string(rc));
        }
    }
    return true;
}


void
reserve_index_scan_result_vectors(const Configuration &config, ScanResults &results) {
    results.index_results_per_thread.resize(config.num_threads);
}

void pre_alloc_index_scan_result_vectors(const Configuration &config, ScanResults &results, const uint8_t *data) {
    pre_alloc_per_thread(data, results.index_results_per_thread, config.num_threads, config.predicate_low,
                         config.predicate_high, config.num_entries_per_thread);
}

void
reserve_bitvector_result_vectors(const Configuration &config, ScanResults &results) {
    auto num_results = config.num_entries / 64;
    results.bitvector_results.resize(num_results);
}

void
create_dict_result_vectors(const Configuration &config, ScanResults &results) {
    results.dict_results_per_thread.resize(config.num_threads);
}

void pre_alloc_dict_scan_result_vectors(const Configuration &config, ScanResults &results, const uint8_t *data) {
    pre_alloc_per_thread(data, results.dict_results_per_thread, config.num_threads, config.predicate_low,
                         config.predicate_high, config.num_entries_per_thread);
}

int initialize_enclave(const char *enclave_file, sgx_enclave_id_t &enclave_id) {
    sgx_status_t ret;

    logger("Initializing enclave.");

    /* Call sgx_create_enclave to initialize an enclave instance */
    /* Debug Support: set 2nd parameter to 1 */
    ret = sgx_create_enclave(enclave_file, 1, nullptr, nullptr, &enclave_id, nullptr);
    if (ret != SGX_SUCCESS) {
        error("Enclave initialization failed!");
        print_error_message(ret);
        return -1;
    }
    ret = ecall_noop(enclave_id);
    if (ret != SGX_SUCCESS) {
        error("First call into enclave failed!");
        print_error_message(ret);
        return -1;
    }
    logger("Done.");
    return 0;
}


int destroy_enclave(sgx_enclave_id_t enclave_id) {
    sgx_status_t ret;

    /* Call sgx_destroy_enclave to destroy  the enclave instance */
    ret = sgx_destroy_enclave(enclave_id);
    if (ret != SGX_SUCCESS) {
        error("Destroying the enclave failed!");
        print_error_message(ret);
        return -1;
    }

    return 0;
}


