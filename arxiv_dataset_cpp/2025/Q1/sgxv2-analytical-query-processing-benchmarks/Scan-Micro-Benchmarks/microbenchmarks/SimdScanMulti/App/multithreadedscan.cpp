#include "multithreadedscan.hpp"

#include "Enclave_u.h"

#include "../shared/ScalarScan.hpp"
#include "Logger.hpp"
#include "SIMD512.hpp"
#include "multithreadscan_utils.hpp"
#include "rdtscpWrapper.h"
#include "sgxerrors.h"


int
run_bitvector_scan(const uint8_t *data, uint64_t start_idx, ScanResults &results, uint64_t output_idx,
                   uint64_t &cpu_cycles_cntr, const Configuration &config, sgx_enclave_id_t eid, Barrier &barrier,
                   int thread_id) {
    if (config.enclave) {
        sgx_status_t ret;
        if (config.preload) {
            ret = ecall_bitvector_scan_preload(eid, config.predicate_low, config.predicate_high, start_idx,
                                               config.num_entries_per_thread, output_idx, &cpu_cycles_cntr,
                                               config.read_config.num_runs, config.read_config.num_warmup_runs,
                                               config.read_config.unique);
            if (ret != SGX_SUCCESS) {
                error("Bitvector scan enclave preload failed!");
                print_error_message(ret);
                return -1;
            } else {
                return 0;
            }
        } else {
            ret = ecall_bitvector_scan_user(
                    eid, config.predicate_low, config.predicate_high, data, config.num_entries_per_thread,
                    results.bitvector_results.data() + output_idx, &cpu_cycles_cntr, config.read_config.num_runs,
                    config.read_config.num_warmup_runs, config.read_config.unique);
            if (ret != SGX_SUCCESS) {
                error("Bitvector scan in enclave no preload failed!");
                print_error_message(ret);
                return -1;
            } else {
                return 0;
            }
        }
    } else {
        auto result_start = reinterpret_cast<__mmask64 *>(results.bitvector_results.data() + output_idx);

        for (auto i = 0; i < config.read_config.num_warmup_runs; ++i) {
            SIMD512::bitvector_scan(config.predicate_low, config.predicate_high,
                                    reinterpret_cast<const __m512i *>(data), config.num_entries_per_thread,
                                    result_start);
        }

        rdtscpWrapper rdtscpWrap(&cpu_cycles_cntr);
        for (auto i = 0; i < config.read_config.num_runs; ++i) {
            SIMD512::bitvector_scan(config.predicate_low, config.predicate_high,
                                    reinterpret_cast<const __m512i *>(data), config.num_entries_per_thread,
                                    result_start);
        }

        return 0;
    }
}


int
run_index_scan(const uint8_t *data, const uint64_t start_idx, ScanResults &results, uint64_t &cpu_cycles_cntr,
               const Configuration &config, sgx_enclave_id_t eid, Barrier &barrier, int thread_id) {
    if (config.enclave) {
        sgx_status_t ret;
        if (config.preload) {
            ret = ecall_index_scan_self_alloc_preload(eid, config.predicate_low, config.predicate_high, start_idx,
                                                      config.num_entries_per_thread, thread_id, &cpu_cycles_cntr,
                                                      config.read_config.num_runs, config.read_config.num_warmup_runs,
                                                      config.read_config.unique, config.pre_alloc);
        } else {
            if (!config.pre_alloc) {
                throw std::invalid_argument("Index scan user can only be run if pre_alloc is activated!");
            }
            ret = ecall_index_scan_user(eid, config.predicate_low, config.predicate_high, data,
                                        config.num_entries_per_thread, &results.index_results_per_thread[thread_id],
                                        &cpu_cycles_cntr, config.read_config.num_runs,
                                        config.read_config.num_warmup_runs, config.read_config.unique);
        }
        if (ret != SGX_SUCCESS) {
            error("Index scan in enclave failed!");
            print_error_message(ret);
            return -1;
        } else {
            return 0;
        }
    } else {
        if (!config.pre_alloc) {
            results.index_results_per_thread[thread_id] = CacheAlignedVector<uint64_t>();
        }

        for (auto i = 0; i < config.read_config.num_warmup_runs; ++i) {
            SIMD512::implicit_index_scan_self_alloc(
                    config.predicate_low, config.predicate_high, reinterpret_cast<const __m512i *>(data),
                    config.num_entries_per_thread, results.index_results_per_thread[thread_id], !config.pre_alloc);
        }

        rdtscpWrapper rdtscpWrap(&cpu_cycles_cntr);
        for (auto i = 0; i < config.read_config.num_runs; ++i) {
            SIMD512::implicit_index_scan_self_alloc(
                    config.predicate_low, config.predicate_high, reinterpret_cast<const __m512i *>(data),
                    config.num_entries_per_thread, results.index_results_per_thread[thread_id], !config.pre_alloc);
        }

        if (config.include_join) {
            // TODO also do this in the enclave for preload and user check
            barrier.wait([&config, &results]() { return reserve_unified_result_vector(config, results); });

            join_results(config, results, thread_id);
        }
    }
    return 0;
}

int
run_index_scan_scalar(const uint8_t *data, const uint64_t start_idx, ScanResults &results, uint64_t &cpu_cycles_cntr,
                      const Configuration &config, sgx_enclave_id_t eid, Barrier &barrier, int thread_id) {
    if (config.enclave) {
        sgx_status_t ret;
        if (config.preload) {
            ret = ecall_scalar_scan_preload(eid, config.predicate_low, config.predicate_high, start_idx,
                                            config.num_entries_per_thread, thread_id, &cpu_cycles_cntr,
                                            config.read_config.num_runs, config.read_config.num_warmup_runs,
                                            config.read_config.unique, config.pre_alloc);
        } else {
            if (!config.pre_alloc) {
                throw std::invalid_argument("Index scan user can only be run if pre_alloc is activated!");
            }
            ret = ecall_scalar_scan_user(eid, config.predicate_low, config.predicate_high, data,
                                         config.num_entries_per_thread, &results.index_results_per_thread[thread_id],
                                         &cpu_cycles_cntr, config.read_config.num_runs,
                                         config.read_config.num_warmup_runs, config.read_config.unique);
        }
        if (ret != SGX_SUCCESS) {
            error("Index scan in enclave failed!");
            print_error_message(ret);
            return -1;
        } else {
            return 0;
        }
    } else {
        if (!config.pre_alloc) {
            results.index_results_per_thread[thread_id] = CacheAlignedVector<uint64_t>();
        }

        for (auto i = 0; i < config.read_config.num_warmup_runs; ++i) {
            scalar_implicit_index_scan(config.predicate_low, config.predicate_high, data, config.num_entries_per_thread,
                                       results.index_results_per_thread[thread_id], !config.pre_alloc);
        }

        rdtscpWrapper rdtscpWrap(&cpu_cycles_cntr);
        for (auto i = 0; i < config.read_config.num_runs; ++i) {
            scalar_implicit_index_scan(config.predicate_low, config.predicate_high, data, config.num_entries_per_thread,
                                       results.index_results_per_thread[thread_id], !config.pre_alloc);
        }

        if (config.include_join) {
            // TODO also do this in the enclave for preload and user check
            barrier.wait([&config, &results]() { return reserve_unified_result_vector(config, results); });

            join_results(config, results, thread_id);
        }
    }
    return 0;
}

int
run_dict_scan(const uint8_t *data, uint64_t start_idx, const int64_t *dict, ScanResults &results,
              uint64_t &cpu_cycles_cntr, const Configuration &config, sgx_enclave_id_t eid, Barrier &barrier,
              int thread_id) {
    if (config.enclave) {
        sgx_status_t ret;
        if (config.preload) {
            ret = ecall_dict_scan_preload(eid, config.predicate_low, config.predicate_high, start_idx,
                                          config.num_entries_per_thread, thread_id, &cpu_cycles_cntr,
                                          config.read_config.num_runs, config.read_config.num_warmup_runs,
                                          config.read_config.unique, config.pre_alloc);
        } else {
            if (!config.pre_alloc) {
                throw std::invalid_argument("Dict scan user can only be run if pre_alloc is activated!");
            }
            ret = ecall_dict_scan_user(eid, config.predicate_low, config.predicate_high, data,
                                       config.num_entries_per_thread, dict, &results.dict_results_per_thread[thread_id],
                                       &cpu_cycles_cntr, config.read_config.num_runs,
                                       config.read_config.num_warmup_runs, config.read_config.unique);
        }
        if (ret != SGX_SUCCESS) {
            error("Dict scan in enclave failed!");
            print_error_message(ret);
            return -1;
        } else {
            return 0;
        }
    } else {
        if (!config.pre_alloc) {
            results.dict_results_per_thread[thread_id] = CacheAlignedVector<int64_t>();
        }

        for (auto i = 0; i < config.read_config.num_warmup_runs; ++i) {
            SIMD512::dict_scan_8bit_64bit_scalar_unroll(
                    config.predicate_low, config.predicate_high, dict, reinterpret_cast<const __m512i *>(data),
                    config.num_entries_per_thread, results.dict_results_per_thread[thread_id], !config.pre_alloc);
        }

        rdtscpWrapper rdtscpWrap(&cpu_cycles_cntr);
        for (auto i = 0; i < config.read_config.num_runs; ++i) {
            SIMD512::dict_scan_8bit_64bit_scalar_unroll(
                    config.predicate_low, config.predicate_high, dict, reinterpret_cast<const __m512i *>(data),
                    config.num_entries_per_thread, results.dict_results_per_thread[thread_id], !config.pre_alloc);
        }

        if (config.include_join) {
            // TODO also do this in the enclave for preload and user check
            barrier.wait([&config, &results]() { return reserve_unified_result_vector(config, results); });

            join_results(config, results, thread_id);
        }
    }
    return 0;
}

void
scan_wrapper(int t, const Configuration &config, ScanResults &results, const uint8_t *data, const int64_t *dict,
             Barrier &barrier, uint64_t &cycle_cntr, std::vector<uint64_t> &cpu_cntr_per_thread,
             sgx_enclave_id_t global_eid) {

    auto thread_offset = t * config.num_entries_per_thread;

    for (size_t rerun_index = 0; rerun_index < config.num_reruns; ++rerun_index) {
        auto run_offset = config.num_entries * rerun_index + thread_offset;
        auto start_position = data + run_offset;

        int ret = 0;
        if (config.mode == Mode::noIndex) {
            ret = run_index_scan(start_position, run_offset, results, cpu_cntr_per_thread[t], config, global_eid,
                                 barrier, t);
        } else if (config.mode == Mode::bitvector) {
            ret = run_bitvector_scan(start_position, run_offset, results, thread_offset / 64, cpu_cntr_per_thread[t],
                                     config, global_eid, barrier, t);
        } else if (config.mode == Mode::dict) {
            ret = run_dict_scan(start_position, run_offset, dict, results, cpu_cntr_per_thread[t], config, global_eid,
                                barrier, t);
        } else if (config.mode == Mode::scalar) {
            ret = run_index_scan_scalar(start_position, run_offset, results, cpu_cntr_per_thread[t], config, global_eid,
                                        barrier, t);
        }


        if (ret != 0) {
            throw std::invalid_argument("Invalid combination of arguments in scans!");
        }

        barrier.wait(
                [&cycle_cntr, &cpu_cntr_per_thread]() { return join_counter_values(cycle_cntr, cpu_cntr_per_thread); });
    }
}
