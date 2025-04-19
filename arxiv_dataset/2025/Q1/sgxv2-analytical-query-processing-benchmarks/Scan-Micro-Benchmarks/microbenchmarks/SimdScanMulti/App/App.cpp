#include <algorithm>
#include <pthread.h>
#include <thread>

#include "Enclave_u.h"
#include "sgxerrors.h"

#include "../shared/Barrier.hpp"
#include "../shared/ScanResults.hpp"
#include "Allocator.hpp"
#include "Logger.hpp"
#include "PerfEvent.hpp"
#include "flags.hpp"
#include "multithreadedscan.hpp"
#include "multithreadscan_utils.hpp"
#include "rdtscpWrapper.h"
#include "types.hpp"

constexpr char ENCLAVE_FILENAME[] = "simdmultienclave.signed.so";

ConfigurationSpectrum create_configuration_spectrum() {
    return {from_string(FLAGS_enclave),
            from_string(FLAGS_preload),
            parse_modes(FLAGS_mode),
            from_string(FLAGS_unique_data),
            static_cast<uint8_t>(FLAGS_num_warmup_runs),
            FLAGS_num_runs,
            from_string(FLAGS_numa),
            FLAGS_num_reruns,
            {static_cast<uint8_t>(FLAGS_min_threads), static_cast<uint8_t>(FLAGS_max_threads), 1, true},
            ParameterSpan<uint64_t>{(FLAGS_min_entries_exp > 0) ? (1ull << FLAGS_min_entries_exp) : FLAGS_min_entries,
                                    (FLAGS_max_entries_exp > 0) ? (1ull << FLAGS_max_entries_exp) : FLAGS_max_entries,
                                    1, true},
            {static_cast<uint8_t>(FLAGS_min_selectivity), static_cast<uint8_t>(FLAGS_max_selectivity),
             static_cast<uint8_t>(FLAGS_step_selectivity), false},
            FLAGS_join,
            FLAGS_pre_alloc};
}

int main(int argc, char *argv[]) {
    gflags::SetUsageMessage("Parallel scan unsafe and SGXv2 benchmark");
    gflags::ParseCommandLineFlags(&argc, &argv, false);

    auto global_config = create_configuration_spectrum();

    BenchmarkParameters params = {};

    // Pin main thread to core 0
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) { error(std::string("Error calling pthread_setaffinity_np: ") + std::to_string(rc)); }

    sgx_enclave_id_t global_eid = 0;
    if (global_config.use_enclave != Trinary::f) {
        auto ret = initialize_enclave(ENCLAVE_FILENAME, global_eid);
        if (ret != 0) { return -1; }
    }
    size_t num_total_entries;
    if (global_config.num_reruns > 0) {
        num_total_entries = global_config.num_entries.max * global_config.num_reruns;
    } else {
        num_total_entries = global_config.num_entries.max;
    }

    auto need_data_array = !(global_config.use_enclave == Trinary::t && global_config.preload == Trinary::t);
    std::unique_ptr<uint8_t[]> data;
    if (need_data_array) {
        logger("Allocating unencrypted scan data.");
        try {
            data = allocate_data_array_aligned<uint8_t, 64>(num_total_entries);
        } catch (const std::bad_alloc &e) {
            error("Allocation failed!");
            return -1;
        }
        logger("Done.");
    }

    auto need_dict =
            std::find(global_config.modes.begin(), global_config.modes.end(), Mode::dict) != global_config.modes.end();
    std::unique_ptr<int64_t[]> dict;
    if (need_dict) {
        logger("Allocating unencrypted dict.");

        try {
            dict = allocate_data_array_aligned<int64_t, 64>(256);// TODO make dict size flexible
        } catch (const std::bad_alloc &e) {
            error("Allocation failed!");
            return -1;
        }

        logger("Done.");
    }

    auto need_preload = global_config.use_enclave != Trinary::f && global_config.preload != Trinary::f;
    if (need_preload) {
        sgx_status_t ret;
        logger("Allocating encrypted scan data (preload).");
        ret = ecall_init_data(global_eid, num_total_entries);
        if (ret != SGX_SUCCESS) {
            error("Preload failed!");
            print_error_message(ret);
            return -1;
        }
        logger("Done.");

        if (need_dict) {
            logger("Allocating encrypted dict (preload).");
            ret = ecall_init_dict(global_eid, 256);
            if (ret != SGX_SUCCESS) {
                error("Dict preload failed!");
                print_error_message(ret);
                return -1;
            }
            logger("Done.");
        }
    }

    auto experiment_configurations = global_config.enumerate();

    auto print_header = true;
    for (const auto &config: experiment_configurations) {
        params.setParam("reruns", config.num_reruns);
        params.setParam("entries", config.num_entries);
        params.setParam("writeMode", std::string{mode_names[static_cast<size_t>(config.mode)]});
        params.setParam("enclaveMode", (config.enclave ? "enclave" : "native"));
        params.setParam("dataLoading", (config.preload ? "preload" : "noPreload"));
        params.setParam("datasizeKiB", config.num_entries / 1024);
        params.setParam("numThreads", config.num_threads);
        params.setParam<double>("selectivity", config.selectivity / 100.0);
        params.setParam("unique", config.read_config.unique);
        params.setParam("numRuns", config.read_config.num_runs);
        params.setParam("warmup", config.read_config.num_warmup_runs);
        params.setParam("numa", config.NUMA ? "yes" : "no");

        uint64_t cycle_cntr = 0;
        std::vector<uint64_t> cpu_cntr_per_thread(config.num_threads);
        std::vector<std::thread> threads;

        ScanResults results{};
        if (!config.preload) {
            if (config.mode == Mode::bitvector) {
                reserve_bitvector_result_vectors(config, results);
            } else if (config.mode == Mode::noIndex || config.mode == Mode::scalar) {
                if (config.pre_alloc) {
                    pre_alloc_index_scan_result_vectors(config, results, data.get());
                } else {
                    reserve_index_scan_result_vectors(config, results);
                }
            } else if (config.mode == Mode::dict) {
                if (config.pre_alloc) {
                    pre_alloc_dict_scan_result_vectors(config, results, data.get());
                } else {
                    create_dict_result_vectors(config, results);
                }
            }
        } else {
            ecall_init_results(global_eid);
            if (config.mode == Mode::bitvector) {
                ecall_init_results_bitvector(global_eid, config.num_entries);
            } else if (config.mode == Mode::noIndex || config.mode == Mode::scalar) {
                if (config.pre_alloc) {
                    ecall_init_results_index_pre_alloc(global_eid, config.num_threads, config.predicate_low,
                                                       config.predicate_high, config.num_entries_per_thread);
                } else {
                    ecall_init_results_index(global_eid, config.num_threads);
                }
            } else if (config.mode == Mode::dict) {
                if (config.pre_alloc) {
                    ecall_init_results_dict_pre_alloc(global_eid, config.num_threads, config.predicate_low,
                                                      config.predicate_high, config.num_entries_per_thread);
                } else {
                    ecall_init_results_dict(global_eid, config.num_threads);
                }
            }
        }

        {
            uint64_t counter = 0;
            PerfEventBlock e(config.num_entries * config.num_reruns * config.read_config.num_runs, params, print_header,
                             &cycle_cntr);
            // after all threads are created, set their affinity and start the
            // counters.
            auto wait_function = [&config, &threads, &e] {
                setThreadAff(threads, config.NUMA);
                e.e->startCounters();
                return true;
            };

            rdtscpWrapper wrapper{&counter};

            // When creating the threads for the other socket, we need an additional slot in our barrier to wait for
            // the main thread. The scan always runs using config.num_threads slots. When we use socket 0, the main
            // thread also participates in the scan, therefore, we do not create a thread 0.
            Barrier creation_barrier{config.num_threads + static_cast<size_t>(config.NUMA)};
            Barrier scan_barrier{config.num_threads};
            for (int t = !config.NUMA; t < config.num_threads; ++t) {
                threads.emplace_back([t, &creation_barrier, &scan_barrier, &config, &results, &wait_function, &data,
                                      &dict, &cycle_cntr, &cpu_cntr_per_thread, global_eid]() {
                    creation_barrier.wait(wait_function);// wait until all threads are created and
                    // affinity is set
                    scan_wrapper(t, config, results, data.get(), dict.get(), scan_barrier, cycle_cntr, cpu_cntr_per_thread,
                                 global_eid);
                });
            }

            creation_barrier.wait(wait_function); // wait until all threads are created and affinity is set

            if (!config.NUMA) {
                scan_wrapper(0, config, results, data.get(), dict.get(), scan_barrier, cycle_cntr, cpu_cntr_per_thread,
                             global_eid);
            }

            for (auto &thread: threads) { thread.join(); }
        }

        print_header = false;
    }

    if (need_preload) {
        auto ret = ecall_free_preload_data(global_eid);
        if (ret != SGX_SUCCESS) {
            error("Free preload failed!");
            print_error_message(ret);
        }
    }
    if (global_config.use_enclave != Trinary::f) { destroy_enclave(global_eid); }
    return 0;
}
