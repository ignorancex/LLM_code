#include <algorithm>
#include <iostream>
#include <random>
#include <gflags/gflags.h>
#include "sgxerrors.h"
#include "PerfEvent.hpp"

#include "Enclave_u.h"

#include "../Shared/partitioning_algorithms.hpp"
#include "../Shared/histogram_algorithms.hpp"
#include "EnclaveInit.hpp"
#include "Logger.hpp"

#include <sys/prctl.h>

constexpr char ENCLAVE_FILENAME[] = "radixpartenc.signed.so";

DEFINE_uint32(r, 128, "Number of tuples in R. Determines number of different keys in S.");
DEFINE_uint32(s, 1 << 27, "Number of tuples in S.");
DEFINE_int32(min_radix_bits, -1, "Overrides the default radix bit calculation.");
DEFINE_int32(max_radix_bits, -1, "Overrides the default radix bit calculation.");
DEFINE_uint32(min_num_keys_exp, 0, "Determines the minimum number of unique keys in the array.");
DEFINE_uint32(max_num_keys_exp, 20, "Determines the maximum number of unique keys in the array.");
DEFINE_uint32(shift, 0, "Dummy");
DEFINE_bool(debug, false, "Debug output on/off");
DEFINE_uint32(unroll, 1, "Unroll partitioning by how much");
DEFINE_bool(user_check, false, "User check mode on/off");
DEFINE_uint32(repeat, 1, "How often to rerun each setting.");
DEFINE_bool(sbb_mitigation, false, "Enable store buffer bypass mitigation outside enclave.");

constexpr int CACHE_LINE_SIZE = 64;
constexpr int L1_CACHE_SIZE = (48 * 1024);
constexpr int L2_CACHE_SIZE = (1280 * 1024);

Logger l{};

constexpr int SMALL_PADDING_TUPLES = 3 * CACHE_LINE_SIZE / sizeof(row);
constexpr int TUPLES_PER_CACHELINE = 64 / sizeof(row);
constexpr int L1_CACHE_TUPLES = L1_CACHE_SIZE / sizeof(row);
constexpr int L2_CACHE_TUPLES = L2_CACHE_SIZE / sizeof(row);

union alignas(64) CacheLineBuffer {
    CacheLineBuffer() {
        data.size = 0;
    };

    struct {
        row tuples[TUPLES_PER_CACHELINE];
    } tuples;

    struct {
        row tuples[TUPLES_PER_CACHELINE - 1];
        uint32_t size;
    } data;

    inline bool add_tuple(const row &tuple) {     // return true if full after adding
        uint32_t size_before = data.size;
        ++data.size;
        tuples.tuples[size_before] = tuple;
        return (size_before == TUPLES_PER_CACHELINE - 1);
    }

    inline void clear() {
        data.size = 0;
    }

    [[nodiscard]] inline uint32_t size() const {
        return data.size;
    }
};

inline uint32_t fanout_pass_1(uint32_t radix_bits, uint32_t num_passes) {
    return 1 << (radix_bits / num_passes);
}

inline uint32_t fanout_pass_2(uint32_t radix_bits, uint32_t num_passes) {
    return 1 << (radix_bits - radix_bits / num_passes);
}

inline uint32_t padding_tuples(uint32_t radix_bits, uint32_t num_passes) {
    return SMALL_PADDING_TUPLES * (fanout_pass_2(radix_bits, num_passes) + 1);
}

inline uint32_t relation_padding(uint32_t radix_bits, uint32_t num_passes) {
    return padding_tuples(radix_bits, num_passes) * fanout_pass_1(radix_bits, num_passes) * sizeof(row);
}

constexpr int partition_size_mode = 0;

uint32_t calc_num_radix_bits(uint64_t num_r, uint64_t num_s, uint64_t nthreads) {

    uint64_t total_tuples = num_r;
    uint64_t max_tuples_in_cache = L2_CACHE_TUPLES / 4;

    // Division rounding up
    uint64_t num_required_partitions = (total_tuples + max_tuples_in_cache - 1) / max_tuples_in_cache;
    num_required_partitions = std::max(num_required_partitions, nthreads);

    uint32_t radix_bits = 0;
    while ((1 << radix_bits) < num_required_partitions) {
        ++radix_bits;
    }

    return radix_bits;
}

int main(int argc, char *argv[]) {
    gflags::SetUsageMessage("Radix partitioning benchmark");
    gflags::ParseCommandLineFlags(&argc, &argv, false);

    l.debug = FLAGS_debug;

    if (FLAGS_sbb_mitigation) {
        // Disable speculative loads after stores (enable store bypass mitigation)
        const int result = prctl(PR_SET_SPECULATION_CTRL, PR_SPEC_STORE_BYPASS, PR_SPEC_DISABLE, 0, 0);

        if (result == 0) {
            l.log("Successfully enabled store bypass mitigation");
        } else {
            l.err(std::string{"Disabling store bypass mitigation failed with error code "} + std::to_string(result));
        }
    }

    auto data_size = FLAGS_s;
    //uint32_t num_keys = FLAGS_r;

    sgx_enclave_id_t eid = 0;
    auto ret = initialize_enclave(ENCLAVE_FILENAME, eid);
    if (ret != 0) {
        return -1;
    }

    l.log("Data size = " + std::to_string(data_size));

    uint32_t min_radix_bits;
    uint32_t max_radix_bits;
    if (FLAGS_min_radix_bits == -1 && FLAGS_max_radix_bits == -1) {
        min_radix_bits = calc_num_radix_bits(FLAGS_r, FLAGS_s, 1);
        max_radix_bits = min_radix_bits;
    } else {
        min_radix_bits = FLAGS_min_radix_bits;
        max_radix_bits = FLAGS_max_radix_bits;
    }

    auto max_fan_out = 1 << max_radix_bits;
    auto max_padding = padding_tuples(max_radix_bits, 1);

    auto data = (row *) std::aligned_alloc(CACHE_LINE_SIZE, data_size * sizeof(row));
    auto partitioned = (row *) std::aligned_alloc(CACHE_LINE_SIZE, sizeof(row) * (data_size + max_fan_out * max_padding));
    memset(partitioned, 42, sizeof(row) * (data_size + max_fan_out * max_padding));

    ret = ecall_init_data(eid, data_size);
    if (ret != SGX_SUCCESS) {
        l.err("init data failed");
    }
    ret = ecall_init_partitioned_buffer(eid, data_size, max_radix_bits, max_padding);
    if (ret != SGX_SUCCESS) {
        l.err("init partitioning output buffer failed");
    }

    std::random_device rd;
    std::mt19937 g(rd());

    bool print_header = true;
    for (uint32_t radix_bits = min_radix_bits; radix_bits <= max_radix_bits; ++radix_bits) {

        int fan_out = 1 << radix_bits;
        auto padding = padding_tuples(radix_bits, 1);

        for (uint32_t num_keys_exp = FLAGS_min_num_keys_exp; num_keys_exp <= FLAGS_max_num_keys_exp; ++num_keys_exp) {
            for (uint32_t repeat = 0; repeat < FLAGS_repeat; ++repeat) {

                uint32_t num_keys = (1 << num_keys_exp);

                for (uint32_t i = 0; i < data_size; ++i) {
                    data[i].key = i % num_keys;
                }

                std::shuffle(data, data + data_size, g);

                auto hist = (uint32_t *) std::aligned_alloc(CACHE_LINE_SIZE, fan_out * sizeof(uint32_t));
                std::memset(hist, 0, fan_out * sizeof(uint32_t));
                uint64_t mask = fan_out - 1;
                uint64_t shift = FLAGS_shift;

                BenchmarkParameters params{};
                params.setParam("num_keys", num_keys);
                params.setParam("num_bins", fan_out);
                params.setParam("rows", data_size);
                params.setParam("data_size", data_size * sizeof(row));
                params.setParam("workload", "hist");
                params.setParam("sgx", "no");
                params.setParam("preload", "no");
                params.setParam("unroll", FLAGS_unroll);

                uint64_t hist_cycle_counter = 0;
                {
                    PerfEventBlock e{data_size, params, print_header, &hist_cycle_counter};
                    histogram_unrolled(data, data_size, mask, shift, hist, &hist_cycle_counter);
                }
                print_header = false;

                if (FLAGS_user_check) {
                    std::memset(hist, 0, fan_out * sizeof(uint32_t));
                    uint64_t hist_cycle_counter_user_check = 0;
                    params.setParam("sgx", "yes");
                    params.setParam("preload", "no");

                    PerfEventBlock e{data_size, params, print_header, &hist_cycle_counter_user_check};
                    ecall_histogram_user_check_unrolled(eid, data, data_size, hist, mask, shift,
                                                        &hist_cycle_counter_user_check);
                }

                auto sum_hist = (uint32_t *) std::aligned_alloc(CACHE_LINE_SIZE, fan_out * sizeof(uint32_t));
                std::memset(sum_hist, 0, fan_out * 4);
                std::partial_sum(hist, hist + fan_out - 1, sum_hist + 1);

                for (uint32_t i = 0; i < fan_out; i++) {
                    sum_hist[i] += i * padding;
                }

                params.setParam("workload", "copy");
                params.setParam("sgx", "no");
                params.setParam("preload", "no");
                uint64_t copy_cycle_counter = 0;
                auto partition_function = choose_function(PartitioningMode::loop, FLAGS_unroll);
                if (!partition_function.has_value()) {
                    l.err("Unknown partitioning function!");
                    return -1;
                }
                {
                    PerfEventBlock e{data_size, params, print_header, &copy_cycle_counter};
                    partition_function.value()(data, data_size, mask, shift, sum_hist, partitioned, &copy_cycle_counter);
                }

                if (FLAGS_user_check) {
                    // Reset sum histogram
                    std::memset(sum_hist, 0, fan_out * 4);
                    std::partial_sum(hist, hist + fan_out - 1, sum_hist + 1);

                    for (uint32_t i = 0; i < fan_out; i++) {
                        sum_hist[i] += i * padding;
                    }

                    uint64_t copy_cycle_counter_user_check = 0;
                    params.setParam("sgx", "yes");
                    params.setParam("preload", "no");

                    {
                        PerfEventBlock e{data_size, params, print_header, &copy_cycle_counter_user_check};
                        ret = ecall_partition_dynamic(eid, data, data_size, sum_hist, partitioned, mask, shift,
                                                      &copy_cycle_counter_user_check, FLAGS_unroll, false,
                                                      static_cast<uint8_t>(PartitioningMode::loop));
                        if (ret != SGX_SUCCESS) {
                            l.err("ecall_partition_user_check_unrolled failed");
                        }
                    }
                }

                params.setParam("sgx", "yes");
                params.setParam("workload", "hist");
                params.setParam("preload", "yes");
                ret = ecall_next_num_keys(eid, data_size, num_keys, radix_bits);
                if (ret != SGX_SUCCESS) {
                    l.err("ecall_next_num_keys failed");
                }
                uint64_t hist_cycle_counter_enclave = 0;
                uint64_t copy_cycle_counter_enclave = 0;
                {
                    PerfEventBlock e{data_size, params, print_header, &hist_cycle_counter_enclave};
                    ret = ecall_histogram_unrolled(eid, data_size, mask, shift, &hist_cycle_counter_enclave);
                    if (ret != SGX_SUCCESS) {
                        l.err("ecall_histogram_unrolled failed");
                    }
                }
                ret = ecall_cumsum_hist(eid, radix_bits, padding);
                if (ret != SGX_SUCCESS) {
                    l.err("ecall_cumsum_hist failed");
                }
                params.setParam("workload", "copy");
                {
                    PerfEventBlock e{data_size, params, print_header, &copy_cycle_counter_enclave};
                    ret = ecall_partition_dynamic(eid, nullptr, data_size, nullptr, nullptr, mask, shift,
                                                      &copy_cycle_counter_enclave, FLAGS_unroll, true,
                                                      static_cast<uint8_t>(PartitioningMode::loop));
                    if (ret != SGX_SUCCESS) {
                        l.err("ecall_partition_unrolled failed");
                    }
                }
                ret = ecall_free_preload_hist(eid);
                if (ret != SGX_SUCCESS) {
                    l.err("ecall_free_preload_hist failed");
                }

                std::free(sum_hist);
                std::free(hist);
            }
        }
    }

    // One of these somehow causes an error on munmap().
    //std::free(partitioned);
    //std::free(data);

    ret = ecall_free_preload_data(eid);
    if (ret != SGX_SUCCESS) {
        l.err("ecall_free_preload_data failed");
    }
    destroy_enclave(eid);
    return 0;
}
