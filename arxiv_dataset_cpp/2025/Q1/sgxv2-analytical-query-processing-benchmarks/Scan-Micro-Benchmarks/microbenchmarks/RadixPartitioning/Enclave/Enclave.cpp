#include "Enclave_t.h"// structs defined in .edl file etc

#include <vector>

#include "../Shared/partitioning_algorithms.hpp"
#include "../Shared/histogram_algorithms.hpp"
#include "../Shared/simple_algorithms.hpp"
#include "Allocator.hpp"
#include "rdtscpWrapper.h"


struct HistogramBenchmarkParameters;
constexpr int CACHE_LINE_SIZE = 64;

uint64_t *preload_array = nullptr;
uint32_t *preload_out = nullptr;

row *preload_data = nullptr;
uint32_t *preload_hist = nullptr;

row *preload_partitioned = nullptr;
uint32_t *preload_sum_hist = nullptr;

extern "C" {

void
ecall_noop() {
    //for benchmarking
}

void
ecall_init_array(const size_t data_size) {
    try {
        preload_array = static_cast<uint64_t *>(aligned_alloc(CACHE_LINE_SIZE, data_size * sizeof(uint64_t)));
    } catch (const std::bad_alloc &) {
        ocall_print_error("Allocation of data buffer inside enclave failed with std::bad_alloc!");
        throw;
    }

    if (preload_array == nullptr) {
        ocall_print_error("Allocation of data buffer inside enclave failed!");
    }
}

void
ecall_fill_and_shuffle_array(const size_t data_size, const uint32_t num_keys, int int_mode, int use_64_bit) {
    const auto mode = static_cast<RandomMode>(int_mode);

    if (preload_array == nullptr) {
        ocall_print_error("preload_data is nullptr!");
        return;
    }
    try {
        std::random_device rd;
        std::mt19937 gen(rd());

        const auto generator_function = choose_random_generator_function(mode);
        generator_function(preload_array, data_size, preload_out, num_keys, gen, static_cast<bool>(use_64_bit));
    } catch (const std::exception &ex) {
        ocall_print_error("Writing and shuffling in enclave failed!");
        ocall_print_error(ex.what());
        throw;
    }
}

void
ecall_alloc_out_array(const uint64_t num_bins) {
    if (preload_out != nullptr) {
        free(preload_out);
    }

    const uint64_t out_size_bytes = num_bins * sizeof(uint32_t);
    preload_out = static_cast<uint32_t *>(aligned_alloc(CACHE_LINE_SIZE, out_size_bytes));
    std::memset(preload_out, 0, out_size_bytes);
}

void
ecall_reset_out_array(const uint64_t num_bins) {
    const uint64_t out_size_bytes = num_bins * sizeof(uint32_t);
    std::memset(preload_out, 0, out_size_bytes);
}

void
ecall_init_data(const size_t data_size) {
    try {
        preload_data = static_cast<row *>(aligned_alloc(CACHE_LINE_SIZE, data_size * sizeof(row)));
    } catch (const std::bad_alloc &) {
        ocall_print_error("Allocation of data buffer inside enclave failed with std::bad_alloc!");
        throw;
    }

    if (preload_data == nullptr) {
        ocall_print_error("Allocation of data buffer inside enclave failed!");
        ocall_print_error(std::to_string(data_size).c_str());
    }
}

void
ecall_fill_and_shuffle(const size_t data_size, const uint32_t num_keys) {
    if (preload_data == nullptr) {
        ocall_print_error("preload_data is nullptr!");
        return;
    }
    try {
        for (uint32_t i = 0; i < data_size; ++i) {
            preload_data[i].key = i % num_keys;
        }

        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(preload_data, preload_data + data_size, g);
    } catch (const std::exception &ex) {
        ocall_print_error("Writing and shuffling in enclave failed!");
        ocall_print_error(ex.what());
        throw;
    }
}

void
ecall_alloc_hist(const uint64_t num_bins) {
    if (preload_hist != nullptr) {
        free(preload_hist);
    }

    uint64_t hist_size_bytes = num_bins * sizeof(uint32_t);
    preload_hist = (uint32_t *) aligned_alloc(CACHE_LINE_SIZE, hist_size_bytes);
    std::memset(preload_hist, 0, hist_size_bytes);
}

void
ecall_reset_hist(const uint64_t num_bins) {
    uint64_t hist_size_bytes = num_bins * sizeof(uint32_t);
    std::memset(preload_hist, 0, hist_size_bytes);
}

void
ecall_init_partitioned_buffer(const size_t data_size, const uint64_t radix_bits, const uint32_t padding) {
    auto fan_out = 1 << radix_bits;
    try {
        preload_partitioned = (row *) aligned_alloc(CACHE_LINE_SIZE, sizeof(row) * (data_size + fan_out * padding));
        memset(preload_partitioned, 42, sizeof(row) * (data_size + fan_out * padding));
    } catch (const std::bad_alloc &error) {
        ocall_print_error("Allocation of partitioned buffer inside enclave failed with std::bad_alloc!");
        throw;
    }
}

void
ecall_next_num_keys(const size_t data_size, const uint32_t num_keys, const uint64_t radix_bits) {
    auto fan_out = 1 << radix_bits;
    for (uint32_t i = 0; i < data_size; ++i) {
        preload_data[i].key = i % num_keys;
    }

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(preload_data, preload_data + data_size, g);

    size_t hist_size_bytes = fan_out * sizeof(uint32_t);
    preload_hist = (uint32_t *) aligned_alloc(CACHE_LINE_SIZE, hist_size_bytes);
    std::memset(preload_hist, 0, hist_size_bytes);
    preload_sum_hist = (uint32_t *) aligned_alloc(CACHE_LINE_SIZE, hist_size_bytes);
    std::memset(preload_sum_hist, 0, hist_size_bytes);
}

void
ecall_array_dynamic(const void *param_void_pointer, uint64_t *cycle_counter) {
    const auto params = static_cast<const SimpleBenchmarkParameters *>(param_void_pointer);
    const auto &[o_data, data_size, o_out] = params->function_parameters;

    uint32_t *output_array = params->preload ? preload_out : o_out;
    if (params->d64) {
        const auto func = choose_function_simple<true>(params->mode, params->unroll_factor);

        const auto input_array = params->preload ? preload_array : o_data;

        if (params->cache_preheat) {
            func.value()(input_array, data_size, output_array, cycle_counter);
            *cycle_counter = 0;
        }

        func.value()(input_array, data_size, output_array, cycle_counter);
    } else {
        const auto func = choose_function_simple<false>(params->mode, params->unroll_factor);

        const auto input_array = reinterpret_cast<const uint32_t *>(params->preload ? preload_array : o_data);

        if (params->cache_preheat) {
            func.value()(input_array, data_size, output_array, cycle_counter);
            *cycle_counter = 0;
        }

        func.value()(input_array, data_size, output_array, cycle_counter);
    }
}

void
ecall_histogram_dynamic(const void *param_void_pointer, uint64_t *cycle_counter) {
    const auto params = static_cast<const HistogramBenchmarkParameters *>(param_void_pointer);
    const auto func = choose_function(params->mode, params->unroll_factor);
    auto const &h_params = params->function_parameters;

    const row *input_array = params->preload ? preload_data : h_params.data;
    uint32_t *output_hist = params->preload ? preload_hist : h_params.hist;

    func.value()(input_array, h_params.data_size, h_params.mask, h_params.shift, output_hist, cycle_counter);
}

void
ecall_partition_dynamic(void *o_data, const size_t data_size, uint32_t *outside_sum_hist, void *o_partitioned,
                        const uint64_t mask, const uint64_t shift, uint64_t *copy_cycle_counter, uint32_t unroll_factor,
                        uint8_t preload, uint8_t mode) {
    const auto func = choose_function(static_cast<PartitioningMode>(mode), unroll_factor);

    auto data = preload ? preload_data : static_cast<row *>(o_data);
    auto sum_hist = preload ? preload_sum_hist : outside_sum_hist;
    auto partitioned = preload ? preload_partitioned : static_cast<row *>(o_partitioned);

    func.value()(data, data_size, mask, shift, sum_hist, partitioned, copy_cycle_counter);
}

void
ecall_histogram(const size_t data_size, const uint64_t mask, const uint64_t shift, uint64_t *hist_cycle_counter) {
    histogram(preload_data, data_size, mask, shift, preload_hist, hist_cycle_counter);
}

void
ecall_histogram_unrolled(const size_t data_size, const uint64_t mask, const uint64_t shift,
                         uint64_t *hist_cycle_counter) {
    histogram_unrolled(preload_data, data_size, mask, shift, preload_hist, hist_cycle_counter);
}

void
ecall_histogram_loop(const size_t data_size, const uint64_t mask, const uint64_t shift, uint64_t *hist_cycle_counter) {
    histogram_loop<8>(preload_data, data_size, mask, shift, preload_hist, hist_cycle_counter);
}

void
ecall_histogram_user_check(void *o_data, const size_t data_size, void *o_hist, const uint64_t mask,
                           const uint64_t shift, uint64_t *hist_cycle_counter) {
    auto outside_data = reinterpret_cast<row *>(o_data);
    auto outside_hist = reinterpret_cast<uint32_t *>(o_hist);
    histogram(outside_data, data_size, mask, shift, outside_hist, hist_cycle_counter);
}

void
ecall_histogram_user_check_unrolled(void *o_data, const size_t data_size, void *o_hist, const uint64_t mask,
                                    const uint64_t shift, uint64_t *hist_cycle_counter) {
    auto outside_data = reinterpret_cast<row *>(o_data);
    auto outside_hist = reinterpret_cast<uint32_t *>(o_hist);
    histogram_unrolled(outside_data, data_size, mask, shift, outside_hist, hist_cycle_counter);
}

void
ecall_histogram_simd(const size_t data_size, const uint64_t mask, const uint64_t shift, uint64_t *hist_cycle_counter) {
    histogram_simd_unrolled(preload_data, data_size, mask, shift, preload_hist, hist_cycle_counter);
}

void
ecall_cumsum_hist(const uint64_t radix_bits, const uint32_t padding) {
    auto fan_out = 1 << radix_bits;
    std::partial_sum(preload_hist, preload_hist + fan_out - 1, preload_sum_hist + 1);

    for (uint32_t i = 0; i < fan_out; i++) {
        preload_sum_hist[i] += i * padding;
    }
}

void
ecall_partition(const size_t data_size, const uint64_t mask, const uint64_t shift, uint64_t *copy_cycle_counter) {
    partition(preload_data, data_size, mask, shift, preload_sum_hist, preload_partitioned, copy_cycle_counter);
}

void
ecall_partition_user_check(void *o_data, const size_t data_size, uint32_t *outside_sum_hist, void *o_partitioned,
                           const uint64_t mask, const uint64_t shift, uint64_t *copy_cycle_counter) {
    auto outside_data = reinterpret_cast<row *>(o_data);
    auto outside_partitioned = reinterpret_cast<row *>(o_partitioned);

    partition(outside_data, data_size, mask, shift, outside_sum_hist, outside_partitioned, copy_cycle_counter);
}

void
ecall_partition_unrolled(const size_t data_size, const uint64_t mask, const uint64_t shift,
                         uint64_t *copy_cycle_counter) {
    partition_unrolled<4>(preload_data, data_size, mask, shift, preload_sum_hist, preload_partitioned, copy_cycle_counter);
}

void
ecall_partition_user_check_unrolled(void *o_data, const size_t data_size, uint32_t *outside_sum_hist,
                                    void *o_partitioned, const uint64_t mask, const uint64_t shift,
                                    uint64_t *copy_cycle_counter) {
    auto outside_data = reinterpret_cast<row *>(o_data);
    auto outside_partitioned = reinterpret_cast<row *>(o_partitioned);

    partition_unrolled<4>(outside_data, data_size, mask, shift, outside_sum_hist, outside_partitioned, copy_cycle_counter);
}

void
ecall_run_benchmark(const size_t data_size, const uint64_t radix_bits, const uint64_t shift, const uint32_t padding,
                    uint64_t *hist_cycle_counter, uint64_t *copy_cycle_counter) {
    uint64_t mask = (1 << radix_bits) - 1;
    auto fan_out = 1 << radix_bits;

    {
        rdtscpWrapper r{hist_cycle_counter};
        for (uint32_t i = 0; i < data_size; ++i) {
            size_t idx = (preload_data[i].key & mask) >> shift;
            ++preload_hist[idx];
        }
    }

    std::partial_sum(preload_hist, preload_hist + fan_out - 1, preload_sum_hist + 1);
    for (uint32_t i = 0; i < fan_out; i++) {
        preload_sum_hist[i] += i * padding;
    }

    {
        rdtscpWrapper r{copy_cycle_counter};
        for (uint32_t i = 0; i < data_size; ++i) {
            size_t idx = (preload_data[i].key & mask) >> shift;
            preload_partitioned[preload_sum_hist[idx]] = preload_data[i];
            ++preload_sum_hist[idx];
        }
    }
}

void
ecall_free_preload_hist() {
    free(preload_hist);
    free(preload_sum_hist);
    preload_hist = nullptr;
    preload_sum_hist = nullptr;
}

void
ecall_free_preload_data() {
    free(preload_data);
    free(preload_partitioned);
    preload_data = nullptr;
    preload_partitioned = nullptr;
}
}