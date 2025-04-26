#include "Enclave_t.h" // structs defined in .edl file etc

#include <vector>
#include <sgx_trts.h>

#include "SIMD512.hpp"
#include "rdtscpWrapper.h"
#include "Allocator.hpp"
#include "../shared/ScanResults.hpp"
#include "../shared/ResultAllocators.hpp"
#include "../shared/ScalarScan.hpp"

std::unique_ptr<uint8_t[]> data_preload;
std::unique_ptr<int64_t[]> dict_preload;
size_t preloaded_num_records = 0;
size_t dict_num_records = 0;
bool pre_allocated = false;
std::unique_ptr<ScanResults> results_preload;

extern "C" {

void ecall_noop() {
    return;
    //for benchmarking
}

void ecall_free_preload_data() {
    if (preloaded_num_records > 0) {
        data_preload.reset();
        preloaded_num_records = 0;
    }
    if (dict_num_records > 0) {
        dict_preload.reset();
        dict_num_records = 0;
    }
}

void ecall_init_data(size_t num_entries) {
    try {
        data_preload = allocate_data_array_aligned<uint8_t, 64>(num_entries);
        preloaded_num_records = num_entries;
    } catch (const std::bad_alloc &error) {
        ocall_print_error("Allocation inside enclave failed with std::bad_alloc!");
        throw error;
    }
}

void ecall_init_dict(size_t num_entries) {
    try {
        dict_preload = allocate_data_array_aligned<int64_t, 64>(num_entries);
        dict_num_records = num_entries;
    } catch (const std::bad_alloc &error) {
        ocall_print_error("Dict allocation inside enclave failed with std::bad_alloc!");
        throw error;
    }
}

void ecall_init_results() {
    results_preload = std::make_unique<ScanResults>();
}

void ecall_init_results_bitvector(size_t num_entries) {
    // results_preload.bitvector_results.clear();
    results_preload->bitvector_results.resize(num_entries / 64);
}

void ecall_init_results_index(size_t num_threads) {
    results_preload->index_results_per_thread.resize(num_threads);
    for (auto &vec: results_preload->index_results_per_thread) {
        vec.clear();
    }
    results_preload->index_results.clear();
    pre_allocated = false;
}

void ecall_init_results_index_pre_alloc(size_t num_threads, uint8_t predicate_low, uint8_t predicate_high,
                                        size_t num_records_per_thread) {
    pre_alloc_per_thread(data_preload.get(), results_preload->index_results_per_thread, num_threads, predicate_low,
                         predicate_high, num_records_per_thread);
    pre_allocated = true;
}

void ecall_init_results_dict(size_t num_threads) {
    results_preload->dict_results_per_thread.resize(num_threads);
    for (auto &vec: results_preload->dict_results_per_thread) {
        vec.clear();
    }
    results_preload->dict_results.clear();
    pre_allocated = false;
}

void ecall_init_results_dict_pre_alloc(size_t num_threads, uint8_t predicate_low, uint8_t predicate_high,
                                       size_t num_records_per_thread) {
    pre_alloc_per_thread(data_preload.get(), results_preload->dict_results_per_thread, num_threads, predicate_low,
                         predicate_high, num_records_per_thread);
    pre_allocated = true;
}

void
ecall_index_scan_user(uint8_t predicate_low,
                      uint8_t predicate_high,
                      const uint8_t *data,
                      size_t num_records,
                      void *output_buffer,
                      uint64_t *cpu_cntr,
                      size_t num_runs,
                      size_t warmup_runs,
                      int unique_data) {
    if (unique_data) {
        num_runs = 1;
        warmup_runs = 0;
    }

    auto output_buffer_vec = reinterpret_cast<CacheAlignedVector<uint64_t> *>(output_buffer);

    for (auto i = 0; i < warmup_runs; ++i) {
        SIMD512::implicit_index_scan_self_alloc(predicate_low,
                                                predicate_high,
                                                reinterpret_cast<const __m512i *>(data),
                                                num_records,
                                                *output_buffer_vec,
                                                false);
    }
    rdtscpWrapper wrap(cpu_cntr);
    for (auto i = 0; i < num_runs; ++i) {
        SIMD512::implicit_index_scan_self_alloc(predicate_low,
                                                predicate_high,
                                                reinterpret_cast<const __m512i *>(data),
                                                num_records,
                                                *output_buffer_vec,
                                                false);
    }
}

void
ecall_index_scan_self_alloc_preload(uint8_t predicate_low,
                                    uint8_t predicate_high,
                                    size_t start_idx,
                                    size_t num_records,
                                    size_t thread_id,
                                    uint64_t *cpu_cntr,
                                    size_t num_runs,
                                    size_t warmup_runs,
                                    int unique_data,
                                    int pre_alloc) {
    if (preloaded_num_records == 0) {
        ocall_print_error("Trying to run preload scan without preload data!");
        return;
    }

    if (unique_data) {
        num_runs = 1;
        warmup_runs = 0;
    }

    if (!pre_alloc) {
        results_preload->index_results_per_thread[thread_id] = CacheAlignedVector<uint64_t>();
    }

    try {
        for (auto i = 0; i < warmup_runs; ++i) {
            SIMD512::implicit_index_scan_self_alloc(predicate_low,
                                                    predicate_high,
                                                    reinterpret_cast<__m512i *>(data_preload.get() + start_idx),
                                                    num_records,
                                                    results_preload->index_results_per_thread[thread_id],
                                                    !pre_alloc);
        }
        rdtscpWrapper wrap(cpu_cntr);
        for (auto i = 0; i < num_runs; ++i) {
            SIMD512::implicit_index_scan_self_alloc(predicate_low,
                                                    predicate_high,
                                                    reinterpret_cast<__m512i *>(data_preload.get() + start_idx),
                                                    num_records,
                                                    results_preload->index_results_per_thread[thread_id],
                                                    !pre_alloc);
        }
    } catch (const std::bad_alloc &error) {
        ocall_print_error("reallocation failed!");
        throw;
    }
}

void
ecall_scalar_scan_user(uint8_t predicate_low,
                       uint8_t predicate_high,
                       const uint8_t *data,
                       size_t num_records,
                       void *output_buffer,
                       uint64_t *cpu_cntr,
                       size_t num_runs,
                       size_t warmup_runs,
                       int unique_data) {
    if (unique_data) {
        num_runs = 1;
        warmup_runs = 0;
    }

    auto output_buffer_vec = reinterpret_cast<CacheAlignedVector<uint64_t> *>(output_buffer);

    for (auto i = 0; i < warmup_runs; ++i) {
        scalar_implicit_index_scan(predicate_low,
                                   predicate_high,
                                   data,
                                   num_records,
                                   *output_buffer_vec,
                                   false);
    }
    rdtscpWrapper wrap(cpu_cntr);
    for (auto i = 0; i < num_runs; ++i) {
        scalar_implicit_index_scan(predicate_low,
                                   predicate_high,
                                   data,
                                   num_records,
                                   *output_buffer_vec,
                                   false);
    }
}

void
ecall_scalar_scan_preload(uint8_t predicate_low,
                          uint8_t predicate_high,
                          size_t start_idx,
                          size_t num_records,
                          size_t thread_id,
                          uint64_t *cpu_cntr,
                          size_t num_runs,
                          size_t warmup_runs,
                          int unique_data,
                          int pre_alloc) {
    if (preloaded_num_records == 0) {
        ocall_print_error("Trying to run preload scan without preload data!");
        return;
    }

    if (unique_data) {
        num_runs = 1;
        warmup_runs = 0;
    }

    if (!pre_alloc) {
        results_preload->index_results_per_thread[thread_id] = CacheAlignedVector<uint64_t>();
    }

    try {
        for (auto i = 0; i < warmup_runs; ++i) {
            scalar_implicit_index_scan(predicate_low,
                                       predicate_high,
                                       data_preload.get() + start_idx,
                                       num_records,
                                       results_preload->index_results_per_thread[thread_id],
                                       !pre_alloc);
        }
        rdtscpWrapper wrap(cpu_cntr);
        for (auto i = 0; i < num_runs; ++i) {
            scalar_implicit_index_scan(predicate_low,
                                       predicate_high,
                                       data_preload.get() + start_idx,
                                       num_records,
                                       results_preload->index_results_per_thread[thread_id],
                                       !pre_alloc);
        }
    } catch (const std::bad_alloc &error) {
        ocall_print_error("reallocation failed!");
        throw;
    }
}

void
ecall_bitvector_scan_user(uint8_t predicate_low,
                          uint8_t predicate_high,
                          const uint8_t *data,
                          size_t num_records,
                          uint64_t *output_buffer,
                          uint64_t *cpu_cntr,
                          size_t num_runs,
                          size_t warmup_runs,
                          int unique_data) {
    if (unique_data) {
        num_runs = 1;
        warmup_runs = 0;
    }

    for (auto i = 0; i < warmup_runs; ++i) {
        SIMD512::bitvector_scan(predicate_low,
                                predicate_high,
                                (const __m512i *) data,
                                num_records,
                                reinterpret_cast<__mmask64 *>(output_buffer));
    }
    rdtscpWrapper wrap(cpu_cntr);
    for (auto i = 0; i < num_runs; ++i) {
        SIMD512::bitvector_scan(predicate_low,
                                predicate_high,
                                (const __m512i *) data,
                                num_records,
                                reinterpret_cast<__mmask64 *>(output_buffer));
    }
}

void
ecall_bitvector_scan_preload(uint8_t predicate_low,
                             uint8_t predicate_high,
                             size_t start_idx,
                             size_t num_records,
                             size_t output_idx,
                             uint64_t *cpu_cntr,
                             size_t num_runs,
                             size_t warmup_runs,
                             int unique_data) {
    if (preloaded_num_records == 0) {
        ocall_print_error("Trying to run preload scan without preload data!");
        return;
    }
    if (unique_data) {
        num_runs = 1;
        warmup_runs = 0;
    }
    auto data_start = reinterpret_cast<__m512i *>(data_preload.get() + start_idx);
    auto result_start = reinterpret_cast<__mmask64 *>(results_preload->bitvector_results.data() + output_idx);

    for (auto i = 0; i < warmup_runs; ++i) {
        SIMD512::bitvector_scan(predicate_low, predicate_high, data_start, num_records, result_start);
    }
    rdtscpWrapper wrap(cpu_cntr);
    for (auto i = 0; i < num_runs; ++i) {
        SIMD512::bitvector_scan(predicate_low, predicate_high, data_start, num_records, result_start);
    }
}

void ecall_dict_scan_user(int64_t predicate_low,
                          int64_t predicate_high,
                          const uint8_t *data,
                          size_t num_records,
                          const int64_t *dict,
                          void *output_buffer,
                          uint64_t *cpu_cntr,
                          size_t num_runs,
                          size_t warmup_runs,
                          int unique_data) {
    // This only works if pre_alloc is activated, because otherwise reallocation of the result array happens inside the
    // enclave, all data is copied to the enclave, and it is not a pure scan outside the enclave anymore.

    if (unique_data) {
        num_runs = 1;
        warmup_runs = 0;
    }

    auto output_buffer_vec = reinterpret_cast<CacheAlignedVector<int64_t> *>(output_buffer);

    for (auto i = 0; i < warmup_runs; ++i) {
        SIMD512::dict_scan_8bit_64bit_scalar_gather_scatter(predicate_low,
                                                            predicate_high,
                                                            dict,
                                                            reinterpret_cast<const __m512i *>(data),
                                                            num_records,
                                                            *output_buffer_vec,
                                                            false);
    }

    rdtscpWrapper rdtscpWrap(cpu_cntr);
    for (auto i = 0; i < num_runs; ++i) {
        SIMD512::dict_scan_8bit_64bit_scalar_gather_scatter(predicate_low,
                                                            predicate_high,
                                                            dict,
                                                            reinterpret_cast<const __m512i *>(data),
                                                            num_records,
                                                            *output_buffer_vec,
                                                            false);
    }
}


void ecall_dict_scan_preload(int64_t predicate_low,
                             int64_t predicate_high,
                             size_t start_idx,
                             size_t num_records,
                             size_t thread_id,
                             uint64_t *cpu_cntr,
                             size_t num_runs,
                             size_t warmup_runs,
                             int unique_data,
                             int pre_alloc) {
    if (preloaded_num_records == 0) {
        ocall_print_error("Trying to run preload scan without preload data!");
        return;
    }
    if (unique_data) {
        num_runs = 1;
        warmup_runs = 0;
    }

    if (!pre_alloc) {
        results_preload->dict_results_per_thread[thread_id] = CacheAlignedVector<int64_t>(); // TODO what do I want to measure here?
    }

    for (auto i = 0; i < warmup_runs; ++i) {
        SIMD512::dict_scan_8bit_64bit_scalar_gather_scatter(predicate_low,
                                                            predicate_high,
                                                            dict_preload.get(),
                                                            reinterpret_cast<const __m512i *>(data_preload.get() +
                                                                                              start_idx),
                                                            num_records,
                                                            results_preload->dict_results_per_thread[thread_id],
                                                            !pre_alloc);
    }

    rdtscpWrapper rdtscpWrap(cpu_cntr);
    for (auto i = 0; i < num_runs; ++i) {
        SIMD512::dict_scan_8bit_64bit_scalar_gather_scatter(predicate_low,
                                                            predicate_high,
                                                            dict_preload.get(),
                                                            reinterpret_cast<const __m512i *>(data_preload.get() +
                                                                                              start_idx),
                                                            num_records,
                                                            results_preload->dict_results_per_thread[thread_id],
                                                            !pre_alloc);
    }
}

}