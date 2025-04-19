#include "Enclave_t.h"// structs defined in .edl file etc

#include "../shared/algorithms.hpp"
#include "Barrier.hpp"
#include <vector>

constexpr size_t CACHE_LINE_SIZE = 64;
constexpr uint64_t PAGE_SIZE = 4096;

void *data = nullptr;

extern "C" {

void
ecall_noop() {
    //for benchmarking
}

void
ecall_init_data(size_t totalDataSize) {
    if (data != nullptr) {
        ocall_print_error("Data is not free!");
        return;
    }
    try {
        data = std::aligned_alloc(PAGE_SIZE, totalDataSize);
        std::memset(data, 42, totalDataSize);
    } catch (const std::bad_alloc &error) {
        ocall_print_error("Allocation inside enclave failed with std::bad_alloc!");
        throw error;
    }
}

void
ecall_random_write(uint64_t mask, uint64_t *cycle_counter) {
    random_write(reinterpret_cast<uint64_t *>(data), mask, cycle_counter);
}

void
ecall_random_write_user_check(void *plain_data, uint64_t mask, uint64_t *cycle_counter) {
    random_write(reinterpret_cast<uint64_t *>(plain_data), mask, cycle_counter);
}

void
ecall_random_write_unrolled(uint64_t mask, uint64_t *cycle_counter) {
    random_write_unrolled(reinterpret_cast<uint64_t *>(data), mask, cycle_counter);
}

void
ecall_random_write_unrolled_user_check(void *plain_data, uint64_t mask, uint64_t *cycle_counter) {
    random_write_unrolled(reinterpret_cast<uint64_t *>(plain_data), mask, cycle_counter);
}

void
ecall_random_write_size(uint64_t mask, uint64_t write_size_byte, uint64_t repetitions, uint64_t *cycle_counter) {
    random_write_size(reinterpret_cast<uint8_t *>(data), mask, write_size_byte, repetitions, cycle_counter);
}

void
ecall_random_write_size_user_check(void *plain_data, uint64_t mask, uint64_t write_size_byte, uint64_t repetitions,
                                   uint64_t *cycle_counter) {
    random_write_size(reinterpret_cast<uint8_t *>(plain_data), mask, write_size_byte, repetitions, cycle_counter);
}

void
ecall_random_increment(uint64_t mask, uint64_t *cycle_counter) {
    random_increment(reinterpret_cast<uint64_t *>(data), mask, cycle_counter);
}

void
ecall_random_read(uint64_t mask, uint64_t *cycle_counter) {
    random_read(reinterpret_cast<uint64_t *>(data), mask, cycle_counter);
}

void
ecall_random_read_user_check(void *plain_data, uint64_t mask, uint64_t *cycle_counter) {
    random_read(reinterpret_cast<uint64_t *>(plain_data), mask, cycle_counter);
}

void
ecall_random_increment_user_check(void *plain_data, uint64_t mask, uint64_t *cycle_counter) {
    random_increment(reinterpret_cast<uint64_t *>(plain_data), mask, cycle_counter);
}

void
ecall_non_temporal_write(uint64_t data_size, uint64_t *cycle_counter) {
    non_temporal_write(reinterpret_cast<uint8_t *>(data), data_size, cycle_counter);
}

void
ecall_free_preload_data() {
    std::free(data);
    data = nullptr;
}
}
