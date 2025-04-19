#include "algorithms.hpp"
#include "RNG.hpp"
#include "rdtscpWrapper.h"

#include "SIMD512.hpp"
#include <algorithm>

template<uint64_t REPETITIONS>
void
random_write(uint64_t *data, uint64_t mask, uint64_t *cycle_counter) {
    LinearCongruentialGenerator64 rand{};

    rdtscpWrapper r{cycle_counter};
    for (uint64_t i = 0; i < REPETITIONS; ++i) {
        uint64_t rand_value = rand();
        data[rand_value & mask] = rand_value;
    }
}

template<uint64_t REPETITIONS>
void
random_write_unrolled(uint64_t *data, uint64_t mask, uint64_t *cycle_counter) {
    static_assert(REPETITIONS % 4 == 0, "Number of repetitions must be dividable by 4");

    LinearCongruentialGenerator64 rand{};

    rdtscpWrapper r{cycle_counter};
    for (uint64_t i = 0; i < REPETITIONS; i += 4) {
        uint64_t rand_value = rand();
        uint64_t rand_value2 = rand();
        uint64_t rand_value3 = rand();
        uint64_t rand_value4 = rand();
        data[rand_value & mask] = rand_value;
        data[rand_value2 & mask] = rand_value2;
        data[rand_value3 & mask] = rand_value3;
        data[rand_value4 & mask] = rand_value4;
    }
}

void
random_write_size(uint8_t *data, uint64_t mask, uint64_t write_size_byte, uint64_t repetitions,
                  uint64_t *cycle_counter) {
    LinearCongruentialGenerator64 rand{};

    rdtscpWrapper r{cycle_counter};
    for (uint64_t i = 0; i < repetitions; ++i) {
        uint64_t rand_value = rand();
        uint64_t start_pos = rand_value & mask;
        auto write_value = static_cast<uint8_t>(rand_value & 0xff);
        for (uint64_t j = 0; j < write_size_byte; ++j) {
            data[start_pos + j] = write_value;
        }
    }
}

template<uint64_t REPETITIONS>
void
random_increment(uint64_t *data, uint64_t mask, uint64_t *cycle_counter) {
    LinearCongruentialGenerator64 rand{};

    rdtscpWrapper r{cycle_counter};
    for (uint64_t i = 0; i < REPETITIONS; ++i) {
        ++data[rand() & mask];
    }
}

template<uint64_t REPETITIONS>
void
random_read(uint64_t *data, uint64_t mask, uint64_t *cycle_counter) {
    LinearCongruentialGenerator64 rand{};

    uint64_t sum = 0;

    rdtscpWrapper r{cycle_counter};
    for (uint64_t i = 0; i < REPETITIONS; ++i) {
        sum += data[rand() & mask];
    }

    data[0] = sum;
}

template<uint64_t REPETITIONS>
void
non_temporal_write(uint8_t *data, uint64_t data_size, uint64_t *cycle_counter) {
    uint64_t numElements = data_size / 64;
    // Move the mask right by 6 bits so that every masked value is a multiple of 64. This makes it possible to remove
    // a "shl" from the write loop.
    uint64_t mask = (numElements - 1) << 6;

    LinearCongruentialGenerator64 rand{};

    auto forty_two = _mm512_set1_epi64(42);

    rdtscpWrapper r{cycle_counter};
    for (uint64_t i = 0; i < REPETITIONS; ++i) {
        // Do the pointer arithmetics in bytes and not cache lines. Instead, make sure that rand() & mask is a
        // multiple of 64, so that the effect is the same.
        _mm512_stream_si512(reinterpret_cast<__m512i *>(data + (rand() & mask)), forty_two);
    }
}

template void
random_write<DEFAULT_REPETITIONS>(uint64_t *data, uint64_t mask, uint64_t *cycle_counter);
template void
random_write_unrolled<DEFAULT_REPETITIONS>(uint64_t *data, uint64_t mask, uint64_t *cycle_counter);
template void
random_increment<DEFAULT_REPETITIONS>(uint64_t *data, uint64_t mask, uint64_t *cycle_counter);
template void
random_read<DEFAULT_REPETITIONS>(uint64_t *data, uint64_t mask, uint64_t *cycle_counter);
template void
non_temporal_write<DEFAULT_REPETITIONS>(uint8_t *data, uint64_t data_size, uint64_t *cycle_counter);
