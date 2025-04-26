#include "partitioning_algorithms.hpp"
#include "rdtscpWrapper.h"

std::optional<PartitioningFunction>
choose_function(const PartitioningMode mode, const uint32_t unroll_factor) {
    switch (mode) {
        case PartitioningMode::scalar:
            if (unroll_factor == 1)
                return partition;
            break;
        case PartitioningMode::unrolled:
            switch (unroll_factor) {
                case 1:
                    return partition_unrolled<1>;
                case 2:
                    return partition_unrolled<2>;
                case 3:
                    return partition_unrolled<3>;
                case 4:
                    return partition_unrolled<4>;
                case 5:
                    return partition_unrolled<5>;
                case 6:
                    return partition_unrolled<6>;
                case 7:
                    return partition_unrolled<7>;
                case 8:
                    return partition_unrolled<8>;
                default:
                    break;
            }
            break;
        case PartitioningMode::loop:
            switch (unroll_factor) {
                case 1:
                    return partition_loop<1>;
                case 2:
                    return partition_loop<2>;
                case 3:
                    return partition_loop<3>;
                case 4:
                    return partition_loop<4>;
                case 5:
                    return partition_loop<5>;
                case 6:
                    return partition_loop<6>;
                case 7:
                    return partition_loop<7>;
                case 8:
                    return partition_loop<8>;
                default:
                    break;
            }
            break;
    }

    return std::nullopt;
}

void
partition(const row *data, const uint32_t data_size, const uint32_t mask, const uint32_t shift, uint32_t *sum_hist,
          row *partitioned, uint64_t *copy_cycle_counter) {
    rdtscpWrapper r{copy_cycle_counter};
    for (uint32_t i = 0; i < data_size; ++i) {
        const uint32_t idx = (data[i].key & mask) >> shift;
        partitioned[sum_hist[idx]] = data[i];
        ++sum_hist[idx];
    }
}

template<int UNROLL_FACTOR>
void
partition_unrolled(const row *data, const uint32_t data_size, const uint32_t mask, const uint32_t shift,
                   uint32_t *sum_hist, row *partitioned, uint64_t *copy_cycle_counter) {
    static_assert(UNROLL_FACTOR >= 1 && UNROLL_FACTOR <= 8,
                  "UNROLL_FACTOR template parameter must be between 1 and 8!");
    rdtscpWrapper r{copy_cycle_counter};
    uint32_t i = 0;
    uint32_t idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7;
    for (; i <= data_size - UNROLL_FACTOR; i += UNROLL_FACTOR) {
        idx0 = (data[i].key & mask) >> shift;
        if constexpr (UNROLL_FACTOR == 1)
            goto inc_cpy;
        idx1 = (data[i + 1].key & mask) >> shift;
        if constexpr (UNROLL_FACTOR == 2)
            goto inc_cpy;
        idx2 = (data[i + 2].key & mask) >> shift;
        if constexpr (UNROLL_FACTOR == 3)
            goto inc_cpy;
        idx3 = (data[i + 3].key & mask) >> shift;
        if constexpr (UNROLL_FACTOR == 4)
            goto inc_cpy;
        idx4 = (data[i + 4].key & mask) >> shift;
        if constexpr (UNROLL_FACTOR == 5)
            goto inc_cpy;
        idx5 = (data[i + 5].key & mask) >> shift;
        if constexpr (UNROLL_FACTOR == 6)
            goto inc_cpy;
        idx6 = (data[i + 6].key & mask) >> shift;
        if constexpr (UNROLL_FACTOR == 7)
            goto inc_cpy;
        idx7 = (data[i + 7].key & mask) >> shift;

    inc_cpy:
        partitioned[sum_hist[idx0]] = data[i];
        ++sum_hist[idx0];
        if constexpr (UNROLL_FACTOR == 1)
            goto endloop;
        partitioned[sum_hist[idx1]] = data[i + 1];
        ++sum_hist[idx1];
        if constexpr (UNROLL_FACTOR == 2)
            goto endloop;
        partitioned[sum_hist[idx2]] = data[i + 2];
        ++sum_hist[idx2];
        if constexpr (UNROLL_FACTOR == 3)
            goto endloop;
        partitioned[sum_hist[idx3]] = data[i + 3];
        ++sum_hist[idx3];
        if constexpr (UNROLL_FACTOR == 4)
            goto endloop;
        partitioned[sum_hist[idx4]] = data[i + 4];
        ++sum_hist[idx4];
        if constexpr (UNROLL_FACTOR == 5)
            goto endloop;
        partitioned[sum_hist[idx5]] = data[i + 5];
        ++sum_hist[idx5];
        if constexpr (UNROLL_FACTOR == 6)
            goto endloop;
        partitioned[sum_hist[idx6]] = data[i + 6];
        ++sum_hist[idx6];
        if constexpr (UNROLL_FACTOR == 7)
            goto endloop;
        partitioned[sum_hist[idx7]] = data[i + 7];
        ++sum_hist[idx7];
    endloop:;
    }
    for (; i < data_size; ++i) {
        const size_t idx = (data[i].key & mask) >> shift;
        partitioned[sum_hist[idx]] = data[i];
        ++sum_hist[idx];
    }
}

void
partition_unrolled_2(const row *data, const uint32_t data_size, const uint32_t mask, const uint32_t shift,
                   uint32_t *sum_hist, row *partitioned, uint64_t *copy_cycle_counter) {
    rdtscpWrapper r{copy_cycle_counter};
    uint32_t i = 0;
    for (; i <= data_size - 2; i += 2) {
        const uint32_t idx = (data[i].key & mask) >> shift;
        const uint32_t idx1 = (data[i + 1].key & mask) >> shift;

        const uint32_t pos = sum_hist[idx]++;
        const uint32_t pos1 = sum_hist[idx1]++;

        partitioned[pos] = data[i];
        partitioned[pos1] = data[i + 1];
    }
    for (; i < data_size; ++i) {
        const size_t idx = (data[i].key & mask) >> shift;
        partitioned[sum_hist[idx]] = data[i];
        ++sum_hist[idx];
    }
}

template<int UNROLL_FACTOR>
void
partition_loop(const row *data, const uint32_t data_size, const uint32_t mask, const uint32_t shift,
                   uint32_t *sum_hist, row *partitioned, uint64_t *copy_cycle_counter) {
    rdtscpWrapper r{copy_cycle_counter};

    alignas(64) uint32_t partition_indexes[UNROLL_FACTOR];
    alignas(64) uint32_t copy_indexes[UNROLL_FACTOR];

    uint32_t i = 0;
    for (; i <= data_size - UNROLL_FACTOR; i += UNROLL_FACTOR) {
        const auto current_data = data + i;
        for (uint32_t j = 0; j < UNROLL_FACTOR; ++j) {
            partition_indexes[j] = (current_data[j].key & mask) >> shift;
        }

        for (uint32_t j = 0; j < UNROLL_FACTOR; ++j) {
            copy_indexes[j] = sum_hist[partition_indexes[j]]++;
        }

        for (uint32_t j = 0; j < UNROLL_FACTOR; ++j) {
            partitioned[copy_indexes[j]] = current_data[j];
        }
    }
    for (; i < data_size; ++i) {
        const size_t idx = (data[i].key & mask) >> shift;
        partitioned[sum_hist[idx]] = data[i];
        ++sum_hist[idx];
    }
}