#include "histogram_algorithms.hpp"
#include "avx_includes.hpp"

#include "rdtscpWrapper.h"


constexpr uint32_t CACHE_LINE_SIZE = 64;
constexpr uint32_t ROWS_PER_512_REGISTER = 8;
constexpr uint32_t KEYS_PER_512_REGISTER = 16;

std::optional<HistogramFunction>
choose_function(const HistogramMode mode, const uint32_t unroll_factor) {
    switch (mode) {
        case HistogramMode::scalar:
            if (unroll_factor == 1)
                return histogram;
            break;
        case HistogramMode::pragma:
            if (unroll_factor == 8)
                return histogram_pragma;
            break;
        case HistogramMode::unrolled:
            switch (unroll_factor) {
                case 1:
                    return histogram_unrolled_max_16<1>;
                case 2:
                    return histogram_unrolled_max_16<2>;
                case 3:
                    return histogram_unrolled_max_16<3>;
                case 4:
                    return histogram_unrolled_max_16<4>;
                case 5:
                    return histogram_unrolled_max_16<5>;
                case 6:
                    return histogram_unrolled_max_16<6>;
                case 7:
                    return histogram_unrolled_max_16<7>;
                case 8:
                    return histogram_unrolled_max_16<8>;
                case 9:
                    return histogram_unrolled_max_16<9>;
                case 10:
                    return histogram_unrolled_max_16<10>;
                case 11:
                    return histogram_unrolled_max_16<11>;
                case 12:
                    return histogram_unrolled_max_16<12>;
                case 13:
                    return histogram_unrolled_max_16<13>;
                case 14:
                    return histogram_unrolled_max_16<14>;
                case 15:
                    return histogram_unrolled_max_16<15>;
                case 16:
                    return histogram_unrolled_max_16<16>;
                default:
                    break;
            }
            break;
        case HistogramMode::shuffle:
            switch (unroll_factor) {
                case 2:
                    return histogram_shuffle_2;
                case 4:
                    return histogram_shuffle_4;
                case 8:
                    return histogram_shuffle_8;
            }
            break;
        case HistogramMode::loop:
            switch (unroll_factor) {
                case 1:
                    return histogram_loop<1>;
                case 2:
                    return histogram_loop<2>;
                case 3:
                    return histogram_loop<3>;
                case 4:
                    return histogram_loop<4>;
                case 5:
                    return histogram_loop<5>;
                case 6:
                    return histogram_loop<6>;
                case 7:
                    return histogram_loop<7>;
                case 8:
                    return histogram_loop<8>;
                case 9:
                    return histogram_loop<9>;
                case 10:
                    return histogram_loop<10>;
                case 11:
                    return histogram_loop<11>;
                case 12:
                    return histogram_loop<12>;
                case 16:
                    return histogram_loop<16>;
                default:
                    break;
            }
            break;
        case HistogramMode::loop_pragma:
            if (unroll_factor == 8)
                return histogram_loop_pragma;
            break;
        case HistogramMode::simd_loop:
            if (unroll_factor == 4) {
                return histogram_simd_loop;
            }
            break;
        case HistogramMode::simd_loop_unrolled:
            switch (unroll_factor) {
                case 1:
                    return histogram_simd_loop_unrolled<1>;
                case 2:
                    return histogram_simd_loop_unrolled<2>;
                case 3:
                    return histogram_simd_loop_unrolled<3>;
                case 4:
                    return histogram_simd_loop_unrolled<4>;
                case 5:
                    return histogram_simd_loop_unrolled<5>;
                case 6:
                    return histogram_simd_loop_unrolled<6>;
                case 7:
                    return histogram_simd_loop_unrolled<7>;
                case 8:
                    return histogram_simd_loop_unrolled<8>;
                default:
                    break;
            }
            break;
        case HistogramMode::simd_loop_unrolled_512:
            switch (unroll_factor) {
                case 1:
                    return histogram_simd_loop_unrolled_512<1>;
                case 2:
                    return histogram_simd_loop_unrolled_512<2>;
                case 3:
                    return histogram_simd_loop_unrolled_512<3>;
                case 4:
                    return histogram_simd_loop_unrolled_512<4>;
                default:
                    break;
            }
            break;
        case HistogramMode::simd_unrolled:
            if (unroll_factor == 4) {
                return histogram_simd_unrolled;
            }
            break;
    }

    return std::nullopt;
}

void
histogram(const row *data, const uint32_t data_size, const uint32_t mask, const uint32_t shift, uint32_t *hist,
          uint64_t *cycle_counter) {
    rdtscpWrapper r{cycle_counter};
    for (uint32_t i = 0; i < data_size; ++i) {
        const uint32_t idx = (data[i].key & mask) >> shift;
        ++hist[idx];
    }
}

void
histogram_pragma(const row *data, const uint32_t data_size, const uint32_t mask, const uint32_t shift, uint32_t *hist,
                 uint64_t *cycle_counter) {
    rdtscpWrapper r{cycle_counter};
#pragma GCC unroll 8
    for (uint32_t i = 0; i < data_size; ++i) {
        const uint32_t idx = (data[i].key & mask) >> shift;
        ++hist[idx];
    }
}

void
histogram_shuffle_2(const row *data, const uint32_t data_size, const uint32_t mask, const uint32_t shift,
                     uint32_t *hist, uint64_t *cycle_counter) {
    rdtscpWrapper r{cycle_counter};
    uint32_t i = 0;
    for (; i <= data_size - 2; i += 2) {
        const uint32_t idx = (data[i].key & mask) >> shift;
        ++hist[idx];
        const uint32_t idx1 = (data[i + 1].key & mask) >> shift;
        ++hist[idx1];
    }
    for (; i < data_size; ++i) {
        const uint32_t idx = (data[i].key & mask) >> shift;
        ++hist[idx];
    }
}

void
histogram_shuffle_4(const row *data, const uint32_t data_size, const uint32_t mask, const uint32_t shift,
                     uint32_t *hist, uint64_t *cycle_counter) {
    rdtscpWrapper r{cycle_counter};
    uint32_t i = 0;
    for (; i <= data_size - 4; i += 4) {
        const uint32_t idx = (data[i].key & mask) >> shift;
        const uint32_t idx1 = (data[i + 1].key & mask) >> shift;
        ++hist[idx];
        ++hist[idx1];

        const uint32_t idx2 = (data[i + 2].key & mask) >> shift;
        const uint32_t idx3 = (data[i + 3].key & mask) >> shift;
        ++hist[idx2];
        ++hist[idx3];
    }
    for (; i < data_size; ++i) {
        const uint32_t idx = (data[i].key & mask) >> shift;
        ++hist[idx];
    }
}

void
histogram_shuffle_8(const row *data, const uint32_t data_size, const uint32_t mask, const uint32_t shift,
                     uint32_t *hist, uint64_t *cycle_counter) {
    rdtscpWrapper r{cycle_counter};
    uint32_t i = 0;
    for (; i <= data_size - 8; i += 8) {
        const uint32_t idx = (data[i].key & mask) >> shift;
        const uint32_t idx1 = (data[i + 1].key & mask) >> shift;
        const uint32_t idx2 = (data[i + 2].key & mask) >> shift;
        const uint32_t idx3 = (data[i + 3].key & mask) >> shift;
        ++hist[idx];
        ++hist[idx1];
        ++hist[idx2];
        ++hist[idx3];
        const uint32_t idx4 = (data[i].key & mask) >> shift;
        const uint32_t idx5 = (data[i + 1].key & mask) >> shift;
        const uint32_t idx6 = (data[i + 2].key & mask) >> shift;
        const uint32_t idx7 = (data[i + 3].key & mask) >> shift;
        ++hist[idx4];
        ++hist[idx5];
        ++hist[idx6];
        ++hist[idx7];
    }
    for (; i < data_size; ++i) {
        const uint32_t idx = (data[i].key & mask) >> shift;
        ++hist[idx];
    }
}

void
histogram_unrolled(const row *data, const uint32_t data_size, const uint32_t mask, const uint32_t shift, uint32_t *hist,
                   uint64_t *cycle_counter) {
    rdtscpWrapper r{cycle_counter};
    uint32_t i = 0;
    for (; i <= data_size - 8; i += 8) {
        const uint32_t idx = (data[i].key & mask) >> shift;
        const uint32_t idx1 = (data[i + 1].key & mask) >> shift;
        const uint32_t idx2 = (data[i + 2].key & mask) >> shift;
        const uint32_t idx3 = (data[i + 3].key & mask) >> shift;
        const uint32_t idx4 = (data[i + 4].key & mask) >> shift;
        const uint32_t idx5 = (data[i + 5].key & mask) >> shift;
        const uint32_t idx6 = (data[i + 6].key & mask) >> shift;
        const uint32_t idx7 = (data[i + 7].key & mask) >> shift;
        ++hist[idx];
        ++hist[idx1];
        ++hist[idx2];
        ++hist[idx3];
        ++hist[idx4];
        ++hist[idx5];
        ++hist[idx6];
        ++hist[idx7];
    }
    for (; i < data_size; ++i) {
        const uint32_t idx = (data[i].key & mask) >> shift;
        ++hist[idx];
    }
}

template<int UNROLL_FACTOR>
void
histogram_unrolled_max_16(const row *data, const uint32_t data_size, const uint32_t mask, const uint32_t shift,
                          uint32_t *hist, uint64_t *cycle_counter) {
    static_assert(UNROLL_FACTOR >= 1 && UNROLL_FACTOR <= 16,
                  "UNROLL_FACTOR template parameter must be between 8 and 16!");
    rdtscpWrapper r{cycle_counter};
    uint32_t i = 0;
    uint32_t idx, idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8, idx9, idx10, idx11, idx12, idx13, idx14, idx15;
    for (; i <= data_size - UNROLL_FACTOR; i += UNROLL_FACTOR) {
        idx = (data[i].key & mask) >> shift;
        if constexpr (UNROLL_FACTOR == 1)
            goto inc;
        idx1 = (data[i + 1].key & mask) >> shift;
        if constexpr (UNROLL_FACTOR == 2)
            goto inc;
        idx2 = (data[i + 2].key & mask) >> shift;
        if constexpr (UNROLL_FACTOR == 3)
            goto inc;
        idx3 = (data[i + 3].key & mask) >> shift;
        if constexpr (UNROLL_FACTOR == 4)
            goto inc;
        idx4 = (data[i + 4].key & mask) >> shift;
        if constexpr (UNROLL_FACTOR == 5)
            goto inc;
        idx5 = (data[i + 5].key & mask) >> shift;
        if constexpr (UNROLL_FACTOR == 6)
            goto inc;
        idx6 = (data[i + 6].key & mask) >> shift;
        if constexpr (UNROLL_FACTOR == 7)
            goto inc;
        idx7 = (data[i + 7].key & mask) >> shift;
        if constexpr (UNROLL_FACTOR == 8)
            goto inc;
        idx8 = (data[i + 8].key & mask) >> shift;
        if constexpr (UNROLL_FACTOR == 9)
            goto inc;
        idx9 = (data[i + 9].key & mask) >> shift;
        if constexpr (UNROLL_FACTOR == 10)
            goto inc;
        idx10 = (data[i + 10].key & mask) >> shift;
        if constexpr (UNROLL_FACTOR == 11)
            goto inc;
        idx11 = (data[i + 11].key & mask) >> shift;
        if constexpr (UNROLL_FACTOR == 12)
            goto inc;
        idx12 = (data[i + 12].key & mask) >> shift;
        if constexpr (UNROLL_FACTOR == 13)
            goto inc;
        idx13 = (data[i + 13].key & mask) >> shift;
        if constexpr (UNROLL_FACTOR == 14)
            goto inc;
        idx14 = (data[i + 14].key & mask) >> shift;
        if constexpr (UNROLL_FACTOR == 15)
            goto inc;
        idx15 = (data[i + 15].key & mask) >> shift;
    inc:
        ++hist[idx];
        if constexpr (UNROLL_FACTOR == 1)
            goto endloop;
        ++hist[idx1];
        if constexpr (UNROLL_FACTOR == 2)
            goto endloop;
        ++hist[idx2];
        if constexpr (UNROLL_FACTOR == 3)
            goto endloop;
        ++hist[idx3];
        if constexpr (UNROLL_FACTOR == 4)
            goto endloop;
        ++hist[idx4];
        if constexpr (UNROLL_FACTOR == 5)
            goto endloop;
        ++hist[idx5];
        if constexpr (UNROLL_FACTOR == 6)
            goto endloop;
        ++hist[idx6];
        if constexpr (UNROLL_FACTOR == 7)
            goto endloop;
        ++hist[idx7];
        if constexpr (UNROLL_FACTOR == 8)
            goto endloop;
        ++hist[idx8];
        if constexpr (UNROLL_FACTOR == 9)
            goto endloop;
        ++hist[idx9];
        if constexpr (UNROLL_FACTOR == 10)
            goto endloop;
        ++hist[idx10];
        if constexpr (UNROLL_FACTOR == 11)
            goto endloop;
        ++hist[idx11];
        if constexpr (UNROLL_FACTOR == 12)
            goto endloop;
        ++hist[idx12];
        if constexpr (UNROLL_FACTOR == 13)
            goto endloop;
        ++hist[idx13];
        if constexpr (UNROLL_FACTOR == 14)
            goto endloop;
        ++hist[idx14];
        if constexpr (UNROLL_FACTOR == 15)
            goto endloop;
        ++hist[idx15];
    endloop:;
    }
    for (; i < data_size; ++i) {
        const uint32_t idx = (data[i].key & mask) >> shift;
        ++hist[idx];
    }
}

template<int UNROLL_FACTOR>
void
histogram_loop(const row *data, const uint32_t data_size, const uint32_t mask, const uint32_t shift, uint32_t *hist,
               uint64_t *cycle_counter) {
    rdtscpWrapper r{cycle_counter};
    alignas(64) uint32_t indexes[UNROLL_FACTOR];
    uint32_t i = 0;
    for (; i <= data_size - UNROLL_FACTOR; i += UNROLL_FACTOR) {
        const auto current_data = data + i;
        for (uint64_t j = 0; j < UNROLL_FACTOR; j++) {
            indexes[j] = (current_data[j].key & mask) >> shift;
        }

        for (uint64_t j = 0; j < UNROLL_FACTOR; j++) {
            ++hist[indexes[j]];
        }
    }
    // handle the remaining entries if data_size is not dividable by unroll_factor
    for (; i < data_size; ++i) {
        const uint32_t idx = (data[i].key & mask) >> shift;
        ++hist[idx];
    }
}

void
histogram_loop_pragma(const row *data, const uint32_t data_size, const uint32_t mask, const uint32_t shift,
                      uint32_t *hist, uint64_t *cycle_counter) {
    rdtscpWrapper r{cycle_counter};
    alignas(64) uint32_t indexes[8];
    uint32_t i = 0;
    for (; i <= data_size - 8; i += 8) {
        const auto current_data = data + i;

#pragma GCC unroll 8
        for (uint64_t j = 0; j < 8; j++) {
            indexes[j] = (current_data[j].key & mask) >> shift;
        }

#pragma GCC unroll 8
        for (uint64_t j = 0; j < 8; j++) {
            ++hist[indexes[j]];
        }
    }
    // handle the remaining entries if data_size is not dividable by unroll_factor
    for (; i < data_size; ++i) {
        const uint32_t idx = (data[i].key & mask) >> shift;
        ++hist[idx];
    }
}

void
histogram_simd_loop(const row *data, const uint32_t data_size, const uint32_t mask, const uint32_t shift,
                    uint32_t *hist, uint64_t *cycle_counter) {
    rdtscpWrapper r{cycle_counter};

    const auto simd_data = reinterpret_cast<__m512i *>(const_cast<row *>(data));
    const auto steps = data_size / 32;

    alignas(64) __m256i mm256i_buffer[4];

    const auto simd_mask = _mm512_set1_epi32(mask);
    const auto simd_shift = _mm512_set1_epi32(shift);

    for (uint32_t i = 0; i < steps; ++i) {
        for (uint32_t j = 0; j < 4; j++) {
            const auto vec = _mm512_stream_load_si512(simd_data + (i * 4) + j);
            const auto masked = _mm512_and_epi32(vec, simd_mask);
            const auto shifted = _mm512_srav_epi32(masked, simd_shift);
            const auto compressed = _mm512_cvtepi64_epi32(shifted);
            _mm256_store_epi32(mm256i_buffer + j, compressed);
        }

        for (uint32_t m256_index = 0; m256_index < 4; ++m256_index) {
#pragma GCC unroll 8
            for (int int_index = 0; int_index < 8; ++int_index) {
                const uint32_t index = _mm256_extract_epi32(mm256i_buffer[m256_index], int_index);
                ++hist[index];
            }
        }
    }
    // if data_size % 32 != 0 then handle remaining rows
    for (uint32_t i = steps * 32; i < data_size; ++i) {
        const uint32_t idx = (data[i].key & mask) >> shift;
        ++hist[idx];
    }
}

template<int UNROLL_FACTOR>
void
histogram_simd_loop_unrolled(const row *data, const uint32_t data_size, const uint32_t mask, const uint32_t shift,
                             uint32_t *hist, uint64_t *cycle_counter) {
    rdtscpWrapper r{cycle_counter};

    const auto simd_data = reinterpret_cast<__m512i *>(const_cast<row *>(data));
    ;
    const auto m512_steps = data_size / (ROWS_PER_512_REGISTER * UNROLL_FACTOR);

    alignas(64) __m256i mm256i_buffer[UNROLL_FACTOR];

    const auto simd_mask = _mm256_set1_epi32(mask);
    const auto simd_shift = _mm256_set1_epi32(shift);

    for (uint32_t m512_index = 0; m512_index < m512_steps; ++m512_index) {
        for (uint32_t unroll_index = 0; unroll_index < UNROLL_FACTOR; unroll_index++) {
            const auto vec = _mm512_load_si512(simd_data + (m512_index * UNROLL_FACTOR) + unroll_index);
            const auto compressed = _mm512_cvtepi64_epi32(vec);
            const auto masked = _mm256_and_si256(compressed, simd_mask);
            auto shifted = _mm256_srav_epi32(masked, simd_shift);
            _mm256_store_epi32(mm256i_buffer + unroll_index, shifted);
        }

        for (uint32_t m256_index = 0; m256_index < UNROLL_FACTOR; ++m256_index) {
#pragma GCC unroll 8
            for (int int_index = 0; int_index < ROWS_PER_512_REGISTER; ++int_index) {
                const uint32_t index = _mm256_extract_epi32(mm256i_buffer[m256_index], int_index);
                ++hist[index];
            }
        }
    }
    // if data_size % 32 != 0 then handle remaining rows
    for (uint32_t i = m512_steps * (ROWS_PER_512_REGISTER * UNROLL_FACTOR); i < data_size; ++i) {
        const uint32_t idx = (data[i].key & mask) >> shift;
        ++hist[idx];
    }
}

template<int UNROLL_FACTOR>
void
histogram_simd_loop_unrolled_512(const row *data, const uint32_t data_size, const uint32_t mask, const uint32_t shift,
                                 uint32_t *hist, uint64_t *cycle_counter) {
    rdtscpWrapper r{cycle_counter};

    const auto simd_data = reinterpret_cast<__m512i *>(const_cast<row *>(data));
    ;
    const auto m512_steps = data_size / (KEYS_PER_512_REGISTER * UNROLL_FACTOR);

    //auto mm256i_buffer = (__m256i *) aligned_alloc(CACHE_LINE_SIZE, sizeof(__m256i) * UNROLL_FACTOR);;
    alignas(64) __m512i m512i_buffer[UNROLL_FACTOR];

    const auto simd_mask = _mm512_set1_epi32(mask);
    const auto simd_shift = _mm512_set1_epi32(shift);
    const __mmask16 blend_mask = 0b1010101010101010;// Take the higher 32 bits of 64 bits from the second vector

    for (uint32_t input_index = 0; input_index < m512_steps; ++input_index) {
        const auto read_start = simd_data + input_index * 2 * UNROLL_FACTOR;
        for (uint32_t unroll_index = 0; unroll_index < UNROLL_FACTOR; unroll_index++) {
            const auto vec1 = _mm512_load_si512(read_start + unroll_index * 2);
            const auto vec2 = _mm512_load_ps(read_start + unroll_index * 2 + 1);
            const auto vec2_doubled = _mm512_moveldup_ps(vec2);
            const auto combined = _mm512_mask_blend_epi32(blend_mask, vec1, (__m512i) vec2_doubled);
            const auto masked = _mm512_and_epi32(combined, simd_mask);
            auto shifted = _mm512_srav_epi32(masked, simd_shift);
            _mm512_store_epi32(m512i_buffer + unroll_index, shifted);
        }

        for (uint32_t buffer_index = 0; buffer_index < UNROLL_FACTOR; ++buffer_index) {
            auto &m512_reg = m512i_buffer[buffer_index];
#pragma GCC unroll 4
            for (int m128_index = 0; m128_index < 4; ++m128_index) {
                auto m128_reg = _mm512_extracti32x4_epi32(m512_reg, m128_index);
#pragma GCC unroll 4
                for (int int_index = 0; int_index < 4; ++int_index) {
                    const uint32_t index = _mm_extract_epi32(m128_reg, int_index);
                    ++hist[index];
                }
            }
        }
    }
    // if data_size % 32 != 0 then handle remaining rows
    for (uint32_t i = m512_steps * (KEYS_PER_512_REGISTER * UNROLL_FACTOR); i < data_size; ++i) {
        const uint32_t idx = (data[i].key & mask) >> shift;
        ++hist[idx];
    }
}

// TODO different simd unrolling factors
void
histogram_simd_unrolled(const row *data, const uint32_t data_size, const uint32_t mask, const uint32_t shift,
                        uint32_t *hist, uint64_t *cycle_counter) {
    rdtscpWrapper r{cycle_counter};

    const auto simd_data = reinterpret_cast<__m512i *>(const_cast<row *>(data));
    const auto steps = data_size / 32;

    alignas(64) uint8_t buffer[128];
    const auto int_buffer = reinterpret_cast<uint32_t *>(buffer);

    const auto simd_mask = _mm512_set1_epi32(mask);
    const auto simd_shift = _mm512_set1_epi32(shift);

    for (uint32_t i = 0; i < steps; ++i) {
        const auto vec = _mm512_stream_load_si512(simd_data + i * 4);
        const auto vec2 = _mm512_stream_load_si512(simd_data + i * 4 + 1);
        const auto vec3 = _mm512_stream_load_si512(simd_data + i * 4 + 2);
        const auto vec4 = _mm512_stream_load_si512(simd_data + i * 4 + 3);
        const auto masked = _mm512_and_epi32(vec, simd_mask);
        const auto masked2 = _mm512_and_epi32(vec2, simd_mask);
        const auto masked3 = _mm512_and_epi32(vec3, simd_mask);
        const auto masked4 = _mm512_and_epi32(vec4, simd_mask);
        const auto shifted = _mm512_srav_epi32(masked, simd_shift);
        const auto shifted2 = _mm512_srav_epi32(masked2, simd_shift);
        const auto shifted3 = _mm512_srav_epi32(masked3, simd_shift);
        const auto shifted4 = _mm512_srav_epi32(masked4, simd_shift);
        const auto compressed = _mm512_cvtepi64_epi32(shifted);
        const auto compressed2 = _mm512_cvtepi64_epi32(shifted2);
        const auto compressed3 = _mm512_cvtepi64_epi32(shifted3);
        const auto compressed4 = _mm512_cvtepi64_epi32(shifted4);
        _mm256_store_epi32(buffer, compressed);
        _mm256_store_epi32(buffer + 32, compressed2);
        _mm256_store_epi32(buffer + 64, compressed3);
        _mm256_store_epi32(buffer + 96, compressed4);

#pragma GCC unroll 32
        for (uint32_t j = 0; j < 32; ++j) {
            ++hist[int_buffer[j]];
        }
    }
    // if data_size % 32 != 0 then handle remaining rows
    for (uint32_t i = steps * 32; i < data_size; ++i) {
        const uint32_t idx = (data[i].key & mask) >> shift;
        ++hist[idx];
    }
}
