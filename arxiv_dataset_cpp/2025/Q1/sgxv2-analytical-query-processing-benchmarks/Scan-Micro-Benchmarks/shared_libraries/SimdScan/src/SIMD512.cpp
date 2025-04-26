#include <algorithm>
#include <vector>
#include "SIMD512.hpp"

namespace SIMD512 {

    size_t count(pred_t predicate_low, pred_t predicate_high, const __m512i *input_compressed, size_t input_size) {
        size_t matches = 0;
        // solution: start
        __m512i low = _mm512_set1_epi8(predicate_low);
        __m512i high = _mm512_set1_epi8(predicate_high);
        if (predicate_low == predicate_high) {
            for (size_t i = 0; i < input_size / 64; ++i) {
                __mmask64 mask = _mm512_cmpeq_epu8_mask(input_compressed[i], low);
                matches += __builtin_popcountll(mask);
            }
        } else {
            for (size_t i = 0; i < input_size / 64; ++i) {
                __mmask64 low_mask = _mm512_cmpge_epu8_mask(input_compressed[i], low);
                __mmask64 high_mask = _mm512_cmple_epu8_mask(input_compressed[i], high);

                // MemoryDump(&input_compressed[i], sizeof(__m512i)).dec<uint8_t>(8);
                //MemoryDump(&low_mask, sizeof(__mmask64)).hex<uint64_t>(8);
                //MemoryDump(&high_mask, sizeof(__mmask64)).hex<uint64_t>(8);
                //std::cout << "popcnt: " << __builtin_popcountll(low_mask & high_mask) << "\n";

                matches += __builtin_popcountll(low_mask & high_mask);
            }
        }
        // solution: end
        return matches;
    }

    size_t sum(pred_t predicate_low, pred_t predicate_high, const __m512i *input_compressed, size_t input_size) {
        size_t sum = 0;
        // solution: start
        __m512i low = _mm512_set1_epi8(predicate_low);
        __m512i high = _mm512_set1_epi8(predicate_high);

        if (predicate_low == predicate_high) {
            for (size_t i = 0; i < input_size / 64; ++i) {
                __mmask64 mask = _mm512_cmpeq_epu8_mask(input_compressed[i], low);
                if (mask == 0) {
                    continue;
                }
                // MemoryDump(&mask, sizeof(__mmask64)).bit<uint64_t>(8);
                for (int j = 0; j < 4; ++j) {
                    __mmask16 part_mask = (mask >> j * 16) & 0xffff;
                    // MemoryDump(&part_mask, sizeof(__mmask16)).bit<uint16_t>(8);
                    if (part_mask == 0) {
                        continue;
                    }
                    __m128i *as_128 = (__m128i *) &input_compressed[i];
                    // MemoryDump(as_128, sizeof(__m128i)).dec<uint8_t>(8);
                    __m512i decompressed = _mm512_cvtepu8_epi32(*(as_128 + j));
                    // MemoryDump(&decompressed, sizeof(__m512i)).dec<uint32_t>(8);
                    // std::cout << "Sum: " << _mm512_mask_reduce_add_epi32(part_mask, decompressed) << "\n";
                    sum += _mm512_mask_reduce_add_epi32(part_mask, decompressed);
                }
            }
        } else {
            for (size_t i = 0; i < input_size / 64; ++i) {
                __mmask64 mask_low = _mm512_cmpge_epu8_mask(input_compressed[i], low);
                __mmask64 mask_high = _mm512_cmple_epu8_mask(input_compressed[i], high);
                __mmask64 mask = mask_high & mask_low;
                // MemoryDump(&input_compressed[i], sizeof(__m512i)).dec<uint8_t>(8);
                // MemoryDump(&mask, sizeof(__mmask64)).bit<uint64_t>(8);
                if (mask == 0) {
                    continue;
                }
                for (int j = 0; j < 4; ++j) {
                    __mmask16 part_mask = (mask >> j * 16) & 0xffff;
                    // MemoryDump(&part_mask, sizeof(__mmask16)).bit<uint16_t>(8);
                    if (part_mask == 0) {
                        continue;
                    }
                    __m128i *as_128 = (__m128i *) &input_compressed[i];
                    // MemoryDump(as_128, sizeof(__m128i)).dec<uint8_t>(8);
                    __m512i decompressed = _mm512_cvtepu8_epi32(*(as_128 + j));
                    // MemoryDump(&decompressed, sizeof(__m512i)).dec<uint32_t>(8);
                    // std::cout << "Sum: " << _mm512_mask_reduce_add_epi32(part_mask, decompressed) << "\n";
                    sum += _mm512_mask_reduce_add_epi32(part_mask, decompressed);
                }
            }
        }
        // solution: end
        return sum;
    }


    size_t scan(pred_t predicate_low, pred_t predicate_high, const __m512i *input_compressed, size_t input_size,
                uint32_t *output_buffer) {
        size_t matches = 0;
        // solution: start
        __m512i low = _mm512_set1_epi8(predicate_low);
        __m512i high = _mm512_set1_epi8(predicate_high);

        if (predicate_low == predicate_high) {
            for (size_t i = 0; i < input_size / 64; ++i) {
                __mmask64 mask = _mm512_cmpeq_epu8_mask(input_compressed[i], low);
                if (mask == 0) {
                    continue;
                }
                // MemoryDump(&mask, sizeof(__mmask64)).bit<uint64_t>(8);
                for (int j = 0; j < 4; ++j) {
                    __mmask16 part_mask = (mask >> j * 16) & 0xffff;
                    // MemoryDump(&part_mask, sizeof(__mmask16)).bit<uint16_t>(8);
                    if (part_mask == 0) {
                        continue;
                    }
                    __m128i *as_128 = (__m128i *) &input_compressed[i];
                    // MemoryDump(as_128, sizeof(__m128i)).dec<uint8_t>(8);
                    __m512i decompressed = _mm512_cvtepu8_epi32(*(as_128 + j));
                    // MemoryDump(&decompressed, sizeof(__m512i)).dec<uint32_t>(8);
                    _mm512_mask_compressstoreu_epi32(static_cast<void *>(output_buffer + matches), part_mask,
                                                     decompressed);
                    // std::cout << "Matches: " << std::popcount(part_mask) << "\n";
                    matches += __builtin_popcount(part_mask);
                }
            }
        } else {
            for (size_t i = 0; i < input_size / 64; ++i) {
                __mmask64 mask_low = _mm512_cmpge_epu8_mask(input_compressed[i], low);
                __mmask64 mask_high = _mm512_cmple_epu8_mask(input_compressed[i], high);
                __mmask64 mask = mask_high & mask_low;
                // MemoryDump(&input_compressed[i], sizeof(__m512i)).dec<uint8_t>(8);
                //MemoryDump(&mask, sizeof(__mmask64)).bit<uint64_t>(8);
                if (mask == 0) {
                    continue;
                }
                for (int j = 0; j < 4; ++j) {
                    __mmask16 part_mask = (mask >> j * 16) & 0xffff;
                    //MemoryDump(&part_mask, sizeof(__mmask16)).bit<uint16_t>(8);
                    if (part_mask == 0) {
                        continue;
                    }
                    __m128i *as_128 = (__m128i *) &input_compressed[i];
                    // MemoryDump(as_128, sizeof(__m128i)).dec<uint8_t>(8);
                    __m512i decompressed = _mm512_cvtepu8_epi32(*(as_128 + j));
                    // MemoryDump(&decompressed, sizeof(__m512i)).dec<uint32_t>(8);
                    _mm512_mask_compressstoreu_epi32(static_cast<void *>(output_buffer + matches), part_mask,
                                                     decompressed);
                    // std::cout << "Matches: " << std::popcount(part_mask) << "\n";
                    matches += __builtin_popcount(part_mask);
                }
            }
        }
        // solution: end
        return matches;
    }

    void explicit_index_scan(pred_t predicate_low, pred_t predicate_high, const __m512i *index_compressed,
                             const __m512i *input_compressed,
                             size_t input_size, size_t *output_buffer) {
        size_t matches = 0;
        // solution: start
        __m512i low = _mm512_set1_epi8(predicate_low);
        __m512i high = _mm512_set1_epi8(predicate_high);

        if (predicate_low == predicate_high) {
            for (size_t i = 0; i < input_size / 64; ++i) {
                __mmask64 mask = _mm512_cmpeq_epu8_mask(input_compressed[i], low);
                if (mask == 0) {
                    continue;
                }
                // MemoryDump(&mask, sizeof(__mmask64)).bit<uint64_t>(8);
                for (int j = 0; j < 8; ++j) {
                    __mmask8 part_mask = (mask >> j * 8) & 0xff;
                    // MemoryDump(&part_mask, sizeof(__mmask16)).bit<uint16_t>(8);
                    if (part_mask == 0) {
                        continue;
                    }
                    __m512i decompressed = index_compressed[i + j];
                    // MemoryDump(&decompressed, sizeof(__m512i)).dec<uint32_t>(8);
                    _mm512_mask_compressstoreu_epi64(static_cast<void *>(output_buffer + matches), part_mask,
                                                     decompressed);
                    // std::cout << "Matches: " << std::popcount(part_mask) << "\n";
                    matches += __builtin_popcount(part_mask);
                }
            }
        } else {
            for (size_t i = 0; i < input_size / 64; ++i) {
                __mmask64 mask_low = _mm512_cmpge_epu8_mask(input_compressed[i], low);
                __mmask64 mask_high = _mm512_cmple_epu8_mask(input_compressed[i], high);
                __mmask64 mask = mask_high & mask_low;
                // MemoryDump(&input_compressed[i], sizeof(__m512i)).dec<uint8_t>(8);
                //MemoryDump(&mask, sizeof(__mmask64)).bit<uint64_t>(8);
                if (mask == 0) {
                    continue;
                }
                for (int j = 0; j < 8; ++j) {
                    __mmask8 part_mask = (mask >> j * 8) & 0xff;
                    //MemoryDump(&part_mask, sizeof(__mmask8)).bit<uint8_t>(8);
                    if (part_mask == 0) {
                        continue;
                    }
                    __m512i decompressed = index_compressed[i + j];
                    // MemoryDump(&decompressed, sizeof(__m512i)).dec<uint32_t>(8);
                    _mm512_mask_compressstoreu_epi64(static_cast<void *>(output_buffer + matches), part_mask,
                                                     decompressed);
                    // std::cout << "Matches: " << std::popcount(part_mask) << "\n";
                    matches += __builtin_popcount(part_mask);
                }
            }
        }
        // solution: end
        // return matches;
    }

    void
    bitvector_scan(pred_t predicate_low, pred_t predicate_high, const __m512i *input_compressed,
                   size_t input_size, __mmask64 *output_buffer) {
        __m512i low = _mm512_set1_epi8(predicate_low);
        __m512i high = _mm512_set1_epi8(predicate_high);

        for (size_t i = 0; i < input_size / 64; ++i) {
            __mmask64 mask_low = _mm512_cmpge_epu8_mask(input_compressed[i], low);
            __mmask64 mask_high = _mm512_cmple_epu8_mask(input_compressed[i], high);
            __mmask64 mask = mask_high & mask_low;
            _store_mask64(output_buffer + i, mask);
        }
    }


    void implicit_index_scan(pred_t predicate_low, pred_t predicate_high, const __m512i *input_compressed,
                             size_t input_size, size_t *output_buffer) {
        __m512i low = _mm512_set1_epi8(predicate_low);
        __m512i high = _mm512_set1_epi8(predicate_high);

        auto off = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
        auto small_step = _mm512_set1_epi64(8);
        auto big_step = _mm512_set1_epi64(64);

        for (size_t i = 0; i < input_size / 64; ++i) {
            __mmask64 mask_low = _mm512_cmpge_epu8_mask(input_compressed[i], low);
            __mmask64 mask_high = _mm512_cmple_epu8_mask(input_compressed[i], high);
            __mmask64 mask = mask_high & mask_low;
            if (mask == 0) {
                off = _mm512_add_epi64(off, big_step);
                continue;
            }
            for (int j = 0; j < 8; ++j) {
                __mmask8 part_mask = (mask >> j * 8) & 0xff;
                _mm512_mask_compressstoreu_epi64(static_cast<void *>(output_buffer), part_mask, off);
                output_buffer += __builtin_popcount(part_mask);
                off = _mm512_add_epi64(off, small_step);
            }
        }
    }

    void implicit_index_scan_self_alloc(pred_t predicate_low, pred_t predicate_high, const __m512i *input_compressed,
                                        size_t input_size, CacheAlignedVector<size_t> &output_buffer, bool cut) {
        constexpr size_t BLOCK_SIZE = 64;

        __m512i low = _mm512_set1_epi8(predicate_low);
        __m512i high = _mm512_set1_epi8(predicate_high);

        auto off = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
        const auto small_step = _mm512_set1_epi64(8);
        const auto big_step = _mm512_set1_epi64(64);

        auto out_index = 0;

        for (size_t i = 0; i < input_size / 64; ++i) {
            if (out_index + BLOCK_SIZE > output_buffer.size()) {
                output_buffer.resize((BLOCK_SIZE + output_buffer.size()) * 3);
            }

            __mmask64 mask_low = _mm512_cmpge_epu8_mask(input_compressed[i], low);
            __mmask64 mask_high = _mm512_cmple_epu8_mask(input_compressed[i], high);
            __mmask64 mask = mask_high & mask_low;
            if (mask == 0) {
                off = _mm512_add_epi64(off, big_step);
                continue;
            }
            for (int j = 0; j < 8; ++j) {
                __mmask8 part_mask = (mask >> j * 8) & 0xff;
                if (part_mask != 0) {
                    _mm512_mask_compressstoreu_epi64(output_buffer.data() + out_index, part_mask, off);
                    out_index += __builtin_popcount(part_mask);
                }
                off = _mm512_add_epi64(off, small_step);
            }
        }

        if (cut) output_buffer.resize(out_index);
    }

    void dict_scan_8bit_64bit(int64_t predicate_low, int64_t predicate_high, const int64_t *dict,
                              const __m512i *input_compressed, size_t input_size, CacheAlignedVector<int64_t> &output_buffer,
                              bool cut) {
        constexpr size_t BLOCK_SIZE = 64;
        constexpr size_t dict_size = 1UL << 8;
        constexpr __mmask64 decompression_mask = 0x0101010101010101ULL;
        const static auto decompression_index_step = _mm512_set1_epi64(8);

        auto low_pointer = std::find_if(dict, dict + dict_size,
                                        [predicate_low](int64_t val) { return val >= predicate_low; });
        auto high_pointer =
                std::find_if(low_pointer, dict + dict_size,
                             [predicate_high](int64_t val) { return val > predicate_high; }) -
                1;

        auto low = _mm512_set1_epi8(static_cast<uint8_t>(low_pointer - dict));
        auto high = _mm512_set1_epi8((uint8_t) ((high_pointer - dict) & 0xff));

        auto out_index = 0;

        for (size_t i = 0; i < input_size / BLOCK_SIZE; ++i) {
            if (out_index + BLOCK_SIZE > output_buffer.size()) {
                output_buffer.resize((BLOCK_SIZE + output_buffer.size()) * 3);
            }
            __mmask64 mask_low = _mm512_cmpge_epu8_mask(input_compressed[i], low);
            __mmask64 mask_high = _mm512_cmple_epu8_mask(input_compressed[i], high);
            __mmask64 mask = mask_high & mask_low;
            if (mask == 0) {
                continue;
            }

            auto popcount = _mm_popcnt_u64(mask);
            auto compressed = _mm512_mask_compress_epi8(input_compressed[i], mask, input_compressed[i]);

            auto decompression_indexes = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);

            for (size_t j = 0; j * 8 < popcount; ++j) {
                //auto scattered_dict_indexes = _mm512_mask_permutexvar_epi8(zero, decompression_mask,
                //                                                           decompression_indexes[j], compressed);
                auto scattered_dict_indexes = _mm512_maskz_permutexvar_epi8(decompression_mask, decompression_indexes,
                                                                            compressed);
                auto values = _mm512_i64gather_epi64(scattered_dict_indexes, reinterpret_cast<const void *>(dict), 8);
                _mm512_storeu_epi64(output_buffer.data() + out_index, values);
                out_index += std::min(popcount - j * 8, 8ULL);
                decompression_indexes = _mm512_add_epi64(decompression_indexes, decompression_index_step);
            }
        }

        if (cut) output_buffer.resize(out_index);
    }

    void dict_scan_8bit_64bit_scalar_gather_scatter(int64_t predicate_low, int64_t predicate_high, const int64_t *dict,
                                                    const __m512i *input_compressed, size_t input_size,
                                                    CacheAlignedVector<int64_t> &output_buffer, bool cut) {
        constexpr size_t BLOCK_SIZE = 64;
        constexpr size_t dict_size = 1UL << 8;
        alignas(64) uint8_t buffer[BLOCK_SIZE] = {0};

        auto low_pointer = std::find_if(dict, dict + dict_size,
                                        [predicate_low](int64_t val) { return val >= predicate_low; });
        auto high_pointer =
                std::find_if(low_pointer, dict + dict_size,
                             [predicate_high](int64_t val) { return val > predicate_high; }) -
                1;

        auto low = _mm512_set1_epi8(static_cast<uint8_t>(low_pointer - dict));
        auto high = _mm512_set1_epi8((uint8_t) ((high_pointer - dict) & 0xff));

        auto out_index = 0;

        for (size_t i = 0; i < input_size / BLOCK_SIZE; ++i) {
            if (out_index + BLOCK_SIZE > output_buffer.size()) {
                output_buffer.resize((BLOCK_SIZE + output_buffer.size()) * 3);
            }
            __mmask64 mask_low = _mm512_cmpge_epu8_mask(input_compressed[i], low);
            __mmask64 mask_high = _mm512_cmple_epu8_mask(input_compressed[i], high);
            __mmask64 mask = mask_high & mask_low;
            if (mask == 0) {
                continue;
            }

            auto popcount = _mm_popcnt_u64(mask);
            auto compressed = _mm512_mask_compress_epi8(input_compressed[i], mask, input_compressed[i]);
            _mm512_storeu_epi8(buffer, compressed);

            for (size_t buffer_index = 0; buffer_index < popcount; ++buffer_index) {
                uint8_t dict_index = buffer[buffer_index];
                output_buffer[out_index] = dict[dict_index];
                ++out_index;
            }
        }

        if (cut) output_buffer.resize(out_index);
    }

    void dict_scan_8bit_64bit_scalar_unroll(int64_t predicate_low, int64_t predicate_high, const int64_t *dict,
                                            const __m512i *input_compressed, size_t input_size,
                                            CacheAlignedVector<int64_t> &output_buffer, bool cut) {
        constexpr size_t BLOCK_SIZE = 64;
        constexpr size_t dict_size = 1UL << 8;
        alignas(64) uint8_t buffer[BLOCK_SIZE] = {0};

        auto low_pointer = std::find_if(dict, dict + dict_size,
                                        [predicate_low](int64_t val) { return val >= predicate_low; });
        auto high_pointer =
                std::find_if(low_pointer, dict + dict_size,
                             [predicate_high](int64_t val) { return val > predicate_high; }) -
                1;

        auto low = _mm512_set1_epi8(static_cast<uint8_t>(low_pointer - dict));
        auto high = _mm512_set1_epi8((uint8_t) ((high_pointer - dict) & 0xff));

        auto out_index = 0;

        for (size_t i = 0; i < input_size / BLOCK_SIZE; ++i) {
            if (out_index + BLOCK_SIZE > output_buffer.size()) {
                output_buffer.resize((BLOCK_SIZE + output_buffer.size()) * 3);
            }
            __mmask64 mask_low = _mm512_cmpge_epu8_mask(input_compressed[i], low);
            __mmask64 mask_high = _mm512_cmple_epu8_mask(input_compressed[i], high);
            __mmask64 mask = mask_high & mask_low;
            if (mask == 0) {
                continue;
            }

            auto popcount = _mm_popcnt_u64(mask);
            auto compressed = _mm512_mask_compress_epi8(input_compressed[i], mask, input_compressed[i]);
            _mm512_store_epi64(buffer, compressed);

            size_t buffer_index = 0;
            for (; buffer_index + 8 <= popcount; buffer_index += 8) {
                uint8_t idx0 = buffer[buffer_index + 0];
                uint8_t idx1 = buffer[buffer_index + 1];
                uint8_t idx2 = buffer[buffer_index + 2];
                uint8_t idx3 = buffer[buffer_index + 3];
                uint8_t idx4 = buffer[buffer_index + 4];
                uint8_t idx5 = buffer[buffer_index + 5];
                uint8_t idx6 = buffer[buffer_index + 6];
                uint8_t idx7 = buffer[buffer_index + 7];
                output_buffer[out_index + 0] = dict[idx0];
                output_buffer[out_index + 1] = dict[idx1];
                output_buffer[out_index + 2] = dict[idx2];
                output_buffer[out_index + 3] = dict[idx3];
                output_buffer[out_index + 4] = dict[idx4];
                output_buffer[out_index + 5] = dict[idx5];
                output_buffer[out_index + 6] = dict[idx6];
                output_buffer[out_index + 7] = dict[idx7];
                out_index += 8;
            }
            for (; buffer_index < popcount; ++buffer_index) {
                uint8_t dict_index = buffer[buffer_index];
                output_buffer[out_index] = dict[dict_index];
                ++out_index;
            }
        }

        if (cut) output_buffer.resize(out_index);
    }

    void dict_scan_8bit_64bit_opt_write(int64_t predicate_low, int64_t predicate_high, const int64_t *dict,
                                                    const __m512i *input_compressed, size_t input_size,
                                                    CacheAlignedVector<int64_t> &output_buffer, bool cut) {
        constexpr size_t BLOCK_SIZE = 64;
        constexpr size_t dict_size = 1UL << 8;
        alignas(64) uint8_t buffer[BLOCK_SIZE] = {0};
        alignas(64) int64_t output_combine_buffer[8] = {0};
        auto output_combine_counter = 0;

        auto low_pointer = std::find_if(dict, dict + dict_size,
                                        [predicate_low](int64_t val) { return val >= predicate_low; });
        auto high_pointer =
                std::find_if(low_pointer, dict + dict_size,
                             [predicate_high](int64_t val) { return val > predicate_high; }) -
                1;

        auto low = _mm512_set1_epi8(static_cast<uint8_t>(low_pointer - dict));
        auto high = _mm512_set1_epi8((uint8_t) ((high_pointer - dict) & 0xff));

        auto out_index = 0;

        for (size_t i = 0; i < input_size / BLOCK_SIZE; ++i) {
            if (out_index + BLOCK_SIZE > output_buffer.size()) {
                output_buffer.resize((BLOCK_SIZE + output_buffer.size()) * 3);
            }
            __mmask64 mask_low = _mm512_cmpge_epu8_mask(input_compressed[i], low);
            __mmask64 mask_high = _mm512_cmple_epu8_mask(input_compressed[i], high);
            __mmask64 mask = mask_high & mask_low;
            if (mask == 0) {
                continue;
            }

            auto popcount = _mm_popcnt_u64(mask);
            auto compressed = _mm512_mask_compress_epi8(input_compressed[i], mask, input_compressed[i]);
            _mm512_storeu_epi8(buffer, compressed);

            for (size_t buffer_index = 0; buffer_index < popcount; ++buffer_index) {
                uint8_t dict_index = buffer[buffer_index];
                output_combine_buffer[output_combine_counter] = dict[dict_index];
                output_combine_counter++;
                if (output_combine_counter == 8) {
                    _mm512_stream_si512(reinterpret_cast<__m512i *>(output_buffer.data() + out_index),
                                        _mm512_stream_load_si512(output_combine_buffer));
                    out_index += 8;
                    output_combine_counter = 0;
                }
            }
        }
        for (uint32_t i = 0; i < output_combine_counter; i++) {
            output_buffer[out_index] = output_combine_buffer[output_combine_counter];
            out_index++;
        }
        _mm_mfence();

        if (cut) output_buffer.resize(out_index);
    }

    void dict_scan_8bit_64bit_scalar(int64_t predicate_low, int64_t predicate_high, const int64_t *dict,
                                     const uint8_t *input_compressed, size_t input_size,
                                     CacheAlignedVector<int64_t> &output_buffer) {
        constexpr size_t dict_size = 1UL << 8;

        auto low_pointer = std::find_if(dict, dict + dict_size,
                                        [predicate_low](int64_t val) { return val >= predicate_low; });
        auto high_pointer =
                std::find_if(low_pointer, dict + dict_size,
                             [predicate_high](int64_t val) { return val > predicate_high; }) -
                1;

        auto low_dict_index = static_cast<uint8_t>(low_pointer - dict);
        auto high_dict_index = static_cast<uint8_t>((high_pointer - dict) & 0xff);

        size_t out_index = 0;

        for (size_t i = 0; i < input_size; ++i) {
            auto dict_index = input_compressed[i];
            if (dict_index >= low_dict_index && dict_index <= high_dict_index) {
                output_buffer[out_index] = dict[dict_index];
                ++out_index;
            }
        }
    }

    void dict_scan_16bit_64bit(int64_t predicate_low, int64_t predicate_high, const int64_t *dict,
                               const __m512i *input_compressed, size_t input_size,
                               CacheAlignedVector<int64_t> &output_buffer) {
        constexpr size_t BLOCK_SIZE = 32; // One register can hold 32 16 bit values
        constexpr size_t dict_size = 1UL << 16;
        constexpr __mmask32 decompression_mask = 0x11111111U;
        const static auto decompression_index_step = _mm512_set1_epi64(8);

        auto low_pointer = std::find_if(dict, dict + dict_size,
                                        [predicate_low](int64_t val) { return val >= predicate_low; });
        auto high_pointer =
                std::find_if(low_pointer, dict + dict_size,
                             [predicate_high](int64_t val) { return val > predicate_high; }) -
                1;

        auto low = _mm512_set1_epi16(static_cast<uint16_t>(low_pointer - dict));
        auto high = _mm512_set1_epi16(static_cast<uint16_t>(high_pointer - dict));

        auto out_index = 0;

        for (size_t i = 0; i < input_size / BLOCK_SIZE; ++i) {
            if (out_index + BLOCK_SIZE >= output_buffer.size()) {
                output_buffer.resize((BLOCK_SIZE + output_buffer.size()) * 3);
            }

            __mmask32 mask_low = _mm512_cmpge_epu16_mask(input_compressed[i], low);
            __mmask32 mask_high = _mm512_cmple_epu16_mask(input_compressed[i], high);
            __mmask32 mask = mask_high & mask_low;
            if (mask == 0) {
                continue;
            }

            auto popcount = _mm_popcnt_u32(mask);
            auto compressed = _mm512_mask_compress_epi16(input_compressed[i], mask, input_compressed[i]);

            auto decompression_indexes = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);

            for (size_t j = 0; j * 8 < popcount; ++j) {
                auto scattered_dict_indexes = _mm512_maskz_permutexvar_epi16(decompression_mask, decompression_indexes,
                                                                             compressed);
                auto values = _mm512_i64gather_epi64(scattered_dict_indexes, reinterpret_cast<const void *>(dict), 8);
                _mm512_storeu_epi64(output_buffer.data() + out_index, values);
                out_index += std::min(popcount - j * 8, 8UL);
                decompression_indexes = _mm512_add_epi64(decompression_indexes, decompression_index_step);
            }
        }

        output_buffer.resize(out_index);
    }

    void dict_scan_32bit_64bit(int64_t predicate_low, int64_t predicate_high, const int64_t *dict, size_t dict_size,
                               const __m512i *input_compressed, size_t input_size,
                               CacheAlignedVector<int64_t> &output_buffer) {
        constexpr size_t BLOCK_SIZE = 16; // One register can hold 16 32 bit values
        constexpr __mmask16 decompression_mask = 0x5555U;
        const static auto decompression_index_step = _mm512_set1_epi64(8);

        auto low_pointer = std::find_if(dict, dict + dict_size,
                                        [predicate_low](int64_t val) { return val >= predicate_low; });
        auto high_pointer =
                std::find_if(low_pointer, dict + dict_size,
                             [predicate_high](int64_t val) { return val > predicate_high; }) -
                1;

        auto low = _mm512_set1_epi32(static_cast<uint16_t>(low_pointer - dict));
        auto high = _mm512_set1_epi32(static_cast<uint16_t>(high_pointer - dict));

        auto out_index = 0;

        for (size_t i = 0; i < input_size / BLOCK_SIZE; ++i) {
            if (out_index + BLOCK_SIZE >= output_buffer.size()) {
                output_buffer.resize((BLOCK_SIZE + output_buffer.size()) * 3);
            }

            __mmask16 mask_low = _mm512_cmpge_epu32_mask(input_compressed[i], low);
            __mmask16 mask_high = _mm512_cmple_epu32_mask(input_compressed[i], high);
            __mmask16 mask = mask_high & mask_low;
            if (mask == 0) {
                continue;
            }

            auto popcount = _mm_popcnt_u32(mask);
            auto compressed = _mm512_mask_compress_epi32(input_compressed[i], mask, input_compressed[i]);

            auto decompression_indexes = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);

            for (size_t j = 0; j * 8 < popcount; ++j) {
                auto scattered_dict_indexes = _mm512_maskz_permutexvar_epi32(decompression_mask, decompression_indexes,
                                                                             compressed);
                auto values = _mm512_i64gather_epi64(scattered_dict_indexes, reinterpret_cast<const void *>(dict), 8);
                _mm512_storeu_epi64(output_buffer.data() + out_index, values);
                out_index += std::min(popcount - j * 8, 8UL);
                decompression_indexes = _mm512_add_epi64(decompression_indexes, decompression_index_step);
            }
        }

        output_buffer.resize(out_index);

    }

    void dict_scan_32bit_64bit_scalar_gather_scatter(int64_t predicate_low, int64_t predicate_high, const int64_t *dict,
                                                     size_t dict_size, const __m512i *input_compressed,
                                                     size_t input_size, CacheAlignedVector<int64_t> &output_buffer, bool cut) {
        constexpr size_t BLOCK_SIZE = 16; // One register can hold 16 32 bit values
        alignas(64) uint32_t buffer[BLOCK_SIZE] = {0};

        auto low_pointer = std::find_if(dict, dict + dict_size,
                                        [predicate_low](int64_t val) { return val >= predicate_low; });
        auto high_pointer =
                std::find_if(low_pointer, dict + dict_size,
                             [predicate_high](int64_t val) { return val > predicate_high; }) -
                1;

        auto low = _mm512_set1_epi32(static_cast<uint16_t>(low_pointer - dict));
        auto high = _mm512_set1_epi32(static_cast<uint16_t>(high_pointer - dict));

        auto out_index = 0;

        for (size_t i = 0; i < input_size / BLOCK_SIZE; ++i) {
            if (out_index + BLOCK_SIZE >= output_buffer.size()) {
                output_buffer.resize((BLOCK_SIZE + output_buffer.size()) * 3);
            }

            __mmask16 mask_low = _mm512_cmpge_epu32_mask(input_compressed[i], low);
            __mmask16 mask_high = _mm512_cmple_epu32_mask(input_compressed[i], high);
            __mmask16 mask = mask_high & mask_low;
            if (mask == 0) {
                continue;
            }

            auto popcount = _mm_popcnt_u64(mask);
            auto compressed = _mm512_mask_compress_epi8(input_compressed[i], mask, input_compressed[i]);
            _mm512_store_epi32(buffer, compressed);

            for (size_t buffer_index = 0; buffer_index < popcount; ++buffer_index) {
                uint32_t dict_index = buffer[buffer_index];
                output_buffer[out_index] = dict[dict_index];
                ++out_index;
            }
        }

        if (cut) output_buffer.resize(out_index);

    }

}