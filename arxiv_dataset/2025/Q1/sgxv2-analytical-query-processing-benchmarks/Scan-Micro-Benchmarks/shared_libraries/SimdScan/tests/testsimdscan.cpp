#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>
#include <SIMD512.hpp>
#include "Allocator.hpp"
#include "rdtscpWrapper.h"

TEST_CASE("Dict 8_64 scan test 1", "[main]") {
    size_t input_size = 1UL << 20;
    auto data_array = allocate_data_array_aligned<uint8_t, 64>(input_size);
    auto dict = allocate_data_array_aligned<int64_t, 64>(256);

    auto predicate_low = 0LL;
    auto predicate_high = 100LL;

    CacheAlignedVector<int64_t> result_vector{};

    SIMD512::dict_scan_8bit_64bit(predicate_low,
                                  predicate_high,
                                  dict.get(),
                                  reinterpret_cast<__m512i *>(data_array.get()),
                                  input_size,
                                  result_vector,
                                  true);

    REQUIRE(result_vector.size() == input_size / 256 * 101);

}

TEST_CASE("Dict 8_64 scan test 2", "[main]") {
    size_t input_size = 1UL << 20;
    auto data_array = allocate_data_array_aligned<uint8_t, 64>(input_size);
    auto dict = allocate_data_array_aligned<int64_t, 64>(256);

    auto predicate_low = 1LL;
    auto predicate_high = 100LL;

    CacheAlignedVector<int64_t> result_vector{};

    SIMD512::dict_scan_8bit_64bit(predicate_low,
                                  predicate_high,
                                  dict.get(),
                                  reinterpret_cast<__m512i *>(data_array.get()),
                                  input_size,
                                  result_vector,
                                  true);

    REQUIRE(result_vector.size() == input_size / 256 * 100);

    for (auto i = 0; i < 100; ++i) {
        REQUIRE(result_vector[i] == i + 1);
    }
    REQUIRE(result_vector[100] == 1UL);
}

TEST_CASE("Dict 8_64 scan test 3", "[main]") {
    size_t input_size = 1UL << 20;
    auto data_array = allocate_data_array_aligned<uint8_t, 64>(input_size);
    auto dict = allocate_data_array_aligned<int64_t, 64>(256);

    std::transform(dict.get(), dict.get() + 256, dict.get(), [](auto in) { return in * 2; });

    auto predicate_low = 0LL;
    auto predicate_high = 98LL;

    CacheAlignedVector<int64_t> result_vector{};

    SIMD512::dict_scan_8bit_64bit(predicate_low,
                                  predicate_high,
                                  dict.get(),
                                  reinterpret_cast<__m512i *>(data_array.get()),
                                  input_size,
                                  result_vector,
                                  true);

    REQUIRE(result_vector.size() == input_size / 256 * 50);

    for (auto i = 0; i < 50; ++i) {
        REQUIRE(result_vector[i] == i * 2);
    }

    REQUIRE(result_vector[99] == 98UL);
    REQUIRE(result_vector[100] == 0UL);
}

TEST_CASE("Dict 8_64 scan test 4", "[main]") {
    size_t input_size = 1UL << 20;

    auto gen = [](uint64_t in) { return static_cast<uint8_t>(in & 3); };

    auto data_array = allocate_data_array_aligned<uint8_t, 64>(input_size, gen);
    auto dict = allocate_data_array_aligned<int64_t, 64>(256);

    auto predicate_low = 0LL;
    auto predicate_high = 2LL;

    CacheAlignedVector<int64_t> result_vector{};

    SIMD512::dict_scan_8bit_64bit(predicate_low,
                                  predicate_high,
                                  dict.get(),
                                  reinterpret_cast<__m512i *>(data_array.get()),
                                  input_size,
                                  result_vector,
                                  true);

    REQUIRE(result_vector.size() == input_size / 4 * 3);

    REQUIRE(result_vector[0] == 0UL);
    REQUIRE(result_vector[1] == 1UL);
    REQUIRE(result_vector[2] == 2UL);
    REQUIRE(result_vector[3] == 0UL);
}

TEST_CASE("Dict 8_64 scan test 5", "[main]") {
    size_t input_size = 1UL << 20;

    auto data_array = allocate_data_array_aligned<uint8_t, 64>(input_size);
    auto dict = allocate_data_array_aligned<int64_t, 64>(256);

    auto predicate_low = 100LL;
    auto predicate_high = 199LL;

    CacheAlignedVector<int64_t> result_vector{};

    SIMD512::dict_scan_8bit_64bit(predicate_low,
                                  predicate_high,
                                  dict.get(),
                                  reinterpret_cast<__m512i *>(data_array.get()),
                                  input_size,
                                  result_vector,
                                  true);

    REQUIRE(result_vector.size() == input_size / 256 * 100);

    REQUIRE(result_vector[0] == 100UL);
    REQUIRE(result_vector[99] == 199UL);
}

TEST_CASE("Dict 8_64 scan test 6", "[main]") {
    size_t input_size = 1UL << 20;

    auto gen = [](uint64_t in) { return in - 128; };

    auto data_array = allocate_data_array_aligned<uint8_t, 64>(input_size);
    auto dict = allocate_data_array_aligned<int64_t, 64>(256, gen);

    auto predicate_low = -10LL;
    auto predicate_high = 0LL;

    CacheAlignedVector<int64_t> result_vector{};

    SIMD512::dict_scan_8bit_64bit(predicate_low,
                                  predicate_high,
                                  dict.get(),
                                  reinterpret_cast<__m512i *>(data_array.get()),
                                  input_size,
                                  result_vector,
                                  true);

    REQUIRE(result_vector.size() == input_size / 256 * 11);

    REQUIRE(result_vector[0] == -10UL);
    REQUIRE(result_vector[10] == 0UL);
}

TEST_CASE("Dict 16_64 scan test 1", "[main]") {
    size_t input_size = 1UL << 20;

    auto data_array = allocate_data_array_aligned<uint16_t, 64>(input_size);
    auto dict = allocate_data_array_aligned<int64_t, 64>(1 << 16);

    auto predicate_low = 0LL;
    auto predicate_high = 299LL;

    CacheAlignedVector<int64_t> result_vector{};

    SIMD512::dict_scan_16bit_64bit(predicate_low,
                                   predicate_high,
                                   dict.get(),
                                   reinterpret_cast<__m512i *>(data_array.get()),
                                   input_size,
                                   result_vector);

    REQUIRE(result_vector.size() == input_size / (1 << 16) * 300);

    for (auto i = 0; i < result_vector.size(); ++i) {
        REQUIRE(result_vector[i] == i % 300);
    }
}

TEST_CASE("Dict 32_64 scan test 1", "[main]") {
    size_t input_size = 1UL << 20;
    size_t dict_size = 1UL << 20;

    auto gen = [](uint64_t in) { return static_cast<uint32_t>(in & ((1 << 8) - 1)); };

    auto data_array = allocate_data_array_aligned<uint32_t, 64>(input_size, gen);
    auto dict = allocate_data_array_aligned<int64_t, 64>(dict_size);

    auto predicate_low = 0LL;
    auto predicate_high = 299LL;

    CacheAlignedVector<int64_t> result_vector{};

    SIMD512::dict_scan_32bit_64bit(predicate_low,
                                   predicate_high,
                                   dict.get(),
                                   dict_size,
                                   reinterpret_cast<__m512i *>(data_array.get()),
                                   input_size,
                                   result_vector);

    REQUIRE(result_vector.size() == input_size);
}

TEST_CASE("Dict 8_64 scan test scalar gather scatter", "[main]") {
    size_t input_size = 1UL << 20;
    auto data_array = allocate_data_array_aligned<uint8_t, 64>(input_size);
    auto dict = allocate_data_array_aligned<int64_t, 64>(256);

    std::transform(dict.get(), dict.get() + 256, dict.get(), [](auto in) { return in * 2; });

    auto predicate_low = 0LL;
    auto predicate_high = 98LL;

    CacheAlignedVector<int64_t> result_vector{};

    SIMD512::dict_scan_8bit_64bit_scalar_gather_scatter(predicate_low,
                                                        predicate_high,
                                                        dict.get(),
                                                        reinterpret_cast<__m512i *>(data_array.get()),
                                                        input_size,
                                                        result_vector,
                                                        true);

    REQUIRE(result_vector.size() == input_size / 256 * 50);

    for (auto i = 0; i < 50; ++i) {
        REQUIRE(result_vector[i] == i * 2);
    }

    REQUIRE(result_vector[99] == 98UL);
    REQUIRE(result_vector[100] == 0UL);
}

TEST_CASE("AVX vs scalar gather scatter", "[bench]") {
    size_t input_size = 1UL << 30;
    auto data_array = allocate_data_array_aligned<uint8_t, 64>(input_size);
    auto dict = allocate_data_array_aligned<int64_t, 64>(256);

    std::transform(dict.get(), dict.get() + 256, dict.get(), [](auto in) { return in * 2; });

    auto predicate_low = 0LL;
    auto predicate_high = 98LL;

    CacheAlignedVector<int64_t> result_vector(input_size);

    uint64_t cycles_avx = 0;
    uint64_t cycles_scalar = 0;

    {
        rdtscpWrapper wrap{&cycles_avx};
        SIMD512::dict_scan_8bit_64bit(predicate_low,
                                      predicate_high,
                                      dict.get(),
                                      reinterpret_cast<__m512i *>(data_array.get()),
                                      input_size,
                                      result_vector,
                                      false);
    }

    {
        rdtscpWrapper wrap{&cycles_scalar};
        SIMD512::dict_scan_8bit_64bit_scalar_gather_scatter(predicate_low,
                                                            predicate_high,
                                                            dict.get(),
                                                            reinterpret_cast<__m512i *>(data_array.get()),
                                                            input_size,
                                                            result_vector,
                                                            false);
    }

    std::cout << "AVX Cycles:    " << cycles_avx << "\n"
              << "Scalar Cycles: " << cycles_scalar << std::endl;

}

TEST_CASE("AVX vs scalar gather scatter low selectivity", "[bench]") {
    size_t input_size = 1UL << 30;
    auto data_array = allocate_data_array_aligned<uint8_t, 64>(input_size);
    auto dict = allocate_data_array_aligned<int64_t, 64>(256);

    std::transform(dict.get(), dict.get() + 256, dict.get(), [](auto in) { return in * 2; });

    auto predicate_low = 0LL;
    auto predicate_high = 4LL;

    CacheAlignedVector<int64_t> result_vector(input_size);

    uint64_t cycles_avx = 0;
    uint64_t cycles_scalar = 0;

    {
        rdtscpWrapper wrap{&cycles_avx};
        SIMD512::dict_scan_8bit_64bit(predicate_low,
                                      predicate_high,
                                      dict.get(),
                                      reinterpret_cast<__m512i *>(data_array.get()),
                                      input_size,
                                      result_vector,
                                      false);
    }

    {
        rdtscpWrapper wrap{&cycles_scalar};
        SIMD512::dict_scan_8bit_64bit_scalar_gather_scatter(predicate_low,
                                                            predicate_high,
                                                            dict.get(),
                                                            reinterpret_cast<__m512i *>(data_array.get()),
                                                            input_size,
                                                            result_vector,
                                                            false);
    }

    std::cout << "AVX Cycles:    " << cycles_avx << "\n"
              << "Scalar Cycles: " << cycles_scalar << std::endl;

}

TEST_CASE("scalar gather scatter vs. unrolled gather scatter", "[bench]") {
    size_t input_size = 1UL << 30;
    auto data_array = allocate_random_uniform<uint8_t>(input_size);
    auto dict = allocate_data_array_aligned<int64_t, 64>(256);

    std::transform(dict.get(), dict.get() + 256, dict.get(), [](auto in) { return in * 2; });

    auto predicate_low = 0LL;
    auto predicate_high = 255LL;

    CacheAlignedVector<int64_t> result_vector(input_size);

    uint64_t cycles_scalar = 0;
    uint64_t cycles_unrolled = 0;

    {
        rdtscpWrapper wrap{&cycles_unrolled};
        SIMD512::dict_scan_8bit_64bit_scalar_unroll(predicate_low,
                                                    predicate_high,
                                                    dict.get(),
                                                    reinterpret_cast<__m512i *>(data_array.get()),
                                                    input_size,
                                                    result_vector,
                                                    false);
    }

    {
        rdtscpWrapper wrap{&cycles_scalar};
        SIMD512::dict_scan_8bit_64bit_scalar_gather_scatter(predicate_low,
                                                            predicate_high,
                                                            dict.get(),
                                                            reinterpret_cast<__m512i *>(data_array.get()),
                                                            input_size,
                                                            result_vector,
                                                            false);
    }

    std::cout << "Unrolled Cycles: " << cycles_unrolled << "\n"
              << "Scalar Cycles:   " << cycles_scalar << std::endl;

}

TEST_CASE("unrolled gather scatter vs. pure scalar", "[bench]") {
    size_t input_size = 1UL << 30;
    auto data_array = allocate_random_uniform<uint8_t>(input_size);
    auto dict = allocate_data_array_aligned<int64_t, 64>(256);

    std::transform(dict.get(), dict.get() + 256, dict.get(), [](auto in) { return in * 2; });

    auto predicate_low = 0LL;
    auto predicate_high = 255LL;

    CacheAlignedVector<int64_t> result_vector(input_size);

    uint64_t cycles_scalar = 0;
    uint64_t cycles_unrolled = 0;

    {
        rdtscpWrapper wrap{&cycles_unrolled};
        SIMD512::dict_scan_8bit_64bit_scalar_unroll(predicate_low,
                                                    predicate_high,
                                                    dict.get(),
                                                    reinterpret_cast<__m512i *>(data_array.get()),
                                                    input_size,
                                                    result_vector,
                                                    false);
    }

    {
        rdtscpWrapper wrap{&cycles_scalar};
        SIMD512::dict_scan_8bit_64bit_scalar(predicate_low,
                                             predicate_high,
                                             dict.get(),
                                             data_array.get(),
                                             input_size,
                                             result_vector);
    }

    std::cout << "Unrolled Cycles: " << cycles_unrolled << "\n"
              << "Pure Scalar Cycles:   " << cycles_scalar << std::endl;

}

TEST_CASE("compare performance all", "[bench]") {
    size_t input_size = 1UL << 30;
    auto data_array = allocate_random_uniform<uint8_t>(input_size);
    auto dict = allocate_data_array_aligned<int64_t, 64>(256);

    auto predicate_low = 0LL;
    auto predicate_high = 2LL;

    CacheAlignedVector<int64_t> result_vector(input_size);

    uint64_t cycles_pure_scalar = 0;
    uint64_t cycles_scalar_gs = 0;
    uint64_t cycles_unrolled = 0;
    uint64_t cycles_AVX = 0;

    {
        rdtscpWrapper wrap{&cycles_unrolled};
        SIMD512::dict_scan_8bit_64bit_scalar_unroll(predicate_low,
                                                    predicate_high,
                                                    dict.get(),
                                                    reinterpret_cast<__m512i *>(data_array.get()),
                                                    input_size,
                                                    result_vector,
                                                    false);
    }

    {
        rdtscpWrapper wrap{&cycles_pure_scalar};
        SIMD512::dict_scan_8bit_64bit_scalar(predicate_low,
                                             predicate_high,
                                             dict.get(),
                                             data_array.get(),
                                             input_size,
                                             result_vector);
    }

    {
        rdtscpWrapper wrap{&cycles_scalar_gs};
        SIMD512::dict_scan_8bit_64bit_scalar_gather_scatter(predicate_low,
                                                            predicate_high,
                                                            dict.get(),
                                                            reinterpret_cast<__m512i *>(data_array.get()),
                                                            input_size,
                                                            result_vector, false);
    }

    {
        rdtscpWrapper wrap{&cycles_AVX};
        SIMD512::dict_scan_8bit_64bit(predicate_low,
                                      predicate_high,
                                      dict.get(),
                                      reinterpret_cast<__m512i *>(data_array.get()),
                                      input_size,
                                      result_vector,
                                      false);
    }

    std::cout << "Unrolled Cycles:    " << cycles_unrolled << "\n"
              << "Pure Scalar Cycles: " << cycles_pure_scalar << "\n"
              << "Scalar GS Cycles:   " << cycles_scalar_gs << "\n"
              << "AVX Cycles:         " << cycles_AVX << std::endl;

}

TEST_CASE("Dict 32_64 scalar gather scatter scan test 1", "[main]") {
    size_t input_size = 1UL << 20;
    size_t dict_size = 1UL << 20;

    auto gen = [](uint64_t in) { return static_cast<uint32_t>(in & ((1 << 8) - 1)); };

    auto data_array = allocate_data_array_aligned<uint32_t, 64>(input_size, gen);
    auto dict = allocate_data_array_aligned<int64_t, 64>(dict_size);

    auto predicate_low = 0LL;
    auto predicate_high = 299LL;

    CacheAlignedVector<int64_t> result_vector{};

    SIMD512::dict_scan_32bit_64bit_scalar_gather_scatter(predicate_low,
                                                         predicate_high,
                                                         dict.get(),
                                                         dict_size,
                                                         reinterpret_cast<__m512i *>(data_array.get()),
                                                         input_size,
                                                         result_vector,
                                                         true);

    REQUIRE(result_vector.size() == input_size);
}

TEST_CASE("Dict 32_64 scalar gather scatter scan test 2", "[main]") {
    size_t input_size = 1UL << 20;
    size_t dict_size = 1UL << 20;

    auto data_array = allocate_data_array_aligned<uint32_t, 64>(input_size);
    auto dict = allocate_data_array_aligned<int64_t, 64>(dict_size);

    auto predicate_low = 0LL;
    auto predicate_high = 99LL;

    CacheAlignedVector<int64_t> result_vector{};

    SIMD512::dict_scan_32bit_64bit_scalar_gather_scatter(predicate_low,
                                                         predicate_high,
                                                         dict.get(),
                                                         dict_size,
                                                         reinterpret_cast<__m512i *>(data_array.get()),
                                                         input_size,
                                                         result_vector,
                                                         true);

    REQUIRE(result_vector.size() == 100);

    REQUIRE(result_vector[0] == 0UL);
    REQUIRE(result_vector[10] == 10UL);
}