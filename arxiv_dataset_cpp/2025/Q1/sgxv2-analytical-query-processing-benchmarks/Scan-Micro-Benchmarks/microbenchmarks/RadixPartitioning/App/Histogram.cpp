#include "PerfEvent.hpp"
#include "sgxerrors.h"
#include <algorithm>
#include <functional>
#include <gflags/gflags.h>
#include <optional>
#include <random>

#include "Enclave_u.h"

#include "../Shared/histogram_algorithms.hpp"
#include "EnclaveInit.hpp"
#include "Logger.hpp"

#include <sys/prctl.h>

constexpr char ENCLAVE_FILENAME[] = "radixpartenc.signed.so";
constexpr int CACHE_LINE_SIZE = 64;

DEFINE_uint64(data_size, 1 << 27, "Number of rows to be filled with random data and added to histogram");
DEFINE_int32(min_radix_bits, 0, "Minimum number of radix bits to test. Determines histogram size.");
DEFINE_int32(max_radix_bits, 20, "Maximum number of radix bits to test. Determines histogram size.");
DEFINE_uint32(min_num_keys_exp, 30, "Determines the minimum number of unique keys in the array.");
DEFINE_uint32(max_num_keys_exp, 30, "Determines the maximum number of unique keys in the array.");
DEFINE_uint32(shift, 0, "Dummy");
DEFINE_bool(debug, false, "Debug output on/off");
DEFINE_string(modes, "scalar", "Comma-separated list of histrogram modes to benchmark");
DEFINE_uint32(min_unrolling_factor, 1, "Minimum unrolling factor to use for modes that allow unrolling");
DEFINE_uint32(max_unrolling_factor, 1, "Maximum unrolling factor to use for modes that allow unrolling");
DEFINE_bool(ssb_mitigation, false, "Enable store buffer bypass mitigation outside enclave.");

Logger l{};

std::string
mode_to_string(const HistogramMode mode) {
    switch (mode) {
        case HistogramMode::scalar:
            return "scalar";
        case HistogramMode::pragma:
            return "pragma";
        case HistogramMode::unrolled:
            return "unrolled";
        case HistogramMode::loop:
            return "loop";
        case HistogramMode::loop_pragma:
            return "loop_pragma";
        case HistogramMode::simd_loop:
            return "simd_loop";
        case HistogramMode::simd_loop_unrolled:
            return "simd_loop_unrolled";
        case HistogramMode::simd_loop_unrolled_512:
            return "simd_loop_unrolled_512";
        case HistogramMode::simd_unrolled:
            return "simd_unrolled";
        case HistogramMode::shuffle:
            return "shuffle";
        default:;
    }
    return "";
}

HistogramMode
string_to_mode(const std::string_view input_string) {
    const static std::vector<std::pair<std::string, HistogramMode>> mode_pairs{
            {"scalar", HistogramMode::scalar},
            {"pragma", HistogramMode::pragma},
            {"shuffle", HistogramMode::shuffle},
            {"unrolled", HistogramMode::unrolled},
            {"loop", HistogramMode::loop},
            {"loop_pragma", HistogramMode::loop_pragma},
            {"simd_loop", HistogramMode::simd_loop},
            {"simd_loop_unrolled", HistogramMode::simd_loop_unrolled},
            {"simd_loop_unrolled_512", HistogramMode::simd_loop_unrolled_512},
            {"simd_unrolled", HistogramMode::simd_unrolled}};

    for (auto &[mode_string, mode]: mode_pairs) {
        if (mode_string == input_string) {
            return mode;
        }
    }

    l.err("Mode string not found. Returning default scalar");

    return HistogramMode::scalar;
}

const std::vector modes{HistogramMode::scalar,
                        HistogramMode::pragma,
                        HistogramMode::unrolled,
                        HistogramMode::loop,
                        HistogramMode::loop_pragma,
                        HistogramMode::simd_loop,
                        HistogramMode::simd_loop_unrolled,
                        HistogramMode::simd_unrolled,
                        HistogramMode::simd_loop_unrolled_512,
                        HistogramMode::shuffle};

std::vector<HistogramMode>
parse_modes(std::string_view input_string) {
    std::vector<HistogramMode> result{};

    if (input_string == "all") {
        std::copy(modes.begin(), modes.end(), std::back_inserter(result));
    } else {
        size_t start = 0;
        size_t comma_pos;

        do {
            comma_pos = input_string.find(',', start);
            if (comma_pos == std::string::npos)
                comma_pos = input_string.size();
            const auto sub = input_string.substr(start, comma_pos - start);

            result.emplace_back(string_to_mode(sub));

            start = comma_pos + 1;
        } while (comma_pos < input_string.size());
    }

    return result;
}

void
run_bench(const HistogramBenchmarkParameters &params) {

    const auto func = choose_function(params.mode, params.unroll_factor);
    if (!func.has_value())
        return;

    BenchmarkParameters pe_params{};
    pe_params.setParam("num_keys", params.num_keys);
    pe_params.setParam("num_bins", params.num_bins);
    pe_params.setParam("rows", FLAGS_data_size);
    pe_params.setParam("data_size", FLAGS_data_size * sizeof(row));
    pe_params.setParam("workload", "hist");
    pe_params.setParam("sgx", params.sgx);
    pe_params.setParam("preload", params.preload);
    pe_params.setParam("mode", mode_to_string(params.mode));
    pe_params.setParam("unroll_factor", params.unroll_factor);
    pe_params.setParam("ssb_mitigation", params.ssb_mitigation);

    auto const &h_params = params.function_parameters;

    uint64_t cycle_counter = 0;
    PerfEventBlock e{FLAGS_data_size, pe_params, params.print_header, &cycle_counter};
    if (params.sgx) {
        // TODO add user_check variant
        auto ret = ecall_histogram_dynamic(params.eid, &params, &cycle_counter);
        if (ret != SGX_SUCCESS) {
            l.err("Running histogram algorithm in enclave failed.");
        }
    } else {
        func.value()(h_params.data, h_params.data_size, h_params.mask, h_params.shift, h_params.hist, &cycle_counter);
        //l.log(std::to_string(*h_params.hist));
    }
}

int
main(int argc, char *argv[]) {
    gflags::SetUsageMessage("Radix histogram benchmark");
    gflags::ParseCommandLineFlags(&argc, &argv, false);

    l.debug = FLAGS_debug;

    // TODO add mitigation parameter to output
    if (FLAGS_ssb_mitigation) {
        // Disable speculative loads after stores (enable store bypass mitigation)
        const int result = prctl(PR_SET_SPECULATION_CTRL, PR_SPEC_STORE_BYPASS, PR_SPEC_DISABLE, 0, 0);

        if (result == 0) {
            l.log("Successfully enabled store bypass mitigation");
        } else {
            l.err(std::string{"Enabling store bypass mitigation failed with error code "} + std::to_string(result));
        }
    } else {
        const int result = prctl(PR_SET_SPECULATION_CTRL, PR_SPEC_STORE_BYPASS, PR_SPEC_ENABLE, 0, 0);

        if (result == 0) {
            l.log("Successfully disabled store bypass mitigation");
        } else {
            l.err(std::string{"Disabling store bypass mitigation failed with error code "} + std::to_string(result));
        }
    }


    sgx_enclave_id_t eid = 0;
    auto ret = initialize_enclave(ENCLAVE_FILENAME, eid);
    if (ret != 0) {
        return -1;
    }

    const uint32_t min_radix_bits = FLAGS_min_radix_bits;
    const uint32_t max_radix_bits = FLAGS_max_radix_bits;

    const auto chosen_modes = parse_modes(FLAGS_modes);

    l.log("Allocating data array.");
    const auto data = static_cast<row *>(std::aligned_alloc(CACHE_LINE_SIZE, FLAGS_data_size * sizeof(row)));
    l.log("Done.");

    l.log("Allocating data array in enclave.");
    ret = ecall_init_data(eid, FLAGS_data_size);
    if (ret != SGX_SUCCESS) {
        l.err("init data failed");
        return -1;
    }
    l.log("Done.");

    std::random_device rd;
    std::mt19937 g(rd());

    bool print_header = true;
    for (uint32_t num_keys_exp = FLAGS_min_num_keys_exp; num_keys_exp <= FLAGS_max_num_keys_exp; ++num_keys_exp) {
        const uint32_t num_keys = (1 << num_keys_exp);

        l.log("Writing new keys.");
        for (uint32_t i = 0; i < FLAGS_data_size; ++i) {
            data[i].key = i % num_keys;
        }
        l.log("Done.");

        l.log("Shuffling.");
        std::shuffle(data, data + FLAGS_data_size, g);
        l.log("Done.");

        for (uint32_t radix_bits = min_radix_bits; radix_bits <= max_radix_bits; ++radix_bits) {
            const uint32_t fan_out = 1 << radix_bits;
            const uint32_t mask = fan_out - 1;
            const uint32_t shift = FLAGS_shift;

            const auto hist = static_cast<uint32_t *>(std::aligned_alloc(CACHE_LINE_SIZE, fan_out * sizeof(uint32_t)));
            ret = ecall_alloc_hist(eid, fan_out);
            if (ret != SGX_SUCCESS) {
                l.err("Allocating histogram in enclave failed!");
                continue;
            }

            for (const auto &mode: chosen_modes) {
                for (uint32_t unroll_factor = FLAGS_min_unrolling_factor; unroll_factor <= FLAGS_max_unrolling_factor;
                     ++unroll_factor) {
                    std::memset(hist, 0, fan_out * sizeof(uint32_t));
                    HistogramParameters hist_params{data, static_cast<uint32_t>(FLAGS_data_size), mask, shift, hist};
                    HistogramBenchmarkParameters bench_params{
                            num_keys, fan_out,       false, false,      print_header, FLAGS_ssb_mitigation,
                            mode,     unroll_factor, eid,   hist_params};
                    run_bench(bench_params);
                    print_header = false;
                }
            }

            for (const auto &mode: chosen_modes) {
                for (uint32_t unroll_factor = FLAGS_min_unrolling_factor; unroll_factor <= FLAGS_max_unrolling_factor;
                     ++unroll_factor) {
                    std::memset(hist, 0, fan_out * sizeof(uint32_t));
                    HistogramParameters hist_params{data, static_cast<uint32_t>(FLAGS_data_size), mask, shift, hist};
                    HistogramBenchmarkParameters bench_params{
                            num_keys, fan_out,       true, false,      print_header, FLAGS_ssb_mitigation,
                            mode,     unroll_factor, eid,  hist_params};
                    run_bench(bench_params);
                }
            }
            free(hist);
        }

        l.log("Writing and shuffling in enclave");
        ret = ecall_fill_and_shuffle(eid, FLAGS_data_size, num_keys);
        if (ret != SGX_SUCCESS) {
            l.err("Writing and shuffling in enclave failed!");
            continue;
        }
        l.log("Done.");

        for (uint32_t radix_bits = min_radix_bits; radix_bits <= max_radix_bits; ++radix_bits) {
            const uint32_t fan_out = 1 << radix_bits;
            const uint32_t mask = fan_out - 1;
            const uint32_t shift = FLAGS_shift;

            const auto hist = static_cast<uint32_t *>(std::aligned_alloc(CACHE_LINE_SIZE, fan_out * sizeof(uint32_t)));
            ret = ecall_alloc_hist(eid, fan_out);
            if (ret != SGX_SUCCESS) {
                l.err("Allocating histogram in enclave failed!");
                continue;
            }

            for (const auto &mode: chosen_modes) {
                for (uint32_t unroll_factor = FLAGS_min_unrolling_factor; unroll_factor <= FLAGS_max_unrolling_factor;
                     ++unroll_factor) {
                    std::memset(hist, 0, fan_out * sizeof(uint32_t));
                    HistogramParameters hist_params{data, static_cast<uint32_t>(FLAGS_data_size), mask, shift, hist};
                    HistogramBenchmarkParameters bench_params{
                            num_keys, fan_out,       true, true,       print_header, FLAGS_ssb_mitigation,
                            mode,     unroll_factor, eid,  hist_params};
                    ret = ecall_reset_hist(eid, fan_out);
                    if (ret != SGX_SUCCESS) {
                        l.err("Resetting histogram in enclave failed!");
                        continue;
                    }
                    run_bench(bench_params);
                }
            }
            free(hist);
        }
    }

    return 0;
}
