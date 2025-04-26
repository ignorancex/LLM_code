#include "PerfEvent.hpp"
#include "sgxerrors.h"
#include <algorithm>
#include <array>
#include <functional>
#include <gflags/gflags.h>
#include <optional>
#include <random>

#include "Enclave_u.h"

#include "../Shared/RNG.hpp"
#include "../Shared/simple_algorithms.hpp"
#include "EnclaveInit.hpp"
#include "Logger.hpp"
#include "sgxerrors.h"

#include <charconv>
#include <sys/prctl.h>

constexpr char ENCLAVE_FILENAME[] = "radixpartenc.signed.so";
constexpr int CACHE_LINE_SIZE = 64;

DEFINE_uint64(data_size, 1 << 27, "Number of rows to be filled with random data and added to histogram");
DEFINE_string(output_sizes, "1000",
              "Comma-separated list of number of different slots to write to. Size of output array and number of "
              "different indexes in input array.");
DEFINE_int32(min_output_size_exp, -1, "Number of bins in 2^n. Replaces num_bins if set.");
DEFINE_int32(max_output_size_exp, -1, "Number of bins in 2^n. Replaces num_bins if set.");
DEFINE_int32(min_write_positions_exp, -1, "Number of positions to start from in random_linear mode.");
DEFINE_int32(max_write_positions_exp, -1, "Number of positions to start from in random_linear mode.");
DEFINE_uint32(shift, 0, "Dummy");
DEFINE_bool(debug, false, "Debug output on/off");
DEFINE_string(modes, "scalar", "Comma-separated list of histrogram modes to benchmark");
DEFINE_string(random_mode, "random", "How to fill the input array with indexes");
DEFINE_uint32(min_unrolling_factor, 1, "Minimum unrolling factor to use for modes that allow unrolling");
DEFINE_uint32(max_unrolling_factor, 1, "Maximum unrolling factor to use for modes that allow unrolling");
DEFINE_bool(cache, false,
            "If true, preheats the input array in the cache by running the algorithm once before measuring the CPU "
            "cycles. Performance counters other than CPU cycles are measured for both executions.");
DEFINE_bool(ssb_mitigation, false, "Enable store buffer bypass mitigation outside enclave.");
DEFINE_bool(d64, false, "Whether input indices should be 64 bit. Must be true for pointer modes.");

Logger l{};

std::string
mode_to_string(const SimpleMode mode) {
    switch (mode) {
        case SimpleMode::scalar:
            return "scalar";
        case SimpleMode::unrolled:
            return "unrolled";
        case SimpleMode::loop:
            return "loop";
        case SimpleMode::loop_pointer:
            return "loop_pointer";
        case SimpleMode::loop_triple:
            return "loop_triple";
        case SimpleMode::loop_const:
            return "loop_const";
        case SimpleMode::ptr_loop:
            return "ptr_loop";
        case SimpleMode::ptr_scalar:
            return "ptr_scalar";
        default:;
    }
    return "";
}

SimpleMode
string_to_mode(const std::string_view input_string) {
    const static std::vector<std::pair<std::string, SimpleMode>> mode_pairs{{"scalar", SimpleMode::scalar},
                                                                            {"unrolled", SimpleMode::unrolled},
                                                                            {"loop", SimpleMode::loop},
                                                                            {"loop_pointer", SimpleMode::loop_pointer},
                                                                            {"loop_triple", SimpleMode::loop_triple},
                                                                            {"loop_const", SimpleMode::loop_const},
                                                                            {"ptr_loop", SimpleMode::ptr_loop},
                                                                            {"ptr_scalar", SimpleMode::ptr_scalar}};

    for (auto &[mode_string, mode]: mode_pairs) {
        if (mode_string == input_string) {
            return mode;
        }
    }

    l.err("Mode string not found. Returning default scalar");

    return SimpleMode::scalar;
}

const std::vector modes{SimpleMode::scalar,      SimpleMode::unrolled,   SimpleMode::loop,     SimpleMode::loop_pointer,
                        SimpleMode::loop_triple, SimpleMode::loop_const, SimpleMode::ptr_loop, SimpleMode::ptr_scalar};

std::vector<SimpleMode>
parse_modes(std::string_view input_string) {
    std::vector<SimpleMode> result{};

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

RandomMode
string_to_random_mode(const std::string_view input_string) {
    const static std::array<std::pair<std::string, RandomMode>, 4> mode_pairs{
            std::pair{"random", RandomMode::random}, std::pair{"random_linear", RandomMode::random_linear},
            std::pair{"linear", RandomMode::linear}, std::pair{"ptr_linear", RandomMode::ptr_linear}};

    for (auto &[mode_string, mode]: mode_pairs) {
        if (mode_string == input_string) {
            return mode;
        }
    }

    l.err("Mode string not found. Returning default scalar");

    return RandomMode::random;
}

std::string
random_mode_to_string(const RandomMode mode) {
    switch (mode) {
        case RandomMode::random:
            return "random";
        case RandomMode::random_linear:
            return "random_linear";
        case RandomMode::linear:
            return "linear";
        case RandomMode::ptr_linear:
            return "ptr_linear";
        default:;
    }
    return "";
}

std::vector<uint32_t>
parse_num_bins(std::string_view input_string) {
    std::vector<uint32_t> result{};

    size_t start = 0;
    size_t comma_pos;

    do {
        comma_pos = input_string.find(',', start);
        if (comma_pos == std::string::npos)
            comma_pos = input_string.size();
        const auto sub = input_string.substr(start, comma_pos - start);

        uint32_t num_bins;
        if (const auto [_, ec] = std::from_chars(sub.data(), sub.data() + sub.size(), num_bins);
            ec == std::errc::invalid_argument) {
            l.err("Could not convert.");
        } else {
            result.emplace_back(num_bins);
        }

        start = comma_pos + 1;
    } while (comma_pos < input_string.size());

    return result;
}

template<bool USE_64_BIT = false>
void
run_bench(const SimpleBenchmarkParameters &params) {

    const auto func = choose_function_simple<USE_64_BIT>(params.mode, params.unroll_factor);

    if (!func.has_value())
        return;

    BenchmarkParameters pe_params{};
    pe_params.setParam("num_keys", FLAGS_data_size);
    pe_params.setParam("num_bins", params.num_bins);
    pe_params.setParam("data_size", FLAGS_data_size * (params.d64 ? sizeof(uint64_t) : sizeof(uint32_t)));
    pe_params.setParam("workload", params.random_mode == RandomMode::ptr_linear ? "ptr" : "simple");
    pe_params.setParam("sgx", params.sgx);
    pe_params.setParam("preload", params.preload);
    pe_params.setParam("mode", mode_to_string(params.mode));
    pe_params.setParam("unroll_factor", params.unroll_factor);
    pe_params.setParam("cache_preheat", params.cache_preheat);
    pe_params.setParam("random_mode", random_mode_to_string(params.random_mode));
    pe_params.setParam("ssb_mitigation", params.ssb_mitigation);
    pe_params.setParam("d64", USE_64_BIT);

    auto const &h_params = params.function_parameters;

    auto data = reinterpret_cast<const std::conditional_t<USE_64_BIT, uint64_t, uint32_t> *>(h_params.data);

    uint64_t cycle_counter = 0;
    PerfEventBlock e{FLAGS_data_size, pe_params, params.print_header, &cycle_counter};
    if (params.sgx) {
        const auto ret = ecall_array_dynamic(params.eid, &params, &cycle_counter);
        if (ret != SGX_SUCCESS) {
            l.err("Running algorithm in enclave failed.");
            print_error_message(ret);
        }
    } else {
        if (params.cache_preheat) {
            func.value()(data, h_params.data_size, h_params.out, &cycle_counter);
            cycle_counter = 0;
        }
        func.value()(data, h_params.data_size, h_params.out, &cycle_counter);
    }
}

int
main(int argc, char *argv[]) {
    gflags::SetUsageMessage("Radix histogram benchmark");
    gflags::ParseCommandLineFlags(&argc, &argv, false);

    l.debug = FLAGS_debug;

    if (FLAGS_ssb_mitigation) {
        // Disable speculative loads after stores (enable store bypass mitigation)
        const int result = prctl(PR_SET_SPECULATION_CTRL, PR_SPEC_STORE_BYPASS, PR_SPEC_DISABLE, 0, 0);

        if (result == 0) {
            l.log("Successfully enabled store bypass mitigation");
        } else {
            l.err(std::string{"Disabling store bypass mitigation failed with error code "} + std::to_string(result));
        }
    }

    sgx_enclave_id_t eid = 0;
    auto init_ret = initialize_enclave(ENCLAVE_FILENAME, eid);
    if (init_ret != 0) {
        return -1;
    }

    const auto chosen_modes = parse_modes(FLAGS_modes);
    const auto random_mode = string_to_random_mode(FLAGS_random_mode);
    if (random_mode == RandomMode::random_linear || FLAGS_max_write_positions_exp != -1 ||
        FLAGS_min_write_positions_exp != -1) {
        l.err("Random number generation mode random_linear ist currently not implemented.");
    }

    const auto random_generator_function = choose_random_generator_function(random_mode);

    std::vector<uint32_t> chosen_bins{};

    if (FLAGS_max_output_size_exp >= 0 && FLAGS_min_output_size_exp >= 0) {
        if (FLAGS_max_output_size_exp > 31) {
            l.err("Number of bins to large. Max is 2^31.");
            return -1;
        }
        for (int exp = FLAGS_min_output_size_exp; exp <= FLAGS_max_output_size_exp; ++exp) {
            chosen_bins.emplace_back(1 << exp);
        }
    } else {
        chosen_bins = parse_num_bins(FLAGS_output_sizes);
    }

    if (std::find_first_of(chosen_modes.begin(), chosen_modes.end(), modes.begin() + 6, modes.end()) !=
                chosen_modes.end() &&
        !FLAGS_d64) {
        l.err("Pointer modes require 64 bit inputs");
        return -1;
    }

    l.log("Allocating data array.");
    const size_t element_size = FLAGS_d64 ? sizeof(uint64_t) : sizeof(uint32_t);
    const auto data = static_cast<uint64_t *>(std::aligned_alloc(CACHE_LINE_SIZE, FLAGS_data_size * element_size));
    l.log("Done.");

    l.log("Allocating data array in enclave.");
    sgx_status_t ret = ecall_init_array(eid, FLAGS_data_size);
    if (ret != SGX_SUCCESS) {
        l.err("init data failed");
        print_error_message(ret);
        return -1;
    }
    l.log("Done.");

    std::random_device rd;
    std::mt19937 g(rd());

    bool print_header = true;
    for (const uint32_t num_bins: chosen_bins) {

        const auto out = static_cast<uint32_t *>(std::aligned_alloc(CACHE_LINE_SIZE, num_bins * sizeof(uint32_t)));

        l.log("Writing new keys.");
        random_generator_function(data, FLAGS_data_size, out, num_bins, g, FLAGS_d64);
        l.log("Done.");

        for (const auto &mode: chosen_modes) {
            for (uint32_t unroll_factor = FLAGS_min_unrolling_factor; unroll_factor <= FLAGS_max_unrolling_factor;
                 ++unroll_factor) {
                std::memset(out, 0, num_bins * sizeof(uint32_t));
                SimpleParameters hist_params{data, static_cast<uint32_t>(FLAGS_data_size), out};
                SimpleBenchmarkParameters bench_params{
                        num_bins,    false,         false,       print_header, FLAGS_ssb_mitigation, FLAGS_d64, mode,
                        random_mode, unroll_factor, FLAGS_cache, eid,          hist_params};
                if (FLAGS_d64) {
                    run_bench<true>(bench_params);
                } else {
                    run_bench<false>(bench_params);
                }
                print_header = false;
            }
        }

        for (const auto &mode: chosen_modes) {
            for (uint32_t unroll_factor = FLAGS_min_unrolling_factor; unroll_factor <= FLAGS_max_unrolling_factor;
                 ++unroll_factor) {
                std::memset(out, 0, num_bins * sizeof(uint32_t));
                SimpleParameters hist_params{data, static_cast<uint32_t>(FLAGS_data_size), out};
                SimpleBenchmarkParameters bench_params{
                        num_bins,    true,          false,       print_header, FLAGS_ssb_mitigation, FLAGS_d64, mode,
                        random_mode, unroll_factor, FLAGS_cache, eid,          hist_params};
                if (FLAGS_d64) {
                    run_bench<true>(bench_params);
                } else {
                    run_bench<false>(bench_params);
                }
            }
        }

        ret = ecall_alloc_out_array(eid, num_bins);
        if (ret != SGX_SUCCESS) {
            l.err("Allocating output in enclave failed!");
            print_error_message(ret);
            continue;
        }

        l.log("Writing new keys in enclave");
        ret = ecall_fill_and_shuffle_array(eid, FLAGS_data_size, num_bins, static_cast<int>(random_mode),
                                           static_cast<int>(FLAGS_d64));
        if (ret != SGX_SUCCESS) {
            l.err("Writing and shuffling in enclave failed!");
            print_error_message(ret);
            continue;
        }
        l.log("Done.");

        for (const auto &mode: chosen_modes) {
            for (uint32_t unroll_factor = FLAGS_min_unrolling_factor; unroll_factor <= FLAGS_max_unrolling_factor;
                 ++unroll_factor) {
                SimpleParameters hist_params{data, static_cast<uint32_t>(FLAGS_data_size), out};
                SimpleBenchmarkParameters bench_params{
                        num_bins,    true,          true,        print_header, FLAGS_ssb_mitigation, FLAGS_d64, mode,
                        random_mode, unroll_factor, FLAGS_cache, eid,          hist_params};
                ret = ecall_reset_out_array(eid, num_bins);
                if (ret != SGX_SUCCESS) {
                    l.err("Resetting output in enclave failed!");
                    print_error_message(ret);
                    continue;
                }
                if (FLAGS_d64) {
                    run_bench<true>(bench_params);
                } else {
                    run_bench<false>(bench_params);
                }
            }
        }

        free(out);
    }
    return 0;
}
