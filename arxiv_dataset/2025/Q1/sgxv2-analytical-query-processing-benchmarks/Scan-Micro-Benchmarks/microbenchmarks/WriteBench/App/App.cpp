#include "../shared/algorithms.hpp"
#include "Enclave_u.h"
#include "Logger.hpp"
#include "PerfEvent.hpp"
#include "sgxerrors.h"
#include <algorithm>
#include <gflags/gflags.h>
#include <memory>

constexpr char ENCLAVE_FILENAME[] = "writebenchenc.signed.so";
constexpr uint64_t CACHE_LINE_SIZE = 64;
constexpr uint64_t PAGE_SIZE = 4096;

DEFINE_bool(debug, false, "Debug output.");
DEFINE_uint64(repeat, 5, "How many times to repeat benchmark");
DEFINE_uint64(size_exp_min, 3, "Minimum array size in bytes to test");
DEFINE_uint64(size_exp_max, 20, "Maximum array size in bytes to test");
DEFINE_uint64(write_size_exp_min, 3, "Minimum array size in bytes to test");
DEFINE_uint64(write_size_exp_max, 20, "Maximum array size in bytes to test");
DEFINE_string(mode, "write", "write | write_u | inc | read");

Logger l{};

int
initialize_enclave(const char *enclave_file, sgx_enclave_id_t &enclave_id) {
    sgx_status_t ret;

    l.log("Initializing enclave.");

    /* Call sgx_create_enclave to initialize an enclave instance */
    /* Debug Support: set 2nd parameter to 1 */
    ret = sgx_create_enclave(enclave_file, 1, nullptr, nullptr, &enclave_id, nullptr);
    if (ret != SGX_SUCCESS) {
        l.err("Enclave initialization failed!");
        print_error_message(ret);
        return -1;
    }
    ret = ecall_noop(enclave_id);
    if (ret != SGX_SUCCESS) {
        l.err("First call into enclave failed!");
        print_error_message(ret);
        return -1;
    }
    l.log("Done.");
    return 0;
}

int
destroy_enclave(sgx_enclave_id_t enclave_id) {
    sgx_status_t ret;

    l.log("Destroying enclave.");

    /* Call sgx_destroy_enclave to destroy  the enclave instance */
    ret = sgx_destroy_enclave(enclave_id);
    if (ret != SGX_SUCCESS) {
        l.err("Destroying the enclave failed!");
        print_error_message(ret);
        return -1;
    }
    l.log("Done.");
    return 0;
}

void *
alloc_data(uint64_t data_size) {
    auto data = std::aligned_alloc(PAGE_SIZE, data_size);
    if (!data) {
        std::cerr << "Failed to allocate aligned memory." << std::endl;
        std::exit(-1);
    }
    return data;
}

[[nodiscard]] uint64_t
uint64_t_mask(uint64_t size_exp) {
    return (1ULL << (size_exp - 3)) - 1;
}

[[nodiscard]] uint64_t
variable_size_mask(uint64_t size_exp, uint64_t write_size_exp) {
    return ((1ULL << size_exp) - 1) - ((1ULL << write_size_exp) - 1);
}

using BenchmarkFunction = decltype(random_write<DEFAULT_REPETITIONS>);
using EnclaveBenchmarkFunction = decltype(ecall_random_write);
using EnclaveUserCheckBenchmarkFunction = decltype(ecall_random_write_user_check);

template<BenchmarkFunction PlainFunction, EnclaveBenchmarkFunction EnclaveFunction,
         EnclaveUserCheckBenchmarkFunction UserCheckFunction>
void
bench(uint64_t *data, const std::string &workload, sgx_enclave_id_t eid) {
    l.log("Starting simple random write benchmark...");

    uint64_t cycle_cntr = 0;
    BenchmarkParameters params = {};
    constexpr uint64_t writeOps = 1'000'000'000;

    params.setParam("workload", workload);
    params.setParam("writeSize", 8);
    params.setParam("writeOps", writeOps);
    params.setParam("pattern", "random");
    params.setParam("preload", "no");
    params.setParam("sgx", "no");

    bool printHeader = true;
    for (size_t size_exp = FLAGS_size_exp_min; size_exp <= FLAGS_size_exp_max; ++size_exp) {
        params.setParam("totalDataSize", 1ULL << size_exp);
        for (auto repeat = 0; repeat < FLAGS_repeat; ++repeat) {
            cycle_cntr = 0;
            PerfEventBlock e{writeOps, params, printHeader, &cycle_cntr};
            PlainFunction(data, uint64_t_mask(size_exp), &cycle_cntr);
            printHeader = false;
        }
    }

    params.setParam("sgx", "yes");
    for (size_t size_exp = FLAGS_size_exp_min; size_exp <= FLAGS_size_exp_max; ++size_exp) {
        params.setParam("totalDataSize", 1ULL << size_exp);
        for (auto repeat = 0; repeat < FLAGS_repeat; ++repeat) {
            cycle_cntr = 0;
            PerfEventBlock e{writeOps, params, false, &cycle_cntr};
            UserCheckFunction(eid, data, uint64_t_mask(size_exp), &cycle_cntr);
        }
    }

    params.setParam("preload", "yes");
    for (size_t size_exp = FLAGS_size_exp_min; size_exp <= FLAGS_size_exp_max; ++size_exp) {
        params.setParam("totalDataSize", 1ULL << size_exp);
        for (auto repeat = 0; repeat < FLAGS_repeat; ++repeat) {
            cycle_cntr = 0;
            PerfEventBlock e{writeOps, params, false, &cycle_cntr};
            EnclaveFunction(eid, uint64_t_mask(size_exp), &cycle_cntr);
        }
    }
    l.log("Benchmark over");
}

void
bench_variable_size(uint64_t *data, const std::string &workload, sgx_enclave_id_t eid) {
    l.log("Starting simple random write benchmark...");

    uint64_t cycle_cntr = 0;
    BenchmarkParameters params = {};
    constexpr uint64_t writeOpsDefault = 100'000'000;
    uint64_t writeOps = writeOpsDefault;

    uint64_t size_exp = FLAGS_size_exp_max;

    params.setParam("workload", workload);
    params.setParam("writeOps", writeOps);
    params.setParam("pattern", "random");
    params.setParam("preload", "no");
    params.setParam("sgx", "no");
    params.setParam("totalDataSize", 1ULL << size_exp);

    bool printHeader = true;
    for (size_t write_size_exp = FLAGS_write_size_exp_min; write_size_exp <= FLAGS_write_size_exp_max;
         ++write_size_exp) {
        uint64_t write_size = 1ULL << write_size_exp;
        params.setParam("writeSize", write_size);
        if (write_size_exp > 5 && write_size_exp % 5 == 0) {
            writeOps /= 10;
            params.setParam("writeOps", writeOps);
        }
        for (auto repeat = 0; repeat < FLAGS_repeat; ++repeat) {
            cycle_cntr = 0;
            PerfEventBlock e{writeOps, params, printHeader, &cycle_cntr};
            random_write_size(reinterpret_cast<uint8_t *>(data), variable_size_mask(size_exp, write_size_exp),
                              write_size, writeOps, &cycle_cntr);
            printHeader = false;
        }
    }

    writeOps = writeOpsDefault;
    params.setParam("writeOps", writeOps);
    params.setParam("sgx", "yes");
    for (size_t write_size_exp = FLAGS_write_size_exp_min; write_size_exp <= FLAGS_write_size_exp_max;
         ++write_size_exp) {
        uint64_t write_size = 1ULL << write_size_exp;
        params.setParam("writeSize", write_size);
        if (write_size_exp > 5 && write_size_exp % 5 == 0) {
            writeOps /= 10;
            params.setParam("writeOps", writeOps);
        }
        for (auto repeat = 0; repeat < FLAGS_repeat; ++repeat) {
            cycle_cntr = 0;
            PerfEventBlock e{writeOps, params, printHeader, &cycle_cntr};
            ecall_random_write_size_user_check(eid, data, variable_size_mask(size_exp, write_size_exp), write_size,
                                               writeOps, &cycle_cntr);
        }
    }

    writeOps = writeOpsDefault;
    params.setParam("writeOps", writeOps);
    params.setParam("preload", "yes");
    for (size_t write_size_exp = FLAGS_write_size_exp_min; write_size_exp <= FLAGS_write_size_exp_max;
         ++write_size_exp) {
        uint64_t write_size = 1ULL << write_size_exp;
        params.setParam("writeSize", write_size);
        if (write_size_exp > 5 && write_size_exp % 5 == 0) {
            writeOps /= 10;
            params.setParam("writeOps", writeOps);
        }
        for (auto repeat = 0; repeat < FLAGS_repeat; ++repeat) {
            cycle_cntr = 0;
            PerfEventBlock e{writeOps, params, printHeader, &cycle_cntr};
            ecall_random_write_size(eid, variable_size_mask(size_exp, write_size_exp), write_size, writeOps,
                                    &cycle_cntr);
        }
    }

    l.log("Benchmark over");
}

int
main(int argc, char *argv[]) {
    gflags::SetUsageMessage("Write benchmark");
    gflags::ParseCommandLineFlags(&argc, &argv, false);

    l.debug = FLAGS_debug;

    uint64_t max_array_size = 1ULL << FLAGS_size_exp_max;
    l.log("Allocating array.");
    auto data = std::unique_ptr<uint64_t>(reinterpret_cast<uint64_t *>(alloc_data(max_array_size)));
    std::memset(data.get(), 42, max_array_size);
    l.log("Done.");

    sgx_enclave_id_t eid;
    initialize_enclave(ENCLAVE_FILENAME, eid);
    sgx_status_t ret = SGX_SUCCESS;
    l.log("Allocating array in enclave.");
    ret = ecall_init_data(eid, max_array_size);
    if (ret != SGX_SUCCESS) {
        l.err("Initializing data in enclave failed!");
        std::exit(-1);
    }
    l.log("Done.");

    if (FLAGS_mode == "write") {
        bench<random_write, ecall_random_write, ecall_random_write_user_check>(data.get(), "write", eid);
    } else if (FLAGS_mode == "write_u") {
        bench<random_write_unrolled, ecall_random_write_unrolled, ecall_random_write_unrolled_user_check>(
                data.get(), "write_u", eid);
    } else if (FLAGS_mode == "inc") {
        bench<random_increment, ecall_random_increment, ecall_random_increment_user_check>(data.get(), "inc", eid);
    } else if (FLAGS_mode == "write_size") {
        bench_variable_size(data.get(), "write_size", eid);
    } else if (FLAGS_mode == "read") {
        bench<random_read, ecall_random_read, ecall_random_read_user_check>(data.get(), "read", eid);
    } else {
        l.err("mode unknown!");
    }

    ret = ecall_free_preload_data(eid);
    if (ret != SGX_SUCCESS) {
        l.err("Free of enclave data failed");
    }

    return 0;
}
