#include "PerfEvent.hpp"
#include "rdtscpWrapper.h"
#include "Logger.hpp"
#include "cpu_mapping.h"

PerfEvent e;

extern "C" {

void ocall_startTimer(u_int64_t *t) {
    *t = rdtscp_s();
}

void ocall_stopTimer(u_int64_t *t) {
    *t = rdtscp_s() - *t;
}

void ocall_start_performance_counters() {
    e.startCounters();
}

void ocall_stop_performance_counters(uint64_t scale_factor, const char *phase) {
    e.stopCounters();
    // e.printReport(std::cout, scale_factor); // Uncomment this to print detailed report
    auto phase_str = phase ? phase : "";
    auto after_phase_space = phase ? " " : "";
    for (size_t i = 3; i < 9; ++i) {
        auto name_c_str = e.names[i].c_str();
        logger(LEVEL::INFO, "%s%s%s: %lf", phase_str, after_phase_space, name_c_str,
               e.events[i].readCounter() / static_cast<double>(scale_factor));
    }
}

void ocall_get_system_micros(uint64_t *t) {
    using clock = std::chrono::steady_clock;
    static_assert(clock::is_steady);
    auto now = clock::now().time_since_epoch();
    auto now_micro = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(now).count());
    *t = now_micro;
}

void ocall_exit(int status) {
    exit(status);
}

void ocall_get_num_active_threads_in_numa(int *res, int numaregionid) {
    *res = get_num_active_threads_in_numa(numaregionid);
}

void ocall_get_thread_index_in_numa(int *res, int logicaltid) {
    *res = get_thread_index_in_numa(logicaltid);
}

void ocall_get_cpu_id(int *res, int thread_id) {
    *res = get_cpu_id(thread_id);
}

void ocall_get_num_numa_regions(int *res) {
    *res = get_num_numa_regions();
}

void ocall_numa_thread_mark_active(int phytid) {
    numa_thread_mark_active(phytid);
}

void ocall_throw(const char *message) {
    logger(ERROR, "%s", message);
    exit(EXIT_FAILURE);
}

void ocall_pin_thread(int tid) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(tid, &cpuset);
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        logger(ERROR, "Error calling pthread_setaffinity_np: %d", rc);
    }
}

void ocall_set_mask(uint64_t min_cid, uint64_t max_cid) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (auto cid = min_cid; cid < max_cid; ++cid) {
        CPU_SET(cid, &cpuset);
    }
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        logger(ERROR, "Error calling pthread_setaffinity_np: %d", rc);
    }
}

}