#include "simple_algorithms.hpp"

#include "rdtscpWrapper.h"


template<bool USE_64_BIT>
std::optional<std::conditional_t<USE_64_BIT, SimpleFunction64, SimpleFunction32>>
choose_function_simple(SimpleMode mode, uint32_t unroll_factor) {
    using T = std::conditional_t<USE_64_BIT, uint64_t, uint32_t>;
    switch (mode) {
        case SimpleMode::scalar:
            if (unroll_factor == 1) {
                return scalar<T>;
            }
            break;
        case SimpleMode::unrolled:
            if (unroll_factor == 4) {
                return unrolled<T>;
            }
            break;
        case SimpleMode::loop:
            switch (unroll_factor) {
                case 1:
                    return simple_loop<T, 1>;
                case 2:
                    return simple_loop<T, 2>;
                case 3:
                    return simple_loop<T, 3>;
                case 4:
                    return simple_loop<T, 4>;
                case 5:
                    return simple_loop<T, 5>;
                case 6:
                    return simple_loop<T, 6>;
                case 7:
                    return simple_loop<T, 7>;
                case 8:
                    return simple_loop<T, 8>;
                case 9:
                    return simple_loop<T, 9>;
                case 10:
                    return simple_loop<T, 10>;
                case 11:
                    return simple_loop<T, 11>;
                case 12:
                    return simple_loop<T, 12>;
                case 16:
                    return simple_loop<T, 16>;
                default:
                    break;
            }
        case SimpleMode::loop_const:
            switch (unroll_factor) {
                case 1:
                    return simple_const_loop<T, 1>;
                case 2:
                    return simple_const_loop<T, 2>;
                case 3:
                    return simple_const_loop<T, 3>;
                case 4:
                    return simple_const_loop<T, 4>;
                case 5:
                    return simple_const_loop<T, 5>;
                case 6:
                    return simple_const_loop<T, 6>;
                case 7:
                    return simple_const_loop<T, 7>;
                case 8:
                    return simple_const_loop<T, 8>;
                case 9:
                    return simple_const_loop<T, 9>;
                case 10:
                    return simple_const_loop<T, 10>;
                case 11:
                    return simple_const_loop<T, 11>;
                case 12:
                    return simple_const_loop<T, 12>;
                case 16:
                    return simple_const_loop<T, 16>;
                default:
                    break;
            }
        case SimpleMode::ptr_loop:
            if constexpr (std::is_same_v<T, uint64_t>) {
                switch (unroll_factor) {
                    case 1:
                        return simple_pointers_loop<1>;
                    case 2:
                        return simple_pointers_loop<2>;
                    case 3:
                        return simple_pointers_loop<3>;
                    case 4:
                        return simple_pointers_loop<4>;
                    case 5:
                        return simple_pointers_loop<5>;
                    case 6:
                        return simple_pointers_loop<6>;
                    case 7:
                        return simple_pointers_loop<7>;
                    case 8:
                        return simple_pointers_loop<8>;
                    case 9:
                        return simple_pointers_loop<9>;
                    case 10:
                        return simple_pointers_loop<10>;
                    case 11:
                        return simple_pointers_loop<11>;
                    case 12:
                        return simple_pointers_loop<12>;
                    case 16:
                        return simple_pointers_loop<16>;
                    default:
                        break;
                }
            }
            break;
        case SimpleMode::ptr_scalar:
            if constexpr (std::is_same_v<T, uint64_t>) {
                if (unroll_factor == 1) {
                    return simple_pointers_scalar;
                }
            }
            break;
        default:
            break;
    }

    return std::nullopt;
}

template std::optional<SimpleFunction32> choose_function_simple<false>(SimpleMode mode, uint32_t unroll_factor);
template std::optional<SimpleFunction64> choose_function_simple<true>(SimpleMode mode, uint32_t unroll_factor);

template<typename T>
void
scalar(const T *data, const uint32_t data_size, uint32_t *out, uint64_t *cycle_counter) {
    rdtscpWrapper r{cycle_counter};
    for (uint32_t i = 0; i < data_size; ++i) {
        const uint32_t idx = data[i];
        out[idx] = idx;
    }
}

template<typename T>
void
unrolled(const T *data, const uint32_t data_size, uint32_t *out, uint64_t *cycle_counter) {
    rdtscpWrapper r{cycle_counter};
    uint32_t i = 0;
    for (; i <= data_size - 4; i += 4) {
        const uint32_t idx = data[i];
        const uint32_t idx1 = data[i + 1];
        const uint32_t idx2 = data[i + 2];
        const uint32_t idx3 = data[i + 3];
        out[idx] = idx;
        out[idx1] = idx1;
        out[idx1] = idx2;
        out[idx3] = idx3;
    }
    for (; i < data_size; ++i) {
        const uint32_t idx = data[i];
        out[idx] = idx;
    }
}

template<typename T, int UNROLL_FACTOR>
void
simple_loop(const T *data, const uint32_t data_size, uint32_t *out, uint64_t *cycle_counter) {
    rdtscpWrapper r{cycle_counter};
    alignas(64) uint32_t indexes[UNROLL_FACTOR];

    uint32_t i = 0;
    for (; i <= data_size - UNROLL_FACTOR; i += UNROLL_FACTOR) {
        for (uint32_t j = 0; j < UNROLL_FACTOR; j++) {
            indexes[j] = data[i + j];
        }

        for (uint32_t j = 0; j < UNROLL_FACTOR; j++) {
            out[indexes[j]] = indexes[j];
        }
    }
    // handle the remaining entries if data_size is not dividable by unroll_factor
    for (; i < data_size; ++i) {
        const uint32_t idx = data[i];
        out[idx] = idx;
    }
}

template<typename T, int UNROLL_FACTOR>
void
simple_const_loop(const T *data, const uint32_t data_size, uint32_t *out, uint64_t *cycle_counter) {
    rdtscpWrapper r{cycle_counter};
    alignas(64) uint32_t indexes[UNROLL_FACTOR];

    uint32_t i = 0;
    for (; i <= data_size - UNROLL_FACTOR; i += UNROLL_FACTOR) {
        for (uint32_t j = 0; j < UNROLL_FACTOR; j++) {
            indexes[j] = data[i + j];
        }

        for (uint32_t j = 0; j < UNROLL_FACTOR; j++) {
            out[indexes[j]] = 0xdeadbeefU;
        }
    }
    // handle the remaining entries if data_size is not dividable by unroll_factor
    for (; i < data_size; ++i) {
        const uint32_t idx = data[i];
        out[idx] = 0xdeadbeefU;
    }
}

void
simple_pointers_scalar(const uint64_t *data, const uint32_t data_size, uint32_t *out, uint64_t *cycle_counter) {
    rdtscpWrapper r{cycle_counter};

    const auto data_pointers = reinterpret_cast<uint32_t * const *>(data);
    // Explicitly do not unroll this loop!
#pragma GCC novector
    for (uint32_t i = 0; i < data_size; ++i) {
        uint32_t *ptr = data_pointers[i];
        *ptr = 0xdeadbeefU;
    }
}

template<int UNROLL_FACTOR>
void
simple_pointers_loop(const uint64_t *data, const uint32_t data_size, uint32_t *out, uint64_t *cycle_counter) {
    rdtscpWrapper r{cycle_counter};

    auto data_pointers = reinterpret_cast<uint32_t * const *>(data);

    alignas(64) uint32_t *pointers[UNROLL_FACTOR];

    uint32_t i = 0;
    for (; i <= data_size - UNROLL_FACTOR; i += UNROLL_FACTOR) {
        for (uint32_t j = 0; j < UNROLL_FACTOR; j++) {
            pointers[j] = data_pointers[i + j];
        }

        for (uint32_t j = 0; j < UNROLL_FACTOR; j++) {
            *pointers[j] = 0xdeadbeefU;
        }
    }
    // handle the remaining entries if data_size is not dividable by unroll_factor
    for (; i < data_size; ++i) {
        uint32_t *ptr = data_pointers[i];
        *ptr = 0xdeadbeefU;
    }
}