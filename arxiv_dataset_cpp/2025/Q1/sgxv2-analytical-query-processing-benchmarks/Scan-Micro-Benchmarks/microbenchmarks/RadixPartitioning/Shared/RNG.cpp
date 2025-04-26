#include "RNG.hpp"

void
generate_random_array(uint64_t *data, const uint64_t data_size, const uint32_t *out, const uint32_t num_bins,
                      std::mt19937 &gen, bool use_64_bit) {
    std::uniform_int_distribution<uint32_t> distribution{0, num_bins - 1};
    if (use_64_bit) {
        std::generate_n(data, data_size, [&gen, &distribution] { return distribution(gen); });
    } else {
        auto data_32 = reinterpret_cast<uint32_t *>(data);
        std::generate_n(data_32, data_size, [&gen, &distribution] { return distribution(gen); });
    }
}

void
generate_random_copy_array(uint64_t *data, const uint64_t data_size, const uint32_t *out, const uint32_t num_bins,
                           std::mt19937 &gen, bool use_64_bit) {
    std::vector<uint32_t> start_positions(num_bins);
    std::vector<uint32_t> end_positions(num_bins);
    const auto step = static_cast<uint32_t>(data_size / static_cast<uint64_t>(num_bins));
    for (int i = 0; i < num_bins; ++i) {
        start_positions[i] = step * i;
        end_positions[i] = step * (i + 1);
    }

    std::uniform_int_distribution<size_t> index_dist{0, num_bins - 1};

    uint64_t output_index = 0;
    for (; output_index < data_size; ++output_index) {
        const size_t index = index_dist(gen);
        if (start_positions[index] == end_positions[index]) [[unlikely]] {
            break;
        }

        data[output_index] = start_positions[index];
        start_positions[index]++;
    }

    std::vector<size_t> remaining{};

    for (int i = 0; i < num_bins; ++i) {
        if (start_positions[i] != end_positions[i]) {
            remaining.push_back(i);
        }
    }

    index_dist = std::uniform_int_distribution<size_t>{0, remaining.size() - 1};

    for (; output_index < data_size; ++output_index) {
        const auto remaining_index = index_dist(gen);
        const size_t bin_index = remaining[remaining_index];
        data[output_index] = start_positions[bin_index];
        start_positions[bin_index]++;
        if (start_positions[bin_index] == end_positions[bin_index]) {
            remaining.erase(remaining.cbegin() + remaining_index);
            index_dist = std::uniform_int_distribution<size_t>{0, remaining.size() - 1};
        }
    }
}

void
generate_linear_array(uint64_t *data, const uint64_t data_size, const uint32_t *out, const uint32_t num_bins,
                      std::mt19937 &gen, bool use_64_bit) {
    if (use_64_bit) {
        for (auto i = 0; i < data_size; ++i) {
            data[i] = i % num_bins;
        }
    } else {
        auto data_32 = reinterpret_cast<uint32_t *>(data);
        for (auto i = 0; i < data_size; ++i) {
            data_32[i] = i % num_bins;
        }
    }

}

void
generate_linear_pointer_array(uint64_t *data, const uint64_t data_size, const uint32_t *out, const uint32_t num_bins,
                              std::mt19937 &gen, bool use_64_bit) {
    for (auto i = 0; i < data_size; ++i) {
        data[i] = reinterpret_cast<uint64_t>(out + i % num_bins);
    }
}

RandomGeneratorFunction
choose_random_generator_function(const RandomMode mode) {
    switch (mode) {
        case RandomMode::random:
            return generate_random_array;
        case RandomMode::random_linear:
            break;
        case RandomMode::linear:
            return generate_linear_array;
        case RandomMode::ptr_linear:
            return generate_linear_pointer_array;
    }

    // Cannot happen. Just added to please the controll flow analysis.
    return generate_random_array;
}