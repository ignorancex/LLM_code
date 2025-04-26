#include "types.hpp"

[[nodiscard]] std::vector<bool> convert_trinary(Trinary t) {
    std::vector<bool> result{};
    if (t == Trinary::f) {
        result.push_back(false);
    } else if (t == Trinary::t) {
        result.push_back(true);
    } else {
        result.push_back(false);
        result.push_back(true);
    }
    return result;
}

[[nodiscard]] Trinary from_bool(bool b) {
    if (b) {
        return Trinary::t;
    } else {
        return Trinary::f;
    }
}

Trinary from_string(const std::string &value) {
    if (value == "t") return Trinary::t;
    if (value == "f") return Trinary::f;
    if (value == "b") return Trinary::b;
    throw std::invalid_argument("Tried to create Trinary from string " + value);
}
