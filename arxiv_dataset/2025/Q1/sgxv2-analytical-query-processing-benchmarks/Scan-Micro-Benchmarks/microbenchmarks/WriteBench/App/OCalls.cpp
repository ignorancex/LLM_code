#include "Logger.hpp"

extern "C" {

/* OCall functions */
void ocall_print_string(const char *str) {
    /* Proxy/Bridge will check the length and null-terminate
     * the input string to prevent buffer overflow.
     */
    logger(str);
}

void ocall_print_error(const char *str) {
    error(str);
}

}