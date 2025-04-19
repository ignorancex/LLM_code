#include <cstdarg>
#include <cstdio>
#include <cstring>
#include "EnclavePrint_t.h"
#include "EnclavePrint.hpp"

int printf(const char* fmt, ...)
{
    char buf[BUFSIZ] = { '\0' };
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    ocall_print_string(buf);
    return (int)strnlen(buf, BUFSIZ - 1) + 1;
}
