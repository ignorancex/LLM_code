// Copyright 2023 Dynatrace LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Portions of this code, as identified in remarks, are provided under the
// Creative Commons BY-SA 4.0 or the MIT license, and are provided without
// any warranty. In each of the remarks, we have provided attribution to the
// original creators and other attribution parties, along with the title of
// the code (if known) a copyright notice and a link to the license, and a
// statement indicating whether or not we have modified the code.

#include "utils.h"
#include "main.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

bool is_http_socket(int fd) {
    return fd > 2 && fd < MAX_SOCKET_FD && SOCKET_PORT_MAPPING[fd] > 0;
}

bool contains_keyword(void* buf, size_t count) {
    for (int i = 0; i < WORD_LIST_LENGTH; i++) {
        if (strstr(buf, WORD_LIST[i]) != NULL) {
            write_log("buffer matches word '%s' from the word list\n", WORD_LIST[i]);
            return true;
        }
    }

    return false;
}

void write_log(const char* format, ...) {
    FILE* file = fopen(LOG_FILE, "a");
    if (file == NULL) {
        fprintf(stderr, "Could not open log file \"%s\"\n", LOG_FILE);
        return;
    }

    va_list args;
    va_start(args, format);
    vfprintf(file, format, args);
    va_end(args);

    fclose(file);
}
