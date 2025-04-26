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

#include "hook.h"
#include "main.h"
#include "utils.h"

#include <arpa/inet.h>
#include <dlfcn.h>
#include <stdio.h>

int accept(int sockfd, struct sockaddr* addr, socklen_t* addrlen) {
    int fd = ((accept_t)dlsym(RTLD_NEXT, "accept"))(sockfd, addr, addrlen);

    // mark the file descriptor as traceable if it is an IPv4 or IPv6 socket with a valid port
    if (sockfd > 2 && sockfd < MAX_SOCKET_FD && fd < MAX_SOCKET_FD && addr != NULL &&
        (addr->sa_family == AF_INET || addr->sa_family == AF_INET6)) {
        struct sockaddr_in* addr_in = (struct sockaddr_in*)addr;
        unsigned short port = htons(addr_in->sin_port);
        if (port > 1) {
            SOCKET_PORT_MAPPING[fd] = SOCKET_FD_TRACEABLE;
        }
    }

    return fd;
}

ssize_t read(int fd, void* buf, size_t count) {
    read_t read_ptr = (read_t)dlsym(RTLD_NEXT, "read");
    ssize_t bytes_read = read_ptr(fd, buf, count);

    if (is_http_socket(fd)) {
        if (contains_keyword(buf, count)) {
            SOCKET_PORT_MAPPING[fd] = SOCKET_FD_TRACEABLE_AND_BLOCKED;
        }
    }

    return bytes_read;
}

int close(int fd) {
    if (is_http_socket(fd)) {
        SOCKET_PORT_MAPPING[fd] = SOCKET_FD_UNTRACEABLE;
    }

    return ((close_t)dlsym(RTLD_NEXT, "close"))(fd);
}
