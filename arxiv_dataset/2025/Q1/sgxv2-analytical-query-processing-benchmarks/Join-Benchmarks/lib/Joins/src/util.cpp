#include "util.hpp"
#include "Logger.hpp"
#include "data-types.h"
#include <cstdlib>

#ifdef ENCLAVE
#include "ocalls_t.h"
#else
#include "ocalls.hpp"
#endif

void malloc_check_ex(void * ptr, char const * file, char const * function, int line)
{
    if (ptr == 0)
    {
        logger(ERROR, "%s:%s:%d Failed to allocate memory", file, function, line);
        ocall_exit(EXIT_FAILURE);
    }
}

void insert_output(output_list_t ** head, type_key key, type_value Rpayload, type_value Spayload)
{
    output_list_t * row = (output_list_t *) malloc(sizeof(output_list_t));
    malloc_check_ex(row, __FILE__, __FUNCTION__, __LINE__);
    row->key = key;
    row->Rpayload = Rpayload;
    row->Spayload = Spayload;
    row->next = *head;
    *head = row;
}
