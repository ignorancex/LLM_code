#include "CrkJoin/JoinWrapper.hpp"
#include "cht/CHTJoinWrapper.hpp"
#include "mway/sortmergejoin_multiway.h"
#include "nl/nested_loop_join.h"
#include "npj/no_partitioning_bucket_chaining_join.hpp"
#include "npj/no_partitioning_hash_join.hpp"
#include "npj/no_partitioning_hash_join_st.hpp"
#include "psm/parallel_sortmerge_join.h"
#include "radix/radix_join.h"
#include "radix/radix_sortmerge_join.hpp"
#include "data-types.h"
#include "Logger.hpp"

#ifndef ENCLAVE
#include "ocalls.hpp"
#else
#include "ocalls_t.h"
#endif

#include "joins.hpp"
#include <cstring>

result_t *
CHT(const table_t *relR, const table_t *relS, const joinconfig_t *config) {
    join_result_t join_result = CHTJ<7>(relR, relS, config);
    result_t *joinresult = (result_t *) malloc(sizeof(result_t));
    joinresult->totalresults = join_result.matches;
    joinresult->nthreads = config->NTHREADS;
    joinresult->materialized = 0;
    return joinresult;
}

const static algorithm_t sgx_algorithms[] = {
        {"PHT",     PHT},
        {"PHT_no",  PHT_no_overflow},
        {"PHT_un",  PHT_unrolled},
        {"PHT_o",   PHT_overflow},
        {"NPO_st",  NPO_single_thread},
        {"NPO_no",  NPO_no_overflow},
        {"NL",      NL},
        {"INL",     INL},
        {"RHO",     RHO},
        {"RHT",     RHT},
        {"PSM",     PSM},
        {"RSM",     RSM},
        {"CHT",     CHT},
        {"MWAY",    MWAY},
        {"CRKJ",    CRKJ},
        {"CrkJoin", CRKJ},
        {"CRKJF",   CRKJF},
        {"CRKJS",   CRKJS},
        {"NPBC_st", NPBC_st},
};

void
run_join(result_t *res, const struct table_t *relR, const struct table_t *relS, const char *algorithm_name,
           const struct joinconfig_t *config) {
    int i = 0;
    int found = 0;
    const algorithm_t *algorithm = nullptr;
    while (sgx_algorithms[i].join) {
        if (strcmp(algorithm_name, sgx_algorithms[i].name) == 0) {
            found = 1;
            algorithm = &sgx_algorithms[i];
            break;
        }
        i++;
    }

    if (found == 0) {
        logger(ERROR, "Algorithm not found: %s", algorithm_name);
        ocall_exit(EXIT_FAILURE);
    }
    result_t *tmp = algorithm->join(relR, relS, config);
    if (tmp) {
        memcpy(res, tmp, sizeof(result_t));
    }
}