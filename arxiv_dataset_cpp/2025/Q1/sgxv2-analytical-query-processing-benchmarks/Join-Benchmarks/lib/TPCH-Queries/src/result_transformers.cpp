#include "result_transformers.hpp"
#include "Logger.hpp"
#include "util.hpp"
#include <cstdlib>
#include <memory>
#include <vector>
#include "pthread.h"

#ifndef ENCLAVE
#include "ocalls.hpp"
#else
#include "ocalls_t.h"
#endif

struct JrToTableArg {
    uint32_t tid;
    output_list_t *threadresult;
    uint64_t nresults;
    tuple_t *out;
    const type_key *keyToExtract;

    //for ST
    int nthreads;
    struct threadresult_t *resultlist;
};

/**
 * Constructs a table that uses param->keyToExtract to translate Rpayload as new key and copies Spayload as payload
 * @param param JrToTableArg*
 * @return nullptr
 */
void *
toTableThread_RpToKeySpST(void *param) {
    auto arg = static_cast<JrToTableArg *>(param);
    int nthreads = arg->nthreads;
    uint64_t output_index = 0;
    for (int thread_index = 0; thread_index < nthreads; thread_index++) {
        output_list_t *tlist = arg->resultlist[thread_index].results;
        uint64_t num_thread_results = arg->resultlist[thread_index].nresults;
        for (uint64_t thread_result_index = 0; thread_result_index < num_thread_results; thread_result_index++) {
            if (tlist == nullptr) {
                logger(ERROR, "Error retrieving join tuple");
            }
            arg->out[output_index].key = arg->keyToExtract[tlist->Rpayload];
            arg->out[output_index].payload = tlist->Spayload;
            output_index++;
            tlist = tlist->next;
        }
    }
    return nullptr;
}

/**
 * Constructs a table that uses param->keyToExtract to translate Rpayload as new key and copies Spayload as payload.
 * Can be run multi-threaded
 * @param param JrToTableArg*
 * @return nullptr
 */
void *
toTableThread_RpToKeySp(void *param) {
    auto *arg = static_cast<JrToTableArg *>(param);
    output_list_t *tmp = arg->threadresult;
    for (uint64_t i = 0; i < arg->nresults; i++) {
        if (tmp == nullptr) {
            logger(ERROR, "Error retrieving join tuple");
        }
        arg->out[i].key = arg->keyToExtract[tmp->Rpayload];
        arg->out[i].payload = tmp->Spayload;
        tmp = tmp->next;
    }
    return nullptr;
}

/**
 * Constructs a table that uses param->keyToExtract to translate Spayload as new key. Payload is left empty.
 * @param param JrToTableArg*
 * @return nullptr
 */
void *
toTableThread_SpToTupleST(void *param) {
    auto arg = static_cast<JrToTableArg *>(param);
    int nthreads = arg->nthreads;
    uint64_t i = 0;
    for (int j = 0; j < nthreads; j++) {
        output_list_t *tlist = arg->resultlist[j].results;
        uint64_t tresults = arg->resultlist[j].nresults;
        for (uint64_t k = 0; k < tresults; k++) {
            if (tlist == nullptr) {
                logger(ERROR, "Error retrieving join tuple");
            }
            arg->out[i].key = arg->keyToExtract[tlist->Spayload];
            i++;
            tlist = tlist->next;
        }
    }
    return nullptr;
}

/**
 * Constructs a table that uses param->keyToExtract to translate Spayload as new key. Payload is left empty.
 * Can be run multi-threaded
 * @param param JrToTableArg*
 * @return nullptr
 */
void *
toTableThread_SpToTuple(void *param) {
    auto *arg = static_cast<JrToTableArg *>(param);
    output_list_t *tmp = arg->threadresult;
    for (uint64_t i = 0; i < arg->nresults; i++) {
        if (tmp == nullptr) {
            logger(ERROR, "Error retrieving join tuple");
        }
        arg->out[i].key = arg->keyToExtract[tmp->Spayload];
        tmp = tmp->next;
    }
    return nullptr;
}

/**
 * Constructs a table that contains the payload of S (the right table in the join) as key and value.
 * @param param JrToTableArg*
 * @return nullptr
 */
void *
toTableThread_SpSpST(void *param) {
    auto arg = static_cast<JrToTableArg *>(param);
    int nthreads = arg->nthreads;
    uint64_t i = 0;
    for (int j = 0; j < nthreads; j++) {
        output_list_t *tlist = arg->resultlist[j].results;
        uint64_t tresults = arg->resultlist[j].nresults;
        for (uint64_t k = 0; k < tresults; k++) {
            if (tlist == nullptr) {
                logger(ERROR, "Error retrieving join tuple");
            }
            arg->out[i].key = tlist->Spayload;
            arg->out[i].payload = tlist->Spayload;
            i++;
            tlist = tlist->next;
        }
    }
    return nullptr;
}

/**
 * Constructs a table that contains the payload of S (the right table in the join) as key and value.
 * Can be run multi-threaded
 * @param param JrToTableArg*
 * @return nullptr
 */
void *
toTableThread_SpSp(void *param) {
    auto *arg = static_cast<JrToTableArg *>(param);
    output_list_t *tmp = arg->threadresult;
    uint64_t i = 0;
    while (tmp != nullptr) {
        arg->out[i].key = tmp->Spayload;
        arg->out[i].payload = tmp->Spayload;
        tmp = tmp->next;
        i++;
    }

    //SGX_ASSERT(i == arg->nresults, "iterated over wrong number of elements");
    return nullptr;
}

/**
 * Constructs a table that contains the join key as key and Spayload as payload.
 * Can be run multi-threaded
 * @param param JrToTableArg*
 * @return nullptr
 */
void *
toTableThread_RkSp(void *param) {
    auto *arg = static_cast<JrToTableArg *>(param);
    output_list_t *tmp = arg->threadresult;
    for (uint64_t i = 0; i < arg->nresults; i++) {
        if (tmp == nullptr) {
            logger(ERROR, "Error retrieving join tuple");
        }
        arg->out[i].key = tmp->key;
        arg->out[i].payload = tmp->Spayload;
        tmp = tmp->next;
    }
    return nullptr;
}

/**
 * Constructs table using the ToTableThread function and keyToExtract if given.
 * @param table [out] written to
 * @param jr
 * @param t_thread
 * @param keyToExtract
 */
void
joinResultToTableST(table_t *table, const result_t *jr, ToTableThread t_thread,
                    const type_key *keyToExtract) {
    if (!jr->materialized) {
        logger(ERROR, "Trying to build table out of non-materialized join result!");
        ocall_exit(EXIT_FAILURE);
    }
    if (jr->result_type != 0) {
        logger(ERROR, "Trying to build table out wrong result type. Is %d expected 0!", jr->result_type);
        ocall_exit(EXIT_FAILURE);
    }
    table->tuples = (tuple_t *) malloc(sizeof(tuple_t) * jr->totalresults);
    malloc_check(table->tuples);
    table->num_tuples = jr->totalresults;
    auto args = std::make_unique<JrToTableArg>();
    args->tid = 0;
    args->nthreads = jr->nthreads;
    args->resultlist = (threadresult_t *) jr->result;
    args->out = table->tuples;
    args->keyToExtract = keyToExtract;
    t_thread(args.get());
}

/**
 * Constructs table using the ToTableThread function and keyToExtract if given. Uses jr->nthreads threads.
 * @param table [out] written to
 * @param jr
 * @param t_thread
 * @param keyToExtract
 */
void
joinResultToTable(table_t *table, const result_t *jr, ToTableThread t_thread, const type_key *keyToExtract) {
    if (!jr->materialized) {
        logger(ERROR, "Trying to build table out of non-materialized join result!");
        ocall_exit(EXIT_FAILURE);
    }
    table->tuples = (tuple_t *) malloc(sizeof(tuple_t) * jr->totalresults);
    malloc_check(table->tuples);
    table->num_tuples = jr->totalresults;
    int nthreads = jr->nthreads;
    std::vector<uint64_t> offset(nthreads);
    offset[0] = 0;
    for (int i = 1; i < nthreads; i++) {
        offset[i] = offset[i - 1] + ((threadresult_t *) jr->result)[i - 1].nresults;
    }

    std::vector<JrToTableArg> args(nthreads);
    std::vector<pthread_t> tid(nthreads);
    int rv;
    for (int i = 0; i < jr->nthreads; i++) {
        auto thread_result = ((threadresult_t *) jr->result)[i];
        args[i].tid = thread_result.threadid;
        args[i].threadresult = thread_result.results;
        args[i].nresults = thread_result.nresults;
        args[i].out = table->tuples + offset[i];
        args[i].keyToExtract = keyToExtract;
        rv = pthread_create(&tid[i], nullptr, t_thread, (void *) &args[i]);
        if (rv) {
            logger(ERROR, "return code from pthread_create() is %d\n", rv);
        }
    }
    for (int i = 0; i < nthreads; i++) {
        pthread_join(tid[i], nullptr);
    }
}

/**
 * Constructs table using the ToTableThread function and keyToExtract if given. Extracts keyToExtract for
 * joinResultToTableST from the tuples in the parameter keyToExtract.
 * @param table [out] written to
 * @param jr
 * @param t_thread
 * @param keyToExtract
 */
void
joinResultToTableST(table_t *table, const result_t *jr, ToTableThread t_thread, const tuple_t *keyToExtract,
                    uint64_t keySize) {
    auto *tmp = (type_key *) malloc(sizeof(type_key) * keySize);
    for (uint64_t i = 0; i < keySize; i++) {
        tmp[i] = keyToExtract[i].key;
    }
    joinResultToTableST(table, jr, t_thread, tmp);
    free(tmp);
}

/**
 * Constructs table using the ToTableThread function and keyToExtract if given. Uses jr->nthreads threads.
 * Extracts keyToExtract for joinResultToTableST from the tuples in the parameter keyToExtract.
 * @param table [out] written to
 * @param jr
 * @param t_thread
 * @param keyToExtract
 */
void
joinResultToTable(table_t *table, const result_t *jr, ToTableThread t_thread, const tuple_t *keyToExtract,
                  uint64_t keySize) {
    auto *tmp = (type_key *) malloc(sizeof(type_key) * keySize);
    for (uint64_t i = 0; i < keySize; i++) {
        tmp[i] = keyToExtract[i].key;
    }
    joinResultToTable(table, jr, t_thread, tmp);
    free(tmp);
}
