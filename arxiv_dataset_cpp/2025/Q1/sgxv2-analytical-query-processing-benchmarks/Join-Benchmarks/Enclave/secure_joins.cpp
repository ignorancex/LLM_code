#include "Enclave_t.h"
#include "Logger.hpp"
#include "data-types.h"
#include "joins.hpp"
#include "rdtscpWrapper.h"
#include "util.hpp"
#include <memory>

void print_relation(const table_t *rel, uint32_t num, uint32_t offset) {
    logger(DBG, "****************** Relation sample ******************");
    for (uint32_t i = offset; i < rel->num_tuples && i < num + offset; i++) {
        logger(DBG, "%u -> %u", rel->tuples[i].key, rel->tuples[i].payload);
    }
    logger(DBG, "******************************************************");
}

extern "C" {

void
ecall_init_logger(uint64_t start) {
    initLogger(start);
}

void
ecall_join(struct result_t *res, const struct table_t *relR, const struct table_t *relS, const char *algorithm_name,
           const struct joinconfig_t *config) {
    run_join(res, relR, relS, algorithm_name, config);
}

table_t preload_relR;
table_t preload_relS;
bool preload = false;

void ecall_preload_relations(struct table_t *relR, struct table_t *relS) {
    preload_relR.num_tuples = relR->num_tuples;
    preload_relR.ratio_holes = relR->ratio_holes;
    preload_relR.sorted = relR->sorted;
    //preload_relS.tuples = new tuple_t;
    preload_relR.tuples = (tuple_t *) malloc(relR->num_tuples * sizeof(tuple_t));
    memcpy(preload_relR.tuples, relR->tuples, relR->num_tuples * sizeof(tuple_t));
    preload_relS.num_tuples = relS->num_tuples;
    preload_relS.ratio_holes = relS->ratio_holes;
    preload_relS.sorted = relS->sorted;
    //preload_relS.tuples = new tuple_t;
    preload_relS.tuples = (tuple_t *) malloc(relS->num_tuples * sizeof(tuple_t));
    memcpy(preload_relS.tuples, relS->tuples, relS->num_tuples * sizeof(tuple_t));

    preload = true;
}

void ecall_join_preload(const char *algorithm_name, const joinconfig_t *config) {
    logger(INFO, "Preload num-tuples r: %lu", preload_relR.num_tuples);
    logger(INFO, "Preload num-tuples s: %lu", preload_relS.num_tuples);
    auto res = std::make_unique<result_t>();
    if (preload)
        ecall_join(res.get(), &preload_relR, &preload_relS, algorithm_name, config);
}

}
