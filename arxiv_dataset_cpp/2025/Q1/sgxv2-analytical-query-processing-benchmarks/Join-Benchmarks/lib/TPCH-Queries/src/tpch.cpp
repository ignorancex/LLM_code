#include "Q10Predicates.hpp"
#include "Q12Predicates.hpp"
#include "Q19Predicates.hpp"
#include "Q3Predicates.hpp"
#include "filters.hpp"

#include "result_transformers.hpp"
#include "time_print.hpp"

#include "Logger.hpp"
#include "joins.hpp"
#include "rdtscpWrapper.h"

#include "ChunkedTable.hpp"
#include "tpch.hpp"
#include <algorithm>

bool
compare_tables(const table_t &a, const table_t &b) {
    if (a.num_tuples != b.num_tuples) {
        logger(ERROR, "Table length unequal");
        return false;
    }
    bool equal = std::equal(a.tuples, a.tuples + a.num_tuples, b.tuples,
                      [](const row_t &a, const row_t &b) { return a.key == b.key && a.payload == b.payload; });

    if (!equal) {
        logger(ERROR, "Table content unequal");
    } else {
        logger(INFO, "Table content equal");
    }

    return equal;
}

void
tpch_q3(result_t *result, const struct CustomerTable *c, const struct OrdersTable *o, const struct LineItemTable *l,
        const char *algorithm, struct joinconfig_t *config) {
    // TODO: copy tables to enclave before starting benchmark
    logger(INFO, "%s", __FUNCTION__);
    logger(INFO, "LineItemTable size: %u, OrdersTable size: %u, CustomerTable size: %u", l->numTuples, o->numTuples,
           c->numTuples);
    TPCHTimers t{};
    auto timer_start = rdtscp_s();

    //selection
#ifndef SIMD
    table_t customers_filtered = filter_table<CustomerTable, q3CustomerPredicate, q3CustomerCopy>(*c);
    auto timer_selections_1 = rdtscp_s();
    table_t orders_filtered = filter_table<OrdersTable, q3OrdersPredicate, q3OrderCopy>(*o);
#else
    table_t customers_filtered = parallel_filter<const row_t *, const uint8_t *>(
            config->NTHREADS, q3_filter_customer_simd, c->numTuples, c->c_custkey, c->c_mktsegment);
    auto timer_selections_1 = rdtscp_s();
    table_t orders_filtered = parallel_filter<const type_key *, const row_t *, const uint64_t *>(
            config->NTHREADS, q3_filter_orders_simd, o->numTuples, o->o_custkey, o->o_orderkey, o->o_orderdate);
#endif
    auto timer_selections_2 = rdtscp_s();
    t.selection_1 = timer_selections_1 - timer_start;
    t.selection_2 = timer_selections_2 - timer_selections_1;


    //join customers and orders
    logger(INFO, "Join customers=%u with orders=%u", customers_filtered.num_tuples, orders_filtered.num_tuples);
    result_t o_c_join_result;
    config->MATERIALIZE = true;
    // TODO: make sure that all required joins correctly return their results. Done: CrkJoin, RHO, PHT
    run_join(&o_c_join_result, &customers_filtered, &orders_filtered, algorithm, config);
    auto timer_join1_end = rdtscp_s();
    t.join_1 = (timer_join1_end - timer_selections_2);
    logger(INFO, "Join 1 timer (cycles) : %lu", t.join_1);
    free(customers_filtered.tuples);
    free(orders_filtered.tuples);

    //transform joinResult to relation_t
    table_t o_c_joined_table{};
    if (o_c_join_result.result_type == 0) {
        joinResultToTableST(&o_c_joined_table, &o_c_join_result, toTableThread_SpSpST);
    } else {
        chunked_to_solid_table<copy_Sp_Sp>(&o_c_joined_table,
                                           reinterpret_cast<chunked_table_t *>(o_c_join_result.result));
        destroy_table(reinterpret_cast<chunked_table_t *>(o_c_join_result.result));
    }
    logger(INFO, "U tuples=%u", o_c_joined_table.num_tuples);
    auto timer_copy_1_end = rdtscp_s();
    t.copy += timer_copy_1_end - timer_join1_end;

    // selection 2
#ifndef SIMD
    table_t lineitem_filtered = filter_table<LineItemTable, q3LineitemPredicate, q3LineitemCopy>(*l);
#else
    table_t lineitem_filtered = parallel_filter<const row_t *, const uint64_t *>(
            config->NTHREADS, q3_filter_lineitem_simd, l->numTuples, l->l_orderkey, l->l_shipdate);
#endif
    auto timer_selection_3_end = rdtscp_s();
    t.selection_3 = timer_selection_3_end - timer_copy_1_end;

    //join with lineitems
    logger(INFO, "Join U=%u with lineitems=%u", o_c_joined_table.num_tuples, lineitem_filtered.num_tuples);
    config->MATERIALIZE = false;
    run_join(result, &o_c_joined_table, &lineitem_filtered, algorithm, config);
    auto timerEnd = rdtscp_s();
    t.join_2 += (timerEnd - timer_selection_3_end);
    logger(INFO, "Join 2 timer (cycles) : %lu", (t.join_2));

    //clean up
    free(lineitem_filtered.tuples);
    free(o_c_joined_table.tuples);

    // print
    uint64_t numTuples = (l->numTuples + o->numTuples + c->numTuples);
    t.total = timerEnd - timer_start;
    print_query_results(timers_to_us(t), numTuples);
    result->throughput = static_cast<double>(numTuples) / static_cast<double>(t.total);
}

void
tpch_q10(result_t *result, const CustomerTable *c, const OrdersTable *o, const LineItemTable *l, const NationTable *n,
         const char *algorithm, joinconfig_t *config) {
    logger(INFO, "%s", __FUNCTION__);
    logger(INFO, "CustomerTable: %u, OrdersTable size: %u, LineItemTable size: %u, NationTable: %u", c->numTuples,
           o->numTuples, l->numTuples, n->numTuples);
    TPCHTimers t{};
    auto timer_start = rdtscp_s();

    //filter customer
    table_t customer_table{c->c_custkey, c->numTuples, 0, 0};
#ifndef SIMD
    table_t filtered_orders = filter_table<OrdersTable, q10OrdersPredicate, q10OrderCopy>(*o);
#else
    table_t filtered_orders = parallel_filter<const type_key *, const row_t *, const uint64_t *>(
            config->NTHREADS, q10_filter_order_simd, o->numTuples, o->o_custkey, o->o_orderkey, o->o_orderdate);
#endif
    auto timer_selection_1_end = rdtscp_s();
    t.selection_1 = timer_selection_1_end - timer_start;

    //join orders and customer
    logger(INFO, "Join Customer=%u with Orders=%u", customer_table.num_tuples, filtered_orders.num_tuples);
    config->MATERIALIZE = true;
    result_t joinResult;
    run_join(&joinResult, &customer_table, &filtered_orders, algorithm, config);
    auto timer_join_1_end = rdtscp_s();
    t.join_1 = timer_join_1_end - timer_selection_1_end;
    logger(INFO, "Join 1 timer : %lu", t.join_1);
    free(filtered_orders.tuples);

    //transform joinResult to relation_t
    table_t c_o_joined{};
    if (joinResult.result_type == 0) {
        joinResultToTableST(&c_o_joined, &joinResult, toTableThread_RpToKeySpST, c->c_nationkey);
    } else {
        auto chunked_table = reinterpret_cast<chunked_table_t *>(joinResult.result);
        logger(INFO, "Join result tuples: %d", joinResult.totalresults);
        logger(INFO, "Chunked Table Tuples: %d", chunked_table->num_tuples);
        chunked_to_solid_table<type_key, copy_RpToKeySp>(&c_o_joined, chunked_table, c->c_nationkey);
        destroy_table(chunked_table);
    }
    logger(INFO, "Solid table tuples=%u", c_o_joined.num_tuples);
    auto timer_copy_1_end = rdtscp_s();
    t.copy += timer_copy_1_end - timer_join_1_end;

    //join nation and previous
    table_t nation_table{n->n_nationkey, n->numTuples, 0, 0};
    config->MATERIALIZE = true;
    logger(INFO, "Join Nation=%u with U=%u", n->numTuples, c_o_joined.num_tuples);
    result_t join_result_2;
    run_join(&join_result_2, &nation_table, &c_o_joined, algorithm, config);
    auto timer_join_2_end = rdtscp_s();
    t.join_2 += (timer_join_2_end - timer_copy_1_end);
    logger(INFO, "Join 2 timer (cycles) : %lu", t.join_2);
    free(c_o_joined.tuples);

    //transform joinResult to relation_t
    table_t c_o_n_joined{};
    if (join_result_2.result_type == 0) {
        joinResultToTableST(&c_o_n_joined, &join_result_2, toTableThread_SpToTupleST, o->o_orderkey, o->numTuples);
    } else {
        auto chunked_table = reinterpret_cast<chunked_table_t *>(join_result_2.result);
        logger(INFO, "Join result tuples: %d", join_result_2.totalresults);
        logger(INFO, "Chunked Table Tuples: %d", chunked_table->num_tuples);
        chunked_to_solid_table<row_t, copy_SpToTupleST>(
                &c_o_n_joined, chunked_table, o->o_orderkey);
        destroy_table(chunked_table);
    }
    logger(INFO, "Solid table tuples=%u", c_o_n_joined.num_tuples);
    auto timer_copy_2_end = rdtscp_s();
    t.copy += timer_copy_2_end - timer_join_2_end;

    //filter lineitem
#ifndef SIMD
    table_t lineitem_filtered = filter_table<LineItemTable, q10LineItemPredicate, q10LineItemCopy>(*l);
#else
    table_t lineitem_filtered = parallel_filter<const row_t *, const char *>(
            config->NTHREADS, q10_filter_lineitem_simd, l->numTuples, l->l_orderkey, l->l_returnflag);
#endif
    auto timer_selection_2_end = rdtscp_s();
    t.selection_2 = timer_selection_2_end - timer_copy_2_end;

    //join lineitem and previous
    logger(INFO, "Join U=%u with LineItem=%u", c_o_n_joined.num_tuples, lineitem_filtered.num_tuples);
    config->MATERIALIZE = false;
    run_join(result, &c_o_n_joined, &lineitem_filtered, algorithm, config);
    auto timer_join_3_end = rdtscp_s();
    t.join_3 += (timer_join_3_end - timer_selection_2_end);
    logger(INFO, "Join 3 timer : %lu", t.join_3);
    logger(INFO, "Join result tuples: %d", result->totalresults);

    // clean + print
    free(c_o_n_joined.tuples);
    free(lineitem_filtered.tuples);

    uint64_t numTuples = (l->numTuples + o->numTuples + c->numTuples + n->numTuples);
    t.total = timer_join_3_end - timer_start;
    print_query_results(timers_to_us(t), numTuples);
    result->throughput = static_cast<double>(numTuples) / static_cast<double>(t.total);
}

void
tpch_q12(result_t *result, const LineItemTable *l, const OrdersTable *o, const char *algorithm, joinconfig_t *config) {
    logger(INFO, "%s", __FUNCTION__);
    logger(INFO, "LineItemTable size: %u, OrdersTable size: %u", l->numTuples, o->numTuples);
    TPCHTimers t{};
    auto timer_start = rdtscp_s();

    // selection 1
    table_t order_table{o->o_orderkey, o->numTuples, 0, 0};
#ifndef SIMD
    table_t lineitem_filtered = filter_table<LineItemTable, q12Predicate, q12Copy>(*l);
#else
    table_t lineitem_filtered =
            parallel_filter<const row_t *, const uint8_t *, const uint64_t *, const uint64_t *, const uint64_t *>(
                    config->NTHREADS, q12_filter_lineitem, l->numTuples, l->l_orderkey, l->l_shipmode, l->l_commitdate,
                    l->l_shipdate, l->l_receiptdate);
#endif
    auto timer_selection_1_end = rdtscp_s();
    t.selection_1 = timer_selection_1_end - timer_start;

    // join l and o
    logger(INFO, "Join lineitem=%u with order=%u tuples", lineitem_filtered.num_tuples, order_table.num_tuples);
    config->MATERIALIZE = false;
    run_join(result, &order_table, &lineitem_filtered, algorithm, config);
    auto timer_join_1_end = rdtscp_s();
    t.join_1 = timer_join_1_end - timer_selection_1_end;
    logger(INFO, "Join 1 timer : %lu", t.join_1);

    // clean up + print
    free(lineitem_filtered.tuples);
    uint64_t numTuples = (l->numTuples + o->numTuples);
    t.total = timer_join_1_end - timer_start;
    print_query_results(timers_to_us(t), numTuples);
    result->throughput = static_cast<double>(numTuples) / static_cast<double>(t.total);
}

void
tpch_q19(result_t *result, const LineItemTable *l, const PartTable *p, const char *algorithm, joinconfig_t *config) {
    logger(INFO, "%s", __FUNCTION__);
    logger(INFO, "LineItemTable size: %u, PartTable size: %u", l->numTuples, p->numTuples);
    TPCHTimers t{};
    auto timer_start = rdtscp_s();

    //selection
#ifndef SIMD
    table_t part_filtered = filter_table<PartTable, q19PartPredicate, q19PartCopy>(*p);
    auto timer_selection_1_end = rdtscp_s();
    table_t lineitem_filtered = filter_table<LineItemTable, q19LineItemPredicate, q19LineItemCopy>(*l);
#else
    table_t part_filtered = parallel_filter<const row_t *, const uint32_t *, const uint8_t *, const uint8_t *>(
            config->NTHREADS, q19_part_filter, p->numTuples, p->p_partkey, p->p_size, p->p_brand, p->p_container);
    auto timer_selection_1_end = rdtscp_s();
    table_t lineitem_filtered =
            parallel_filter<const row_t *, const type_key *, const float *, const uint8_t *, const uint8_t *>(
                    config->NTHREADS, q19_lineitem_filter, l->numTuples, l->l_orderkey, l->l_partkey, l->l_quantity,
                    l->l_shipmode, l->l_shipinstruct);
#endif
    auto timer_selection_2_end = rdtscp_s();
    t.selection_1 = timer_selection_1_end - timer_start;
    t.selection_2 = timer_selection_2_end - timer_selection_1_end;

    //join 1
    logger(INFO, "Join Part=%u with LineItem=%u", part_filtered.num_tuples, lineitem_filtered.num_tuples);
    config->MATERIALIZE = true;
    run_join(result, &part_filtered, &lineitem_filtered, algorithm, config);
    auto timer_join_1_end = rdtscp_s();
    t.join_1 = timer_join_1_end - timer_selection_2_end;
    logger(INFO, "Join 1 timer : %lu", t.join_1);

    //filter the join results
    uint64_t matches;
    if (result->result_type == 0) {
        matches = q19FilterJoinResults(result, l, p);
    } else {
        auto chunked_table = reinterpret_cast<chunked_table_t *>(result->result);
        logger(WARN, "Number of Tuples: %d", chunked_table->num_tuples);
        logger(WARN, "Number of Chunks: %d", chunked_table->num_chunks);
        matches = q19FilterJoinResultsChunked(result, l, p);
    }
    auto timer_selection_3_end = rdtscp_s();
    t.selection_3 = timer_selection_3_end - timer_join_1_end;
    logger(INFO, "Total matches = %u", matches);
    uint64_t timerEnd = timer_selection_3_end;

    // clean + print
    free(part_filtered.tuples);
    free(lineitem_filtered.tuples);
    uint64_t numTuples = (l->numTuples + p->numTuples);
    t.total = timerEnd - timer_start;
    print_query_results(timers_to_us(t), numTuples);
    result->throughput = static_cast<double>(numTuples) / static_cast<double>(t.total);
}
