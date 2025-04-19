#include "Logger.hpp"
#include "TpcHCommons.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <ranges>
#include <span>
#include <thread>

template<class T>
[[gnu::always_inline]] inline char *
byte_cast(T *in) {
    return reinterpret_cast<char *>(in);  //NOSONAR
}

void
store_binary(std::span<char> data, const std::string &path) {
    std::basic_ofstream<char> output{path, std::ios::binary};
    std::ranges::copy(data, std::ostreambuf_iterator<char>(output));
}

void
store_size(const std::string &path, uint64_t size) {
    std::ofstream size_file{path + "/size"};
    size_file << size;
    size_file.close();
}

void
store_orders(const OrdersTable &o, int scale_factor) {
    if (o.numTuples == 0) {
        logger(INFO, "Skipping Orders");
        return;
    }

    logger(INFO, "Writing Orders Table");

    auto path = getPath(scale_factor, ORDERS_TBL) + ".dir";
    std::filesystem::create_directory(path);
    store_size(path, o.numTuples);

    if (o.o_orderkey != nullptr) {
        store_binary({byte_cast(o.o_orderkey), o.numTuples * sizeof(tuple_t)},
                     path + "/o_orderkey.bin");
    }
    if (o.o_custkey != nullptr) {
        store_binary({byte_cast(o.o_custkey), o.numTuples * sizeof(type_key)},
                     path + "/o_custkey.bin");
    }
    if (o.o_orderdate != nullptr) {
        store_binary({byte_cast(o.o_orderdate), o.numTuples * sizeof(uint64_t)},
                     path + "/o_orderdate.bin");
    }

    logger(INFO, "Done");
}

void
store_customers(const CustomerTable &c, int scale_factor) {
    if (c.numTuples == 0) {
        logger(INFO, "Skipping Customers");
        return;
    }

    logger(INFO, "Writing Customers Table");

    auto path = getPath(scale_factor, CUSTOMER_TBL) + ".dir";
    std::filesystem::create_directory(path);
    store_size(path, c.numTuples);

    if (c.c_custkey != nullptr) {
        store_binary({byte_cast(c.c_custkey), c.numTuples * sizeof(tuple_t)},
                     path + "/c_custkey.bin");
    }
    if (c.c_mktsegment != nullptr) {
        store_binary({byte_cast(c.c_mktsegment), c.numTuples * sizeof(uint8_t)},
                     path + "/c_mktsegment.bin");
    }
    if (c.c_nationkey != nullptr) {
        store_binary({byte_cast(c.c_nationkey), c.numTuples * sizeof(type_key)},
                     path + "/c_nationkey.bin");
    }

    logger(INFO, "Done");
}

void
store_parts(const PartTable &p, int scale_factor) {
    if (p.numTuples == 0) {
        logger(INFO, "Skipping Parts");
        return;
    }

    logger(INFO, "Writing Parts Table");

    auto path = getPath(scale_factor, PART_TBL) + ".dir";
    std::filesystem::create_directory(path);
    store_size(path, p.numTuples);

    if (p.p_partkey != nullptr) {
        store_binary({byte_cast(p.p_partkey), p.numTuples * sizeof(tuple_t)},
                     path + "/p_partkey.bin");
    }
    if (p.p_brand != nullptr) {
        store_binary({byte_cast(p.p_brand), p.numTuples * sizeof(uint8_t)}, path + "/p_brand.bin");
    }
    if (p.p_container != nullptr) {
        store_binary({byte_cast(p.p_container), p.numTuples * sizeof(uint8_t)},
                     path + "/p_container.bin");
    }
    if (p.p_size != nullptr) {
        store_binary({byte_cast(p.p_size), p.numTuples * sizeof(uint32_t)}, path + "/p_size.bin");
    }

    logger(INFO, "Done");
}

void
store_nations(const NationTable &n, int scale_factor) {
    if (n.numTuples == 0) {
        logger(INFO, "Skipping Nations");
        return;
    }

    logger(INFO, "Writing Nations Table");

    auto path = getPath(scale_factor, NATION_TBL) + ".dir";
    std::filesystem::create_directory(path);
    store_size(path, n.numTuples);

    if (n.n_nationkey != nullptr) {
        store_binary({byte_cast(n.n_nationkey), n.numTuples * sizeof(tuple_t)},
                     path + "/n_nationkey.bin");
    }

    logger(INFO, "Done");
}

void
store_lineitem(const LineItemTable &l, int scale_factor) {
    if (l.numTuples == 0) {
        logger(INFO, "Skipping LineItems");
        return;
    }

    logger(INFO, "Writing Lineitem Table");

    auto path = getPath(scale_factor, LINEITEM_TBL) + ".dir";
    std::filesystem::create_directory(path);
    store_size(path, l.numTuples);

    if (l.l_orderkey != nullptr) {
        store_binary({byte_cast(l.l_orderkey), l.numTuples * sizeof(tuple_t)},
                     path + "/l_orderkey.bin");
    }
    if (l.l_shipmode != nullptr) {
        store_binary({byte_cast(l.l_shipmode), l.numTuples * sizeof(uint8_t)},
                     path + "/l_shipmode.bin");
    }
    if (l.l_shipinstruct != nullptr) {
        store_binary({byte_cast(l.l_shipinstruct), l.numTuples * sizeof(uint8_t)},
                     path + "/l_shipinstruct.bin");
    }
    if (l.l_returnflag != nullptr) {
        store_binary({byte_cast(l.l_returnflag), l.numTuples * sizeof(char)},
                     path + "/l_returnflag.bin");
    }
    if (l.l_quantity != nullptr) {
        store_binary({byte_cast(l.l_quantity), l.numTuples * sizeof(float)},
                     path + "/l_quantity.bin");
    }
    if (l.l_partkey != nullptr) {
        store_binary({byte_cast(l.l_partkey), l.numTuples * sizeof(type_key)},
                     path + "/l_partkey.bin");
    }
    if (l.l_commitdate != nullptr) {
        store_binary({byte_cast(l.l_commitdate), l.numTuples * sizeof(uint64_t)},
                     path + "/l_commitdate.bin");
    }
    if (l.l_shipdate != nullptr) {
        store_binary({byte_cast(l.l_shipdate), l.numTuples * sizeof(uint64_t)},
                     path + "/l_shipdate.bin");
    }
    if (l.l_receiptdate != nullptr) {
        store_binary({byte_cast(l.l_receiptdate), l.numTuples * sizeof(uint64_t)},
                     path + "/l_receiptdate.bin");
    }

    logger(INFO, "Done");
}

int
main(int argc, char *argv[]) {
    initLogger();
    logger(INFO, "************* CSV To Binary Converter *************");

    tcph_args_t params;
    tpch_parse_args(argc, argv, &params);

    logger(INFO, "Transforming columns for query %d scale factor %d", params.query, params.scale);

    std::jthread lineitem_thread{[&params] {
        LineItemTable l{};
        load_lineitem_from_csv(&l, params.scale);
        store_lineitem(l, params.scale);
        free_lineitem(&l);
    }};

    if (!params.parallel) {
        lineitem_thread.join();
    }

    std::jthread orders_thread {[&params] {
        OrdersTable o{};
        load_orders_from_csv(&o, params.scale);
        store_orders(o, params.scale);
        free_orders(&o);
    }};

    if (!params.parallel) {
        orders_thread.join();
    }

    std::jthread customer_thread {[&params] {
        CustomerTable c{};
        load_customer_from_csv(&c, params.scale);
        store_customers(c, params.scale);
        free_customer(&c);
    }};

    if (!params.parallel) {
        customer_thread.join();
    }

    std::jthread part_thread{[&params] {
        PartTable p{};
        load_part_from_csv(&p, params.scale);
        store_parts(p, params.scale);
        free_part(&p);
    }};

    if (!params.parallel) {
        part_thread.join();
    }

    std::jthread nation_thread{[&params] {
        NationTable n{};
        load_nation_from_csv(&n, params.scale);
        store_nations(n, params.scale);
        free_nation(&n);
    }};

    if (!params.parallel) {
        nation_thread.join();
    }
}