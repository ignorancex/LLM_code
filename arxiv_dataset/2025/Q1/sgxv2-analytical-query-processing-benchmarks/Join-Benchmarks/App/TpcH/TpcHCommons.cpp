#include "TpcHCommons.hpp"
#include "Logger.hpp"
#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <istream>
#include <sstream>
#include <string>

#include "csv.hpp"

#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

class CSVRow {
public:
    std::string m_line;

    std::string_view operator[](std::size_t index) const {
        return std::string_view(&m_line[m_data[index] + 1], m_data[index + 1] - (m_data[index] + 1));
    }

    std::size_t size() const {
        return m_data.size() - 1;
    }

    void readNextRow(std::istream &str) {
        std::getline(str, m_line);

        m_data.clear();
        m_data.emplace_back(-1);
        std::string::size_type pos = 0;
        while ((pos = m_line.find('|', pos)) != std::string::npos) {
            m_data.emplace_back(pos);
            ++pos;
        }
        // This checks for a trailing comma with no data after it.
        pos = m_line.size();
        m_data.emplace_back(pos);
    }

private:
    std::vector<int> m_data;
};

std::istream &operator>>(std::istream &str, CSVRow &data) {
    data.readNextRow(str);
    return str;
}

template<typename T>
void
vton(const std::string_view &view, T &out) {
    std::from_chars(view.data(), view.data() + view.size(), out);
}

void
vtof(const std::string_view &view, float &out) {
    std::from_chars(view.data(), view.data() + view.size(), out, std::chars_format::general);
}

template<typename T>
void
vton_debug(const std::string_view &view, T &out) {
    auto ret = std::from_chars(view.data(), view.data() + view.size(), out);
    if (ret.ec == std::errc::invalid_argument) {
        throw std::runtime_error{"vton failed with invalid argument"};
    } else if (ret.ec == std::errc::result_out_of_range) {
        throw std::runtime_error{"vton failed with out of range"};
    } else if (ret.ptr != view.data() + view.size()) {
        throw std::runtime_error{"vton was unable to parse whole range"};
    }
}

uint8_t
parsePartBrand(const std::string &value);

uint8_t
parsePartContainer(const std::string &value);

uint8_t
parseLineitemShipinstruct(const std::string &value);

void
tpch_parse_args(int argc, char **argv, tcph_args_t *params) {
    int c;
    while (1) {
        static struct option long_options[] = {
            //                        {"alg", required_argument, 0, 'a'},
            //                        {"query", required_argument, 0, 'q'},
            //                        {"threads", required_argument, 0, 'n'}
        };

        int option_index = 0;

        c = getopt_long(argc, argv, "a:b:m:n:q:s:p", long_options, &option_index);

        if (c == -1)
            break;
        switch (c) {
            case 'a':
                strcpy(params->algorithm_name, optarg);
                break;
            case 'b':
                params->bits = (uint8_t) atoi(optarg);
                break;
            case 'n':
                params->nthreads = (uint8_t) atoi(optarg);
                break;
            case 'q':
                params->query = (uint8_t) atoi(optarg);
                break;
            case 's':
                params->scale = (uint8_t) atoi(optarg);
                break;
            case 'p':
                params->parallel = true;
                break;
            default:
                break;
        }
    }

    if (optind < argc) {
        logger(INFO, "remaining arguments: ");
        while (optind < argc) {
            logger(INFO, "%s", argv[optind++]);
        }
    }
}

uint8_t
parseShipMode(const std::string &value) {
    if (value == "MAIL") {
        return L_SHIPMODE_MAIL;
    } else if (value == "SHIP") {
        return L_SHIPMODE_SHIP;
    } else if (value == "AIR") {
        return L_SHIPMODE_AIR;
    } else if (value == "AIR REG") {
        return L_SHIPMODE_AIR_REG;
    } else {
        return 0;
    }
}

uint64_t
parseDateToLong(const std::string &value) {
    std::tm tm = {};
    std::istringstream ss(value + " 00:00:00");
    ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
    std::time_t time_stamp = mktime(&tm) - timezone;
    return static_cast<uint64_t>(time_stamp);
}

uint64_t
parseDateToLong_2(const std::string &&value) {
    static std::tm tm{};
    static auto time_getter = std::get_time(&tm, "%Y-%m-%d");

    std::istringstream ss(value);
    ss >> time_getter;
    const std::time_t time_stamp = mktime(&tm) - timezone;
    return static_cast<uint64_t>(time_stamp);
}

uint8_t
parseMktSegment(const std::string &value) {
    if (value == "BUILDING") {
        return MKT_BUILDING;
    } else {
        return 0;
    }
}

uint64_t
getNumberOfLines(const std::string &filename) {
    std::ifstream ifs(filename);
    auto lines = std::count(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>(), '\n');
    logger(INFO, "File %s has %lu lines", filename.c_str(), lines);
    return lines;
}

std::string
getPath(int scale, const std::string &tbl) {
    std::stringstream ss;
    ss << std::setw(3) << std::setfill('0') << scale;
    return "../data/scale" + ss.str() + "/" + tbl;
}

uint64_t
read_size_file(const std::string &path) {
    std::ifstream size_file{path + "/size"};
    uint64_t size;
    size_file >> size;
    size_file.close();
    return size;
}

void
read_binary(char *buffer, const uint64_t size, const std::string &path) {
    std::ifstream binary_file{path, std::ios::in | std::ios::binary};
    binary_file.read(buffer, size);
    binary_file.close();
}

std::string
row_to_string(const csv::CSVRow &row) {
    std::ostringstream row_stream{};
    std::for_each(row.begin(), row.end(),
                  [&row_stream](const csv::CSVField &field) { row_stream << field.get_sv() << '|'; });
    return row_stream.str();
}

void
handle_parse_error(const csv::CSVRow &row, const uint64_t i, const std::runtime_error &error) {
    logger(ERROR, "Failed to parse row %lld\nRow: %s\nException: %s", i, row_to_string(row).c_str(), error.what());
}

void
handle_parse_error(const CSVRow &row, const uint64_t i, const std::runtime_error &error) {
    logger(ERROR, "Failed to parse row %lld\nRow: %s\nException: %s", i, row.m_line.c_str(), error.what());
}

int
load_lineitems_from_binary(LineItemTable *l_table, uint8_t query, uint8_t scale) {
    if (query != 3 && query != 10 && query != 12 && query != 19) {
        return 0;
    }

    const std::string PATH = getPath(scale, LINEITEM_TBL) + ".dir";
    uint64_t numTuples = read_size_file(PATH);

    l_table->numTuples = numTuples;

    int rv = 0;
    rv |= posix_memalign((void **) &(l_table->l_orderkey), 64, sizeof(tuple_t) * numTuples);
    if (query == 3) {
        rv |= posix_memalign((void **) &(l_table->l_shipdate), 64, sizeof(uint64_t) * numTuples);
    } else if (query == 10) {
        rv |= posix_memalign((void **) &(l_table->l_returnflag), 64, sizeof(char) * numTuples);
    } else if (query == 12) {
        rv |= posix_memalign((void **) &(l_table->l_shipdate), 64, sizeof(uint64_t) * numTuples);
        rv |= posix_memalign((void **) &(l_table->l_commitdate), 64, sizeof(uint64_t) * numTuples);
        rv |= posix_memalign((void **) &(l_table->l_receiptdate), 64, sizeof(uint64_t) * numTuples);
        rv |= posix_memalign((void **) &(l_table->l_shipmode), 64, sizeof(uint8_t) * numTuples);
    } else if (query == 19) {
        rv |= posix_memalign((void **) &(l_table->l_partkey), 64, sizeof(type_key) * numTuples);
        rv |= posix_memalign((void **) &(l_table->l_quantity), 64, sizeof(float) * numTuples);
        rv |= posix_memalign((void **) &(l_table->l_shipinstruct), 64, sizeof(uint8_t) * numTuples);
        rv |= posix_memalign((void **) &(l_table->l_shipmode), 64, sizeof(uint8_t) * numTuples);
    }
    if (rv != 0) {
        logger(ERROR, "memalign error");
        return 1;
    }

    read_binary(reinterpret_cast<char *>(l_table->l_orderkey), numTuples * sizeof(tuple_t), PATH + "/l_orderkey.bin");
    if (query == 3) {
        read_binary(reinterpret_cast<char *>(l_table->l_shipdate), numTuples * sizeof(uint64_t),
                    PATH + "/l_shipdate.bin");
    } else if (query == 10) {
        read_binary(reinterpret_cast<char *>(l_table->l_returnflag), numTuples * sizeof(char),
                    PATH + "/l_returnflag.bin");
    } else if (query == 12) {
        read_binary(reinterpret_cast<char *>(l_table->l_shipdate), numTuples * sizeof(uint64_t),
                    PATH + "/l_shipdate.bin");
        read_binary(reinterpret_cast<char *>(l_table->l_commitdate), numTuples * sizeof(uint64_t),
                    PATH + "/l_commitdate.bin");
        read_binary(reinterpret_cast<char *>(l_table->l_receiptdate), numTuples * sizeof(uint64_t),
                    PATH + "/l_receiptdate.bin");
        read_binary(reinterpret_cast<char *>(l_table->l_shipmode), numTuples * sizeof(uint8_t),
                    PATH + "/l_shipmode.bin");
    } else if (query == 19) {
        read_binary(reinterpret_cast<char *>(l_table->l_partkey), numTuples * sizeof(type_key),
                    PATH + "/l_partkey.bin");
        read_binary(reinterpret_cast<char *>(l_table->l_quantity), numTuples * sizeof(float), PATH + "/l_quantity.bin");
        read_binary(reinterpret_cast<char *>(l_table->l_shipinstruct), numTuples * sizeof(uint8_t),
                    PATH + "/l_shipinstruct.bin");
        read_binary(reinterpret_cast<char *>(l_table->l_shipmode), numTuples * sizeof(uint8_t),
                    PATH + "/l_shipmode.bin");
    }

    return 0;
}

int
load_lineitem_from_csv(LineItemTable *l_table, uint8_t scale) {
    logger(INFO, "Loading Lineitem");

    const std::string PATH = getPath(scale, LINEITEM_TBL);

    const uint64_t num_tuples = getNumberOfLines(PATH);

    l_table->numTuples = num_tuples;
    int rv = 0;
    rv |= posix_memalign((void **) &(l_table->l_orderkey), 64, sizeof(tuple_t) * num_tuples);
    rv |= posix_memalign((void **) &(l_table->l_shipdate), 64, sizeof(uint64_t) * num_tuples);
    rv |= posix_memalign((void **) &(l_table->l_returnflag), 64, sizeof(char) * num_tuples);
    rv |= posix_memalign((void **) &(l_table->l_commitdate), 64, sizeof(uint64_t) * num_tuples);
    rv |= posix_memalign((void **) &(l_table->l_receiptdate), 64, sizeof(uint64_t) * num_tuples);
    rv |= posix_memalign((void **) &(l_table->l_shipmode), 64, sizeof(uint8_t) * num_tuples);
    rv |= posix_memalign((void **) &(l_table->l_partkey), 64, sizeof(type_key) * num_tuples);
    rv |= posix_memalign((void **) &(l_table->l_quantity), 64, sizeof(float) * num_tuples);
    rv |= posix_memalign((void **) &(l_table->l_shipinstruct), 64, sizeof(uint8_t) * num_tuples);
    if (rv != 0) {
        logger(ERROR, "memalign error");
        return 1;
    }

    std::ifstream csv_file_stream{PATH};
    CSVRow row;

    for (uint64_t i = 0; i < num_tuples; ++i) {
        csv_file_stream >> row;

        if ((i & (1 << 20) - 1) == 0) {
            logger(INFO, "Processed %lld rows of lineitem", i);
        }

        try {
            vton_debug<type_key>(row[0], l_table->l_orderkey[i].key);
            l_table->l_orderkey[i].payload = static_cast<type_value>(i);
            vton_debug<type_key>(row[1], l_table->l_partkey[i]);
            vton_debug<float>(row[4], l_table->l_quantity[i]);
            l_table->l_returnflag[i] = row[8][0];
            l_table->l_shipdate[i] = parseDateToLong_2(std::string{row[10]});
            l_table->l_commitdate[i] = parseDateToLong_2(std::string{row[11]});
            l_table->l_receiptdate[i] = parseDateToLong_2(std::string{row[12]});
            l_table->l_shipinstruct[i] = parseLineitemShipinstruct(std::string{row[13]});
            l_table->l_shipmode[i] = parseShipMode(std::string{row[14]});
        } catch (const std::runtime_error &error) { handle_parse_error(row, i, error); }
    }
    logger(INFO, "lineitem table parsed");
    return 0;
}

uint8_t
parseLineitemShipinstruct(const std::string &value) {
    if (value == "DELIVER IN PERSON")
        return L_SHIPINSTRUCT_DELIVER_IN_PERSON;
    else
        return 0;
}

template<typename T>
void
checked_free(T * &ptr) {
    if (ptr != nullptr) {
        free(ptr);
        ptr = nullptr;
    }
}

void
free_lineitem(LineItemTable *l_table) {
    l_table->numTuples = 0;
    checked_free(l_table->l_orderkey);
    checked_free(l_table->l_shipdate);
    checked_free(l_table->l_returnflag);
    checked_free(l_table->l_commitdate);
    checked_free(l_table->l_receiptdate);
    checked_free(l_table->l_partkey);
    checked_free(l_table->l_quantity);
    checked_free(l_table->l_shipinstruct);
    checked_free(l_table->l_shipmode);
}

int
load_orders_from_csv(OrdersTable *o_table, uint8_t scale) {
    logger(INFO, "Loading Orders");

    const std::string PATH = getPath(scale, ORDERS_TBL);

    const uint64_t num_tuples = getNumberOfLines(PATH);

    o_table->numTuples = num_tuples;
    int rv = posix_memalign((void **) &(o_table->o_orderkey), 64, sizeof(tuple_t) * num_tuples);
    rv |= posix_memalign((void **) &(o_table->o_orderdate), 64, sizeof(uint64_t) * num_tuples);
    rv |= posix_memalign((void **) &(o_table->o_custkey), 64, sizeof(type_key) * num_tuples);
    if (rv != 0) {
        logger(ERROR, "memalign error");
        return 1;
    }

    csv::CSVFormat format{};

    format.delimiter('|');
    format.no_header();
    format.quote(false);

    csv::CSVReader reader{PATH, format};
    csv::CSVRow row;

    for (uint64_t i = 0; i < num_tuples; ++i) {
        reader.read_row(row);

        if ((i & (1 << 20) - 1) == 0) {
            logger(INFO, "Processed %lld rows of orders", i);
        }

        try {
            o_table->o_orderkey[i].key = row[0].get<type_key>();
            o_table->o_orderkey[i].payload = static_cast<type_value>(i);
            o_table->o_custkey[i] = row[1].get<type_key>();
            o_table->o_orderdate[i] = parseDateToLong_2(row[4].get());
        } catch (std::runtime_error &error) { handle_parse_error(row, i, error); }
    }
    logger(INFO, "orders table parsed");
    return 0;
}

int
load_orders_from_binary(OrdersTable *o_table, uint8_t query, uint8_t scale) {
    if (query != 3 && query != 10 && query != 12) {
        return 0;
    }

    const std::string PATH = getPath(scale, ORDERS_TBL) + ".dir";
    uint64_t numTuples = read_size_file(PATH);

    o_table->numTuples = numTuples;
    int rv = posix_memalign((void **) &(o_table->o_orderkey), 64, sizeof(tuple_t) * numTuples);
    if (query == 3 || query == 10) {
        rv |= posix_memalign((void **) &(o_table->o_orderdate), 64, sizeof(uint64_t) * numTuples);
        rv |= posix_memalign((void **) &(o_table->o_custkey), 64, sizeof(type_key) * numTuples);
    }
    if (rv != 0) {
        logger(ERROR, "memalign error");
        return 1;
    }

    read_binary(reinterpret_cast<char *>(o_table->o_orderkey), numTuples * sizeof(tuple_t), PATH + "/o_orderkey.bin");
    if (query == 3 || query == 10) {
        read_binary(reinterpret_cast<char *>(o_table->o_orderdate), numTuples * sizeof(uint64_t),
                    PATH + "/o_orderdate.bin");
        read_binary(reinterpret_cast<char *>(o_table->o_custkey), numTuples * sizeof(type_key),
                    PATH + "/o_custkey.bin");
    }

    return 0;
}

void
free_orders(OrdersTable *o_table) {
    o_table->numTuples = 0;
    checked_free(o_table->o_orderkey);
    checked_free(o_table->o_custkey);
    checked_free(o_table->o_orderdate);
}

int
load_customer_from_csv(CustomerTable *c_table, uint8_t scale) {
    logger(INFO, "Loading Customers");

    const std::string PATH = getPath(scale, CUSTOMER_TBL);

    const uint64_t num_tuples = getNumberOfLines(PATH);

    c_table->numTuples = num_tuples;
    int rv = posix_memalign((void **) &(c_table->c_custkey), 64, sizeof(tuple_t) * num_tuples);
    rv |= posix_memalign((void **) &(c_table->c_mktsegment), 64, sizeof(uint8_t) * num_tuples);
    rv |= posix_memalign((void **) &(c_table->c_nationkey), 64, sizeof(type_key) * num_tuples);

    if (rv != 0) {
        logger(ERROR, "memalign error");
        return 1;
    }
    csv::CSVFormat format{};

    format.delimiter('|');
    format.no_header();
    format.quote(false);

    csv::CSVReader reader{PATH, format};
    csv::CSVRow row;

    for (uint64_t i = 0; i < num_tuples; i++) {
        reader.read_row(row);

        if ((i & (1 << 20) - 1) == 0) {
            logger(INFO, "Processed %lld rows of customer", i);
        }

        try {
            c_table->c_custkey[i].key = row[0].get<type_key>();
            c_table->c_custkey[i].payload = static_cast<type_value>(i);
            c_table->c_mktsegment[i] = parseMktSegment(row[6].get());
            c_table->c_nationkey[i] = row[3].get<type_key>();
        } catch (std::runtime_error &error) { handle_parse_error(row, i, error); }
    }
    logger(INFO, "customer table parsed");
    return 0;
}

int
load_customers_from_binary(CustomerTable *c_table, uint8_t query, uint8_t scale) {
    if (query != 3 && query != 10) {
        return 0;
    }

    const std::string PATH = getPath(scale, CUSTOMER_TBL) + ".dir";
    uint64_t numTuples = read_size_file(PATH);

    c_table->numTuples = numTuples;

    int rv = posix_memalign(reinterpret_cast<void **>(&c_table->c_custkey), 64, sizeof(tuple_t) * numTuples);
    if (query == 3) {
        rv |= posix_memalign((void **) &c_table->c_mktsegment, 64, sizeof(uint8_t) * numTuples);
    } else if (query == 10) {
        rv |= posix_memalign((void **) &c_table->c_nationkey, 64, sizeof(type_key) * numTuples);
    }
    if (rv != 0) {
        logger(ERROR, "memalign error");
        return 1;
    }

    read_binary(reinterpret_cast<char *>(c_table->c_custkey), numTuples * sizeof(tuple_t), PATH + "/c_custkey.bin");
    if (query == 3) {
        read_binary(reinterpret_cast<char *>(c_table->c_mktsegment), numTuples * sizeof(uint8_t),
                    PATH + "/c_mktsegment.bin");
    } else if (query == 10) {
        read_binary(reinterpret_cast<char *>(c_table->c_nationkey), numTuples * sizeof(type_key),
                    PATH + "/c_nationkey.bin");
    }

    return 0;
}

void
free_customer(CustomerTable *c_table) {
    c_table->numTuples = 0;
    checked_free(c_table->c_custkey);
    checked_free(c_table->c_mktsegment);
    checked_free(c_table->c_nationkey);
}

int
load_part_from_csv(PartTable *p, uint8_t scale) {
    logger(INFO, "Loading Parts");

    const std::string PATH = getPath(scale, PART_TBL);

    const uint64_t num_tuples = getNumberOfLines(PATH);

    p->numTuples = num_tuples;
    int rv = posix_memalign((void **) &(p->p_partkey), 64, sizeof(tuple_t) * num_tuples);
    rv |= posix_memalign((void **) &(p->p_brand), 64, sizeof(uint8_t) * num_tuples);
    rv |= posix_memalign((void **) &(p->p_size), 64, sizeof(uint32_t) * num_tuples);
    rv |= posix_memalign((void **) &(p->p_container), 64, sizeof(uint8_t) * num_tuples);
    if (rv != 0) {
        logger(ERROR, "memalign error");
        return 1;
    }

    csv::CSVFormat format{};

    format.delimiter('|');
    format.no_header();
    format.quote(false);

    csv::CSVReader reader{PATH, format};
    csv::CSVRow row;

    for (uint64_t i = 0; i < num_tuples; i++) {
        reader.read_row(row);

        if ((i & (1 << 20) - 1) == 0) {
            logger(INFO, "Processed %lld rows of part", i);
        }

        try {
            p->p_partkey[i].key = row[0].get<type_key>();
            p->p_partkey[i].payload = static_cast<type_value>(i);
            p->p_brand[i] = parsePartBrand(row[3].get());
            p->p_size[i] = row[5].get<uint32_t>();
            p->p_container[i] = parsePartContainer(row[6].get());
        } catch (std::runtime_error &error) { handle_parse_error(row, i, error); }
    }
    logger(INFO, "part table parsed");
    return 0;
}

int
load_parts_from_binary(PartTable *p, uint8_t query, uint8_t scale) {
    if (query != 19) {
        return 0;
    }

    const std::string PATH = getPath(scale, PART_TBL) + ".dir";
    uint64_t numTuples = read_size_file(PATH);

    p->numTuples = numTuples;

    int rv = posix_memalign((void **) &(p->p_partkey), 64, sizeof(tuple_t) * numTuples);
    if (query == 19) {
        rv |= posix_memalign((void **) &(p->p_brand), 64, sizeof(uint8_t) * numTuples);
        rv |= posix_memalign((void **) &(p->p_container), 64, sizeof(uint8_t) * numTuples);
        rv |= posix_memalign((void **) &(p->p_size), 64, sizeof(uint32_t) * numTuples);
    }
    if (rv != 0) {
        logger(ERROR, "memalign error");
        return 1;
    }

    read_binary(reinterpret_cast<char *>(p->p_partkey), numTuples * sizeof(tuple_t), PATH + "/p_partkey.bin");
    if (query == 19) {
        read_binary(reinterpret_cast<char *>(p->p_brand), numTuples * sizeof(uint8_t), PATH + "/p_brand.bin");
        read_binary(reinterpret_cast<char *>(p->p_container), numTuples * sizeof(uint8_t), PATH + "/p_container.bin");
        read_binary(reinterpret_cast<char *>(p->p_size), numTuples * sizeof(uint32_t), PATH + "/p_size.bin");
    }

    return 0;
}

uint8_t
parsePartContainer(const std::string &value) {
    if (value == "SM CASE")
        return P_CONTAINER_SM_CASE;
    else if (value == "SM BOX")
        return P_CONTAINER_SM_BOX;
    else if (value == "SM PACK")
        return P_CONTAINER_SM_PACK;
    else if (value == "SM PKG")
        return P_CONTAINER_SM_PKG;
    else if (value == "MED BAG")
        return P_CONTAINER_MED_BAG;
    else if (value == "MED BOX")
        return P_CONTAINER_MED_BOX;
    else if (value == "MED PKG")
        return P_CONTAINER_MED_PKG;
    else if (value == "MED PACK")
        return P_CONTAINER_MED_PACK;
    else if (value == "LG CASE")
        return P_CONTAINER_LG_CASE;
    else if (value == "LG BOX")
        return P_CONTAINER_LG_BOX;
    else if (value == "LG PACK")
        return P_CONTAINER_LG_PACK;
    else if (value == "LG PKG")
        return P_CONTAINER_LG_PKG;
    else
        return 0;
}

uint8_t
parsePartBrand(const std::string &value) {
    if (value == "Brand#12")
        return P_BRAND_12;
    else if (value == "Brand#23")
        return P_BRAND_23;
    else if (value == "Brand#34")
        return P_BRAND_34;
    else
        return 0;
}

void
free_part(PartTable *p) {
    p->numTuples = 0;
    checked_free(p->p_partkey);
    checked_free(p->p_brand);
    checked_free(p->p_size);
    checked_free(p->p_container);
}

int
load_nation_from_csv(NationTable *n, uint8_t scale) {
    logger(INFO, "Loading Nation");

    const std::string PATH = getPath(scale, NATION_TBL);

    const uint64_t num_tuples = getNumberOfLines(PATH);

    n->numTuples = num_tuples;
    int rv = posix_memalign((void **) &(n->n_nationkey), 64, sizeof(tuple_t) * num_tuples);
    if (rv != 0) {
        logger(ERROR, "memalign error");
        return 1;
    }

    csv::CSVFormat format{};

    format.delimiter('|');
    format.no_header();
    format.quote(false);

    csv::CSVReader reader{PATH, format};
    csv::CSVRow row;

    for (uint64_t i = 0; i < num_tuples; i++) {
        reader.read_row(row);

        if ((i & (1 << 20) - 1) == 0) {
            logger(INFO, "Processed %lld rows of nation", i);
        }
        try {
            n->n_nationkey[i].key = row[0].get<type_key>();
            n->n_nationkey[i].payload = static_cast<type_value>(i);
        } catch (std::runtime_error &error) { handle_parse_error(row, i, error); }
    }
    logger(INFO, "nation table parsed");
    return 0;
}

int
load_nations_from_binary(NationTable *n, uint8_t query, uint8_t scale) {
    if (query != 10) {
        return 0;
    }

    const std::string PATH = getPath(scale, NATION_TBL) + ".dir";
    uint64_t numTuples = read_size_file(PATH);

    n->numTuples = numTuples;

    int rv = posix_memalign((void **) &(n->n_nationkey), 64, sizeof(tuple_t) * numTuples);
    if (rv != 0) {
        logger(ERROR, "memalign error");
        return 1;
    }

    read_binary(reinterpret_cast<char *>(n->n_nationkey), numTuples * sizeof(tuple_t), PATH + "/n_nationkey.bin");

    return 0;
}

void
free_nation(NationTable *n) {
    n->numTuples = 0;
    checked_free(n->n_nationkey);
}
