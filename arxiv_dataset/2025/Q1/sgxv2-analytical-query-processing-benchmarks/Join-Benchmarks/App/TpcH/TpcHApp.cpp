#include "Enclave_u.h"
#include "ErrorSupport.h"
#include "Logger.hpp"
#include "TpcHCommons.hpp"
#include "sgx_urts.h"

constexpr std::string_view ENCLAVE_FILENAME = "joinenclave.signed.so";

sgx_enclave_id_t global_eid = 0;
char experiment_filename [512];
int write_to_file = 0;

int initialize_enclave(void)
{
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;

    logger(INFO, "Initialize enclave");
    /* Call sgx_create_enclave to initialize an enclave instance */
    /* Debug Support: set 2nd parameter to 1 */
    ret = sgx_create_enclave(std::string{ENCLAVE_FILENAME}.data(), 1, nullptr, nullptr, &global_eid, nullptr);
    if (ret != SGX_SUCCESS) {
        ret_error_support(ret);
        return -1;
    }
    logger(INFO, "Enclave id = %d", global_eid);
    return 0;
}

int main(int argc, char *argv[])
{
    initLogger();
    joinconfig_t joinconfig;
    logger(INFO, "************* TPC-H APP *************");
    // 1. Parse args
    tcph_args_t params;
    tpch_parse_args(argc, argv, &params);
    uint8_t query = params.query;
    joinconfig.NTHREADS = (int) params.nthreads;
    joinconfig.RADIXBITS = -1;
    logger(INFO, "Run Q%d (scale %d) with join algorithm %s (%d threads)",
           query, params.scale, params.algorithm_name, joinconfig.NTHREADS);

    // 2. load required TPC-H tables
    LineItemTable l;
    OrdersTable o;
    CustomerTable c;
    PartTable p;
    NationTable n;

    load_orders_from_binary(&o, query, params.scale);
    load_customers_from_binary(&c, query, params.scale);
    load_parts_from_binary(&p, query, params.scale);
    load_nations_from_binary(&n, query, params.scale);
    load_lineitems_from_binary(&l, query, params.scale);

    // 3. init enclave
    initialize_enclave();
    ecall_init_logger(global_eid, logger_get_start());

    // 4. execute specified query using an ECALL
    sgx_status_t ret = SGX_SUCCESS, retval;
    result_t result{};
    switch(query) {
        case 3:
            ret = ecall_tpch_q3(global_eid,
                                &retval,
                                &result,
                                &c,
                                &o,
                                &l,
                                params.algorithm_name,
                                &joinconfig);
            break;
        case 10:
            ret = ecall_tpch_q10(global_eid,
                                 &retval,
                                 &result,
                                 &c,
                                 &o,
                                 &l,
                                 &n,
                                 params.algorithm_name,
                                 &joinconfig);
            break;
        case 12:
            ret = ecall_tpch_q12(global_eid,
                                 &retval,
                                 &result,
                                 &l,
                                 &o,
                                 params.algorithm_name,
                                 &joinconfig);
            break;
        case 19:
            ret = ecall_tpch_q19(global_eid,
                                 &retval,
                                 &result,
                                 &l,
                                 &p,
                                 params.algorithm_name,
                                 &joinconfig);
            break;
        default:
            logger(ERROR, "TPC-H Q%d is not supported", query);
    }
    // 4. report and clean up
    if (ret != SGX_SUCCESS) {
        ret_error_support(ret);
    } else {
        logger(INFO, "Query completed");
    }

    free_orders(&o);
    free_part(&p);
    free_customer(&c);
    free_lineitem(&l);
    free_nation(&n);

    // 5. destroy enclave
    ret = sgx_destroy_enclave(global_eid);
    if (ret == SGX_SUCCESS) {
        logger(INFO, "Enclave destroyed");
    }

}