#include "Enclave_t.h"
#include "TpcHTypes.hpp"
#include "data-types.h"
#include "tpch.hpp"
#include "sgx_error.h"

extern "C" {

sgx_status_t
ecall_tpch_q3(result_t *result, const struct CustomerTable *c, const struct OrdersTable *o,
              const struct LineItemTable *l, const char *algorithm, struct joinconfig_t *config) {
    tpch_q3(result, c, o, l, algorithm, config);
    return SGX_SUCCESS;
}

sgx_status_t
ecall_tpch_q10(result_t *result, const CustomerTable *c, const OrdersTable *o, const LineItemTable *l,
               const NationTable *n, const char *algorithm, joinconfig_t *config) {
    tpch_q10(result, c, o, l, n, algorithm, config);
    return SGX_SUCCESS;
}

sgx_status_t
ecall_tpch_q12(result_t *result, const LineItemTable *l, const OrdersTable *o, const char *algorithm,
               joinconfig_t *config) {
    tpch_q12(result, l, o, algorithm, config);
    return SGX_SUCCESS;
}

sgx_status_t
ecall_tpch_q19(result_t *result, const LineItemTable *l, const PartTable *p, const char *algorithm,
               joinconfig_t *config) {
    tpch_q19(result, l, p, algorithm, config);
    return SGX_SUCCESS;
}

}
