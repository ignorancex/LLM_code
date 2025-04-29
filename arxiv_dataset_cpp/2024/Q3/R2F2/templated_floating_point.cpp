#include "dcl.h"

using namespace std;


void float2myFP(float *f, myFP *h, int EB, int MB, bool *overflow)
{
    ap_uint<std_FP32_TOTAL_BIT>* ptr = (ap_uint<std_FP32_TOTAL_BIT>*)f;
    uint8_t myFP_total_bit = EB + MB + 1;

    (*h)[myFP_total_bit-1] = (*ptr)[std_FP32_TOTAL_BIT-1]; // sign bit
    ap_uint<std_FP32_EXPN_BIT + 2> e1 = (*ptr).range(std_FP32_TOTAL_BIT-2, std_FP32_MANT_BIT); // expn bits

    uint8_t bias_myFP = ((uint8_t)1 << (EB-1)) - 1; // my type bias
    ap_uint<std_FP32_EXPN_BIT + 2> e2 = 0;
    e2 = e1 - std_FP32_BIAS + bias_myFP;
    if( e2[EB+1] == 0 && e2[EB] == 1 ) {    // overflow: new expn is larger than max
        e2 = ~0;
        *overflow = true;
    }
    else if ( e2[EB+1] == 1 ) { // underflow: new expn is smaller than 0
        e2 = 0;
        *overflow = true;
    }
    //printf("in float2myFP: e1: %s, e2: %s\n", e1.to_string().c_str(), e2.to_string().c_str());
        
    (*h).range(myFP_total_bit-2, myFP_total_bit-1-EB) = e2.range(EB-1, 0); // expn bits
    (*h).range(MB-1, 0) = (*ptr).range(std_FP32_MANT_BIT-1, std_FP32_MANT_BIT-MB); // mant bits

    //printf("my float after conversion: %s\n", (*h).to_string().c_str());

}


float myFP2float(const myFP h, int EB, int MB)
{
    ap_uint<std_FP32_TOTAL_BIT> f = 0;
    
    uint8_t myFP_total_bit = EB + MB + 1;
    f[std_FP32_TOTAL_BIT-1] = h[myFP_total_bit - 1]; // sign bit

    uint8_t bias_myFP = ((uint8_t)1 << (EB-1)) - 1; // my type bias
    ap_uint<std_FP32_EXPN_BIT + 2> e1 = 0;
    e1.range(EB-1, 0) = h.range(myFP_total_bit-2, myFP_total_bit-1-EB);
    ap_uint<std_FP32_EXPN_BIT + 2> e2 = e1 - bias_myFP + std_FP32_BIAS;

    //printf("in myFP2float: e1: %s, e2: %s\n", e1.to_string().c_str(), e2.to_string().c_str());

    f.range(std_FP32_TOTAL_BIT-2, std_FP32_MANT_BIT) = e2.range(std_FP32_EXPN_BIT-1, 0);
    f.range(std_FP32_MANT_BIT-1, std_FP32_MANT_BIT-MB) = h.range(MB-1, 0);
    // remaining bits are pre-filled with zeros

    //printf("float converted back from myFP: %f (%s)\n", *(float*)(&f), f.to_string().c_str());

    return *(float*)(&f);
}



// Templated floating point multiplication
// Here, "templated" means that, during simulation,
// the data and intermediate results are still stored in maximum width for simplicity,
// but the computations use the actual bitwidth, where:
// 1 bit sign, EB bits exponent, MB bits mantissa

float templated_float_multiply(float a, float b, bool *overflow, int EB_in, int MB_in) {
    
    ap_uint<4> EB = EB_in;
    ap_uint<4> MB = MB_in;

    myFP myFP_a = 0;
    myFP myFP_b = 0;
    myFP_sign sign_a, sign_b;
    myFP_expn expn_a, expn_b;
    myFP_mant mant_a, mant_b;

    myFP result = 0;
    myFP_sign sign_res = 0;
    myFP_expn expn_res = 0;
    myFP_mant mant_res = 0;

    *overflow = false;

    float2myFP(&a, &myFP_a, EB, MB, overflow);
    float2myFP(&b, &myFP_b, EB, MB, overflow);

    uint8_t myFP_total_bit = EB + MB + 1;
    sign_a = myFP_a[myFP_total_bit-1];
    expn_a = 0;
    expn_a.range(myFP_total_bit-2-MB, 0) = myFP_a.range(myFP_total_bit-2, MB);
    mant_a = 0;
    mant_a.range(MB-1, 0) = myFP_a.range(MB-1, 0);
    mant_a[MB] = 1;

    sign_b = myFP_b[myFP_total_bit-1];
    expn_b = 0;
    expn_b.range(myFP_total_bit-2-MB, 0) = myFP_b.range(myFP_total_bit-2, MB);
    mant_b = 0;
    mant_b.range(MB-1, 0) = myFP_b.range(MB-1, 0);
    mant_b[MB] = 1;

    // Compute sign of the result
    sign_res = sign_a ^ sign_b;

    // Compute mantissa of the result
    ap_uint<myFP_MAX_MANT_MUL_BIT> mul;
    
    
    #pragma HLS BIND_OP variable=mul op=mul impl=fabric
    mul = mant_a * mant_b;

    ap_uint<1> mantissa_carry = 0;
    ap_uint<2> mantissa_shift = 0;

    // Handle overflow in mantissa
    if( mul[MB*2 + 1] == (ap_uint<1>)(0)) {
        // highest bit is 0: no mantissa overflow
        mul = mul << 2;
        mantissa_shift = 2;
    }
    else if ((mul[MB*2 + 1] == (ap_uint<1>)(1)) && (*overflow == false)) {
        // highest bit is 1: mantissa overflow -- need to shift exponent
        mul = mul << 1;
        mantissa_carry = 1;
        mantissa_shift = 1;
    }
    mul[MB*2 + 1 + mantissa_shift] = 0;

    // Handle mantissa rounding
    // GRS: gaurd, round, and sticky bits (https://pages.cs.wisc.edu/~markhill/cs354/Fall2008/notes/flpt.apprec.html)
    // In this implementation, not considering round bit and sticky bit
    ap_uint<1> guard = mul[MB + 1];
    ap_uint<1> round = mul[MB];
    ap_uint<1> sticky = mul[MB - 1] | mul[MB - 2];
    
    if((guard == 1)) {
        // mantissa rounded
        mul += (ap_uint<myFP_MAX_MANT_MUL_BIT>)1 << (MB + 1);  
    }   
    
    // check if mantissa rounding result in mantissa overflow
    if( mul[MB*2 + 1 + mantissa_shift] == 1 ) {
        // mantissa rounded results in first-bit overflow
        mul = mul << 1;
        mantissa_carry = 1;
    }
    mant_res.range(MB-1, 0) = mul.range(MB*2+1, MB+2);
    
    // Compute exponent of the result
    uint8_t bias_myFP = ((uint8_t)1 << (EB-1)) - 1; // my type bias
    myFP_expn _exp = (expn_a - bias_myFP) + (expn_b - bias_myFP) + bias_myFP + mantissa_carry;

    if( _exp[EB+1] == 0 && _exp[EB] == 1 ) { // overflow: new expn is larger than max or smaller than min
        _exp = ~0;
        *overflow = true;
    }
    else if (_exp[EB+1] == 1 ) {
        _exp = 0;
        *overflow = true;
    }
    expn_res.range(EB-1, 0) = _exp.range(EB-1, 0);

    // Assemble the results
    if( *overflow == false) {
        result[myFP_total_bit-1] = sign_res;
        result.range(myFP_total_bit-2, MB) = expn_res;
        result.range(MB-1, 0) = mant_res.range(MB-1, 0);
    }
    else {
        // result overflow
        result[myFP_total_bit-1] = 0;
        result.range(myFP_total_bit-2, MB) = expn_res;
        result.range(MB-1, 0) = 0;
    }
 
    // Convert back to float; this is to verify the result
    float float_result = myFP2float(result, EB, MB);

    return float_result;
}