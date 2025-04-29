#include "dcl.h"

using namespace std;

// Flexible run-time reconfigurable floating point multiplication
// 1 bit sign, EB bits exponent, MB bits mantissa

float flexible_float_multiply(float a, float b, bool *overflow, int EB_in, int MB_in, bool adjust) {

    ap_uint<4> EB = EB_in;
    ap_uint<4> MB = MB_in;
    
    ap_uint<1> flex_type_sign = 0;
    ap_uint<FIX_EXPN> fix_exp_a = 0;
    ap_uint<FIX_EXPN> fix_exp_b = 0;
    ap_uint<FIX_EXPN+1> fix_exp_res = 0;
    ap_uint<FLEX_BIT> flex_bits_a = 0;
    ap_uint<FLEX_BIT> flex_bits_b = 0;
    ap_uint<FLEX_BIT+1> flex_bits_res = 0;
    ap_uint<FIX_MANT+1> fix_mant_a = 0;
    ap_uint<FIX_MANT+1> fix_mant_b = 0;
    ap_uint<FIX_MANT> fix_mant_res = 0;
    ap_uint<FLEX_BIT> mask = 0;

    ap_int<4> EXPN_residual = EB-FIX_EXPN-1;
    ap_int<4> MANT_residual = MB-FIX_MANT-1;

    mask = ~0;
    if( MANT_residual >= 0) {
        mask.range(MANT_residual, 0) = 0;
    }

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


    // get the correct representations of those flexible floating point numbers
    fix_exp_a.range(FIX_EXPN-1, 0) = expn_a.range(EB-1, EB-FIX_EXPN);
    fix_exp_b.range(FIX_EXPN-1, 0) = expn_b.range(EB-1, EB-FIX_EXPN);
    
    fix_mant_a.range(FIX_MANT-1, 0) = mant_a.range(MB-1, MB-FIX_MANT);
    fix_mant_b.range(FIX_MANT-1, 0) = mant_b.range(MB-1, MB-FIX_MANT);

    // the implicit one before mantissa
    fix_mant_a[FIX_MANT] = 1;
    fix_mant_b[FIX_MANT] = 1;

    // leftovers from expn or mantissa bits are filled into the flexible bit region (FLEX_BIT)
    if( EXPN_residual >= 0 ) { 
        flex_bits_a.range(FLEX_BIT-1, FLEX_BIT-1-(EXPN_residual)) = expn_a.range(EXPN_residual, 0);
        flex_bits_b.range(FLEX_BIT-1, FLEX_BIT-1-(EXPN_residual)) = expn_b.range(EXPN_residual, 0);
    }
    if( MANT_residual >= 0 ) {
        flex_bits_a.range(MANT_residual, 0) = mant_a(MANT_residual, 0);
        flex_bits_b.range(MANT_residual, 0) = mant_b(MANT_residual, 0);
    }


    // Compute mantissa multiplication result

    // compute fixed region first
    ap_uint< (FIX_MANT+1)*2 + MANT_APPX > fix_mant_res_im = 0;  // intermediate for mantissa result
    fix_mant_res_im = fix_mant_a * fix_mant_b;
    fix_mant_res_im = fix_mant_res_im << MANT_APPX;

    // compute flexible region
    for(ap_int<3> m = MANT_residual; m >= 0; m--) {
        ap_uint<1> b1, b2;
        b1 = flex_bits_a[MB-FIX_MANT-m-1];
        b2 = flex_bits_b[MB-FIX_MANT-m-1];
        fix_mant_res_im += ((ap_uint<FIX_MANT + 1 + MANT_APPX>)(fix_mant_a * b2)) << (2-m);
        fix_mant_res_im += ((ap_uint<FIX_MANT + 1 + MANT_APPX>)(fix_mant_b * b1)) << (2-m);

        if( m == 0) {
            fix_mant_res_im += ((ap_uint<MANT_APPX>)(b1 & b2)) << 1;
        }
        else if( m == 1) {
            fix_mant_res_im += b1 & flex_bits_b[1] + b2 & flex_bits_a[1];
        }
    }

    // handle overflow in mantissa
    ap_uint<1> mantissa_carry = 0;
    ap_uint<2> mantissa_shift = 0;
    if( fix_mant_res_im[(FIX_MANT+1)*2 + MANT_APPX - 1] == (ap_uint<1>)(0) ) {
        mantissa_shift = 2;
    }
    else if( fix_mant_res_im[(FIX_MANT+1)*2 + MANT_APPX - 1] == (ap_uint<1>)(1) ) {
        mantissa_carry = 1;
        mantissa_shift = 1;
    }
    fix_mant_res_im[(FIX_MANT+1)*2 + MANT_APPX - 1] = 0;

    // handle mantissa rounding
    ap_uint<1> flex_guard = fix_mant_res_im[ FIX_MANT*2 + MANT_APPX - MB + 1 - mantissa_shift];

    if( flex_guard == 1) {
        // flexible mantissa rounded
        fix_mant_res_im += (ap_uint< (FIX_MANT+1)*2 + MANT_APPX >)1 << ( FIX_MANT*2 + MANT_APPX - MB + 1 - mantissa_shift );
    }
    // check if mantissa rounding result in mantissa overflow
    if( fix_mant_res_im[(FIX_MANT+1)*2 + MANT_APPX - 1] == 1 ) {
        // mantissa rounded results in first-bit overflow
        mantissa_carry = 1;
        mantissa_shift++;
    }
    fix_mant_res_im = fix_mant_res_im << mantissa_shift;

    // copy mantissa intermediate result back to mantissa flexible and fix fields
    fix_mant_res.range(FIX_MANT-1, 0) = fix_mant_res_im.range((FIX_MANT+1)*2 + MANT_APPX - 1, (FIX_MANT+1)*2 + MANT_APPX - FIX_MANT);
    if(MANT_residual >=0) {
        flex_bits_res.range(MANT_residual, 0) = fix_mant_res_im.range((FIX_MANT+1)*2 + MANT_APPX - FIX_MANT - 1, (FIX_MANT+1)*2 + MANT_APPX - MB);
    }
    
    // compute exponent

    // compute flexible region
    uint8_t bias_flex_FP = ((uint8_t)1 << (EB-1)) - 1; // my type bias
    fix_exp_res = fix_exp_a + fix_exp_b;

    if( EXPN_residual >= 0 ) {   // only if there are leftovers from expn
        // minus bias: 11 (3EB) or 111 (4EB) or 1111 (5EB) or 11111 (6EB)
        // the same as -100+1 or -1000+1 or ...
        // the minus only happens at FIX_EXPN field: EB-1-(EB-FIX_EXPN)
                
        ap_uint< FLEX_BIT+1 > e_tmp = (flex_bits_a & mask) + (flex_bits_b & mask) + ((ap_uint<FLEX_BIT>)1 << (FLEX_BIT+FIX_EXPN-EB));
        flex_bits_res.range( FLEX_BIT, MB-FIX_MANT ) = e_tmp.range( FLEX_BIT, MB-FIX_MANT ); // not to tamper flexible mantissa bits

        fix_exp_res = fix_exp_res + flex_bits_res[FLEX_BIT] - ( ((ap_uint<FIX_EXPN>)1) << (FIX_EXPN-1) ); // add the carry from flexible bits, and then subtract 0b100; 
        flex_bits_res[FLEX_BIT] = 0; // set the carry bit to be zero
    }
    else {
        fix_exp_res = fix_exp_res - bias_flex_FP;
    }

    if( EXPN_residual >= 0 ) {   // if there are expn bits from flexible field
        flex_bits_res += ((ap_uint<FLEX_BIT>)mantissa_carry) << (MB-FIX_MANT);
        fix_exp_res = fix_exp_res + flex_bits_res[FLEX_BIT];
        flex_bits_res[FLEX_BIT] = 0; // set the carry bit to be zero
    }
    else {  // mantissa overflow bit (mantissa_carry) goes to fixed field
        fix_exp_res += mantissa_carry;
    }


    if( fix_exp_res[FIX_EXPN] == 1 ) {
        *overflow = true;
    }


    // // precision adjustment //
    // if ( FIX_EXPN >= 3 && EB > 3 && *overflow == false && adjust == true ) {
    //     bool reduce = false;
    //     if (fix_exp_a.range(FIX_EXPN-1, FIX_EXPN-3).to_string() == "0b011" && 
    //         fix_exp_b.range(FIX_EXPN-1, FIX_EXPN-3).to_string() == "0b011" &&
    //         fix_exp_res.range(FIX_EXPN-1, FIX_EXPN-3).to_string() == "0b011") {
    //             reduce = true;
    //     }
    //     else if (fix_exp_a.range(FIX_EXPN-1, FIX_EXPN-3).to_string() == "0b100" && 
    //         fix_exp_b.range(FIX_EXPN-1, FIX_EXPN-3).to_string() == "0b100" &&
    //         fix_exp_res.range(FIX_EXPN-1, FIX_EXPN-3).to_string() == "0b100") {
    //             reduce = true;
    //     }
    //     if( reduce == true ) {
    //         // adjust the precision by reducing exponent by one bit
    //         // it uses with real applications
    //         // the code will become messy so it is omitted here
    //     }
    // }

    // assemble the results back to check the correctness; just for verification purpose
    // code inside "if"0 is commented out during synthesis
    myFP flex_result = 0;
    float float_result = 0;
    if( *overflow == false) {
        flex_result[myFP_total_bit-1] = sign_a ^ sign_b;

        // copy fix exponent field
        flex_result.range(myFP_total_bit-2, myFP_total_bit-1-FIX_EXPN) = fix_exp_res.range(FIX_EXPN-1, 0);

        // copy flexible exponent bits
        if(EXPN_residual >= 0) {
            flex_result.range(myFP_total_bit-2-FIX_EXPN, MB) = flex_bits_res.range( FLEX_BIT-1, FLEX_BIT - (EB - FIX_EXPN) );
        }
        
        // copy fix mantissa field
        flex_result.range(MB-1, MB-FIX_MANT) = fix_mant_res_im.range((FIX_MANT+1)*2 + MANT_APPX-1, (FIX_MANT+1)*2 + MANT_APPX - FIX_MANT);
        
        // copy flexible mantissa field
        if( MANT_residual >= 0) {
            flex_result.range( MANT_residual, 0) = flex_bits_res.range(MANT_residual, 0);
        }
    }

    float_result = myFP2float(flex_result, EB, MB);
    return float_result;
    
}


float flexible_float_multiply_retry(float a, float b, bool adjust)
{
    // this part of the code does precision adjustment by increasing the exponent bits if overflows
    // precision adjustment will be used in real applications
    // the code will become messy so omitted
    return 1.0;
}