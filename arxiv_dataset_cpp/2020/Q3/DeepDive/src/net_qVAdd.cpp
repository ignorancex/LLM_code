#include "hw/QVector_Add.hpp"

void QVector_Add(dType_8u const *inp_0,
                 dType_8u const *inp_1,
                 dType_8u *out_0,
                 dType_Reg skipline_ipzp1,
                 dType_Reg skipline_ipzp2,
                 dType_Reg skipline_opzp1,
                 dType_8u xMultiplier,
                 dType_8u xShift,
                 dType_8u yMultiplier,
                 dType_8u yShift,
                 dType_Reg output_size)
{
#pragma HLS DATAFLOW
    for (dType_32u i = 0; i < output_size; i++)
    {
#pragma HLS PIPELINE
        dType_Reg inPipe1 = inp_1[i] - skipline_ipzp1;
        dType_Reg inPipe2 = inp_0[i] - skipline_ipzp2;
        dType_Reg inPipe1_m = (inPipe1 * xMultiplier) >> xShift;
        dType_Reg inPipe2_m = (inPipe2 * yMultiplier) >> yShift;
        dType_Reg addi = inPipe1_m + inPipe2_m + skipline_opzp1;
        dType_8uf sum = addi;
        out_0[i] = sum;
    }
}
