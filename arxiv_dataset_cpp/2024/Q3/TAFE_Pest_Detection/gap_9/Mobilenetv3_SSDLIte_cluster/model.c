#include <stdint.h>
#include <stdio.h>
#include "AutoTilerLib.h"
#include "CNN_Generators_SQ8.h"

#include "CNN_Copy_Generators.h"

void load_expressions_kernels() {
    LibKernelTemplate(
        "s372_kernel_args_t",
        CArgs(8,
            TCArg("unsigned int", "I0"),
            TCArg("signed char *__restrict__ ", "expr_0_in_0"),
            TCArg("signed char *__restrict__ ", "expr_0_in_1"),
            TCArg("signed char *__restrict__ ", "expr_0_in_2"),
            TCArg("signed char *__restrict__ ", "expr_0_in_3"),
            TCArg("signed char *__restrict__ ", "expr_0_in_4"),
            TCArg("signed char *__restrict__ ", "expr_0_out_0"),
            TCArg("signed char *__restrict__ ", "expr_0_out_1")
        )
    );
    
    LibKernel(
        "s372_kernel",
        CALL_PARALLEL,
        0,
        "s372_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s366_kernel_args_t",
        CArgs(8,
            TCArg("unsigned int", "I0"),
            TCArg("signed char *__restrict__ ", "expr_1_in_0"),
            TCArg("signed char *__restrict__ ", "expr_1_in_1"),
            TCArg("signed char *__restrict__ ", "expr_1_in_2"),
            TCArg("signed char *__restrict__ ", "expr_1_in_3"),
            TCArg("signed char *__restrict__ ", "expr_1_in_4"),
            TCArg("signed char *__restrict__ ", "expr_1_out_0"),
            TCArg("signed char *__restrict__ ", "expr_1_out_1")
        )
    );
    
    LibKernel(
        "s366_kernel",
        CALL_PARALLEL,
        0,
        "s366_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s144_kernel_args_t",
        CArgs(5,
            TCArg("unsigned int", "I0"),
            TCArg("unsigned int", "I1"),
            TCArg("signed char *__restrict__ ", "expr_11_in_0"),
            TCArg("signed char *__restrict__ ", "expr_11_in_1"),
            TCArg("signed char *__restrict__ ", "expr_11_out_0")
        )
    );
    
    LibKernel(
        "s144_kernel",
        CALL_PARALLEL,
        0,
        "s144_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s164_kernel_args_t",
        CArgs(5,
            TCArg("unsigned int", "I0"),
            TCArg("unsigned int", "I1"),
            TCArg("signed char *__restrict__ ", "expr_14_in_0"),
            TCArg("signed char *__restrict__ ", "expr_14_in_1"),
            TCArg("signed char *__restrict__ ", "expr_14_out_0")
        )
    );
    
    LibKernel(
        "s164_kernel",
        CALL_PARALLEL,
        0,
        "s164_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s45_kernel_args_t",
        CArgs(5,
            TCArg("unsigned int", "I0"),
            TCArg("unsigned int", "I1"),
            TCArg("signed char *__restrict__ ", "expr_16_in_0"),
            TCArg("signed char *__restrict__ ", "expr_16_in_1"),
            TCArg("signed char *__restrict__ ", "expr_16_out_0")
        )
    );
    
    LibKernel(
        "s45_kernel",
        CALL_PARALLEL,
        0,
        "s45_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s64_kernel_args_t",
        CArgs(5,
            TCArg("unsigned int", "I0"),
            TCArg("unsigned int", "I1"),
            TCArg("signed char *__restrict__ ", "expr_17_in_0"),
            TCArg("signed char *__restrict__ ", "expr_17_in_1"),
            TCArg("signed char *__restrict__ ", "expr_17_out_0")
        )
    );
    
    LibKernel(
        "s64_kernel",
        CALL_PARALLEL,
        0,
        "s64_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s84_kernel_args_t",
        CArgs(5,
            TCArg("unsigned int", "I0"),
            TCArg("unsigned int", "I1"),
            TCArg("signed char *__restrict__ ", "expr_18_in_0"),
            TCArg("signed char *__restrict__ ", "expr_18_in_1"),
            TCArg("signed char *__restrict__ ", "expr_18_out_0")
        )
    );
    
    LibKernel(
        "s84_kernel",
        CALL_PARALLEL,
        0,
        "s84_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s184_kernel_args_t",
        CArgs(5,
            TCArg("unsigned int", "I0"),
            TCArg("unsigned int", "I1"),
            TCArg("signed char *__restrict__ ", "expr_26_in_0"),
            TCArg("signed char *__restrict__ ", "expr_26_in_1"),
            TCArg("signed char *__restrict__ ", "expr_26_out_0")
        )
    );
    
    LibKernel(
        "s184_kernel",
        CALL_PARALLEL,
        0,
        "s184_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s204_kernel_args_t",
        CArgs(5,
            TCArg("unsigned int", "I0"),
            TCArg("unsigned int", "I1"),
            TCArg("signed char *__restrict__ ", "expr_29_in_0"),
            TCArg("signed char *__restrict__ ", "expr_29_in_1"),
            TCArg("signed char *__restrict__ ", "expr_29_out_0")
        )
    );
    
    LibKernel(
        "s204_kernel",
        CALL_PARALLEL,
        0,
        "s204_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s224_kernel_args_t",
        CArgs(5,
            TCArg("unsigned int", "I0"),
            TCArg("unsigned int", "I1"),
            TCArg("signed char *__restrict__ ", "expr_32_in_0"),
            TCArg("signed char *__restrict__ ", "expr_32_in_1"),
            TCArg("signed char *__restrict__ ", "expr_32_out_0")
        )
    );
    
    LibKernel(
        "s224_kernel",
        CALL_PARALLEL,
        0,
        "s224_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "expr_6_args_t",
        CArgs(5,
            TCArg("signed char *__restrict__ ", "expr_6_in_0"),
            TCArg("signed char *__restrict__ ", "expr_6_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_6",
        CALL_PARALLEL,
        0,
        "expr_6_args_t",
        0
    );
    LibKernelTemplate(
        "expr_19_args_t",
        CArgs(5,
            TCArg("signed char *__restrict__ ", "expr_19_in_0"),
            TCArg("signed char *__restrict__ ", "expr_19_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_19",
        CALL_PARALLEL,
        0,
        "expr_19_args_t",
        0
    );
    LibKernelTemplate(
        "expr_20_args_t",
        CArgs(5,
            TCArg("signed char *__restrict__ ", "expr_20_in_0"),
            TCArg("signed char *__restrict__ ", "expr_20_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_20",
        CALL_PARALLEL,
        0,
        "expr_20_args_t",
        0
    );
    LibKernelTemplate(
        "expr_21_args_t",
        CArgs(5,
            TCArg("signed char *__restrict__ ", "expr_21_in_0"),
            TCArg("signed char *__restrict__ ", "expr_21_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_21",
        CALL_PARALLEL,
        0,
        "expr_21_args_t",
        0
    );
    LibKernelTemplate(
        "expr_22_args_t",
        CArgs(5,
            TCArg("signed char *__restrict__ ", "expr_22_in_0"),
            TCArg("signed char *__restrict__ ", "expr_22_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_22",
        CALL_PARALLEL,
        0,
        "expr_22_args_t",
        0
    );
    LibKernelTemplate(
        "expr_23_args_t",
        CArgs(5,
            TCArg("signed char *__restrict__ ", "expr_23_in_0"),
            TCArg("signed char *__restrict__ ", "expr_23_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_23",
        CALL_PARALLEL,
        0,
        "expr_23_args_t",
        0
    );
    LibKernelTemplate(
        "expr_24_args_t",
        CArgs(5,
            TCArg("signed char *__restrict__ ", "expr_24_in_0"),
            TCArg("signed char *__restrict__ ", "expr_24_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_24",
        CALL_PARALLEL,
        0,
        "expr_24_args_t",
        0
    );
    LibKernelTemplate(
        "expr_7_args_t",
        CArgs(5,
            TCArg("signed char *__restrict__ ", "expr_7_in_0"),
            TCArg("signed char *__restrict__ ", "expr_7_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_7",
        CALL_PARALLEL,
        0,
        "expr_7_args_t",
        0
    );
    LibKernelTemplate(
        "expr_8_args_t",
        CArgs(5,
            TCArg("signed char *__restrict__ ", "expr_8_in_0"),
            TCArg("signed char *__restrict__ ", "expr_8_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_8",
        CALL_PARALLEL,
        0,
        "expr_8_args_t",
        0
    );
    LibKernelTemplate(
        "expr_9_args_t",
        CArgs(5,
            TCArg("signed char *__restrict__ ", "expr_9_in_0"),
            TCArg("signed char *__restrict__ ", "expr_9_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_9",
        CALL_PARALLEL,
        0,
        "expr_9_args_t",
        0
    );
    LibKernelTemplate(
        "expr_10_args_t",
        CArgs(5,
            TCArg("signed char *__restrict__ ", "expr_10_in_0"),
            TCArg("signed char *__restrict__ ", "expr_10_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_10",
        CALL_PARALLEL,
        0,
        "expr_10_args_t",
        0
    );
    LibKernelTemplate(
        "expr_12_args_t",
        CArgs(5,
            TCArg("signed char *__restrict__ ", "expr_12_in_0"),
            TCArg("signed char *__restrict__ ", "expr_12_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_12",
        CALL_PARALLEL,
        0,
        "expr_12_args_t",
        0
    );
    LibKernelTemplate(
        "expr_13_args_t",
        CArgs(5,
            TCArg("signed char *__restrict__ ", "expr_13_in_0"),
            TCArg("signed char *__restrict__ ", "expr_13_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_13",
        CALL_PARALLEL,
        0,
        "expr_13_args_t",
        0
    );
    LibKernelTemplate(
        "expr_15_args_t",
        CArgs(5,
            TCArg("signed char *__restrict__ ", "expr_15_in_0"),
            TCArg("signed char *__restrict__ ", "expr_15_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_15",
        CALL_PARALLEL,
        0,
        "expr_15_args_t",
        0
    );
    LibKernelTemplate(
        "expr_25_args_t",
        CArgs(5,
            TCArg("signed char *__restrict__ ", "expr_25_in_0"),
            TCArg("signed char *__restrict__ ", "expr_25_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_25",
        CALL_PARALLEL,
        0,
        "expr_25_args_t",
        0
    );
    LibKernelTemplate(
        "expr_27_args_t",
        CArgs(5,
            TCArg("signed char *__restrict__ ", "expr_27_in_0"),
            TCArg("signed char *__restrict__ ", "expr_27_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_27",
        CALL_PARALLEL,
        0,
        "expr_27_args_t",
        0
    );
    LibKernelTemplate(
        "expr_28_args_t",
        CArgs(5,
            TCArg("signed char *__restrict__ ", "expr_28_in_0"),
            TCArg("signed char *__restrict__ ", "expr_28_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_28",
        CALL_PARALLEL,
        0,
        "expr_28_args_t",
        0
    );
    LibKernelTemplate(
        "expr_30_args_t",
        CArgs(5,
            TCArg("signed char *__restrict__ ", "expr_30_in_0"),
            TCArg("signed char *__restrict__ ", "expr_30_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_30",
        CALL_PARALLEL,
        0,
        "expr_30_args_t",
        0
    );
    LibKernelTemplate(
        "expr_31_args_t",
        CArgs(5,
            TCArg("signed char *__restrict__ ", "expr_31_in_0"),
            TCArg("signed char *__restrict__ ", "expr_31_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_31",
        CALL_PARALLEL,
        0,
        "expr_31_args_t",
        0
    );
    LibKernelTemplate(
        "expr_33_args_t",
        CArgs(5,
            TCArg("signed char *__restrict__ ", "expr_33_in_0"),
            TCArg("signed char *__restrict__ ", "expr_33_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_33",
        CALL_PARALLEL,
        0,
        "expr_33_args_t",
        0
    );
    LibKernelTemplate(
        "expr_2_args_t",
        CArgs(5,
            TCArg("signed char *__restrict__ ", "expr_2_in_0"),
            TCArg("signed char *__restrict__ ", "expr_2_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_2",
        CALL_PARALLEL,
        0,
        "expr_2_args_t",
        0
    );
    LibKernelTemplate(
        "expr_2_1_args_t",
        CArgs(5,
            TCArg("signed char *__restrict__ ", "expr_2_in_0"),
            TCArg("signed char *__restrict__ ", "expr_2_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_2_1",
        CALL_PARALLEL,
        0,
        "expr_2_1_args_t",
        0
    );
    LibKernelTemplate(
        "expr_2_2_args_t",
        CArgs(5,
            TCArg("signed char *__restrict__ ", "expr_2_in_0"),
            TCArg("signed char *__restrict__ ", "expr_2_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_2_2",
        CALL_PARALLEL,
        0,
        "expr_2_2_args_t",
        0
    );
    LibKernelTemplate(
        "expr_2_3_args_t",
        CArgs(5,
            TCArg("signed char *__restrict__ ", "expr_2_in_0"),
            TCArg("signed char *__restrict__ ", "expr_2_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_2_3",
        CALL_PARALLEL,
        0,
        "expr_2_3_args_t",
        0
    );
    LibKernelTemplate(
        "expr_2_4_args_t",
        CArgs(5,
            TCArg("signed char *__restrict__ ", "expr_2_in_0"),
            TCArg("signed char *__restrict__ ", "expr_2_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_2_4",
        CALL_PARALLEL,
        0,
        "expr_2_4_args_t",
        0
    );
    LibKernelTemplate(
        "expr_2_5_args_t",
        CArgs(5,
            TCArg("signed char *__restrict__ ", "expr_2_in_0"),
            TCArg("signed char *__restrict__ ", "expr_2_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_2_5",
        CALL_PARALLEL,
        0,
        "expr_2_5_args_t",
        0
    );
}



int s372_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (1, 1, 2574) spaces: ((2,),) 
        // parametric_spaces: ((2,),) 
        // exterior_shape: (2574.0,) 
        KernelIterSpace(2, IterParSpace(KER_ITER_D0, 2574, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(7,
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_0_in_0"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_0_in_1"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_0_in_2"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_0_in_3"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_0_in_4"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_0_out_0"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_0_out_1")
        ),
        Calls(1,
            Call("s372_kernel", LOC_D0,
                Bindings(8,
                    K_ArgPar("expr_0_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_Arg("expr_0_in_0", KER_ARG_TILE),
                    K_Arg("expr_0_in_1", KER_ARG_TILE),
                    K_Arg("expr_0_in_2", KER_ARG_TILE),
                    K_Arg("expr_0_in_3", KER_ARG_TILE),
                    K_Arg("expr_0_in_4", KER_ARG_TILE),
                    K_Arg("expr_0_out_0", KER_ARG_TILE),
                    K_Arg("expr_0_out_1", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_0_in_0 axes: (0,)
        // var: expr_0_in_1 axes: (0,)
        // var: expr_0_in_2 axes: (0,)
        // var: expr_0_in_4 axes: (0,)
        // var: expr_0_in_3 axes: (0,)
        // var: expr_0_out_0 axes: (0,)
        // var: expr_0_out_1 axes: (0,)
        KerArgs(7,
            KerArg("expr_0_in_0",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_0_in_0"),
            KerArg("expr_0_in_1",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_0_in_1"),
            KerArg("expr_0_in_2",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_0_in_2"),
            KerArg("expr_0_in_4",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_0_in_4"),
            KerArg("expr_0_in_3",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_0_in_3"),
            KerArg("expr_0_out_0", KerArgSpace(1, KER_ITER_D0), O_OUT|O_DB, 1, 1, 1, 0, 0, 0, "expr_0_out_0"),
            KerArg("expr_0_out_1", KerArgSpace(1, KER_ITER_D0), O_OUT|O_DB, 1, 1, 1, 0, 0, 0, "expr_0_out_1")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 87516, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 18018, 0);
        AddKernelArgDim(Name, "expr_0_in_0",  2, 2574, 1);
        AddKernelArgDim(Name, "expr_0_in_1",  2, 2574, 1);
        AddKernelArgDim(Name, "expr_0_in_2",  2, 2574, 1);
        AddKernelArgDim(Name, "expr_0_in_3",  2, 2574, 1);
        AddKernelArgDim(Name, "expr_0_in_4",  2, 2574, 1);
        AddKernelArgDim(Name, "expr_0_out_0", 2, 2574, 1);
        AddKernelArgDim(Name, "expr_0_out_1", 2, 2574, 1);
    }
    return (Kernel!=0);
}
int s366_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (1, 1, 2574) spaces: ((2,),) 
        // parametric_spaces: ((2,),) 
        // exterior_shape: (2574.0,) 
        KernelIterSpace(2, IterParSpace(KER_ITER_D0, 2574, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(7,
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_1_in_0"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_1_in_1"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_1_in_2"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_1_in_3"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_1_in_4"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_1_out_0"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_1_out_1")
        ),
        Calls(1,
            Call("s366_kernel", LOC_D0,
                Bindings(8,
                    K_ArgPar("expr_1_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_Arg("expr_1_in_0", KER_ARG_TILE),
                    K_Arg("expr_1_in_1", KER_ARG_TILE),
                    K_Arg("expr_1_in_2", KER_ARG_TILE),
                    K_Arg("expr_1_in_3", KER_ARG_TILE),
                    K_Arg("expr_1_in_4", KER_ARG_TILE),
                    K_Arg("expr_1_out_0", KER_ARG_TILE),
                    K_Arg("expr_1_out_1", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_1_in_0 axes: (0,)
        // var: expr_1_in_1 axes: (0,)
        // var: expr_1_in_2 axes: (0,)
        // var: expr_1_in_4 axes: (0,)
        // var: expr_1_in_3 axes: (0,)
        // var: expr_1_out_0 axes: (0,)
        // var: expr_1_out_1 axes: (0,)
        KerArgs(7,
            KerArg("expr_1_in_0",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_1_in_0"),
            KerArg("expr_1_in_1",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_1_in_1"),
            KerArg("expr_1_in_2",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_1_in_2"),
            KerArg("expr_1_in_4",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_1_in_4"),
            KerArg("expr_1_in_3",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_1_in_3"),
            KerArg("expr_1_out_0", KerArgSpace(1, KER_ITER_D0), O_OUT|O_DB, 1, 1, 1, 0, 0, 0, "expr_1_out_0"),
            KerArg("expr_1_out_1", KerArgSpace(1, KER_ITER_D0), O_OUT|O_DB, 1, 1, 1, 0, 0, 0, "expr_1_out_1")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 87516, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 18018, 0);
        AddKernelArgDim(Name, "expr_1_in_0",  2, 2574, 1);
        AddKernelArgDim(Name, "expr_1_in_1",  2, 2574, 1);
        AddKernelArgDim(Name, "expr_1_in_2",  2, 2574, 1);
        AddKernelArgDim(Name, "expr_1_in_3",  2, 2574, 1);
        AddKernelArgDim(Name, "expr_1_in_4",  2, 2574, 1);
        AddKernelArgDim(Name, "expr_1_out_0", 2, 2574, 1);
        AddKernelArgDim(Name, "expr_1_out_1", 2, 2574, 1);
    }
    return (Kernel!=0);
}
int s144_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (16, 20, 480) spaces: ((0, 1), (2,)) 
        // parametric_spaces: ((0, 1), (2,)) 
        // exterior_shape: (320, 480.0) 
        KernelIterSpace(3, IterParSpace(KER_ITER_D0, 320, 1), IterParSpace(KER_ITER_D1, 480, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(3,
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_11_in_0"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_11_in_1"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_11_out_0")
        ),
        Calls(1,
            Call("s144_kernel", LOC_D1,
                Bindings(5,
                    K_ArgPar("expr_11_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_ArgPar("expr_11_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D1),
                    K_Arg("expr_11_in_0", KER_ARG_TILE),
                    K_Arg("expr_11_in_1", KER_ARG_TILE),
                    K_Arg("expr_11_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_11_out_0 axes: (0, 1)
        // var: expr_11_in_0 axes: (1,)
        // var: expr_11_in_1 axes: (0, 1)
        KerArgs(3,
            KerArg("expr_11_out_0", KerArgSpace(2, KER_ITER_D0, KER_ITER_D1), O_OUT|O_DB, 1, 1, 1, 0, 0, 0, "expr_11_out_0"),
            KerArg("expr_11_in_0",  KerArgSpace(1, KER_ITER_D1),              O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_11_in_0"),
            KerArg("expr_11_in_1",  KerArgSpace(2, KER_ITER_D0, KER_ITER_D1), O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_11_in_1")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 1075200, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 307680, 0);
        AddKernelArgDim(Name, "expr_11_in_0",  2, 480,         1);
        AddKernelArgDim(Name, "expr_11_in_1",  4, 16, 20, 480, 1);
        AddKernelArgDim(Name, "expr_11_out_0", 4, 16, 20, 480, 1);
    }
    return (Kernel!=0);
}
int s164_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (16, 20, 672) spaces: ((0, 1), (2,)) 
        // parametric_spaces: ((0, 1), (2,)) 
        // exterior_shape: (320, 672.0) 
        KernelIterSpace(3, IterParSpace(KER_ITER_D0, 320, 1), IterParSpace(KER_ITER_D1, 672, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(3,
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_14_in_0"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_14_in_1"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_14_out_0")
        ),
        Calls(1,
            Call("s164_kernel", LOC_D1,
                Bindings(5,
                    K_ArgPar("expr_14_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_ArgPar("expr_14_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D1),
                    K_Arg("expr_14_in_0", KER_ARG_TILE),
                    K_Arg("expr_14_in_1", KER_ARG_TILE),
                    K_Arg("expr_14_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_14_out_0 axes: (0, 1)
        // var: expr_14_in_0 axes: (1,)
        // var: expr_14_in_1 axes: (0, 1)
        KerArgs(3,
            KerArg("expr_14_out_0", KerArgSpace(2, KER_ITER_D0, KER_ITER_D1), O_OUT|O_DB, 1, 1, 1, 0, 0, 0, "expr_14_out_0"),
            KerArg("expr_14_in_0",  KerArgSpace(1, KER_ITER_D1),              O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_14_in_0"),
            KerArg("expr_14_in_1",  KerArgSpace(2, KER_ITER_D0, KER_ITER_D1), O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_14_in_1")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 1505280, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 430752, 0);
        AddKernelArgDim(Name, "expr_14_in_0",  2, 672,         1);
        AddKernelArgDim(Name, "expr_14_in_1",  4, 16, 20, 672, 1);
        AddKernelArgDim(Name, "expr_14_out_0", 4, 16, 20, 672, 1);
    }
    return (Kernel!=0);
}
int s45_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (32, 40, 72) spaces: ((0, 1), (2,)) 
        // parametric_spaces: ((0, 1), (2,)) 
        // exterior_shape: (1280, 72.0) 
        KernelIterSpace(3, IterParSpace(KER_ITER_D0, 1280, 1), IterParSpace(KER_ITER_D1, 72, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(3,
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_16_in_0"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_16_in_1"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_16_out_0")
        ),
        Calls(1,
            Call("s45_kernel", LOC_D1,
                Bindings(5,
                    K_ArgPar("expr_16_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_ArgPar("expr_16_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D1),
                    K_Arg("expr_16_in_0", KER_ARG_TILE),
                    K_Arg("expr_16_in_1", KER_ARG_TILE),
                    K_Arg("expr_16_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_16_out_0 axes: (0, 1)
        // var: expr_16_in_1 axes: (1,)
        // var: expr_16_in_0 axes: (0, 1)
        KerArgs(3,
            KerArg("expr_16_out_0", KerArgSpace(2, KER_ITER_D0, KER_ITER_D1), O_OUT|O_DB, 1, 1, 1, 0, 0, 0, "expr_16_out_0"),
            KerArg("expr_16_in_1",  KerArgSpace(1, KER_ITER_D1),              O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_16_in_1"),
            KerArg("expr_16_in_0",  KerArgSpace(2, KER_ITER_D0, KER_ITER_D1), O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_16_in_0")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 645120, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 184392, 0);
        AddKernelArgDim(Name, "expr_16_in_0",  4, 32, 40, 72, 1);
        AddKernelArgDim(Name, "expr_16_in_1",  2, 72,         1);
        AddKernelArgDim(Name, "expr_16_out_0", 4, 32, 40, 72, 1);
    }
    return (Kernel!=0);
}
int s64_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (32, 40, 120) spaces: ((0, 1), (2,)) 
        // parametric_spaces: ((0, 1), (2,)) 
        // exterior_shape: (1280, 120.0) 
        KernelIterSpace(3, IterParSpace(KER_ITER_D0, 1280, 1), IterParSpace(KER_ITER_D1, 120, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(3,
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_17_in_0"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_17_in_1"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_17_out_0")
        ),
        Calls(1,
            Call("s64_kernel", LOC_D1,
                Bindings(5,
                    K_ArgPar("expr_17_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_ArgPar("expr_17_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D1),
                    K_Arg("expr_17_in_0", KER_ARG_TILE),
                    K_Arg("expr_17_in_1", KER_ARG_TILE),
                    K_Arg("expr_17_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_17_out_0 axes: (0, 1)
        // var: expr_17_in_1 axes: (1,)
        // var: expr_17_in_0 axes: (0, 1)
        KerArgs(3,
            KerArg("expr_17_out_0", KerArgSpace(2, KER_ITER_D0, KER_ITER_D1), O_OUT|O_DB, 1, 1, 1, 0, 0, 0, "expr_17_out_0"),
            KerArg("expr_17_in_1",  KerArgSpace(1, KER_ITER_D1),              O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_17_in_1"),
            KerArg("expr_17_in_0",  KerArgSpace(2, KER_ITER_D0, KER_ITER_D1), O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_17_in_0")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 1075200, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 307320, 0);
        AddKernelArgDim(Name, "expr_17_in_0",  4, 32, 40, 120, 1);
        AddKernelArgDim(Name, "expr_17_in_1",  2, 120,         1);
        AddKernelArgDim(Name, "expr_17_out_0", 4, 32, 40, 120, 1);
    }
    return (Kernel!=0);
}
int s84_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (32, 40, 120) spaces: ((0, 1), (2,)) 
        // parametric_spaces: ((0, 1), (2,)) 
        // exterior_shape: (1280, 120.0) 
        KernelIterSpace(3, IterParSpace(KER_ITER_D0, 1280, 1), IterParSpace(KER_ITER_D1, 120, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(3,
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_18_in_0"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_18_in_1"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_18_out_0")
        ),
        Calls(1,
            Call("s84_kernel", LOC_D1,
                Bindings(5,
                    K_ArgPar("expr_18_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_ArgPar("expr_18_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D1),
                    K_Arg("expr_18_in_0", KER_ARG_TILE),
                    K_Arg("expr_18_in_1", KER_ARG_TILE),
                    K_Arg("expr_18_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_18_out_0 axes: (0, 1)
        // var: expr_18_in_1 axes: (1,)
        // var: expr_18_in_0 axes: (0, 1)
        KerArgs(3,
            KerArg("expr_18_out_0", KerArgSpace(2, KER_ITER_D0, KER_ITER_D1), O_OUT|O_DB, 1, 1, 1, 0, 0, 0, "expr_18_out_0"),
            KerArg("expr_18_in_1",  KerArgSpace(1, KER_ITER_D1),              O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_18_in_1"),
            KerArg("expr_18_in_0",  KerArgSpace(2, KER_ITER_D0, KER_ITER_D1), O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_18_in_0")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 1075200, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 307320, 0);
        AddKernelArgDim(Name, "expr_18_in_0",  4, 32, 40, 120, 1);
        AddKernelArgDim(Name, "expr_18_in_1",  2, 120,         1);
        AddKernelArgDim(Name, "expr_18_out_0", 4, 32, 40, 120, 1);
    }
    return (Kernel!=0);
}
int s184_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (8, 10, 672) spaces: ((0, 1), (2,)) 
        // parametric_spaces: ((0, 1), (2,)) 
        // exterior_shape: (80, 672.0) 
        KernelIterSpace(3, IterParSpace(KER_ITER_D0, 80, 1), IterParSpace(KER_ITER_D1, 672, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(3,
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_26_in_0"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_26_in_1"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_26_out_0")
        ),
        Calls(1,
            Call("s184_kernel", LOC_D1,
                Bindings(5,
                    K_ArgPar("expr_26_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_ArgPar("expr_26_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D1),
                    K_Arg("expr_26_in_0", KER_ARG_TILE),
                    K_Arg("expr_26_in_1", KER_ARG_TILE),
                    K_Arg("expr_26_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_26_out_0 axes: (0, 1)
        // var: expr_26_in_0 axes: (1,)
        // var: expr_26_in_1 axes: (0, 1)
        KerArgs(3,
            KerArg("expr_26_out_0", KerArgSpace(2, KER_ITER_D0, KER_ITER_D1), O_OUT|O_DB, 1, 1, 1, 0, 0, 0, "expr_26_out_0"),
            KerArg("expr_26_in_0",  KerArgSpace(1, KER_ITER_D1),              O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_26_in_0"),
            KerArg("expr_26_in_1",  KerArgSpace(2, KER_ITER_D0, KER_ITER_D1), O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_26_in_1")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 376320, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 108192, 0);
        AddKernelArgDim(Name, "expr_26_in_0",  2, 672,        1);
        AddKernelArgDim(Name, "expr_26_in_1",  4, 8, 10, 672, 1);
        AddKernelArgDim(Name, "expr_26_out_0", 4, 8, 10, 672, 1);
    }
    return (Kernel!=0);
}
int s204_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (8, 10, 480) spaces: ((0, 1), (2,)) 
        // parametric_spaces: ((0, 1), (2,)) 
        // exterior_shape: (80, 480.0) 
        KernelIterSpace(3, IterParSpace(KER_ITER_D0, 80, 1), IterParSpace(KER_ITER_D1, 480, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(3,
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_29_in_0"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_29_in_1"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_29_out_0")
        ),
        Calls(1,
            Call("s204_kernel", LOC_D1,
                Bindings(5,
                    K_ArgPar("expr_29_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_ArgPar("expr_29_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D1),
                    K_Arg("expr_29_in_0", KER_ARG_TILE),
                    K_Arg("expr_29_in_1", KER_ARG_TILE),
                    K_Arg("expr_29_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_29_out_0 axes: (0, 1)
        // var: expr_29_in_0 axes: (1,)
        // var: expr_29_in_1 axes: (0, 1)
        KerArgs(3,
            KerArg("expr_29_out_0", KerArgSpace(2, KER_ITER_D0, KER_ITER_D1), O_OUT|O_DB, 1, 1, 1, 0, 0, 0, "expr_29_out_0"),
            KerArg("expr_29_in_0",  KerArgSpace(1, KER_ITER_D1),              O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_29_in_0"),
            KerArg("expr_29_in_1",  KerArgSpace(2, KER_ITER_D0, KER_ITER_D1), O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_29_in_1")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 268800, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 77280, 0);
        AddKernelArgDim(Name, "expr_29_in_0",  2, 480,        1);
        AddKernelArgDim(Name, "expr_29_in_1",  4, 8, 10, 480, 1);
        AddKernelArgDim(Name, "expr_29_out_0", 4, 8, 10, 480, 1);
    }
    return (Kernel!=0);
}
int s224_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (8, 10, 480) spaces: ((0, 1), (2,)) 
        // parametric_spaces: ((0, 1), (2,)) 
        // exterior_shape: (80, 480.0) 
        KernelIterSpace(3, IterParSpace(KER_ITER_D0, 80, 1), IterParSpace(KER_ITER_D1, 480, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(3,
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_32_in_0"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_32_in_1"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_32_out_0")
        ),
        Calls(1,
            Call("s224_kernel", LOC_D1,
                Bindings(5,
                    K_ArgPar("expr_32_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_ArgPar("expr_32_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D1),
                    K_Arg("expr_32_in_0", KER_ARG_TILE),
                    K_Arg("expr_32_in_1", KER_ARG_TILE),
                    K_Arg("expr_32_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_32_out_0 axes: (0, 1)
        // var: expr_32_in_0 axes: (1,)
        // var: expr_32_in_1 axes: (0, 1)
        KerArgs(3,
            KerArg("expr_32_out_0", KerArgSpace(2, KER_ITER_D0, KER_ITER_D1), O_OUT|O_DB, 1, 1, 1, 0, 0, 0, "expr_32_out_0"),
            KerArg("expr_32_in_0",  KerArgSpace(1, KER_ITER_D1),              O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_32_in_0"),
            KerArg("expr_32_in_1",  KerArgSpace(2, KER_ITER_D0, KER_ITER_D1), O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_32_in_1")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 268800, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 77280, 0);
        AddKernelArgDim(Name, "expr_32_in_0",  2, 480,        1);
        AddKernelArgDim(Name, "expr_32_in_1",  4, 8, 10, 480, 1);
        AddKernelArgDim(Name, "expr_32_out_0", 4, 8, 10, 480, 1);
    }
    return (Kernel!=0);
}

void Mobv3SSDLiteModel(unsigned int L1Memory, unsigned int L2Memory, unsigned int L3Memory, unsigned int L3Flash)
{
    KernelOper_T Cop = KOP_CONV;

    // SetKernelOpts(KER_OPT_NONE, KER_OPT_BUFFER_PROMOTE);
    SetSymbolDynamics();

    SetUsedFilesNames(0, 5, "Gap.h", "Mobv3SSDLite.h", "CNN_BasicKernels_SQ8.h", "CNN_BasicKernels_SQ8.h", "Expression_Kernels.h");
    SetGeneratedFilesNames("Mobv3SSDLiteKernels.c", "Mobv3SSDLiteKernels.h");
    AT_SetGraphCtrl(AT_GRAPH_MONITOR_CYCLES, AT_OPT_ON);
    AT_SetGraphCtrl(AT_GRAPH_PRODUCE_NODE_NAMES, AT_OPT_ON);
    AT_SetGraphCtrl(AT_GRAPH_PRODUCE_OPERINFOS, AT_OPT_ON);

    SetMemoryDeviceInfos(4,
        AT_MEM_L1, L1Memory, "Mobv3SSDLite_L1_Memory", 0, 0,
        AT_MEM_L2, L2Memory, "Mobv3SSDLite_L2_Memory", 0, 1,
        AT_MEM_L3_DEFAULTRAM, L3Memory, "Mobv3SSDLite_L3_Memory", 0, 0,
        AT_MEM_L3_DEFAULTFLASH, L3Flash, "Mobv3SSDLite_L3_Flash", "Mobv3SSDLite_L3_Flash_Const.dat", 0
    );

    LoadCNN_SQ8_Library();
    LoadCNN_Copy_Library();
    load_expressions_kernels();

    CNN_GenControl_T gen_ctrl_S3_Conv2d_16x3x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S3_Conv2d_16x3x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S3_Conv2d_16x3x3x3_Custom, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S3_Conv2d_16x3x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_6");
    // generator for _backbone_features_0_features_0_0_features_0_0_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S3_Conv2d_16x3x3x3_Custom", &gen_ctrl_S3_Conv2d_16x3x3x3_Custom,
                               4, 1,
                               3, 16, 320, 256,
                               KOP_CONV, 3, 3, 1, 1, 2, 2, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S6_Conv2d_16x3x3x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S6_Conv2d_16x3x3x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S6_Conv2d_16x3x3x1_Relu, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_1_block_block_0_block_0_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S6_Conv2d_16x3x3x1_Relu", &gen_ctrl_S6_Conv2d_16x3x3x1_Relu,
                               4, 1,
                               16, 16, 160, 128,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S9_Conv2d_16x1x1x16;
    CNN_InitGenCtrl(&gen_ctrl_S9_Conv2d_16x1x1x16);
    CNN_SetGenCtrl(&gen_ctrl_S9_Conv2d_16x1x1x16, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_1_block_block_1_block_1_0_Conv
    CNN_ConvolutionPoolAct_SQ8("S9_Conv2d_16x1x1x16", &gen_ctrl_S9_Conv2d_16x1x1x16,
                               4, 1,
                               16, 16, 160, 128,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    
    // generator for _backbone_features_0_features_0_1_Add
    CNN_MatAddAct_SQ8("S10_MatAdd_128x160x16", 0, 128, 160, 16, KOP_MATADD, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S13_Conv2d_64x1x1x16_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S13_Conv2d_64x1x1x16_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S13_Conv2d_64x1x1x16_Relu, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_2_block_block_0_block_0_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S13_Conv2d_64x1x1x16_Relu", &gen_ctrl_S13_Conv2d_64x1x1x16_Relu,
                               4, 1,
                               16, 64, 160, 128,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S16_Conv2d_64x3x3x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S16_Conv2d_64x3x3x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S16_Conv2d_64x3x3x1_Relu, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_2_block_block_1_block_1_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S16_Conv2d_64x3x3x1_Relu", &gen_ctrl_S16_Conv2d_64x3x3x1_Relu,
                               4, 1,
                               64, 64, 160, 128,
                               KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S19_Conv2d_24x1x1x64;
    CNN_InitGenCtrl(&gen_ctrl_S19_Conv2d_24x1x1x64);
    CNN_SetGenCtrl(&gen_ctrl_S19_Conv2d_24x1x1x64, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_2_block_block_2_block_2_0_Conv
    CNN_ConvolutionPoolAct_SQ8("S19_Conv2d_24x1x1x64", &gen_ctrl_S19_Conv2d_24x1x1x64,
                               4, 1,
                               64, 24, 80, 64,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S22_Conv2d_72x1x1x24_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S22_Conv2d_72x1x1x24_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S22_Conv2d_72x1x1x24_Relu, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_3_block_block_0_block_0_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S22_Conv2d_72x1x1x24_Relu", &gen_ctrl_S22_Conv2d_72x1x1x24_Relu,
                               4, 1,
                               24, 72, 80, 64,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S25_Conv2d_72x3x3x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S25_Conv2d_72x3x3x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S25_Conv2d_72x3x3x1_Relu, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_3_block_block_1_block_1_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S25_Conv2d_72x3x3x1_Relu", &gen_ctrl_S25_Conv2d_72x3x3x1_Relu,
                               4, 1,
                               72, 72, 80, 64,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S28_Conv2d_24x1x1x72;
    CNN_InitGenCtrl(&gen_ctrl_S28_Conv2d_24x1x1x72);
    CNN_SetGenCtrl(&gen_ctrl_S28_Conv2d_24x1x1x72, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_3_block_block_2_block_2_0_Conv
    CNN_ConvolutionPoolAct_SQ8("S28_Conv2d_24x1x1x72", &gen_ctrl_S28_Conv2d_24x1x1x72,
                               4, 1,
                               72, 24, 80, 64,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    
    // generator for _backbone_features_0_features_0_3_Add
    CNN_MatAddAct_SQ8("S29_MatAdd_64x80x24", 0, 64, 80, 24, KOP_MATADD, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S32_Conv2d_72x1x1x24_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S32_Conv2d_72x1x1x24_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S32_Conv2d_72x1x1x24_Relu, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_4_block_block_0_block_0_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S32_Conv2d_72x1x1x24_Relu", &gen_ctrl_S32_Conv2d_72x1x1x24_Relu,
                               4, 1,
                               24, 72, 80, 64,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S35_Conv2d_72x5x5x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S35_Conv2d_72x5x5x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S35_Conv2d_72x5x5x1_Relu, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_4_block_block_1_block_1_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S35_Conv2d_72x5x5x1_Relu", &gen_ctrl_S35_Conv2d_72x5x5x1_Relu,
                               4, 1,
                               72, 72, 80, 64,
                               KOP_CONV_DW, 5, 5, 1, 1, 2, 2, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    
    // generator for _backbone_features_0_features_0_4_block_block_2_avgpool_GlobalAveragePool_trans Transpose 32x40x72 -> 72x32x40 ((1, 0))
    CNN_MatTranspose("S36_Op__backbone_features_0_features_0_4_block_block_2_avgpool_GlobalAveragePool_trans", 0, 1,
                      1, 72, 1280, KOP_MATTRANSP);
    
    // generator for _backbone_features_0_features_0_4_block_block_2_avgpool_GlobalAveragePool
    CNN_GlobalPoolAct_SQ8("S37_Op__backbone_features_0_features_0_4_block_block_2_avgpool_GlobalAveragePool", 0,
                          72, 16, 80,
                          KOP_GLOBAL_AVGPOOL, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S41_Conv2d_24x1x1x72_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S41_Conv2d_24x1x1x72_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S41_Conv2d_24x1x1x72_Relu, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_4_block_block_2_fc1_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S41_Conv2d_24x1x1x72_Relu", &gen_ctrl_S41_Conv2d_24x1x1x72_Relu,
                               4, 1,
                               72, 24, 1, 1,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S44_Conv2d_72x1x1x24_Hsigmoid;
    CNN_InitGenCtrl(&gen_ctrl_S44_Conv2d_72x1x1x24_Hsigmoid);
    CNN_SetGenCtrl(&gen_ctrl_S44_Conv2d_72x1x1x24_Hsigmoid, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_4_block_block_2_fc2_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S44_Conv2d_72x1x1x24_Hsigmoid", &gen_ctrl_S44_Conv2d_72x1x1x24_Hsigmoid,
                               4, 1,
                               24, 72, 1, 1,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_HSIGMOID);
    
    
    // generator for expr_16
    s45_kernel_gen("S45_Op_expr_16");
    
    CNN_GenControl_T gen_ctrl_S48_Conv2d_40x1x1x72;
    CNN_InitGenCtrl(&gen_ctrl_S48_Conv2d_40x1x1x72);
    CNN_SetGenCtrl(&gen_ctrl_S48_Conv2d_40x1x1x72, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_4_block_block_3_block_3_0_Conv
    CNN_ConvolutionPoolAct_SQ8("S48_Conv2d_40x1x1x72", &gen_ctrl_S48_Conv2d_40x1x1x72,
                               4, 1,
                               72, 40, 40, 32,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S51_Conv2d_120x1x1x40_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S51_Conv2d_120x1x1x40_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S51_Conv2d_120x1x1x40_Relu, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_5_block_block_0_block_0_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S51_Conv2d_120x1x1x40_Relu", &gen_ctrl_S51_Conv2d_120x1x1x40_Relu,
                               4, 1,
                               40, 120, 40, 32,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S54_Conv2d_120x5x5x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S54_Conv2d_120x5x5x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S54_Conv2d_120x5x5x1_Relu, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_5_block_block_1_block_1_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S54_Conv2d_120x5x5x1_Relu", &gen_ctrl_S54_Conv2d_120x5x5x1_Relu,
                               4, 1,
                               120, 120, 40, 32,
                               KOP_CONV_DW, 5, 5, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    
    // generator for _backbone_features_0_features_0_5_block_block_2_avgpool_GlobalAveragePool_trans Transpose 32x40x120 -> 120x32x40 ((1, 0))
    CNN_MatTranspose("S55_Op__backbone_features_0_features_0_5_block_block_2_avgpool_GlobalAveragePool_trans", 0, 1,
                      1, 120, 1280, KOP_MATTRANSP);
    
    // generator for _backbone_features_0_features_0_5_block_block_2_avgpool_GlobalAveragePool
    CNN_GlobalPoolAct_SQ8("S56_Op__backbone_features_0_features_0_5_block_block_2_avgpool_GlobalAveragePool", 0,
                          120, 16, 80,
                          KOP_GLOBAL_AVGPOOL, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S60_Conv2d_32x1x1x120_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S60_Conv2d_32x1x1x120_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S60_Conv2d_32x1x1x120_Relu, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_5_block_block_2_fc1_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S60_Conv2d_32x1x1x120_Relu", &gen_ctrl_S60_Conv2d_32x1x1x120_Relu,
                               4, 1,
                               120, 32, 1, 1,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S63_Conv2d_120x1x1x32_Hsigmoid;
    CNN_InitGenCtrl(&gen_ctrl_S63_Conv2d_120x1x1x32_Hsigmoid);
    CNN_SetGenCtrl(&gen_ctrl_S63_Conv2d_120x1x1x32_Hsigmoid, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_5_block_block_2_fc2_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S63_Conv2d_120x1x1x32_Hsigmoid", &gen_ctrl_S63_Conv2d_120x1x1x32_Hsigmoid,
                               4, 1,
                               32, 120, 1, 1,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_HSIGMOID);
    
    
    // generator for expr_17
    s64_kernel_gen("S64_Op_expr_17");
    
    CNN_GenControl_T gen_ctrl_S67_Conv2d_40x1x1x120;
    CNN_InitGenCtrl(&gen_ctrl_S67_Conv2d_40x1x1x120);
    CNN_SetGenCtrl(&gen_ctrl_S67_Conv2d_40x1x1x120, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_5_block_block_3_block_3_0_Conv
    CNN_ConvolutionPoolAct_SQ8("S67_Conv2d_40x1x1x120", &gen_ctrl_S67_Conv2d_40x1x1x120,
                               4, 1,
                               120, 40, 40, 32,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    
    // generator for _backbone_features_0_features_0_5_Add
    CNN_MatAddAct_SQ8("S68_MatAdd_32x40x40", 0, 32, 40, 40, KOP_MATADD, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S71_Conv2d_120x1x1x40_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S71_Conv2d_120x1x1x40_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S71_Conv2d_120x1x1x40_Relu, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_6_block_block_0_block_0_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S71_Conv2d_120x1x1x40_Relu", &gen_ctrl_S71_Conv2d_120x1x1x40_Relu,
                               4, 1,
                               40, 120, 40, 32,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S74_Conv2d_120x5x5x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S74_Conv2d_120x5x5x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S74_Conv2d_120x5x5x1_Relu, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_6_block_block_1_block_1_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S74_Conv2d_120x5x5x1_Relu", &gen_ctrl_S74_Conv2d_120x5x5x1_Relu,
                               4, 1,
                               120, 120, 40, 32,
                               KOP_CONV_DW, 5, 5, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    
    // generator for _backbone_features_0_features_0_6_block_block_2_avgpool_GlobalAveragePool_trans Transpose 32x40x120 -> 120x32x40 ((1, 0))
    CNN_MatTranspose("S75_Op__backbone_features_0_features_0_6_block_block_2_avgpool_GlobalAveragePool_trans", 0, 1,
                      1, 120, 1280, KOP_MATTRANSP);
    
    // generator for _backbone_features_0_features_0_6_block_block_2_avgpool_GlobalAveragePool
    CNN_GlobalPoolAct_SQ8("S76_Op__backbone_features_0_features_0_6_block_block_2_avgpool_GlobalAveragePool", 0,
                          120, 16, 80,
                          KOP_GLOBAL_AVGPOOL, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S80_Conv2d_32x1x1x120_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S80_Conv2d_32x1x1x120_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S80_Conv2d_32x1x1x120_Relu, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_6_block_block_2_fc1_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S80_Conv2d_32x1x1x120_Relu", &gen_ctrl_S80_Conv2d_32x1x1x120_Relu,
                               4, 1,
                               120, 32, 1, 1,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S83_Conv2d_120x1x1x32_Hsigmoid;
    CNN_InitGenCtrl(&gen_ctrl_S83_Conv2d_120x1x1x32_Hsigmoid);
    CNN_SetGenCtrl(&gen_ctrl_S83_Conv2d_120x1x1x32_Hsigmoid, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_6_block_block_2_fc2_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S83_Conv2d_120x1x1x32_Hsigmoid", &gen_ctrl_S83_Conv2d_120x1x1x32_Hsigmoid,
                               4, 1,
                               32, 120, 1, 1,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_HSIGMOID);
    
    
    // generator for expr_18
    s84_kernel_gen("S84_Op_expr_18");
    
    CNN_GenControl_T gen_ctrl_S87_Conv2d_40x1x1x120;
    CNN_InitGenCtrl(&gen_ctrl_S87_Conv2d_40x1x1x120);
    CNN_SetGenCtrl(&gen_ctrl_S87_Conv2d_40x1x1x120, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_6_block_block_3_block_3_0_Conv
    CNN_ConvolutionPoolAct_SQ8("S87_Conv2d_40x1x1x120", &gen_ctrl_S87_Conv2d_40x1x1x120,
                               4, 1,
                               120, 40, 40, 32,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    
    // generator for _backbone_features_0_features_0_6_Add
    CNN_MatAddAct_SQ8("S88_MatAdd_32x40x40", 0, 32, 40, 40, KOP_MATADD, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S91_Conv2d_240x1x1x40_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S91_Conv2d_240x1x1x40_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S91_Conv2d_240x1x1x40_Custom, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S91_Conv2d_240x1x1x40_Custom, "CUSTOM_ACTIVATION_NAME", "expr_19");
    // generator for _backbone_features_0_features_0_7_block_block_0_block_0_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S91_Conv2d_240x1x1x40_Custom", &gen_ctrl_S91_Conv2d_240x1x1x40_Custom,
                               4, 1,
                               40, 240, 40, 32,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S94_Conv2d_240x3x3x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S94_Conv2d_240x3x3x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S94_Conv2d_240x3x3x1_Custom, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S94_Conv2d_240x3x3x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_20");
    // generator for _backbone_features_0_features_0_7_block_block_1_block_1_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S94_Conv2d_240x3x3x1_Custom", &gen_ctrl_S94_Conv2d_240x3x3x1_Custom,
                               4, 1,
                               240, 240, 40, 32,
                               KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S97_Conv2d_80x1x1x240;
    CNN_InitGenCtrl(&gen_ctrl_S97_Conv2d_80x1x1x240);
    CNN_SetGenCtrl(&gen_ctrl_S97_Conv2d_80x1x1x240, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_7_block_block_2_block_2_0_Conv
    CNN_ConvolutionPoolAct_SQ8("S97_Conv2d_80x1x1x240", &gen_ctrl_S97_Conv2d_80x1x1x240,
                               4, 1,
                               240, 80, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    
    // generator for _backbone_features_0_features_0_8_block_block_0_block_0_0_Conv_fusion_qin0
    CNN_Convert("S98_Op__backbone_features_0_features_0_8_block_block_0_block_0_0_Conv_fusion_qin0", 1, 1, 25600, KOP_CONVERT_FP_FP_SCALE);
    
    CNN_GenControl_T gen_ctrl_S101_Conv2d_200x1x1x80_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S101_Conv2d_200x1x1x80_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S101_Conv2d_200x1x1x80_Custom, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S101_Conv2d_200x1x1x80_Custom, "CUSTOM_ACTIVATION_NAME", "expr_21");
    // generator for _backbone_features_0_features_0_8_block_block_0_block_0_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S101_Conv2d_200x1x1x80_Custom", &gen_ctrl_S101_Conv2d_200x1x1x80_Custom,
                               4, 1,
                               80, 200, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S104_Conv2d_200x3x3x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S104_Conv2d_200x3x3x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S104_Conv2d_200x3x3x1_Custom, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S104_Conv2d_200x3x3x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_22");
    // generator for _backbone_features_0_features_0_8_block_block_1_block_1_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S104_Conv2d_200x3x3x1_Custom", &gen_ctrl_S104_Conv2d_200x3x3x1_Custom,
                               4, 1,
                               200, 200, 20, 16,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S107_Conv2d_80x1x1x200;
    CNN_InitGenCtrl(&gen_ctrl_S107_Conv2d_80x1x1x200);
    CNN_SetGenCtrl(&gen_ctrl_S107_Conv2d_80x1x1x200, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_8_block_block_2_block_2_0_Conv
    CNN_ConvolutionPoolAct_SQ8("S107_Conv2d_80x1x1x200", &gen_ctrl_S107_Conv2d_80x1x1x200,
                               4, 1,
                               200, 80, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    
    // generator for _backbone_features_0_features_0_8_Add
    CNN_MatAddAct_SQ8("S108_MatAdd_16x20x80", 0, 16, 20, 80, KOP_MATADD, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S111_Conv2d_184x1x1x80_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S111_Conv2d_184x1x1x80_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S111_Conv2d_184x1x1x80_Custom, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S111_Conv2d_184x1x1x80_Custom, "CUSTOM_ACTIVATION_NAME", "expr_23");
    // generator for _backbone_features_0_features_0_9_block_block_0_block_0_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S111_Conv2d_184x1x1x80_Custom", &gen_ctrl_S111_Conv2d_184x1x1x80_Custom,
                               4, 1,
                               80, 184, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S114_Conv2d_184x3x3x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S114_Conv2d_184x3x3x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S114_Conv2d_184x3x3x1_Custom, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S114_Conv2d_184x3x3x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_24");
    // generator for _backbone_features_0_features_0_9_block_block_1_block_1_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S114_Conv2d_184x3x3x1_Custom", &gen_ctrl_S114_Conv2d_184x3x3x1_Custom,
                               4, 1,
                               184, 184, 20, 16,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S117_Conv2d_80x1x1x184;
    CNN_InitGenCtrl(&gen_ctrl_S117_Conv2d_80x1x1x184);
    CNN_SetGenCtrl(&gen_ctrl_S117_Conv2d_80x1x1x184, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_9_block_block_2_block_2_0_Conv
    CNN_ConvolutionPoolAct_SQ8("S117_Conv2d_80x1x1x184", &gen_ctrl_S117_Conv2d_80x1x1x184,
                               4, 1,
                               184, 80, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    
    // generator for _backbone_features_0_features_0_9_Add
    CNN_MatAddAct_SQ8("S118_MatAdd_16x20x80", 0, 16, 20, 80, KOP_MATADD, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S121_Conv2d_184x1x1x80_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S121_Conv2d_184x1x1x80_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S121_Conv2d_184x1x1x80_Custom, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S121_Conv2d_184x1x1x80_Custom, "CUSTOM_ACTIVATION_NAME", "expr_7");
    // generator for _backbone_features_0_features_0_10_block_block_0_block_0_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S121_Conv2d_184x1x1x80_Custom", &gen_ctrl_S121_Conv2d_184x1x1x80_Custom,
                               4, 1,
                               80, 184, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S124_Conv2d_184x3x3x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S124_Conv2d_184x3x3x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S124_Conv2d_184x3x3x1_Custom, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S124_Conv2d_184x3x3x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_8");
    // generator for _backbone_features_0_features_0_10_block_block_1_block_1_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S124_Conv2d_184x3x3x1_Custom", &gen_ctrl_S124_Conv2d_184x3x3x1_Custom,
                               4, 1,
                               184, 184, 20, 16,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S127_Conv2d_80x1x1x184;
    CNN_InitGenCtrl(&gen_ctrl_S127_Conv2d_80x1x1x184);
    CNN_SetGenCtrl(&gen_ctrl_S127_Conv2d_80x1x1x184, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_10_block_block_2_block_2_0_Conv
    CNN_ConvolutionPoolAct_SQ8("S127_Conv2d_80x1x1x184", &gen_ctrl_S127_Conv2d_80x1x1x184,
                               4, 1,
                               184, 80, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    
    // generator for _backbone_features_0_features_0_10_Add
    CNN_MatAddAct_SQ8("S128_MatAdd_16x20x80", 0, 16, 20, 80, KOP_MATADD, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S131_Conv2d_480x1x1x80_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S131_Conv2d_480x1x1x80_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S131_Conv2d_480x1x1x80_Custom, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S131_Conv2d_480x1x1x80_Custom, "CUSTOM_ACTIVATION_NAME", "expr_9");
    // generator for _backbone_features_0_features_0_11_block_block_0_block_0_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S131_Conv2d_480x1x1x80_Custom", &gen_ctrl_S131_Conv2d_480x1x1x80_Custom,
                               4, 1,
                               80, 480, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S134_Conv2d_480x3x3x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S134_Conv2d_480x3x3x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S134_Conv2d_480x3x3x1_Custom, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S134_Conv2d_480x3x3x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_10");
    // generator for _backbone_features_0_features_0_11_block_block_1_block_1_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S134_Conv2d_480x3x3x1_Custom", &gen_ctrl_S134_Conv2d_480x3x3x1_Custom,
                               4, 1,
                               480, 480, 20, 16,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    
    // generator for _backbone_features_0_features_0_11_block_block_2_avgpool_GlobalAveragePool_trans Transpose 16x20x480 -> 480x16x20 ((1, 0))
    CNN_MatTranspose("S135_Op__backbone_features_0_features_0_11_block_block_2_avgpool_GlobalAveragePool_trans", 0, 1,
                      1, 480, 320, KOP_MATTRANSP);
    
    // generator for _backbone_features_0_features_0_11_block_block_2_avgpool_GlobalAveragePool
    CNN_GlobalPoolAct_SQ8("S136_Op__backbone_features_0_features_0_11_block_block_2_avgpool_GlobalAveragePool", 0,
                          480, 8, 40,
                          KOP_GLOBAL_AVGPOOL, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S140_Conv2d_120x1x1x480_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S140_Conv2d_120x1x1x480_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S140_Conv2d_120x1x1x480_Relu, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_11_block_block_2_fc1_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S140_Conv2d_120x1x1x480_Relu", &gen_ctrl_S140_Conv2d_120x1x1x480_Relu,
                               4, 1,
                               480, 120, 1, 1,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S143_Conv2d_480x1x1x120_Hsigmoid;
    CNN_InitGenCtrl(&gen_ctrl_S143_Conv2d_480x1x1x120_Hsigmoid);
    CNN_SetGenCtrl(&gen_ctrl_S143_Conv2d_480x1x1x120_Hsigmoid, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_11_block_block_2_fc2_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S143_Conv2d_480x1x1x120_Hsigmoid", &gen_ctrl_S143_Conv2d_480x1x1x120_Hsigmoid,
                               4, 1,
                               120, 480, 1, 1,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_HSIGMOID);
    
    
    // generator for expr_11
    s144_kernel_gen("S144_Op_expr_11");
    
    CNN_GenControl_T gen_ctrl_S147_Conv2d_112x1x1x480;
    CNN_InitGenCtrl(&gen_ctrl_S147_Conv2d_112x1x1x480);
    CNN_SetGenCtrl(&gen_ctrl_S147_Conv2d_112x1x1x480, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_11_block_block_3_block_3_0_Conv
    CNN_ConvolutionPoolAct_SQ8("S147_Conv2d_112x1x1x480", &gen_ctrl_S147_Conv2d_112x1x1x480,
                               4, 1,
                               480, 112, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    
    // generator for _backbone_features_0_features_0_12_block_block_0_block_0_0_Conv_fusion_qin0
    CNN_Convert("S148_Op__backbone_features_0_features_0_12_block_block_0_block_0_0_Conv_fusion_qin0", 1, 1, 35840, KOP_CONVERT_FP_FP_SCALE);
    
    CNN_GenControl_T gen_ctrl_S151_Conv2d_672x1x1x112_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S151_Conv2d_672x1x1x112_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S151_Conv2d_672x1x1x112_Custom, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S151_Conv2d_672x1x1x112_Custom, "CUSTOM_ACTIVATION_NAME", "expr_12");
    // generator for _backbone_features_0_features_0_12_block_block_0_block_0_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S151_Conv2d_672x1x1x112_Custom", &gen_ctrl_S151_Conv2d_672x1x1x112_Custom,
                               4, 1,
                               112, 672, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S154_Conv2d_672x3x3x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S154_Conv2d_672x3x3x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S154_Conv2d_672x3x3x1_Custom, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S154_Conv2d_672x3x3x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_13");
    // generator for _backbone_features_0_features_0_12_block_block_1_block_1_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S154_Conv2d_672x3x3x1_Custom", &gen_ctrl_S154_Conv2d_672x3x3x1_Custom,
                               4, 1,
                               672, 672, 20, 16,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    
    // generator for _backbone_features_0_features_0_12_block_block_2_avgpool_GlobalAveragePool_trans Transpose 16x20x672 -> 672x16x20 ((1, 0))
    CNN_MatTranspose("S155_Op__backbone_features_0_features_0_12_block_block_2_avgpool_GlobalAveragePool_trans", 0, 1,
                      1, 672, 320, KOP_MATTRANSP);
    
    // generator for _backbone_features_0_features_0_12_block_block_2_avgpool_GlobalAveragePool
    CNN_GlobalPoolAct_SQ8("S156_Op__backbone_features_0_features_0_12_block_block_2_avgpool_GlobalAveragePool", 0,
                          672, 8, 40,
                          KOP_GLOBAL_AVGPOOL, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S160_Conv2d_168x1x1x672_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S160_Conv2d_168x1x1x672_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S160_Conv2d_168x1x1x672_Relu, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_12_block_block_2_fc1_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S160_Conv2d_168x1x1x672_Relu", &gen_ctrl_S160_Conv2d_168x1x1x672_Relu,
                               4, 1,
                               672, 168, 1, 1,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S163_Conv2d_672x1x1x168_Hsigmoid;
    CNN_InitGenCtrl(&gen_ctrl_S163_Conv2d_672x1x1x168_Hsigmoid);
    CNN_SetGenCtrl(&gen_ctrl_S163_Conv2d_672x1x1x168_Hsigmoid, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_12_block_block_2_fc2_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S163_Conv2d_672x1x1x168_Hsigmoid", &gen_ctrl_S163_Conv2d_672x1x1x168_Hsigmoid,
                               4, 1,
                               168, 672, 1, 1,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_HSIGMOID);
    
    
    // generator for expr_14
    s164_kernel_gen("S164_Op_expr_14");
    
    CNN_GenControl_T gen_ctrl_S167_Conv2d_112x1x1x672;
    CNN_InitGenCtrl(&gen_ctrl_S167_Conv2d_112x1x1x672);
    CNN_SetGenCtrl(&gen_ctrl_S167_Conv2d_112x1x1x672, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_12_block_block_3_block_3_0_Conv
    CNN_ConvolutionPoolAct_SQ8("S167_Conv2d_112x1x1x672", &gen_ctrl_S167_Conv2d_112x1x1x672,
                               4, 1,
                               672, 112, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    
    // generator for _backbone_features_0_features_0_12_Add
    CNN_MatAddAct_SQ8("S168_MatAdd_16x20x112", 0, 16, 20, 112, KOP_MATADD, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S171_Conv2d_672x1x1x112_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S171_Conv2d_672x1x1x112_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S171_Conv2d_672x1x1x112_Custom, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S171_Conv2d_672x1x1x112_Custom, "CUSTOM_ACTIVATION_NAME", "expr_15");
    // generator for _backbone_features_0_features_0_13_features_0_13_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S171_Conv2d_672x1x1x112_Custom", &gen_ctrl_S171_Conv2d_672x1x1x112_Custom,
                               4, 1,
                               112, 672, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S174_Conv2d_672x5x5x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S174_Conv2d_672x5x5x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S174_Conv2d_672x5x5x1_Custom, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S174_Conv2d_672x5x5x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_25");
    // generator for _backbone_features_1_features_1_0_features_1_0_1_features_1_0_1_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S174_Conv2d_672x5x5x1_Custom", &gen_ctrl_S174_Conv2d_672x5x5x1_Custom,
                               4, 1,
                               672, 672, 20, 16,
                               KOP_CONV_DW, 5, 5, 1, 1, 2, 2, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    
    // generator for _backbone_features_1_features_1_0_features_1_0_2_avgpool_GlobalAveragePool_trans Transpose 8x10x672 -> 672x8x10 ((1, 0))
    CNN_MatTranspose("S175_Op__backbone_features_1_features_1_0_features_1_0_2_avgpool_GlobalAveragePool_trans", 0, 1,
                      1, 672, 80, KOP_MATTRANSP);
    
    // generator for _backbone_features_1_features_1_0_features_1_0_2_avgpool_GlobalAveragePool
    CNN_GlobalPoolAct_SQ8("S176_Op__backbone_features_1_features_1_0_features_1_0_2_avgpool_GlobalAveragePool", 0,
                          672, 4, 20,
                          KOP_GLOBAL_AVGPOOL, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S180_Conv2d_168x1x1x672_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S180_Conv2d_168x1x1x672_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S180_Conv2d_168x1x1x672_Relu, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_1_features_1_0_features_1_0_2_fc1_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S180_Conv2d_168x1x1x672_Relu", &gen_ctrl_S180_Conv2d_168x1x1x672_Relu,
                               4, 1,
                               672, 168, 1, 1,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S183_Conv2d_672x1x1x168_Hsigmoid;
    CNN_InitGenCtrl(&gen_ctrl_S183_Conv2d_672x1x1x168_Hsigmoid);
    CNN_SetGenCtrl(&gen_ctrl_S183_Conv2d_672x1x1x168_Hsigmoid, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_1_features_1_0_features_1_0_2_fc2_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S183_Conv2d_672x1x1x168_Hsigmoid", &gen_ctrl_S183_Conv2d_672x1x1x168_Hsigmoid,
                               4, 1,
                               168, 672, 1, 1,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_HSIGMOID);
    
    
    // generator for expr_26
    s184_kernel_gen("S184_Op_expr_26");
    
    CNN_GenControl_T gen_ctrl_S187_Conv2d_80x1x1x672;
    CNN_InitGenCtrl(&gen_ctrl_S187_Conv2d_80x1x1x672);
    CNN_SetGenCtrl(&gen_ctrl_S187_Conv2d_80x1x1x672, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_1_features_1_0_features_1_0_3_features_1_0_3_0_Conv
    CNN_ConvolutionPoolAct_SQ8("S187_Conv2d_80x1x1x672", &gen_ctrl_S187_Conv2d_80x1x1x672,
                               4, 1,
                               672, 80, 10, 8,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    
    // generator for _backbone_features_1_features_1_1_block_block_0_block_0_0_Conv_fusion_qin0
    CNN_Convert("S188_Op__backbone_features_1_features_1_1_block_block_0_block_0_0_Conv_fusion_qin0", 1, 1, 6400, KOP_CONVERT_FP_FP_SCALE);
    
    CNN_GenControl_T gen_ctrl_S191_Conv2d_480x1x1x80_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S191_Conv2d_480x1x1x80_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S191_Conv2d_480x1x1x80_Custom, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S191_Conv2d_480x1x1x80_Custom, "CUSTOM_ACTIVATION_NAME", "expr_27");
    // generator for _backbone_features_1_features_1_1_block_block_0_block_0_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S191_Conv2d_480x1x1x80_Custom", &gen_ctrl_S191_Conv2d_480x1x1x80_Custom,
                               4, 1,
                               80, 480, 10, 8,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S194_Conv2d_480x5x5x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S194_Conv2d_480x5x5x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S194_Conv2d_480x5x5x1_Custom, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S194_Conv2d_480x5x5x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_28");
    // generator for _backbone_features_1_features_1_1_block_block_1_block_1_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S194_Conv2d_480x5x5x1_Custom", &gen_ctrl_S194_Conv2d_480x5x5x1_Custom,
                               4, 1,
                               480, 480, 10, 8,
                               KOP_CONV_DW, 5, 5, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    
    // generator for _backbone_features_1_features_1_1_block_block_2_avgpool_GlobalAveragePool_trans Transpose 8x10x480 -> 480x8x10 ((1, 0))
    CNN_MatTranspose("S195_Op__backbone_features_1_features_1_1_block_block_2_avgpool_GlobalAveragePool_trans", 0, 1,
                      1, 480, 80, KOP_MATTRANSP);
    
    // generator for _backbone_features_1_features_1_1_block_block_2_avgpool_GlobalAveragePool
    CNN_GlobalPoolAct_SQ8("S196_Op__backbone_features_1_features_1_1_block_block_2_avgpool_GlobalAveragePool", 0,
                          480, 4, 20,
                          KOP_GLOBAL_AVGPOOL, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S200_Conv2d_120x1x1x480_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S200_Conv2d_120x1x1x480_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S200_Conv2d_120x1x1x480_Relu, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_1_features_1_1_block_block_2_fc1_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S200_Conv2d_120x1x1x480_Relu", &gen_ctrl_S200_Conv2d_120x1x1x480_Relu,
                               4, 1,
                               480, 120, 1, 1,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S203_Conv2d_480x1x1x120_Hsigmoid;
    CNN_InitGenCtrl(&gen_ctrl_S203_Conv2d_480x1x1x120_Hsigmoid);
    CNN_SetGenCtrl(&gen_ctrl_S203_Conv2d_480x1x1x120_Hsigmoid, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_1_features_1_1_block_block_2_fc2_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S203_Conv2d_480x1x1x120_Hsigmoid", &gen_ctrl_S203_Conv2d_480x1x1x120_Hsigmoid,
                               4, 1,
                               120, 480, 1, 1,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_HSIGMOID);
    
    
    // generator for expr_29
    s204_kernel_gen("S204_Op_expr_29");
    
    CNN_GenControl_T gen_ctrl_S207_Conv2d_80x1x1x480;
    CNN_InitGenCtrl(&gen_ctrl_S207_Conv2d_80x1x1x480);
    CNN_SetGenCtrl(&gen_ctrl_S207_Conv2d_80x1x1x480, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_1_features_1_1_block_block_3_block_3_0_Conv
    CNN_ConvolutionPoolAct_SQ8("S207_Conv2d_80x1x1x480", &gen_ctrl_S207_Conv2d_80x1x1x480,
                               4, 1,
                               480, 80, 10, 8,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    
    // generator for _backbone_features_1_features_1_1_Add
    CNN_MatAddAct_SQ8("S208_MatAdd_8x10x80", 0, 8, 10, 80, KOP_MATADD, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S211_Conv2d_480x1x1x80_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S211_Conv2d_480x1x1x80_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S211_Conv2d_480x1x1x80_Custom, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S211_Conv2d_480x1x1x80_Custom, "CUSTOM_ACTIVATION_NAME", "expr_30");
    // generator for _backbone_features_1_features_1_2_block_block_0_block_0_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S211_Conv2d_480x1x1x80_Custom", &gen_ctrl_S211_Conv2d_480x1x1x80_Custom,
                               4, 1,
                               80, 480, 10, 8,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S214_Conv2d_480x5x5x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S214_Conv2d_480x5x5x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S214_Conv2d_480x5x5x1_Custom, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S214_Conv2d_480x5x5x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_31");
    // generator for _backbone_features_1_features_1_2_block_block_1_block_1_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S214_Conv2d_480x5x5x1_Custom", &gen_ctrl_S214_Conv2d_480x5x5x1_Custom,
                               4, 1,
                               480, 480, 10, 8,
                               KOP_CONV_DW, 5, 5, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    
    // generator for _backbone_features_1_features_1_2_block_block_2_avgpool_GlobalAveragePool_trans Transpose 8x10x480 -> 480x8x10 ((1, 0))
    CNN_MatTranspose("S215_Op__backbone_features_1_features_1_2_block_block_2_avgpool_GlobalAveragePool_trans", 0, 1,
                      1, 480, 80, KOP_MATTRANSP);
    
    // generator for _backbone_features_1_features_1_2_block_block_2_avgpool_GlobalAveragePool
    CNN_GlobalPoolAct_SQ8("S216_Op__backbone_features_1_features_1_2_block_block_2_avgpool_GlobalAveragePool", 0,
                          480, 4, 20,
                          KOP_GLOBAL_AVGPOOL, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S220_Conv2d_120x1x1x480_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S220_Conv2d_120x1x1x480_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S220_Conv2d_120x1x1x480_Relu, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_1_features_1_2_block_block_2_fc1_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S220_Conv2d_120x1x1x480_Relu", &gen_ctrl_S220_Conv2d_120x1x1x480_Relu,
                               4, 1,
                               480, 120, 1, 1,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S223_Conv2d_480x1x1x120_Hsigmoid;
    CNN_InitGenCtrl(&gen_ctrl_S223_Conv2d_480x1x1x120_Hsigmoid);
    CNN_SetGenCtrl(&gen_ctrl_S223_Conv2d_480x1x1x120_Hsigmoid, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_1_features_1_2_block_block_2_fc2_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S223_Conv2d_480x1x1x120_Hsigmoid", &gen_ctrl_S223_Conv2d_480x1x1x120_Hsigmoid,
                               4, 1,
                               120, 480, 1, 1,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_HSIGMOID);
    
    
    // generator for expr_32
    s224_kernel_gen("S224_Op_expr_32");
    
    CNN_GenControl_T gen_ctrl_S227_Conv2d_80x1x1x480;
    CNN_InitGenCtrl(&gen_ctrl_S227_Conv2d_80x1x1x480);
    CNN_SetGenCtrl(&gen_ctrl_S227_Conv2d_80x1x1x480, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_1_features_1_2_block_block_3_block_3_0_Conv
    CNN_ConvolutionPoolAct_SQ8("S227_Conv2d_80x1x1x480", &gen_ctrl_S227_Conv2d_80x1x1x480,
                               4, 1,
                               480, 80, 10, 8,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    
    // generator for _backbone_features_1_features_1_2_Add
    CNN_MatAddAct_SQ8("S228_MatAdd_8x10x80", 0, 8, 10, 80, KOP_MATADD, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S231_Conv2d_480x1x1x80_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S231_Conv2d_480x1x1x80_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S231_Conv2d_480x1x1x80_Custom, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S231_Conv2d_480x1x1x80_Custom, "CUSTOM_ACTIVATION_NAME", "expr_33");
    // generator for _backbone_features_1_features_1_3_features_1_3_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S231_Conv2d_480x1x1x80_Custom", &gen_ctrl_S231_Conv2d_480x1x1x80_Custom,
                               4, 1,
                               80, 480, 10, 8,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S234_Conv2d_256x1x1x480_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S234_Conv2d_256x1x1x480_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S234_Conv2d_256x1x1x480_Relu6, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_extra_0_extra_0_0_extra_0_0_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S234_Conv2d_256x1x1x480_Relu6", &gen_ctrl_S234_Conv2d_256x1x1x480_Relu6,
                               4, 1,
                               480, 256, 10, 8,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S237_Conv2d_256x3x3x1_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S237_Conv2d_256x3x3x1_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S237_Conv2d_256x3x3x1_Relu6, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_extra_0_extra_0_1_extra_0_1_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S237_Conv2d_256x3x3x1_Relu6", &gen_ctrl_S237_Conv2d_256x3x3x1_Relu6,
                               4, 1,
                               256, 256, 10, 8,
                               KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    CNN_GenControl_T gen_ctrl_S240_Conv2d_512x1x1x256_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S240_Conv2d_512x1x1x256_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S240_Conv2d_512x1x1x256_Relu6, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_extra_0_extra_0_2_extra_0_2_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S240_Conv2d_512x1x1x256_Relu6", &gen_ctrl_S240_Conv2d_512x1x1x256_Relu6,
                               4, 1,
                               256, 512, 5, 4,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S243_Conv2d_128x1x1x512_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S243_Conv2d_128x1x1x512_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S243_Conv2d_128x1x1x512_Relu6, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_extra_1_extra_1_0_extra_1_0_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S243_Conv2d_128x1x1x512_Relu6", &gen_ctrl_S243_Conv2d_128x1x1x512_Relu6,
                               4, 1,
                               512, 128, 5, 4,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S246_Conv2d_128x3x3x1_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S246_Conv2d_128x3x3x1_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S246_Conv2d_128x3x3x1_Relu6, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_extra_1_extra_1_1_extra_1_1_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S246_Conv2d_128x3x3x1_Relu6", &gen_ctrl_S246_Conv2d_128x3x3x1_Relu6,
                               4, 1,
                               128, 128, 5, 4,
                               KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    CNN_GenControl_T gen_ctrl_S249_Conv2d_256x1x1x128_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S249_Conv2d_256x1x1x128_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S249_Conv2d_256x1x1x128_Relu6, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_extra_1_extra_1_2_extra_1_2_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S249_Conv2d_256x1x1x128_Relu6", &gen_ctrl_S249_Conv2d_256x1x1x128_Relu6,
                               4, 1,
                               128, 256, 3, 2,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S252_Conv2d_128x1x1x256_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S252_Conv2d_128x1x1x256_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S252_Conv2d_128x1x1x256_Relu6, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_extra_2_extra_2_0_extra_2_0_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S252_Conv2d_128x1x1x256_Relu6", &gen_ctrl_S252_Conv2d_128x1x1x256_Relu6,
                               4, 1,
                               256, 128, 3, 2,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S255_Conv2d_128x3x3x1_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S255_Conv2d_128x3x3x1_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S255_Conv2d_128x3x3x1_Relu6, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_extra_2_extra_2_1_extra_2_1_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S255_Conv2d_128x3x3x1_Relu6", &gen_ctrl_S255_Conv2d_128x3x3x1_Relu6,
                               4, 1,
                               128, 128, 3, 2,
                               KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    CNN_GenControl_T gen_ctrl_S258_Conv2d_256x1x1x128_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S258_Conv2d_256x1x1x128_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S258_Conv2d_256x1x1x128_Relu6, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_extra_2_extra_2_2_extra_2_2_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S258_Conv2d_256x1x1x128_Relu6", &gen_ctrl_S258_Conv2d_256x1x1x128_Relu6,
                               4, 1,
                               128, 256, 2, 1,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S261_Conv2d_64x1x1x256_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S261_Conv2d_64x1x1x256_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S261_Conv2d_64x1x1x256_Relu6, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_extra_3_extra_3_0_extra_3_0_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S261_Conv2d_64x1x1x256_Relu6", &gen_ctrl_S261_Conv2d_64x1x1x256_Relu6,
                               4, 1,
                               256, 64, 2, 1,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S264_Conv2d_64x3x3x1_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S264_Conv2d_64x3x3x1_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S264_Conv2d_64x3x3x1_Relu6, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_extra_3_extra_3_1_extra_3_1_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S264_Conv2d_64x3x3x1_Relu6", &gen_ctrl_S264_Conv2d_64x3x3x1_Relu6,
                               4, 1,
                               64, 64, 2, 1,
                               KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    CNN_GenControl_T gen_ctrl_S267_Conv2d_128x1x1x64_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S267_Conv2d_128x1x1x64_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S267_Conv2d_128x1x1x64_Relu6, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_extra_3_extra_3_2_extra_3_2_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S267_Conv2d_128x1x1x64_Relu6", &gen_ctrl_S267_Conv2d_128x1x1x64_Relu6,
                               4, 1,
                               64, 128, 1, 1,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S270_Conv2d_672x3x3x1_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S270_Conv2d_672x3x3x1_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S270_Conv2d_672x3x3x1_Relu6, "HWC", AT_OPT_VAL(1));
    // generator for _head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S270_Conv2d_672x3x3x1_Relu6", &gen_ctrl_S270_Conv2d_672x3x3x1_Relu6,
                               4, 1,
                               672, 672, 20, 16,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S273_Conv2d_24x1x1x672_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S273_Conv2d_24x1x1x672_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S273_Conv2d_24x1x1x672_Custom, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S273_Conv2d_24x1x1x672_Custom, "CUSTOM_ACTIVATION_NAME", "expr_2");
    // generator for _head_regression_head_module_list_0_module_list_0_1_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S273_Conv2d_24x1x1x672_Custom", &gen_ctrl_S273_Conv2d_24x1x1x672_Custom,
                               4, 1,
                               672, 24, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S277_Conv2d_480x3x3x1_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S277_Conv2d_480x3x3x1_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S277_Conv2d_480x3x3x1_Relu6, "HWC", AT_OPT_VAL(1));
    // generator for _head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S277_Conv2d_480x3x3x1_Relu6", &gen_ctrl_S277_Conv2d_480x3x3x1_Relu6,
                               4, 1,
                               480, 480, 10, 8,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    CNN_GenControl_T gen_ctrl_S281_Conv2d_512x3x3x1_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S281_Conv2d_512x3x3x1_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S281_Conv2d_512x3x3x1_Relu6, "HWC", AT_OPT_VAL(1));
    // generator for _head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S281_Conv2d_512x3x3x1_Relu6", &gen_ctrl_S281_Conv2d_512x3x3x1_Relu6,
                               4, 1,
                               512, 512, 5, 4,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S285_Conv2d_256x3x3x1_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S285_Conv2d_256x3x3x1_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S285_Conv2d_256x3x3x1_Relu6, "HWC", AT_OPT_VAL(1));
    // generator for _head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S285_Conv2d_256x3x3x1_Relu6", &gen_ctrl_S285_Conv2d_256x3x3x1_Relu6,
                               4, 1,
                               256, 256, 3, 2,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    CNN_GenControl_T gen_ctrl_S291_Conv2d_128x3x3x1_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S291_Conv2d_128x3x3x1_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S291_Conv2d_128x3x3x1_Relu6, "HWC", AT_OPT_VAL(1));
    // generator for _head_regression_head_module_list_5_module_list_5_0_module_list_5_0_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S291_Conv2d_128x3x3x1_Relu6", &gen_ctrl_S291_Conv2d_128x3x3x1_Relu6,
                               4, 1,
                               128, 128, 1, 1,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    CNN_GenControl_T gen_ctrl_S311_Conv2d_672x3x3x1_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S311_Conv2d_672x3x3x1_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S311_Conv2d_672x3x3x1_Relu6, "HWC", AT_OPT_VAL(1));
    // generator for _head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S311_Conv2d_672x3x3x1_Relu6", &gen_ctrl_S311_Conv2d_672x3x3x1_Relu6,
                               4, 1,
                               672, 672, 20, 16,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    CNN_GenControl_T gen_ctrl_S312_Conv2d_186x1x1x672;
    CNN_InitGenCtrl(&gen_ctrl_S312_Conv2d_186x1x1x672);
    CNN_SetGenCtrl(&gen_ctrl_S312_Conv2d_186x1x1x672, "HWC", AT_OPT_VAL(1));
    // generator for _head_classification_head_module_list_0_module_list_0_1_Conv
    CNN_ConvolutionPoolAct_SQ8("S312_Conv2d_186x1x1x672", &gen_ctrl_S312_Conv2d_186x1x1x672,
                               4, 1,
                               672, 186, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S315_Conv2d_24x1x1x480_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S315_Conv2d_24x1x1x480_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S315_Conv2d_24x1x1x480_Custom, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S315_Conv2d_24x1x1x480_Custom, "CUSTOM_ACTIVATION_NAME", "expr_2_1");
    // generator for _head_regression_head_module_list_1_module_list_1_1_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S315_Conv2d_24x1x1x480_Custom", &gen_ctrl_S315_Conv2d_24x1x1x480_Custom,
                               4, 1,
                               480, 24, 10, 8,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S318_Conv2d_24x1x1x512_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S318_Conv2d_24x1x1x512_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S318_Conv2d_24x1x1x512_Custom, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S318_Conv2d_24x1x1x512_Custom, "CUSTOM_ACTIVATION_NAME", "expr_2_2");
    // generator for _head_regression_head_module_list_2_module_list_2_1_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S318_Conv2d_24x1x1x512_Custom", &gen_ctrl_S318_Conv2d_24x1x1x512_Custom,
                               4, 1,
                               512, 24, 5, 4,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S321_Conv2d_24x1x1x256_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S321_Conv2d_24x1x1x256_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S321_Conv2d_24x1x1x256_Custom, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S321_Conv2d_24x1x1x256_Custom, "CUSTOM_ACTIVATION_NAME", "expr_2_3");
    // generator for _head_regression_head_module_list_3_module_list_3_1_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S321_Conv2d_24x1x1x256_Custom", &gen_ctrl_S321_Conv2d_24x1x1x256_Custom,
                               4, 1,
                               256, 24, 3, 2,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S325_Conv2d_24x1x1x128_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S325_Conv2d_24x1x1x128_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S325_Conv2d_24x1x1x128_Custom, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S325_Conv2d_24x1x1x128_Custom, "CUSTOM_ACTIVATION_NAME", "expr_2_5");
    // generator for _head_regression_head_module_list_5_module_list_5_1_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S325_Conv2d_24x1x1x128_Custom", &gen_ctrl_S325_Conv2d_24x1x1x128_Custom,
                               4, 1,
                               128, 24, 1, 1,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S328_Conv2d_480x3x3x1_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S328_Conv2d_480x3x3x1_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S328_Conv2d_480x3x3x1_Relu6, "HWC", AT_OPT_VAL(1));
    // generator for _head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S328_Conv2d_480x3x3x1_Relu6", &gen_ctrl_S328_Conv2d_480x3x3x1_Relu6,
                               4, 1,
                               480, 480, 10, 8,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    CNN_GenControl_T gen_ctrl_S330_Conv2d_512x3x3x1_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S330_Conv2d_512x3x3x1_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S330_Conv2d_512x3x3x1_Relu6, "HWC", AT_OPT_VAL(1));
    // generator for _head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S330_Conv2d_512x3x3x1_Relu6", &gen_ctrl_S330_Conv2d_512x3x3x1_Relu6,
                               4, 1,
                               512, 512, 5, 4,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    CNN_GenControl_T gen_ctrl_S332_Conv2d_256x3x3x1_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S332_Conv2d_256x3x3x1_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S332_Conv2d_256x3x3x1_Relu6, "HWC", AT_OPT_VAL(1));
    // generator for _head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S332_Conv2d_256x3x3x1_Relu6", &gen_ctrl_S332_Conv2d_256x3x3x1_Relu6,
                               4, 1,
                               256, 256, 2, 1,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    CNN_GenControl_T gen_ctrl_S333_Conv2d_24x1x1x256_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S333_Conv2d_24x1x1x256_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S333_Conv2d_24x1x1x256_Custom, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S333_Conv2d_24x1x1x256_Custom, "CUSTOM_ACTIVATION_NAME", "expr_2_4");
    // generator for _head_regression_head_module_list_4_module_list_4_1_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S333_Conv2d_24x1x1x256_Custom", &gen_ctrl_S333_Conv2d_24x1x1x256_Custom,
                               4, 1,
                               256, 24, 2, 1,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    
    // generator for _Split
    CNN_SplitLastAxis_Generator("S336_Op__Split", 0, 1, 2574, 1, 1, 1, 1, KOP_SPLIT);
    
    CNN_GenControl_T gen_ctrl_S342_Conv2d_256x3x3x1_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S342_Conv2d_256x3x3x1_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S342_Conv2d_256x3x3x1_Relu6, "HWC", AT_OPT_VAL(1));
    // generator for _head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S342_Conv2d_256x3x3x1_Relu6", &gen_ctrl_S342_Conv2d_256x3x3x1_Relu6,
                               4, 1,
                               256, 256, 3, 2,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    CNN_GenControl_T gen_ctrl_S344_Conv2d_256x3x3x1_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S344_Conv2d_256x3x3x1_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S344_Conv2d_256x3x3x1_Relu6, "HWC", AT_OPT_VAL(1));
    // generator for _head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S344_Conv2d_256x3x3x1_Relu6", &gen_ctrl_S344_Conv2d_256x3x3x1_Relu6,
                               4, 1,
                               256, 256, 2, 1,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    CNN_GenControl_T gen_ctrl_S346_Conv2d_128x3x3x1_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S346_Conv2d_128x3x3x1_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S346_Conv2d_128x3x3x1_Relu6, "HWC", AT_OPT_VAL(1));
    // generator for _head_classification_head_module_list_5_module_list_5_0_module_list_5_0_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S346_Conv2d_128x3x3x1_Relu6", &gen_ctrl_S346_Conv2d_128x3x3x1_Relu6,
                               4, 1,
                               128, 128, 1, 1,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    CNN_GenControl_T gen_ctrl_S348_Conv2d_186x1x1x480;
    CNN_InitGenCtrl(&gen_ctrl_S348_Conv2d_186x1x1x480);
    CNN_SetGenCtrl(&gen_ctrl_S348_Conv2d_186x1x1x480, "HWC", AT_OPT_VAL(1));
    // generator for _head_classification_head_module_list_1_module_list_1_1_Conv
    CNN_ConvolutionPoolAct_SQ8("S348_Conv2d_186x1x1x480", &gen_ctrl_S348_Conv2d_186x1x1x480,
                               4, 1,
                               480, 186, 10, 8,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S351_Conv2d_186x1x1x512;
    CNN_InitGenCtrl(&gen_ctrl_S351_Conv2d_186x1x1x512);
    CNN_SetGenCtrl(&gen_ctrl_S351_Conv2d_186x1x1x512, "HWC", AT_OPT_VAL(1));
    // generator for _head_classification_head_module_list_2_module_list_2_1_Conv
    CNN_ConvolutionPoolAct_SQ8("S351_Conv2d_186x1x1x512", &gen_ctrl_S351_Conv2d_186x1x1x512,
                               4, 1,
                               512, 186, 5, 4,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S354_Conv2d_186x1x1x256;
    CNN_InitGenCtrl(&gen_ctrl_S354_Conv2d_186x1x1x256);
    CNN_SetGenCtrl(&gen_ctrl_S354_Conv2d_186x1x1x256, "HWC", AT_OPT_VAL(1));
    // generator for _head_classification_head_module_list_3_module_list_3_1_Conv
    CNN_ConvolutionPoolAct_SQ8("S354_Conv2d_186x1x1x256", &gen_ctrl_S354_Conv2d_186x1x1x256,
                               4, 1,
                               256, 186, 3, 2,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S357_Conv2d_186x1x1x256;
    CNN_InitGenCtrl(&gen_ctrl_S357_Conv2d_186x1x1x256);
    CNN_SetGenCtrl(&gen_ctrl_S357_Conv2d_186x1x1x256, "HWC", AT_OPT_VAL(1));
    // generator for _head_classification_head_module_list_4_module_list_4_1_Conv
    CNN_ConvolutionPoolAct_SQ8("S357_Conv2d_186x1x1x256", &gen_ctrl_S357_Conv2d_186x1x1x256,
                               4, 1,
                               256, 186, 2, 1,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S360_Conv2d_186x1x1x128;
    CNN_InitGenCtrl(&gen_ctrl_S360_Conv2d_186x1x1x128);
    CNN_SetGenCtrl(&gen_ctrl_S360_Conv2d_186x1x1x128, "HWC", AT_OPT_VAL(1));
    // generator for _head_classification_head_module_list_5_module_list_5_1_Conv
    CNN_ConvolutionPoolAct_SQ8("S360_Conv2d_186x1x1x128", &gen_ctrl_S360_Conv2d_186x1x1x128,
                               4, 1,
                               128, 186, 1, 1,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    // generator for _Softmax
    CNN_SoftMax2D_SQ8("S363_SoftMax", 0, 2574, 31, KOP_SOFTMAX);
    
    
    // generator for expr_1
    s366_kernel_gen("S366_Op_expr_1");
    
    
    // generator for expr_0
    s372_kernel_gen("S372_Op_expr_0");
    

#define GRAPH
#ifdef GRAPH
    CreateGraph("Mobv3SSDLiteCNN",
        /* Arguments either passed or globals */
            CArgs(524,
                TCArgInfo("signed char * __restrict__", "Input_1", ARG_SCOPE_ARG_ALLOC, ARG_DIR_IN, AT_MEM_L2, AT_MEM_L2, 0),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_0_features_0_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_0_features_0_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1387", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1387.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S3_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S3_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S3_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S3_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.08763 out: 0.08433  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S3_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S3_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_1_block_block_0_block_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_1_block_block_0_block_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1390", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1390.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S6_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S6_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S6_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S6_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01610 out: 0.01610  actscale: [1] actscalen: [0] a0: [-128] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S6_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S6_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_1_block_block_1_block_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_1_block_block_1_block_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1393", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1393.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S9_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S9_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S9_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S9_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S9_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S9_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [247] IN1SCALEN: [7] OUTSCALE: [29] OUTSCALEN: [6]
                TCArgInfo("signed char * __restrict__", "S10_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S10_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_2_block_block_0_block_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_2_block_block_0_block_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1396", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1396.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S13_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S13_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S13_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S13_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01997 out: 0.01997  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S13_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S13_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_2_block_block_1_block_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_2_block_block_1_block_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1399", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1399.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S16_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S16_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S16_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S16_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01776 out: 0.01776  actscale: [1] actscalen: [0] a0: [-128] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S16_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S16_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_2_block_block_2_block_2_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_2_block_block_2_block_2_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1402", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1402.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S19_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S19_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S19_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S19_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S19_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S19_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_3_block_block_0_block_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_3_block_block_0_block_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1405", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1405.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S22_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S22_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S22_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S22_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01332 out: 0.01332  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S22_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S22_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_3_block_block_1_block_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_3_block_block_1_block_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1408", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1408.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S25_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S25_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S25_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S25_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01166 out: 0.01166  actscale: [1] actscalen: [0] a0: [-128] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S25_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S25_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_3_block_block_2_block_2_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_3_block_block_2_block_2_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1411", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1411.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S28_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S28_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S28_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S28_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S28_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S28_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [65] IN1SCALEN: [6] OUTSCALE: [71] OUTSCALEN: [7]
                TCArgInfo("signed char * __restrict__", "S29_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S29_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_4_block_block_0_block_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_4_block_block_0_block_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1414", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1414.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S32_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S32_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S32_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S32_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01978 out: 0.01978  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S32_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S32_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_4_block_block_1_block_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_4_block_block_1_block_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1417", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1417.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S35_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S35_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S35_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S35_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.09262 out: 0.09262  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S35_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S35_Infos.tensor", 1, 1, 8, 0)),
                // no activation ACTSCALE: 0 ACTSCALEN: 0 GLOBAL_SUM_SCALE: [169] GLOBAL_SUM_SCALEN: [13]
                TCArgInfo("signed char * __restrict__", "S37_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S37_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_4_block_block_2_fc1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_4_block_block_2_fc1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_backbone_features_0_4_block_2_fc1_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_backbone_features_0_4_block_2_fc1_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S41_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S41_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S41_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S41_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00801 out: 0.00801  actscale: [1] actscalen: [0] a0: [-128] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S41_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S41_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_4_block_block_2_fc2_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_4_block_block_2_fc2_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_backbone_features_0_4_block_2_fc2_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_backbone_features_0_4_block_2_fc2_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S44_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S44_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S44_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S44_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.00787  actscale: [1] actscalen: [0] a0: [127] b0: [64] c0: [0] BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S44_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S44_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_4_block_block_3_block_3_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_4_block_block_3_block_3_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1420", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1420.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S48_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S48_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S48_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S48_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S48_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S48_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_5_block_block_0_block_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_5_block_block_0_block_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1423", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1423.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S51_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S51_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S51_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S51_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01047 out: 0.01047  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S51_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S51_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_5_block_block_1_block_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_5_block_block_1_block_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1426", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1426.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S54_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S54_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S54_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S54_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02624 out: 0.02624  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S54_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S54_Infos.tensor", 1, 1, 8, 0)),
                // no activation ACTSCALE: 0 ACTSCALEN: 0 GLOBAL_SUM_SCALE: [9] GLOBAL_SUM_SCALEN: [9]
                TCArgInfo("signed char * __restrict__", "S56_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S56_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_5_block_block_2_fc1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_5_block_block_2_fc1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_backbone_features_0_5_block_2_fc1_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_backbone_features_0_5_block_2_fc1_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S60_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S60_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S60_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S60_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00692 out: 0.00692  actscale: [1] actscalen: [0] a0: [-128] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S60_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S60_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_5_block_block_2_fc2_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_5_block_block_2_fc2_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_backbone_features_0_5_block_2_fc2_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_backbone_features_0_5_block_2_fc2_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S63_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S63_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S63_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S63_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.00787  actscale: [1] actscalen: [0] a0: [127] b0: [64] c0: [0] BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S63_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S63_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_5_block_block_3_block_3_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_5_block_block_3_block_3_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1429", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1429.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S67_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S67_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S67_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S67_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S67_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S67_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [163] IN1SCALEN: [7] OUTSCALE: [203] OUTSCALEN: [8]
                TCArgInfo("signed char * __restrict__", "S68_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S68_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_6_block_block_0_block_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_6_block_block_0_block_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1432", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1432.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S71_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S71_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S71_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S71_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00896 out: 0.00896  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S71_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S71_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_6_block_block_1_block_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_6_block_block_1_block_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1435", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1435.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S74_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S74_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S74_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S74_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02952 out: 0.02952  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S74_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S74_Infos.tensor", 1, 1, 8, 0)),
                // no activation ACTSCALE: 0 ACTSCALEN: 0 GLOBAL_SUM_SCALE: [117] GLOBAL_SUM_SCALEN: [13]
                TCArgInfo("signed char * __restrict__", "S76_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S76_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_6_block_block_2_fc1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_6_block_block_2_fc1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_backbone_features_0_6_block_2_fc1_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_backbone_features_0_6_block_2_fc1_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S80_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S80_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S80_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S80_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00810 out: 0.00810  actscale: [1] actscalen: [0] a0: [-128] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S80_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S80_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_6_block_block_2_fc2_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_6_block_block_2_fc2_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_backbone_features_0_6_block_2_fc2_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_backbone_features_0_6_block_2_fc2_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S83_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S83_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S83_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S83_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.00787  actscale: [1] actscalen: [0] a0: [127] b0: [64] c0: [0] BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S83_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S83_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_6_block_block_3_block_3_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_6_block_block_3_block_3_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1438", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1438.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S87_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S87_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S87_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S87_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S87_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S87_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [95] IN1SCALEN: [6] OUTSCALE: [131] OUTSCALEN: [8]
                TCArgInfo("signed char * __restrict__", "S88_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S88_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_7_block_block_0_block_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_7_block_block_0_block_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1441", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1441.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S91_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S91_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S91_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S91_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04935 out: 0.04149  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S91_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S91_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_7_block_block_1_block_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_7_block_block_1_block_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1444", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1444.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S94_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S94_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S94_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S94_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03243 out: 0.03842  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S94_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S94_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_7_block_block_2_block_2_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_7_block_block_2_block_2_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1447", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1447.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S97_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S97_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S97_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S97_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S97_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S97_Infos.tensor", 1, 1, 8, 0)),
                // in q: -2.49<(i8-0.00)*0.01944289<2.47 forced out_q: -2.49<(i8-18.00)*0.01705981<1.86
                TCArgInfo("signed char * __restrict__", "S98_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S98_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_8_block_block_0_block_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_8_block_block_0_block_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1450", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1450.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S101_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S101_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S101_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S101_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03928 out: 0.05716  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S101_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S101_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_8_block_block_1_block_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_8_block_block_1_block_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1453", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1453.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S104_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S104_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S104_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S104_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02395 out: 0.01581  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S104_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S104_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_8_block_block_2_block_2_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_8_block_block_2_block_2_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1456", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1456.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S107_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S107_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S107_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S107_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S107_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S107_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [223] IN1SCALEN: [7] OUTSCALE: [139] OUTSCALEN: [8]
                TCArgInfo("signed char * __restrict__", "S108_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S108_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_9_block_block_0_block_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_9_block_block_0_block_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1459", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1459.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S111_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S111_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S111_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S111_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02833 out: 0.03415  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S111_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S111_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_9_block_block_1_block_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_9_block_block_1_block_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1462", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1462.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S114_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S114_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S114_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S114_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02326 out: 0.02538  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S114_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S114_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_9_block_block_2_block_2_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_9_block_block_2_block_2_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1465", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1465.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S117_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S117_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S117_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S117_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S117_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S117_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [175] IN1SCALEN: [7] OUTSCALE: [173] OUTSCALEN: [8]
                TCArgInfo("signed char * __restrict__", "S118_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S118_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_10_block_block_0_block_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_10_block_block_0_block_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1468", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1468.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S121_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S121_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S121_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S121_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02412 out: 0.02837  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S121_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S121_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_10_block_block_1_block_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_10_block_block_1_block_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1471", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1471.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S124_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S124_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S124_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S124_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02057 out: 0.02182  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S124_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S124_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_10_block_block_2_block_2_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_10_block_block_2_block_2_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1474", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1474.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S127_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S127_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S127_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S127_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S127_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S127_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [103] IN1SCALEN: [6] OUTSCALE: [159] OUTSCALEN: [8]
                TCArgInfo("signed char * __restrict__", "S128_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S128_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_11_block_block_0_block_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_11_block_block_0_block_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1477", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1477.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S131_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S131_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S131_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S131_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02837 out: 0.03095  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S131_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S131_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_11_block_block_1_block_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_11_block_block_1_block_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1480", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1480.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S134_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S134_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S134_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S134_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04079 out: 0.02990  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S134_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S134_Infos.tensor", 1, 1, 8, 0)),
                // no activation ACTSCALE: 0 ACTSCALEN: 0 GLOBAL_SUM_SCALE: [127] GLOBAL_SUM_SCALEN: [12]
                TCArgInfo("signed char * __restrict__", "S136_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S136_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_11_block_block_2_fc1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_11_block_block_2_fc1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_backbone_features_0_11_block_2_fc1_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_backbone_features_0_11_block_2_fc1_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S140_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S140_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S140_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S140_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00959 out: 0.00959  actscale: [1] actscalen: [0] a0: [-128] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S140_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S140_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_11_block_block_2_fc2_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_11_block_block_2_fc2_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_backbone_features_0_11_block_2_fc2_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_backbone_features_0_11_block_2_fc2_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S143_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S143_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S143_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S143_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.00787  actscale: [1] actscalen: [0] a0: [127] b0: [64] c0: [0] BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S143_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S143_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_11_block_block_3_block_3_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_11_block_block_3_block_3_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1483", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1483.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S147_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S147_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S147_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S147_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S147_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S147_Infos.tensor", 1, 1, 8, 0)),
                // in q: -0.84<(i8-0.00)*0.00659090<0.84 forced out_q: -0.78<(i8--5.00)*0.00634119<0.84
                TCArgInfo("signed char * __restrict__", "S148_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S148_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_12_block_block_0_block_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_12_block_block_0_block_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1486", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1486.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S151_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S151_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S151_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S151_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01972 out: 0.01742  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S151_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S151_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_12_block_block_1_block_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_12_block_block_1_block_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1489", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1489.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S154_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S154_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S154_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S154_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02587 out: 0.02651  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S154_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S154_Infos.tensor", 1, 1, 8, 0)),
                // no activation ACTSCALE: 0 ACTSCALEN: 0 GLOBAL_SUM_SCALE: [239] GLOBAL_SUM_SCALEN: [13]
                TCArgInfo("signed char * __restrict__", "S156_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S156_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_12_block_block_2_fc1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_12_block_block_2_fc1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_backbone_features_0_12_block_2_fc1_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_backbone_features_0_12_block_2_fc1_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S160_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S160_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S160_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S160_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00879 out: 0.00879  actscale: [1] actscalen: [0] a0: [-128] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S160_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S160_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_12_block_block_2_fc2_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_12_block_block_2_fc2_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_backbone_features_0_12_block_2_fc2_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_backbone_features_0_12_block_2_fc2_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S163_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S163_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S163_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S163_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.00787  actscale: [1] actscalen: [0] a0: [127] b0: [64] c0: [0] BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S163_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S163_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_12_block_block_3_block_3_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_12_block_block_3_block_3_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1492", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1492.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S167_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S167_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S167_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S167_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S167_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S167_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [195] IN1SCALEN: [7] OUTSCALE: [157] OUTSCALEN: [8]
                TCArgInfo("signed char * __restrict__", "S168_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S168_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_13_features_0_13_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_13_features_0_13_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1495", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1495.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S171_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S171_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S171_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S171_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01810 out: 0.01998  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S171_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S171_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_1_features_1_0_features_1_0_1_features_1_0_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_1_features_1_0_features_1_0_1_features_1_0_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1498", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1498.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S174_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S174_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S174_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S174_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01851 out: 0.02083  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S174_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S174_Infos.tensor", 1, 1, 8, 0)),
                // no activation ACTSCALE: 0 ACTSCALEN: 0 GLOBAL_SUM_SCALE: [3] GLOBAL_SUM_SCALEN: [7]
                TCArgInfo("signed char * __restrict__", "S176_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S176_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_1_features_1_0_features_1_0_2_fc1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_1_features_1_0_features_1_0_2_fc1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_backbone_features_1_0_2_fc1_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_backbone_features_1_0_2_fc1_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S180_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S180_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S180_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S180_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00469 out: 0.00469  actscale: [1] actscalen: [0] a0: [-128] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S180_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S180_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_1_features_1_0_features_1_0_2_fc2_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_1_features_1_0_features_1_0_2_fc2_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_backbone_features_1_0_2_fc2_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_backbone_features_1_0_2_fc2_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S183_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S183_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S183_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S183_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.00787  actscale: [1] actscalen: [0] a0: [127] b0: [64] c0: [0] BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S183_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S183_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_1_features_1_0_features_1_0_3_features_1_0_3_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_1_features_1_0_features_1_0_3_features_1_0_3_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1501", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1501.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S187_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S187_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S187_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S187_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S187_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S187_Infos.tensor", 1, 1, 8, 0)),
                // in q: -0.69<(i8-0.00)*0.00539086<0.68 forced out_q: -0.69<(i8-0.00)*0.00537912<0.68
                TCArgInfo("signed char * __restrict__", "S188_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S188_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_1_features_1_1_block_block_0_block_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_1_features_1_1_block_block_0_block_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1504", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1504.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S191_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S191_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S191_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S191_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01122 out: 0.00701  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S191_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S191_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_1_features_1_1_block_block_1_block_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_1_features_1_1_block_block_1_block_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1507", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1507.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S194_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S194_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S194_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S194_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01377 out: 0.00629  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S194_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S194_Infos.tensor", 1, 1, 8, 0)),
                // no activation ACTSCALE: 0 ACTSCALEN: 0 GLOBAL_SUM_SCALE: [143] GLOBAL_SUM_SCALEN: [13]
                TCArgInfo("signed char * __restrict__", "S196_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S196_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_1_features_1_1_block_block_2_fc1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_1_features_1_1_block_block_2_fc1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_backbone_features_1_1_block_2_fc1_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_backbone_features_1_1_block_2_fc1_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S200_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S200_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S200_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S200_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00202 out: 0.00202  actscale: [1] actscalen: [0] a0: [-128] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S200_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S200_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_1_features_1_1_block_block_2_fc2_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_1_features_1_1_block_block_2_fc2_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_backbone_features_1_1_block_2_fc2_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_backbone_features_1_1_block_2_fc2_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S203_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S203_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S203_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S203_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.00787  actscale: [1] actscalen: [0] a0: [127] b0: [64] c0: [0] BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S203_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S203_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_1_features_1_1_block_block_3_block_3_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_1_features_1_1_block_block_3_block_3_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1510", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1510.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S207_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S207_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S207_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S207_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S207_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S207_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [67] IN1SCALEN: [6] OUTSCALE: [203] OUTSCALEN: [8]
                TCArgInfo("signed char * __restrict__", "S208_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S208_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_1_features_1_2_block_block_0_block_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_1_features_1_2_block_block_0_block_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1513", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1513.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S211_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S211_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S211_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S211_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01235 out: 0.00707  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S211_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S211_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_1_features_1_2_block_block_1_block_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_1_features_1_2_block_block_1_block_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1516", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1516.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S214_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S214_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S214_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S214_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01220 out: 0.00533  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S214_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S214_Infos.tensor", 1, 1, 8, 0)),
                // no activation ACTSCALE: 0 ACTSCALEN: 0 GLOBAL_SUM_SCALE: [135] GLOBAL_SUM_SCALEN: [13]
                TCArgInfo("signed char * __restrict__", "S216_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S216_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_1_features_1_2_block_block_2_fc1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_1_features_1_2_block_block_2_fc1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_backbone_features_1_2_block_2_fc1_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_backbone_features_1_2_block_2_fc1_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S220_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S220_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S220_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S220_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00316 out: 0.00316  actscale: [1] actscalen: [0] a0: [-128] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S220_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S220_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_1_features_1_2_block_block_2_fc2_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_1_features_1_2_block_block_2_fc2_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_backbone_features_1_2_block_2_fc2_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_backbone_features_1_2_block_2_fc2_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S223_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S223_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S223_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S223_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.00787  actscale: [1] actscalen: [0] a0: [127] b0: [64] c0: [0] BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S223_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S223_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_1_features_1_2_block_block_3_block_3_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_1_features_1_2_block_block_3_block_3_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1519", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1519.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S227_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S227_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S227_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S227_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S227_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S227_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [65] IN1SCALEN: [6] OUTSCALE: [185] OUTSCALEN: [8]
                TCArgInfo("signed char * __restrict__", "S228_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S228_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_1_features_1_3_features_1_3_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_1_features_1_3_features_1_3_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1522", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1522.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S231_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S231_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S231_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S231_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01304 out: 0.01073  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S231_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S231_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_extra_0_extra_0_0_extra_0_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_extra_0_extra_0_0_extra_0_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1525", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1525.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S234_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S234_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S234_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S234_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00412 out: 0.00412  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S234_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S234_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_extra_0_extra_0_1_extra_0_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_extra_0_extra_0_1_extra_0_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1528", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1528.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S237_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S237_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S237_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S237_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [-128] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S237_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S237_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_extra_0_extra_0_2_extra_0_2_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_extra_0_extra_0_2_extra_0_2_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1531", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1531.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S240_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S240_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S240_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S240_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00690 out: 0.00690  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S240_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S240_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_extra_1_extra_1_0_extra_1_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_extra_1_extra_1_0_extra_1_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1534", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1534.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S243_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S243_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S243_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S243_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00368 out: 0.00368  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S243_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S243_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_extra_1_extra_1_1_extra_1_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_extra_1_extra_1_1_extra_1_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1537", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1537.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S246_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S246_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S246_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S246_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [-128] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S246_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S246_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_extra_1_extra_1_2_extra_1_2_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_extra_1_extra_1_2_extra_1_2_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1540", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1540.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S249_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S249_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S249_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S249_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00819 out: 0.00819  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S249_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S249_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_extra_2_extra_2_0_extra_2_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_extra_2_extra_2_0_extra_2_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1543", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1543.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S252_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S252_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S252_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S252_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00428 out: 0.00428  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S252_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S252_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_extra_2_extra_2_1_extra_2_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_extra_2_extra_2_1_extra_2_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1546", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1546.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S255_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S255_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S255_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S255_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [-128] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S255_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S255_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_extra_2_extra_2_2_extra_2_2_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_extra_2_extra_2_2_extra_2_2_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1549", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1549.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S258_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S258_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S258_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S258_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00471 out: 0.00471  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S258_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S258_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_extra_3_extra_3_0_extra_3_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_extra_3_extra_3_0_extra_3_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1552", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1552.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S261_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S261_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S261_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S261_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00308 out: 0.00308  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S261_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S261_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_extra_3_extra_3_1_extra_3_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_extra_3_extra_3_1_extra_3_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1555", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1555.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S264_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S264_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S264_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S264_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [-128] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S264_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S264_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_extra_3_extra_3_2_extra_3_2_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_extra_3_extra_3_2_extra_3_2_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1558", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1558.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S267_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S267_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S267_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S267_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00875 out: 0.00875  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S267_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S267_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1561", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1561.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S270_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S270_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S270_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S270_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.04724  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S270_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S270_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_head_regression_head_module_list_0_module_list_0_1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_regression_head_module_list_0_module_list_0_1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_head_regression_head_module_list_0_1_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_head_regression_head_module_list_0_1_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S273_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S273_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S273_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S273_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00025 out: 0.00003  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S273_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S273_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1564", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1564.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S277_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S277_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S277_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S277_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [-128] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S277_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S277_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_head_regression_head_module_list_1_module_list_1_1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_regression_head_module_list_1_module_list_1_1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1567", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1567.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S281_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S281_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S281_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S281_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.04724  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S281_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S281_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_head_regression_head_module_list_2_module_list_2_1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_regression_head_module_list_2_module_list_2_1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1570", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1570.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S285_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S285_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S285_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S285_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [-128] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S285_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S285_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_head_regression_head_module_list_3_module_list_3_1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_regression_head_module_list_3_module_list_3_1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_head_regression_head_module_list_4_module_list_4_1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_regression_head_module_list_4_module_list_4_1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_head_regression_head_module_list_5_module_list_5_0_module_list_5_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_regression_head_module_list_5_module_list_5_0_module_list_5_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1576", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1576.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S291_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S291_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S291_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S291_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [-128] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S291_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S291_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_head_regression_head_module_list_5_module_list_5_1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_regression_head_module_list_5_module_list_5_1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_head_classification_head_module_list_0_module_list_0_1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_classification_head_module_list_0_module_list_0_1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_head_classification_head_module_list_0_1_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_head_classification_head_module_list_0_1_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed char * __restrict__", "_head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_head_classification_head_module_list_1_module_list_1_1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_classification_head_module_list_1_module_list_1_1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_head_classification_head_module_list_2_module_list_2_1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_classification_head_module_list_2_module_list_2_1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_head_classification_head_module_list_3_module_list_3_1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_classification_head_module_list_3_module_list_3_1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_head_classification_head_module_list_4_module_list_4_1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_classification_head_module_list_4_module_list_4_1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_head_classification_head_module_list_5_module_list_5_0_module_list_5_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_classification_head_module_list_5_module_list_5_0_module_list_5_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_head_classification_head_module_list_5_module_list_5_1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_classification_head_module_list_5_module_list_5_1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_sub", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_sub.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_add", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_add.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_sub_1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_sub_1.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_add_1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_add_1.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1561_1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1561_1.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S311_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S311_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S311_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S311_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [-128] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S311_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S311_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S312_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S312_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S312_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S312_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S312_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S312_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_head_regression_head_module_list_0_1_bias_1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_head_regression_head_module_list_0_1_bias_1.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S315_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S315_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S315_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S315_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00013 out: 0.00003  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S315_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S315_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_head_regression_head_module_list_0_1_bias_2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_head_regression_head_module_list_0_1_bias_2.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S318_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S318_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S318_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S318_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00011 out: 0.00003  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S318_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S318_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_head_regression_head_module_list_0_1_bias_3", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_head_regression_head_module_list_0_1_bias_3.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S321_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S321_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S321_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S321_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00012 out: 0.00003  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S321_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S321_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_head_regression_head_module_list_0_1_bias_4", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_head_regression_head_module_list_0_1_bias_4.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_head_regression_head_module_list_0_1_bias_5", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_head_regression_head_module_list_0_1_bias_5.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S325_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S325_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S325_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S325_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00005 out: 0.00003  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S325_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S325_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1564_1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1564_1.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S328_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S328_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S328_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S328_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [-128] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S328_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S328_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1567_1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1567_1.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S330_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S330_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S330_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S330_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [-128] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S330_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S330_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1570_1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1570_1.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S332_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S332_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S332_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S332_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [-128] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S332_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S332_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S333_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S333_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S333_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S333_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00005 out: 0.00003  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S333_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S333_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1570_2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1570_2.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S342_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S342_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S342_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S342_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [-128] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S342_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S342_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1570_3", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1570_3.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S344_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S344_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S344_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S344_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [-128] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S344_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S344_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1576_1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1576_1.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S346_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S346_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S346_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S346_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [-128] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S346_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S346_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_head_classification_head_module_list_0_1_bias_1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_head_classification_head_module_list_0_1_bias_1.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S348_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S348_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S348_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S348_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S348_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S348_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_head_classification_head_module_list_0_1_bias_2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_head_classification_head_module_list_0_1_bias_2.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S351_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S351_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S351_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S351_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S351_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S351_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_head_classification_head_module_list_0_1_bias_3", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_head_classification_head_module_list_0_1_bias_3.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S354_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S354_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S354_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S354_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S354_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S354_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_head_classification_head_module_list_0_1_bias_4", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_head_classification_head_module_list_0_1_bias_4.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S357_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S357_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S357_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S357_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S357_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S357_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_head_classification_head_module_list_0_1_bias_5", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_head_classification_head_module_list_0_1_bias_5.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S360_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S360_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S360_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S360_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S360_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S360_Infos.tensor", 1, 1, 8, 0)),
                // in: 0.00049 out: 0.00003  BIASL_SM: [4]
                TCArgInfo("signed char * __restrict__", "S363_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S363_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_sub_1_1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_sub_1_1.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_sub_1_2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_sub_1_2.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Output_1", ARG_SCOPE_ARG_ALLOC, ARG_DIR_OUT, AT_MEM_L2, AT_MEM_L2, 0),
                TCArgInfo("signed char * __restrict__", "Output_2", ARG_SCOPE_ARG_ALLOC, ARG_DIR_OUT, AT_MEM_L2, AT_MEM_L2, 0),
                TCArgInfo("signed char * __restrict__", "Output_3", ARG_SCOPE_ARG_ALLOC, ARG_DIR_OUT, AT_MEM_L2, AT_MEM_L2, 0),
                TCArgInfo("signed char * __restrict__", "Output_4", ARG_SCOPE_ARG_ALLOC, ARG_DIR_OUT, AT_MEM_L2, AT_MEM_L2, 0),
                TCArgInfo("signed short * __restrict__", "Output_5", ARG_SCOPE_ARG_ALLOC, ARG_DIR_OUT, AT_MEM_L2, AT_MEM_L2, 0)
            ),
        /* Locals, allocated dynamically */
        CArgs(129,
            TCArgInfo("signed char * __restrict__", "S3_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S6_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S9_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S10_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S13_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S16_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S19_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S22_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S25_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S28_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S29_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S32_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S35_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S36_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S37_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S41_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S44_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S45_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S48_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S51_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S54_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S55_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S56_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S60_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S63_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S64_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S67_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S68_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S71_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S74_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S75_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S76_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S80_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S83_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S84_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S87_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S88_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S91_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S94_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S97_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S98_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S101_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S104_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S107_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S108_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S111_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S114_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S117_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S118_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S121_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S124_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S127_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S128_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S131_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S134_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S135_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S136_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S140_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S143_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S144_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S147_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S148_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S151_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S154_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S155_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S156_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S160_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S163_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S164_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S167_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S168_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S171_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S174_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S175_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S176_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S180_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S183_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S184_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S187_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S188_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S191_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S194_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S195_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S196_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S200_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S203_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S204_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S207_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S208_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S211_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S214_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S215_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S216_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S220_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S223_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S224_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S227_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S228_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S231_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S234_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S237_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S240_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S243_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S246_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S249_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S252_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S255_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S258_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S261_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S264_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S267_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S270_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S277_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S281_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S285_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S291_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S311_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S328_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S330_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S332_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S335_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S336_Output_0", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S336_Output_3", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S336_Output_2", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S336_Output_1", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S342_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S344_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S346_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S362_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0)
        )
    );

    // Stacked tensors for concats and splits
    AddStackedTensors("S335_Output", 6, "S273_Output", "S315_Output", "S318_Output", "S321_Output", "S333_Output", "S325_Output");
    AddStackedTensors("S362_Output", 6, "S312_Output", "S348_Output", "S351_Output", "S354_Output", "S357_Output", "S360_Output");

    // Node S3_Conv2d_16x3x3x3_Custom inq -1.00<(i8-0.00)*0.00781250<0.99 forced weightsq chan<(i8-0.00)*chan<chan outq -10.79<(i8-0.00)*0.08432552<10.71 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S3_Conv2d_16x3x3x3_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "Input_1", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_0_features_0_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1387", 0),
            GNodeArg(GNA_OUT, "S3_Output", 0),
            GNodeArg(GNA_IN, "S3_Mul_scale", 0),
            GNodeArg(GNA_IN, "S3_Mul_shift", 0),
            GNodeArg(GNA_IN, "S3_Infos", 0)
        )
    );
    // Node S6_Conv2d_16x3x3x1_Relu inq -10.79<(i8-0.00)*0.08432552<10.71 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.01610397<4.11 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S6_Conv2d_16x3x3x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S3_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_1_block_block_0_block_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1390", 0),
            GNodeArg(GNA_OUT, "S6_Output", 0),
            GNodeArg(GNA_IN, "S6_Mul_scale", 0),
            GNodeArg(GNA_IN, "S6_Mul_shift", 0),
            GNodeArg(GNA_IN, "S6_Infos", 0)
        )
    );
    // Node S9_Conv2d_16x1x1x16 inq 0.00<(i8--128.00)*0.01610397<4.11 weightsq chan<(i8-0.00)*chan<chan outq -5.59<(i8-0.00)*0.04365827<5.54 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S9_Conv2d_16x1x1x16",
        Bindings(7,
            GNodeArg(GNA_IN, "S6_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_1_block_block_1_block_1_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1393", 0),
            GNodeArg(GNA_OUT, "S9_Output", 0),
            GNodeArg(GNA_IN, "S9_Mul_scale", 0),
            GNodeArg(GNA_IN, "S9_Mul_shift", 0),
            GNodeArg(GNA_IN, "S9_Infos", 0)
        )
    );
    // Node S10_MatAdd_128x160x16 in1q -10.79<(i8-0.00)*0.08432552<10.71 forced
    //   in2q -5.59<(i8-0.00)*0.04365827<5.54 forced
    //   outq -12.35<(i8-0.00)*0.09650712<12.26 forced scaled input 0 is node input 1
    AddNode("S10_MatAdd_128x160x16",
        Bindings(4,
            GNodeArg(GNA_IN, "S3_Output", 0),
            GNodeArg(GNA_IN, "S9_Output", 0),
            GNodeArg(GNA_OUT, "S10_Output", 0),
            GNodeArg(GNA_IN, "S10_Infos", 0)
        )
    );
    // Node S13_Conv2d_64x1x1x16_Relu inq -12.35<(i8-0.00)*0.09650712<12.26 forced weightsq chan<(i8-0.00)*chan<chan outq -2.56<(i8-0.00)*0.01997015<2.54 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S13_Conv2d_64x1x1x16_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S10_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_2_block_block_0_block_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1396", 0),
            GNodeArg(GNA_OUT, "S13_Output", 0),
            GNodeArg(GNA_IN, "S13_Mul_scale", 0),
            GNodeArg(GNA_IN, "S13_Mul_shift", 0),
            GNodeArg(GNA_IN, "S13_Infos", 0)
        )
    );
    // Node S16_Conv2d_64x3x3x1_Relu inq -2.56<(i8-0.00)*0.01997015<2.54 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.01775541<4.53 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S16_Conv2d_64x3x3x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S13_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_2_block_block_1_block_1_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1399", 0),
            GNodeArg(GNA_OUT, "S16_Output", 0),
            GNodeArg(GNA_IN, "S16_Mul_scale", 0),
            GNodeArg(GNA_IN, "S16_Mul_shift", 0),
            GNodeArg(GNA_IN, "S16_Infos", 0)
        )
    );
    // Node S19_Conv2d_24x1x1x64 inq 0.00<(i8--128.00)*0.01775541<4.53 weightsq chan<(i8-0.00)*chan<chan outq -3.75<(i8-0.00)*0.02926593<3.72 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S19_Conv2d_24x1x1x64",
        Bindings(7,
            GNodeArg(GNA_IN, "S16_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_2_block_block_2_block_2_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1402", 0),
            GNodeArg(GNA_OUT, "S19_Output", 0),
            GNodeArg(GNA_IN, "S19_Mul_scale", 0),
            GNodeArg(GNA_IN, "S19_Mul_shift", 0),
            GNodeArg(GNA_IN, "S19_Infos", 0)
        )
    );
    // Node S22_Conv2d_72x1x1x24_Relu inq -3.75<(i8-0.00)*0.02926593<3.72 forced weightsq chan<(i8-0.00)*chan<chan outq -1.71<(i8-0.00)*0.01332306<1.69 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S22_Conv2d_72x1x1x24_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S19_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_3_block_block_0_block_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1405", 0),
            GNodeArg(GNA_OUT, "S22_Output", 0),
            GNodeArg(GNA_IN, "S22_Mul_scale", 0),
            GNodeArg(GNA_IN, "S22_Mul_shift", 0),
            GNodeArg(GNA_IN, "S22_Infos", 0)
        )
    );
    // Node S25_Conv2d_72x3x3x1_Relu inq -1.71<(i8-0.00)*0.01332306<1.69 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.01166134<2.97 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S25_Conv2d_72x3x3x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S22_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_3_block_block_1_block_1_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1408", 0),
            GNodeArg(GNA_OUT, "S25_Output", 0),
            GNodeArg(GNA_IN, "S25_Mul_scale", 0),
            GNodeArg(GNA_IN, "S25_Mul_shift", 0),
            GNodeArg(GNA_IN, "S25_Infos", 0)
        )
    );
    // Node S28_Conv2d_24x1x1x72 inq 0.00<(i8--128.00)*0.01166134<2.97 weightsq chan<(i8-0.00)*chan<chan outq -3.80<(i8-0.00)*0.02969123<3.77 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S28_Conv2d_24x1x1x72",
        Bindings(7,
            GNodeArg(GNA_IN, "S25_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_3_block_block_2_block_2_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1411", 0),
            GNodeArg(GNA_OUT, "S28_Output", 0),
            GNodeArg(GNA_IN, "S28_Mul_scale", 0),
            GNodeArg(GNA_IN, "S28_Mul_shift", 0),
            GNodeArg(GNA_IN, "S28_Infos", 0)
        )
    );
    // Node S29_MatAdd_64x80x24 in1q -3.80<(i8-0.00)*0.02969123<3.77 forced
    //   in2q -3.75<(i8-0.00)*0.02926593<3.72 forced
    //   outq -6.78<(i8-0.00)*0.05293358<6.72 forced scaled input 0 is node input 0
    AddNode("S29_MatAdd_64x80x24",
        Bindings(4,
            GNodeArg(GNA_IN, "S28_Output", 0),
            GNodeArg(GNA_IN, "S19_Output", 0),
            GNodeArg(GNA_OUT, "S29_Output", 0),
            GNodeArg(GNA_IN, "S29_Infos", 0)
        )
    );
    // Node S32_Conv2d_72x1x1x24_Relu inq -6.78<(i8-0.00)*0.05293358<6.72 forced weightsq chan<(i8-0.00)*chan<chan outq -2.53<(i8-0.00)*0.01978011<2.51 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S32_Conv2d_72x1x1x24_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S29_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_4_block_block_0_block_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1414", 0),
            GNodeArg(GNA_OUT, "S32_Output", 0),
            GNodeArg(GNA_IN, "S32_Mul_scale", 0),
            GNodeArg(GNA_IN, "S32_Mul_shift", 0),
            GNodeArg(GNA_IN, "S32_Infos", 0)
        )
    );
    // Node S35_Conv2d_72x5x5x1_Relu inq -2.53<(i8-0.00)*0.01978011<2.51 forced weightsq chan<(i8-0.00)*chan<chan outq -11.86<(i8-0.00)*0.09261950<11.76 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S35_Conv2d_72x5x5x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S32_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_4_block_block_1_block_1_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1417", 0),
            GNodeArg(GNA_OUT, "S35_Output", 0),
            GNodeArg(GNA_IN, "S35_Mul_scale", 0),
            GNodeArg(GNA_IN, "S35_Mul_shift", 0),
            GNodeArg(GNA_IN, "S35_Infos", 0)
        )
    );
    // Node _backbone_features_0_features_0_4_block_block_2_avgpool_GlobalAveragePool_trans inq -11.86<(i8-0.00)*0.09261950<11.76 forced outq -11.86<(i8-0.00)*0.09261950<11.76 forced
    AddNode("S36_Op__backbone_features_0_features_0_4_block_block_2_avgpool_GlobalAveragePool_trans",
        Bindings(2,
            GNodeArg(GNA_IN, "S35_Output", 0),
            GNodeArg(GNA_OUT, "S36_Output", 0)
        )
    );
    // Node _backbone_features_0_features_0_4_block_block_2_avgpool_GlobalAveragePool inq -11.86<(i8-0.00)*0.09261950<11.76 forced outq -4.50<(i8-0.00)*0.03516799<4.47
    AddNode("S37_Op__backbone_features_0_features_0_4_block_block_2_avgpool_GlobalAveragePool",
        Bindings(3,
            GNodeArg(GNA_IN, "S36_Output", 0),
            GNodeArg(GNA_OUT, "S37_Output", 0),
            GNodeArg(GNA_IN, "S37_Infos", 0)
        )
    );
    // Node S41_Conv2d_24x1x1x72_Relu inq -4.50<(i8-0.00)*0.03516799<4.47 weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.00801153<2.04 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S41_Conv2d_24x1x1x72_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S37_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_4_block_block_2_fc1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_backbone_features_0_4_block_2_fc1_bias", 0),
            GNodeArg(GNA_OUT, "S41_Output", 0),
            GNodeArg(GNA_IN, "S41_Mul_scale", 0),
            GNodeArg(GNA_IN, "S41_Mul_shift", 0),
            GNodeArg(GNA_IN, "S41_Infos", 0)
        )
    );
    // Node S44_Conv2d_72x1x1x24_Hsigmoid inq 0.00<(i8--128.00)*0.00801153<2.04 weightsq chan<(i8-0.00)*chan<chan outq -1.01<(i8-0.00)*0.00787402<1.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S44_Conv2d_72x1x1x24_Hsigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S41_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_4_block_block_2_fc2_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_backbone_features_0_4_block_2_fc2_bias", 0),
            GNodeArg(GNA_OUT, "S44_Output", 0),
            GNodeArg(GNA_IN, "S44_Mul_scale", 0),
            GNodeArg(GNA_IN, "S44_Mul_shift", 0),
            GNodeArg(GNA_IN, "S44_Infos", 0)
        )
    );
    // Node expr_16 in_qs [-11.86<(i8-0.00)*0.09261950<11.76 forced,-1.01<(i8-0.00)*0.00787402<1.00] out_qs [-1.49<(i8-0.00)*0.01161895<1.48]
    AddNode("S45_Op_expr_16",
        Bindings(3,
            GNodeArg(GNA_IN, "S35_Output", 0),
            GNodeArg(GNA_IN, "S44_Output", 0),
            GNodeArg(GNA_OUT, "S45_Output", 0)
        )
    );
    // Node S48_Conv2d_40x1x1x72 inq -1.49<(i8-0.00)*0.01161895<1.48 weightsq chan<(i8-0.00)*chan<chan outq -3.02<(i8-0.00)*0.02356693<2.99 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S48_Conv2d_40x1x1x72",
        Bindings(7,
            GNodeArg(GNA_IN, "S45_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_4_block_block_3_block_3_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1420", 0),
            GNodeArg(GNA_OUT, "S48_Output", 0),
            GNodeArg(GNA_IN, "S48_Mul_scale", 0),
            GNodeArg(GNA_IN, "S48_Mul_shift", 0),
            GNodeArg(GNA_IN, "S48_Infos", 0)
        )
    );
    // Node S51_Conv2d_120x1x1x40_Relu inq -3.02<(i8-0.00)*0.02356693<2.99 forced weightsq chan<(i8-0.00)*chan<chan outq -1.34<(i8-0.00)*0.01047194<1.33 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S51_Conv2d_120x1x1x40_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S48_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_5_block_block_0_block_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1423", 0),
            GNodeArg(GNA_OUT, "S51_Output", 0),
            GNodeArg(GNA_IN, "S51_Mul_scale", 0),
            GNodeArg(GNA_IN, "S51_Mul_shift", 0),
            GNodeArg(GNA_IN, "S51_Infos", 0)
        )
    );
    // Node S54_Conv2d_120x5x5x1_Relu inq -1.34<(i8-0.00)*0.01047194<1.33 forced weightsq chan<(i8-0.00)*chan<chan outq -3.36<(i8-0.00)*0.02623591<3.33 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S54_Conv2d_120x5x5x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S51_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_5_block_block_1_block_1_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1426", 0),
            GNodeArg(GNA_OUT, "S54_Output", 0),
            GNodeArg(GNA_IN, "S54_Mul_scale", 0),
            GNodeArg(GNA_IN, "S54_Mul_shift", 0),
            GNodeArg(GNA_IN, "S54_Infos", 0)
        )
    );
    // Node _backbone_features_0_features_0_5_block_block_2_avgpool_GlobalAveragePool_trans inq -3.36<(i8-0.00)*0.02623591<3.33 forced outq -3.36<(i8-0.00)*0.02623591<3.33 forced
    AddNode("S55_Op__backbone_features_0_features_0_5_block_block_2_avgpool_GlobalAveragePool_trans",
        Bindings(2,
            GNodeArg(GNA_IN, "S54_Output", 0),
            GNodeArg(GNA_OUT, "S55_Output", 0)
        )
    );
    // Node _backbone_features_0_features_0_5_block_block_2_avgpool_GlobalAveragePool inq -3.36<(i8-0.00)*0.02623591<3.33 forced outq -1.49<(i8-0.00)*0.01167470<1.48
    AddNode("S56_Op__backbone_features_0_features_0_5_block_block_2_avgpool_GlobalAveragePool",
        Bindings(3,
            GNodeArg(GNA_IN, "S55_Output", 0),
            GNodeArg(GNA_OUT, "S56_Output", 0),
            GNodeArg(GNA_IN, "S56_Infos", 0)
        )
    );
    // Node S60_Conv2d_32x1x1x120_Relu inq -1.49<(i8-0.00)*0.01167470<1.48 weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.00691821<1.76 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S60_Conv2d_32x1x1x120_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S56_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_5_block_block_2_fc1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_backbone_features_0_5_block_2_fc1_bias", 0),
            GNodeArg(GNA_OUT, "S60_Output", 0),
            GNodeArg(GNA_IN, "S60_Mul_scale", 0),
            GNodeArg(GNA_IN, "S60_Mul_shift", 0),
            GNodeArg(GNA_IN, "S60_Infos", 0)
        )
    );
    // Node S63_Conv2d_120x1x1x32_Hsigmoid inq 0.00<(i8--128.00)*0.00691821<1.76 weightsq chan<(i8-0.00)*chan<chan outq -1.01<(i8-0.00)*0.00787402<1.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S63_Conv2d_120x1x1x32_Hsigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S60_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_5_block_block_2_fc2_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_backbone_features_0_5_block_2_fc2_bias", 0),
            GNodeArg(GNA_OUT, "S63_Output", 0),
            GNodeArg(GNA_IN, "S63_Mul_scale", 0),
            GNodeArg(GNA_IN, "S63_Mul_shift", 0),
            GNodeArg(GNA_IN, "S63_Infos", 0)
        )
    );
    // Node expr_17 in_qs [-3.36<(i8-0.00)*0.02623591<3.33 forced,-1.01<(i8-0.00)*0.00787402<1.00] out_qs [-1.00<(i8-0.00)*0.00783870<1.00]
    AddNode("S64_Op_expr_17",
        Bindings(3,
            GNodeArg(GNA_IN, "S54_Output", 0),
            GNodeArg(GNA_IN, "S63_Output", 0),
            GNodeArg(GNA_OUT, "S64_Output", 0)
        )
    );
    // Node S67_Conv2d_40x1x1x120 inq -1.00<(i8-0.00)*0.00783870<1.00 weightsq chan<(i8-0.00)*chan<chan outq -3.83<(i8-0.00)*0.02995418<3.80 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S67_Conv2d_40x1x1x120",
        Bindings(7,
            GNodeArg(GNA_IN, "S64_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_5_block_block_3_block_3_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1429", 0),
            GNodeArg(GNA_OUT, "S67_Output", 0),
            GNodeArg(GNA_IN, "S67_Mul_scale", 0),
            GNodeArg(GNA_IN, "S67_Mul_shift", 0),
            GNodeArg(GNA_IN, "S67_Infos", 0)
        )
    );
    // Node S68_MatAdd_32x40x40 in1q -3.83<(i8-0.00)*0.02995418<3.80 forced
    //   in2q -3.02<(i8-0.00)*0.02356693<2.99 forced
    //   outq -3.80<(i8-0.00)*0.02966005<3.77 forced scaled input 0 is node input 0
    AddNode("S68_MatAdd_32x40x40",
        Bindings(4,
            GNodeArg(GNA_IN, "S67_Output", 0),
            GNodeArg(GNA_IN, "S48_Output", 0),
            GNodeArg(GNA_OUT, "S68_Output", 0),
            GNodeArg(GNA_IN, "S68_Infos", 0)
        )
    );
    // Node S71_Conv2d_120x1x1x40_Relu inq -3.80<(i8-0.00)*0.02966005<3.77 forced weightsq chan<(i8-0.00)*chan<chan outq -1.15<(i8-0.00)*0.00896385<1.14 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S71_Conv2d_120x1x1x40_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S68_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_6_block_block_0_block_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1432", 0),
            GNodeArg(GNA_OUT, "S71_Output", 0),
            GNodeArg(GNA_IN, "S71_Mul_scale", 0),
            GNodeArg(GNA_IN, "S71_Mul_shift", 0),
            GNodeArg(GNA_IN, "S71_Infos", 0)
        )
    );
    // Node S74_Conv2d_120x5x5x1_Relu inq -1.15<(i8-0.00)*0.00896385<1.14 forced weightsq chan<(i8-0.00)*chan<chan outq -3.78<(i8-0.00)*0.02951551<3.75 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S74_Conv2d_120x5x5x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S71_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_6_block_block_1_block_1_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1435", 0),
            GNodeArg(GNA_OUT, "S74_Output", 0),
            GNodeArg(GNA_IN, "S74_Mul_scale", 0),
            GNodeArg(GNA_IN, "S74_Mul_shift", 0),
            GNodeArg(GNA_IN, "S74_Infos", 0)
        )
    );
    // Node _backbone_features_0_features_0_6_block_block_2_avgpool_GlobalAveragePool_trans inq -3.78<(i8-0.00)*0.02951551<3.75 forced outq -3.78<(i8-0.00)*0.02951551<3.75 forced
    AddNode("S75_Op__backbone_features_0_features_0_6_block_block_2_avgpool_GlobalAveragePool_trans",
        Bindings(2,
            GNodeArg(GNA_IN, "S74_Output", 0),
            GNodeArg(GNA_OUT, "S75_Output", 0)
        )
    );
    // Node _backbone_features_0_features_0_6_block_block_2_avgpool_GlobalAveragePool inq -3.78<(i8-0.00)*0.02951551<3.75 forced outq -2.07<(i8-0.00)*0.01617079<2.05
    AddNode("S76_Op__backbone_features_0_features_0_6_block_block_2_avgpool_GlobalAveragePool",
        Bindings(3,
            GNodeArg(GNA_IN, "S75_Output", 0),
            GNodeArg(GNA_OUT, "S76_Output", 0),
            GNodeArg(GNA_IN, "S76_Infos", 0)
        )
    );
    // Node S80_Conv2d_32x1x1x120_Relu inq -2.07<(i8-0.00)*0.01617079<2.05 weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.00809516<2.06 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S80_Conv2d_32x1x1x120_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S76_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_6_block_block_2_fc1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_backbone_features_0_6_block_2_fc1_bias", 0),
            GNodeArg(GNA_OUT, "S80_Output", 0),
            GNodeArg(GNA_IN, "S80_Mul_scale", 0),
            GNodeArg(GNA_IN, "S80_Mul_shift", 0),
            GNodeArg(GNA_IN, "S80_Infos", 0)
        )
    );
    // Node S83_Conv2d_120x1x1x32_Hsigmoid inq 0.00<(i8--128.00)*0.00809516<2.06 weightsq chan<(i8-0.00)*chan<chan outq -1.01<(i8-0.00)*0.00787402<1.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S83_Conv2d_120x1x1x32_Hsigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S80_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_6_block_block_2_fc2_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_backbone_features_0_6_block_2_fc2_bias", 0),
            GNodeArg(GNA_OUT, "S83_Output", 0),
            GNodeArg(GNA_IN, "S83_Mul_scale", 0),
            GNodeArg(GNA_IN, "S83_Mul_shift", 0),
            GNodeArg(GNA_IN, "S83_Infos", 0)
        )
    );
    // Node expr_18 in_qs [-3.78<(i8-0.00)*0.02951551<3.75 forced,-1.01<(i8-0.00)*0.00787402<1.00] out_qs [-0.87<(i8-0.00)*0.00683511<0.87]
    AddNode("S84_Op_expr_18",
        Bindings(3,
            GNodeArg(GNA_IN, "S74_Output", 0),
            GNodeArg(GNA_IN, "S83_Output", 0),
            GNodeArg(GNA_OUT, "S84_Output", 0)
        )
    );
    // Node S87_Conv2d_40x1x1x120 inq -0.87<(i8-0.00)*0.00683511<0.87 weightsq chan<(i8-0.00)*chan<chan outq -2.56<(i8-0.00)*0.01998399<2.54 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S87_Conv2d_40x1x1x120",
        Bindings(7,
            GNodeArg(GNA_IN, "S84_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_6_block_block_3_block_3_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1438", 0),
            GNodeArg(GNA_OUT, "S87_Output", 0),
            GNodeArg(GNA_IN, "S87_Mul_scale", 0),
            GNodeArg(GNA_IN, "S87_Mul_shift", 0),
            GNodeArg(GNA_IN, "S87_Infos", 0)
        )
    );
    // Node S88_MatAdd_32x40x40 in1q -3.80<(i8-0.00)*0.02966005<3.77 forced
    //   in2q -2.56<(i8-0.00)*0.01998399<2.54 forced
    //   outq -5.02<(i8-0.00)*0.03918556<4.98 forced scaled input 0 is node input 1
    AddNode("S88_MatAdd_32x40x40",
        Bindings(4,
            GNodeArg(GNA_IN, "S68_Output", 0),
            GNodeArg(GNA_IN, "S87_Output", 0),
            GNodeArg(GNA_OUT, "S88_Output", 0),
            GNodeArg(GNA_IN, "S88_Infos", 0)
        )
    );
    // Node S91_Conv2d_240x1x1x40_Custom inq -5.02<(i8-0.00)*0.03918556<4.98 forced weightsq chan<(i8-0.00)*chan<chan outq -5.31<(i8-0.00)*0.04149349<5.27 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S91_Conv2d_240x1x1x40_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S88_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_7_block_block_0_block_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1441", 0),
            GNodeArg(GNA_OUT, "S91_Output", 0),
            GNodeArg(GNA_IN, "S91_Mul_scale", 0),
            GNodeArg(GNA_IN, "S91_Mul_shift", 0),
            GNodeArg(GNA_IN, "S91_Infos", 0)
        )
    );
    // Node S94_Conv2d_240x3x3x1_Custom inq -5.31<(i8-0.00)*0.04149349<5.27 forced weightsq chan<(i8-0.00)*chan<chan outq -4.92<(i8-0.00)*0.03842371<4.88 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S94_Conv2d_240x3x3x1_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S91_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_7_block_block_1_block_1_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1444", 0),
            GNodeArg(GNA_OUT, "S94_Output", 0),
            GNodeArg(GNA_IN, "S94_Mul_scale", 0),
            GNodeArg(GNA_IN, "S94_Mul_shift", 0),
            GNodeArg(GNA_IN, "S94_Infos", 0)
        )
    );
    // Node S97_Conv2d_80x1x1x240 inq -4.92<(i8-0.00)*0.03842371<4.88 weightsq chan<(i8-0.00)*chan<chan outq -2.49<(i8-0.00)*0.01944289<2.47 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S97_Conv2d_80x1x1x240",
        Bindings(7,
            GNodeArg(GNA_IN, "S94_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_7_block_block_2_block_2_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1447", 0),
            GNodeArg(GNA_OUT, "S97_Output", 0),
            GNodeArg(GNA_IN, "S97_Mul_scale", 0),
            GNodeArg(GNA_IN, "S97_Mul_shift", 0),
            GNodeArg(GNA_IN, "S97_Infos", 0)
        )
    );
    // Node _backbone_features_0_features_0_8_block_block_0_block_0_0_Conv_fusion_qin0 inq -2.49<(i8-0.00)*0.01944289<2.47 forced outq -2.49<(i8-18.00)*0.01705981<1.86
    AddNode("S98_Op__backbone_features_0_features_0_8_block_block_0_block_0_0_Conv_fusion_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S97_Output", 0),
            GNodeArg(GNA_OUT, "S98_Output", 0),
            GNodeArg(GNA_IN, "S98_Infos", 0)
        )
    );
    // Node S101_Conv2d_200x1x1x80_Custom inq -2.49<(i8-0.00)*0.01944289<2.47 forced weightsq chan<(i8-0.00)*chan<chan outq -7.32<(i8-0.00)*0.05715969<7.26 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S101_Conv2d_200x1x1x80_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S98_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_8_block_block_0_block_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1450", 0),
            GNodeArg(GNA_OUT, "S101_Output", 0),
            GNodeArg(GNA_IN, "S101_Mul_scale", 0),
            GNodeArg(GNA_IN, "S101_Mul_shift", 0),
            GNodeArg(GNA_IN, "S101_Infos", 0)
        )
    );
    // Node S104_Conv2d_200x3x3x1_Custom inq -7.32<(i8-0.00)*0.05715969<7.26 forced weightsq chan<(i8-0.00)*chan<chan outq -2.02<(i8-0.00)*0.01580674<2.01 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S104_Conv2d_200x3x3x1_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S101_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_8_block_block_1_block_1_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1453", 0),
            GNodeArg(GNA_OUT, "S104_Output", 0),
            GNodeArg(GNA_IN, "S104_Mul_scale", 0),
            GNodeArg(GNA_IN, "S104_Mul_shift", 0),
            GNodeArg(GNA_IN, "S104_Infos", 0)
        )
    );
    // Node S107_Conv2d_80x1x1x200 inq -2.02<(i8-0.00)*0.01580674<2.01 weightsq chan<(i8-0.00)*chan<chan outq -1.43<(i8-0.00)*0.01116935<1.42 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S107_Conv2d_80x1x1x200",
        Bindings(7,
            GNodeArg(GNA_IN, "S104_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_8_block_block_2_block_2_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1456", 0),
            GNodeArg(GNA_OUT, "S107_Output", 0),
            GNodeArg(GNA_IN, "S107_Mul_scale", 0),
            GNodeArg(GNA_IN, "S107_Mul_shift", 0),
            GNodeArg(GNA_IN, "S107_Infos", 0)
        )
    );
    // Node S108_MatAdd_16x20x80 in1q -2.49<(i8-0.00)*0.01944289<2.47 forced
    //   in2q -1.43<(i8-0.00)*0.01116935<1.42 forced
    //   outq -2.63<(i8-0.00)*0.02053471<2.61 forced scaled input 0 is node input 1
    AddNode("S108_MatAdd_16x20x80",
        Bindings(4,
            GNodeArg(GNA_IN, "S97_Output", 0),
            GNodeArg(GNA_IN, "S107_Output", 0),
            GNodeArg(GNA_OUT, "S108_Output", 0),
            GNodeArg(GNA_IN, "S108_Infos", 0)
        )
    );
    // Node S111_Conv2d_184x1x1x80_Custom inq -2.63<(i8-0.00)*0.02053471<2.61 forced weightsq chan<(i8-0.00)*chan<chan outq -4.37<(i8-0.00)*0.03414755<4.34 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S111_Conv2d_184x1x1x80_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S108_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_9_block_block_0_block_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1459", 0),
            GNodeArg(GNA_OUT, "S111_Output", 0),
            GNodeArg(GNA_IN, "S111_Mul_scale", 0),
            GNodeArg(GNA_IN, "S111_Mul_shift", 0),
            GNodeArg(GNA_IN, "S111_Infos", 0)
        )
    );
    // Node S114_Conv2d_184x3x3x1_Custom inq -4.37<(i8-0.00)*0.03414755<4.34 forced weightsq chan<(i8-0.00)*chan<chan outq -3.25<(i8-0.00)*0.02538453<3.22 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S114_Conv2d_184x3x3x1_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S111_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_9_block_block_1_block_1_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1462", 0),
            GNodeArg(GNA_OUT, "S114_Output", 0),
            GNodeArg(GNA_IN, "S114_Mul_scale", 0),
            GNodeArg(GNA_IN, "S114_Mul_shift", 0),
            GNodeArg(GNA_IN, "S114_Infos", 0)
        )
    );
    // Node S117_Conv2d_80x1x1x184 inq -3.25<(i8-0.00)*0.02538453<3.22 weightsq chan<(i8-0.00)*chan<chan outq -1.92<(i8-0.00)*0.01502119<1.91 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S117_Conv2d_80x1x1x184",
        Bindings(7,
            GNodeArg(GNA_IN, "S114_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_9_block_block_2_block_2_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1465", 0),
            GNodeArg(GNA_OUT, "S117_Output", 0),
            GNodeArg(GNA_IN, "S117_Mul_scale", 0),
            GNodeArg(GNA_IN, "S117_Mul_shift", 0),
            GNodeArg(GNA_IN, "S117_Infos", 0)
        )
    );
    // Node S118_MatAdd_16x20x80 in1q -2.63<(i8-0.00)*0.02053471<2.61 forced
    //   in2q -1.92<(i8-0.00)*0.01502119<1.91 forced
    //   outq -2.85<(i8-0.00)*0.02226518<2.83 forced scaled input 0 is node input 1
    AddNode("S118_MatAdd_16x20x80",
        Bindings(4,
            GNodeArg(GNA_IN, "S108_Output", 0),
            GNodeArg(GNA_IN, "S117_Output", 0),
            GNodeArg(GNA_OUT, "S118_Output", 0),
            GNodeArg(GNA_IN, "S118_Infos", 0)
        )
    );
    // Node S121_Conv2d_184x1x1x80_Custom inq -2.85<(i8-0.00)*0.02226518<2.83 forced weightsq chan<(i8-0.00)*chan<chan outq -3.63<(i8-0.00)*0.02836571<3.60 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S121_Conv2d_184x1x1x80_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S118_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_10_block_block_0_block_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1468", 0),
            GNodeArg(GNA_OUT, "S121_Output", 0),
            GNodeArg(GNA_IN, "S121_Mul_scale", 0),
            GNodeArg(GNA_IN, "S121_Mul_shift", 0),
            GNodeArg(GNA_IN, "S121_Infos", 0)
        )
    );
    // Node S124_Conv2d_184x3x3x1_Custom inq -3.63<(i8-0.00)*0.02836571<3.60 forced weightsq chan<(i8-0.00)*chan<chan outq -2.79<(i8-0.00)*0.02181650<2.77 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S124_Conv2d_184x3x3x1_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S121_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_10_block_block_1_block_1_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1471", 0),
            GNodeArg(GNA_OUT, "S124_Output", 0),
            GNodeArg(GNA_IN, "S124_Mul_scale", 0),
            GNodeArg(GNA_IN, "S124_Mul_shift", 0),
            GNodeArg(GNA_IN, "S124_Infos", 0)
        )
    );
    // Node S127_Conv2d_80x1x1x184 inq -2.79<(i8-0.00)*0.02181650<2.77 weightsq chan<(i8-0.00)*chan<chan outq -1.77<(i8-0.00)*0.01385752<1.76 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S127_Conv2d_80x1x1x184",
        Bindings(7,
            GNodeArg(GNA_IN, "S124_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_10_block_block_2_block_2_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1474", 0),
            GNodeArg(GNA_OUT, "S127_Output", 0),
            GNodeArg(GNA_IN, "S127_Mul_scale", 0),
            GNodeArg(GNA_IN, "S127_Mul_shift", 0),
            GNodeArg(GNA_IN, "S127_Infos", 0)
        )
    );
    // Node S128_MatAdd_16x20x80 in1q -2.85<(i8-0.00)*0.02226518<2.83 forced
    //   in2q -1.77<(i8-0.00)*0.01385752<1.76 forced
    //   outq -2.86<(i8-0.00)*0.02231201<2.83 forced scaled input 0 is node input 1
    AddNode("S128_MatAdd_16x20x80",
        Bindings(4,
            GNodeArg(GNA_IN, "S118_Output", 0),
            GNodeArg(GNA_IN, "S127_Output", 0),
            GNodeArg(GNA_OUT, "S128_Output", 0),
            GNodeArg(GNA_IN, "S128_Infos", 0)
        )
    );
    // Node S131_Conv2d_480x1x1x80_Custom inq -2.86<(i8-0.00)*0.02231201<2.83 forced weightsq chan<(i8-0.00)*chan<chan outq -3.96<(i8-0.00)*0.03094595<3.93 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S131_Conv2d_480x1x1x80_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S128_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_11_block_block_0_block_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1477", 0),
            GNodeArg(GNA_OUT, "S131_Output", 0),
            GNodeArg(GNA_IN, "S131_Mul_scale", 0),
            GNodeArg(GNA_IN, "S131_Mul_shift", 0),
            GNodeArg(GNA_IN, "S131_Infos", 0)
        )
    );
    // Node S134_Conv2d_480x3x3x1_Custom inq -3.96<(i8-0.00)*0.03094595<3.93 forced weightsq chan<(i8-0.00)*chan<chan outq -3.83<(i8-0.00)*0.02989934<3.80 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S134_Conv2d_480x3x3x1_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S131_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_11_block_block_1_block_1_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1480", 0),
            GNodeArg(GNA_OUT, "S134_Output", 0),
            GNodeArg(GNA_IN, "S134_Mul_scale", 0),
            GNodeArg(GNA_IN, "S134_Mul_shift", 0),
            GNodeArg(GNA_IN, "S134_Infos", 0)
        )
    );
    // Node _backbone_features_0_features_0_11_block_block_2_avgpool_GlobalAveragePool_trans inq -3.83<(i8-0.00)*0.02989934<3.80 outq -3.83<(i8-0.00)*0.02989934<3.80
    AddNode("S135_Op__backbone_features_0_features_0_11_block_block_2_avgpool_GlobalAveragePool_trans",
        Bindings(2,
            GNodeArg(GNA_IN, "S134_Output", 0),
            GNodeArg(GNA_OUT, "S135_Output", 0)
        )
    );
    // Node _backbone_features_0_features_0_11_block_block_2_avgpool_GlobalAveragePool inq -3.83<(i8-0.00)*0.02989934<3.80 outq -0.97<(i8-0.00)*0.00754014<0.96
    AddNode("S136_Op__backbone_features_0_features_0_11_block_block_2_avgpool_GlobalAveragePool",
        Bindings(3,
            GNodeArg(GNA_IN, "S135_Output", 0),
            GNodeArg(GNA_OUT, "S136_Output", 0),
            GNodeArg(GNA_IN, "S136_Infos", 0)
        )
    );
    // Node S140_Conv2d_120x1x1x480_Relu inq -0.97<(i8-0.00)*0.00754014<0.96 weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.00958847<2.45 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S140_Conv2d_120x1x1x480_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S136_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_11_block_block_2_fc1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_backbone_features_0_11_block_2_fc1_bias", 0),
            GNodeArg(GNA_OUT, "S140_Output", 0),
            GNodeArg(GNA_IN, "S140_Mul_scale", 0),
            GNodeArg(GNA_IN, "S140_Mul_shift", 0),
            GNodeArg(GNA_IN, "S140_Infos", 0)
        )
    );
    // Node S143_Conv2d_480x1x1x120_Hsigmoid inq 0.00<(i8--128.00)*0.00958847<2.45 weightsq chan<(i8-0.00)*chan<chan outq -1.01<(i8-0.00)*0.00787402<1.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S143_Conv2d_480x1x1x120_Hsigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S140_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_11_block_block_2_fc2_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_backbone_features_0_11_block_2_fc2_bias", 0),
            GNodeArg(GNA_OUT, "S143_Output", 0),
            GNodeArg(GNA_IN, "S143_Mul_scale", 0),
            GNodeArg(GNA_IN, "S143_Mul_shift", 0),
            GNodeArg(GNA_IN, "S143_Infos", 0)
        )
    );
    // Node expr_11 in_qs [-1.01<(i8-0.00)*0.00787402<1.00,-3.83<(i8-0.00)*0.02989934<3.80] out_qs [-0.49<(i8-0.00)*0.00379978<0.48]
    AddNode("S144_Op_expr_11",
        Bindings(3,
            GNodeArg(GNA_IN, "S143_Output", 0),
            GNodeArg(GNA_IN, "S134_Output", 0),
            GNodeArg(GNA_OUT, "S144_Output", 0)
        )
    );
    // Node S147_Conv2d_112x1x1x480 inq -0.49<(i8-0.00)*0.00379978<0.48 weightsq chan<(i8-0.00)*chan<chan outq -0.84<(i8-0.00)*0.00659090<0.84 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S147_Conv2d_112x1x1x480",
        Bindings(7,
            GNodeArg(GNA_IN, "S144_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_11_block_block_3_block_3_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1483", 0),
            GNodeArg(GNA_OUT, "S147_Output", 0),
            GNodeArg(GNA_IN, "S147_Mul_scale", 0),
            GNodeArg(GNA_IN, "S147_Mul_shift", 0),
            GNodeArg(GNA_IN, "S147_Infos", 0)
        )
    );
    // Node _backbone_features_0_features_0_12_block_block_0_block_0_0_Conv_fusion_qin0 inq -0.84<(i8-0.00)*0.00659090<0.84 forced outq -0.78<(i8--5.00)*0.00634119<0.84
    AddNode("S148_Op__backbone_features_0_features_0_12_block_block_0_block_0_0_Conv_fusion_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S147_Output", 0),
            GNodeArg(GNA_OUT, "S148_Output", 0),
            GNodeArg(GNA_IN, "S148_Infos", 0)
        )
    );
    // Node S151_Conv2d_672x1x1x112_Custom inq -0.84<(i8-0.00)*0.00659090<0.84 forced weightsq chan<(i8-0.00)*chan<chan outq -2.23<(i8-0.00)*0.01741716<2.21 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S151_Conv2d_672x1x1x112_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S148_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_12_block_block_0_block_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1486", 0),
            GNodeArg(GNA_OUT, "S151_Output", 0),
            GNodeArg(GNA_IN, "S151_Mul_scale", 0),
            GNodeArg(GNA_IN, "S151_Mul_shift", 0),
            GNodeArg(GNA_IN, "S151_Infos", 0)
        )
    );
    // Node S154_Conv2d_672x3x3x1_Custom inq -2.23<(i8-0.00)*0.01741716<2.21 forced weightsq chan<(i8-0.00)*chan<chan outq -3.39<(i8-0.00)*0.02651143<3.37 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S154_Conv2d_672x3x3x1_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S151_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_12_block_block_1_block_1_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1489", 0),
            GNodeArg(GNA_OUT, "S154_Output", 0),
            GNodeArg(GNA_IN, "S154_Mul_scale", 0),
            GNodeArg(GNA_IN, "S154_Mul_shift", 0),
            GNodeArg(GNA_IN, "S154_Infos", 0)
        )
    );
    // Node _backbone_features_0_features_0_12_block_block_2_avgpool_GlobalAveragePool_trans inq -3.39<(i8-0.00)*0.02651143<3.37 outq -3.39<(i8-0.00)*0.02651143<3.37
    AddNode("S155_Op__backbone_features_0_features_0_12_block_block_2_avgpool_GlobalAveragePool_trans",
        Bindings(2,
            GNodeArg(GNA_IN, "S154_Output", 0),
            GNodeArg(GNA_OUT, "S155_Output", 0)
        )
    );
    // Node _backbone_features_0_features_0_12_block_block_2_avgpool_GlobalAveragePool inq -3.39<(i8-0.00)*0.02651143<3.37 outq -0.91<(i8-0.00)*0.00710094<0.90
    AddNode("S156_Op__backbone_features_0_features_0_12_block_block_2_avgpool_GlobalAveragePool",
        Bindings(3,
            GNodeArg(GNA_IN, "S155_Output", 0),
            GNodeArg(GNA_OUT, "S156_Output", 0),
            GNodeArg(GNA_IN, "S156_Infos", 0)
        )
    );
    // Node S160_Conv2d_168x1x1x672_Relu inq -0.91<(i8-0.00)*0.00710094<0.90 weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.00879244<2.24 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S160_Conv2d_168x1x1x672_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S156_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_12_block_block_2_fc1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_backbone_features_0_12_block_2_fc1_bias", 0),
            GNodeArg(GNA_OUT, "S160_Output", 0),
            GNodeArg(GNA_IN, "S160_Mul_scale", 0),
            GNodeArg(GNA_IN, "S160_Mul_shift", 0),
            GNodeArg(GNA_IN, "S160_Infos", 0)
        )
    );
    // Node S163_Conv2d_672x1x1x168_Hsigmoid inq 0.00<(i8--128.00)*0.00879244<2.24 weightsq chan<(i8-0.00)*chan<chan outq -1.01<(i8-0.00)*0.00787402<1.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S163_Conv2d_672x1x1x168_Hsigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S160_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_12_block_block_2_fc2_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_backbone_features_0_12_block_2_fc2_bias", 0),
            GNodeArg(GNA_OUT, "S163_Output", 0),
            GNodeArg(GNA_IN, "S163_Mul_scale", 0),
            GNodeArg(GNA_IN, "S163_Mul_shift", 0),
            GNodeArg(GNA_IN, "S163_Infos", 0)
        )
    );
    // Node expr_14 in_qs [-1.01<(i8-0.00)*0.00787402<1.00,-3.39<(i8-0.00)*0.02651143<3.37] out_qs [-0.35<(i8-0.00)*0.00270446<0.34]
    AddNode("S164_Op_expr_14",
        Bindings(3,
            GNodeArg(GNA_IN, "S163_Output", 0),
            GNodeArg(GNA_IN, "S154_Output", 0),
            GNodeArg(GNA_OUT, "S164_Output", 0)
        )
    );
    // Node S167_Conv2d_112x1x1x672 inq -0.35<(i8-0.00)*0.00270446<0.34 weightsq chan<(i8-0.00)*chan<chan outq -1.29<(i8-0.00)*0.01006242<1.28 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S167_Conv2d_112x1x1x672",
        Bindings(7,
            GNodeArg(GNA_IN, "S164_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_12_block_block_3_block_3_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1492", 0),
            GNodeArg(GNA_OUT, "S167_Output", 0),
            GNodeArg(GNA_IN, "S167_Mul_scale", 0),
            GNodeArg(GNA_IN, "S167_Mul_shift", 0),
            GNodeArg(GNA_IN, "S167_Infos", 0)
        )
    );
    // Node S168_MatAdd_16x20x112 in1q -1.29<(i8-0.00)*0.01006242<1.28 forced
    //   in2q -0.84<(i8-0.00)*0.00659090<0.84 forced
    //   outq -1.38<(i8-0.00)*0.01075022<1.37 forced scaled input 0 is node input 0
    AddNode("S168_MatAdd_16x20x112",
        Bindings(4,
            GNodeArg(GNA_IN, "S167_Output", 0),
            GNodeArg(GNA_IN, "S147_Output", 0),
            GNodeArg(GNA_OUT, "S168_Output", 0),
            GNodeArg(GNA_IN, "S168_Infos", 0)
        )
    );
    // Node S171_Conv2d_672x1x1x112_Custom inq -1.38<(i8-0.00)*0.01075022<1.37 forced weightsq chan<(i8-0.00)*chan<chan outq -2.56<(i8-0.00)*0.01998149<2.54 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S171_Conv2d_672x1x1x112_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S168_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_13_features_0_13_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1495", 0),
            GNodeArg(GNA_OUT, "S171_Output", 0),
            GNodeArg(GNA_IN, "S171_Mul_scale", 0),
            GNodeArg(GNA_IN, "S171_Mul_shift", 0),
            GNodeArg(GNA_IN, "S171_Infos", 0)
        )
    );
    // Node S174_Conv2d_672x5x5x1_Custom inq -2.56<(i8-0.00)*0.01998149<2.54 forced weightsq chan<(i8-0.00)*chan<chan outq -2.67<(i8-0.00)*0.02082715<2.65 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S174_Conv2d_672x5x5x1_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S171_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_1_features_1_0_features_1_0_1_features_1_0_1_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1498", 0),
            GNodeArg(GNA_OUT, "S174_Output", 0),
            GNodeArg(GNA_IN, "S174_Mul_scale", 0),
            GNodeArg(GNA_IN, "S174_Mul_shift", 0),
            GNodeArg(GNA_IN, "S174_Infos", 0)
        )
    );
    // Node _backbone_features_1_features_1_0_features_1_0_2_avgpool_GlobalAveragePool_trans inq -2.67<(i8-0.00)*0.02082715<2.65 outq -2.67<(i8-0.00)*0.02082715<2.65
    AddNode("S175_Op__backbone_features_1_features_1_0_features_1_0_2_avgpool_GlobalAveragePool_trans",
        Bindings(2,
            GNodeArg(GNA_IN, "S174_Output", 0),
            GNodeArg(GNA_OUT, "S175_Output", 0)
        )
    );
    // Node _backbone_features_1_features_1_0_features_1_0_2_avgpool_GlobalAveragePool inq -2.67<(i8-0.00)*0.02082715<2.65 outq -0.89<(i8-0.00)*0.00694913<0.88
    AddNode("S176_Op__backbone_features_1_features_1_0_features_1_0_2_avgpool_GlobalAveragePool",
        Bindings(3,
            GNodeArg(GNA_IN, "S175_Output", 0),
            GNodeArg(GNA_OUT, "S176_Output", 0),
            GNodeArg(GNA_IN, "S176_Infos", 0)
        )
    );
    // Node S180_Conv2d_168x1x1x672_Relu inq -0.89<(i8-0.00)*0.00694913<0.88 weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.00468616<1.19 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S180_Conv2d_168x1x1x672_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S176_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_1_features_1_0_features_1_0_2_fc1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_backbone_features_1_0_2_fc1_bias", 0),
            GNodeArg(GNA_OUT, "S180_Output", 0),
            GNodeArg(GNA_IN, "S180_Mul_scale", 0),
            GNodeArg(GNA_IN, "S180_Mul_shift", 0),
            GNodeArg(GNA_IN, "S180_Infos", 0)
        )
    );
    // Node S183_Conv2d_672x1x1x168_Hsigmoid inq 0.00<(i8--128.00)*0.00468616<1.19 weightsq chan<(i8-0.00)*chan<chan outq -1.01<(i8-0.00)*0.00787402<1.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S183_Conv2d_672x1x1x168_Hsigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S180_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_1_features_1_0_features_1_0_2_fc2_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_backbone_features_1_0_2_fc2_bias", 0),
            GNodeArg(GNA_OUT, "S183_Output", 0),
            GNodeArg(GNA_IN, "S183_Mul_scale", 0),
            GNodeArg(GNA_IN, "S183_Mul_shift", 0),
            GNodeArg(GNA_IN, "S183_Infos", 0)
        )
    );
    // Node expr_26 in_qs [-1.01<(i8-0.00)*0.00787402<1.00,-2.67<(i8-0.00)*0.02082715<2.65] out_qs [-0.49<(i8-0.00)*0.00384131<0.49]
    AddNode("S184_Op_expr_26",
        Bindings(3,
            GNodeArg(GNA_IN, "S183_Output", 0),
            GNodeArg(GNA_IN, "S174_Output", 0),
            GNodeArg(GNA_OUT, "S184_Output", 0)
        )
    );
    // Node S187_Conv2d_80x1x1x672 inq -0.49<(i8-0.00)*0.00384131<0.49 weightsq chan<(i8-0.00)*chan<chan outq -0.69<(i8-0.00)*0.00539086<0.68 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S187_Conv2d_80x1x1x672",
        Bindings(7,
            GNodeArg(GNA_IN, "S184_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_1_features_1_0_features_1_0_3_features_1_0_3_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1501", 0),
            GNodeArg(GNA_OUT, "S187_Output", 0),
            GNodeArg(GNA_IN, "S187_Mul_scale", 0),
            GNodeArg(GNA_IN, "S187_Mul_shift", 0),
            GNodeArg(GNA_IN, "S187_Infos", 0)
        )
    );
    // Node _backbone_features_1_features_1_1_block_block_0_block_0_0_Conv_fusion_qin0 inq -0.69<(i8-0.00)*0.00539086<0.68 forced outq -0.69<(i8-0.00)*0.00537912<0.68
    AddNode("S188_Op__backbone_features_1_features_1_1_block_block_0_block_0_0_Conv_fusion_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S187_Output", 0),
            GNodeArg(GNA_OUT, "S188_Output", 0),
            GNodeArg(GNA_IN, "S188_Infos", 0)
        )
    );
    // Node S191_Conv2d_480x1x1x80_Custom inq -0.69<(i8-0.00)*0.00539086<0.68 forced weightsq chan<(i8-0.00)*chan<chan outq -0.90<(i8-0.00)*0.00701237<0.89 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S191_Conv2d_480x1x1x80_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S188_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_1_features_1_1_block_block_0_block_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1504", 0),
            GNodeArg(GNA_OUT, "S191_Output", 0),
            GNodeArg(GNA_IN, "S191_Mul_scale", 0),
            GNodeArg(GNA_IN, "S191_Mul_shift", 0),
            GNodeArg(GNA_IN, "S191_Infos", 0)
        )
    );
    // Node S194_Conv2d_480x5x5x1_Custom inq -0.90<(i8-0.00)*0.00701237<0.89 forced weightsq chan<(i8-0.00)*chan<chan outq -0.80<(i8-0.00)*0.00628620<0.80 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S194_Conv2d_480x5x5x1_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S191_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_1_features_1_1_block_block_1_block_1_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1507", 0),
            GNodeArg(GNA_OUT, "S194_Output", 0),
            GNodeArg(GNA_IN, "S194_Mul_scale", 0),
            GNodeArg(GNA_IN, "S194_Mul_shift", 0),
            GNodeArg(GNA_IN, "S194_Infos", 0)
        )
    );
    // Node _backbone_features_1_features_1_1_block_block_2_avgpool_GlobalAveragePool_trans inq -0.80<(i8-0.00)*0.00628620<0.80 outq -0.80<(i8-0.00)*0.00628620<0.80
    AddNode("S195_Op__backbone_features_1_features_1_1_block_block_2_avgpool_GlobalAveragePool_trans",
        Bindings(2,
            GNodeArg(GNA_IN, "S194_Output", 0),
            GNodeArg(GNA_OUT, "S195_Output", 0)
        )
    );
    // Node _backbone_features_1_features_1_1_block_block_2_avgpool_GlobalAveragePool inq -0.80<(i8-0.00)*0.00628620<0.80 outq -0.36<(i8-0.00)*0.00281091<0.36
    AddNode("S196_Op__backbone_features_1_features_1_1_block_block_2_avgpool_GlobalAveragePool",
        Bindings(3,
            GNodeArg(GNA_IN, "S195_Output", 0),
            GNodeArg(GNA_OUT, "S196_Output", 0),
            GNodeArg(GNA_IN, "S196_Infos", 0)
        )
    );
    // Node S200_Conv2d_120x1x1x480_Relu inq -0.36<(i8-0.00)*0.00281091<0.36 weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.00201539<0.51 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S200_Conv2d_120x1x1x480_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S196_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_1_features_1_1_block_block_2_fc1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_backbone_features_1_1_block_2_fc1_bias", 0),
            GNodeArg(GNA_OUT, "S200_Output", 0),
            GNodeArg(GNA_IN, "S200_Mul_scale", 0),
            GNodeArg(GNA_IN, "S200_Mul_shift", 0),
            GNodeArg(GNA_IN, "S200_Infos", 0)
        )
    );
    // Node S203_Conv2d_480x1x1x120_Hsigmoid inq 0.00<(i8--128.00)*0.00201539<0.51 weightsq chan<(i8-0.00)*chan<chan outq -1.01<(i8-0.00)*0.00787402<1.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S203_Conv2d_480x1x1x120_Hsigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S200_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_1_features_1_1_block_block_2_fc2_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_backbone_features_1_1_block_2_fc2_bias", 0),
            GNodeArg(GNA_OUT, "S203_Output", 0),
            GNodeArg(GNA_IN, "S203_Mul_scale", 0),
            GNodeArg(GNA_IN, "S203_Mul_shift", 0),
            GNodeArg(GNA_IN, "S203_Infos", 0)
        )
    );
    // Node expr_29 in_qs [-1.01<(i8-0.00)*0.00787402<1.00,-0.80<(i8-0.00)*0.00628620<0.80] out_qs [-0.43<(i8-0.00)*0.00337987<0.43]
    AddNode("S204_Op_expr_29",
        Bindings(3,
            GNodeArg(GNA_IN, "S203_Output", 0),
            GNodeArg(GNA_IN, "S194_Output", 0),
            GNodeArg(GNA_OUT, "S204_Output", 0)
        )
    );
    // Node S207_Conv2d_80x1x1x480 inq -0.43<(i8-0.00)*0.00337987<0.43 weightsq chan<(i8-0.00)*chan<chan outq -0.72<(i8-0.00)*0.00566036<0.72 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S207_Conv2d_80x1x1x480",
        Bindings(7,
            GNodeArg(GNA_IN, "S204_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_1_features_1_1_block_block_3_block_3_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1510", 0),
            GNodeArg(GNA_OUT, "S207_Output", 0),
            GNodeArg(GNA_IN, "S207_Mul_scale", 0),
            GNodeArg(GNA_IN, "S207_Mul_shift", 0),
            GNodeArg(GNA_IN, "S207_Infos", 0)
        )
    );
    // Node S208_MatAdd_8x10x80 in1q -0.72<(i8-0.00)*0.00566036<0.72 forced
    //   in2q -0.69<(i8-0.00)*0.00539086<0.68 forced
    //   outq -0.87<(i8-0.00)*0.00678719<0.86 forced scaled input 0 is node input 0
    AddNode("S208_MatAdd_8x10x80",
        Bindings(4,
            GNodeArg(GNA_IN, "S207_Output", 0),
            GNodeArg(GNA_IN, "S187_Output", 0),
            GNodeArg(GNA_OUT, "S208_Output", 0),
            GNodeArg(GNA_IN, "S208_Infos", 0)
        )
    );
    // Node S211_Conv2d_480x1x1x80_Custom inq -0.87<(i8-0.00)*0.00678719<0.86 forced weightsq chan<(i8-0.00)*chan<chan outq -0.91<(i8-0.00)*0.00707043<0.90 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S211_Conv2d_480x1x1x80_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S208_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_1_features_1_2_block_block_0_block_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1513", 0),
            GNodeArg(GNA_OUT, "S211_Output", 0),
            GNodeArg(GNA_IN, "S211_Mul_scale", 0),
            GNodeArg(GNA_IN, "S211_Mul_shift", 0),
            GNodeArg(GNA_IN, "S211_Infos", 0)
        )
    );
    // Node S214_Conv2d_480x5x5x1_Custom inq -0.91<(i8-0.00)*0.00707043<0.90 forced weightsq chan<(i8-0.00)*chan<chan outq -0.68<(i8-0.00)*0.00533327<0.68 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S214_Conv2d_480x5x5x1_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S211_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_1_features_1_2_block_block_1_block_1_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1516", 0),
            GNodeArg(GNA_OUT, "S214_Output", 0),
            GNodeArg(GNA_IN, "S214_Mul_scale", 0),
            GNodeArg(GNA_IN, "S214_Mul_shift", 0),
            GNodeArg(GNA_IN, "S214_Infos", 0)
        )
    );
    // Node _backbone_features_1_features_1_2_block_block_2_avgpool_GlobalAveragePool_trans inq -0.68<(i8-0.00)*0.00533327<0.68 outq -0.68<(i8-0.00)*0.00533327<0.68
    AddNode("S215_Op__backbone_features_1_features_1_2_block_block_2_avgpool_GlobalAveragePool_trans",
        Bindings(2,
            GNodeArg(GNA_IN, "S214_Output", 0),
            GNodeArg(GNA_OUT, "S215_Output", 0)
        )
    );
    // Node _backbone_features_1_features_1_2_block_block_2_avgpool_GlobalAveragePool inq -0.68<(i8-0.00)*0.00533327<0.68 outq -0.32<(i8-0.00)*0.00252972<0.32
    AddNode("S216_Op__backbone_features_1_features_1_2_block_block_2_avgpool_GlobalAveragePool",
        Bindings(3,
            GNodeArg(GNA_IN, "S215_Output", 0),
            GNodeArg(GNA_OUT, "S216_Output", 0),
            GNodeArg(GNA_IN, "S216_Infos", 0)
        )
    );
    // Node S220_Conv2d_120x1x1x480_Relu inq -0.32<(i8-0.00)*0.00252972<0.32 weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.00316410<0.81 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S220_Conv2d_120x1x1x480_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S216_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_1_features_1_2_block_block_2_fc1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_backbone_features_1_2_block_2_fc1_bias", 0),
            GNodeArg(GNA_OUT, "S220_Output", 0),
            GNodeArg(GNA_IN, "S220_Mul_scale", 0),
            GNodeArg(GNA_IN, "S220_Mul_shift", 0),
            GNodeArg(GNA_IN, "S220_Infos", 0)
        )
    );
    // Node S223_Conv2d_480x1x1x120_Hsigmoid inq 0.00<(i8--128.00)*0.00316410<0.81 weightsq chan<(i8-0.00)*chan<chan outq -1.01<(i8-0.00)*0.00787402<1.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S223_Conv2d_480x1x1x120_Hsigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S220_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_1_features_1_2_block_block_2_fc2_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_backbone_features_1_2_block_2_fc2_bias", 0),
            GNodeArg(GNA_OUT, "S223_Output", 0),
            GNodeArg(GNA_IN, "S223_Mul_scale", 0),
            GNodeArg(GNA_IN, "S223_Mul_shift", 0),
            GNodeArg(GNA_IN, "S223_Infos", 0)
        )
    );
    // Node expr_32 in_qs [-1.01<(i8-0.00)*0.00787402<1.00,-0.68<(i8-0.00)*0.00533327<0.68] out_qs [-0.34<(i8-0.00)*0.00269067<0.34]
    AddNode("S224_Op_expr_32",
        Bindings(3,
            GNodeArg(GNA_IN, "S223_Output", 0),
            GNodeArg(GNA_IN, "S214_Output", 0),
            GNodeArg(GNA_OUT, "S224_Output", 0)
        )
    );
    // Node S227_Conv2d_80x1x1x480 inq -0.34<(i8-0.00)*0.00269067<0.34 weightsq chan<(i8-0.00)*chan<chan outq -0.88<(i8-0.00)*0.00687430<0.87 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S227_Conv2d_80x1x1x480",
        Bindings(7,
            GNodeArg(GNA_IN, "S224_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_1_features_1_2_block_block_3_block_3_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1519", 0),
            GNodeArg(GNA_OUT, "S227_Output", 0),
            GNodeArg(GNA_IN, "S227_Mul_scale", 0),
            GNodeArg(GNA_IN, "S227_Mul_shift", 0),
            GNodeArg(GNA_IN, "S227_Infos", 0)
        )
    );
    // Node S228_MatAdd_8x10x80 in1q -0.88<(i8-0.00)*0.00687430<0.87 forced
    //   in2q -0.87<(i8-0.00)*0.00678719<0.86 forced
    //   outq -1.20<(i8-0.00)*0.00940348<1.19 forced scaled input 0 is node input 0
    AddNode("S228_MatAdd_8x10x80",
        Bindings(4,
            GNodeArg(GNA_IN, "S227_Output", 0),
            GNodeArg(GNA_IN, "S208_Output", 0),
            GNodeArg(GNA_OUT, "S228_Output", 0),
            GNodeArg(GNA_IN, "S228_Infos", 0)
        )
    );
    // Node S231_Conv2d_480x1x1x80_Custom inq -1.20<(i8-0.00)*0.00940348<1.19 forced weightsq chan<(i8-0.00)*chan<chan outq -1.37<(i8-0.00)*0.01072699<1.36 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S231_Conv2d_480x1x1x80_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S228_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_1_features_1_3_features_1_3_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1522", 0),
            GNodeArg(GNA_OUT, "S231_Output", 0),
            GNodeArg(GNA_IN, "S231_Mul_scale", 0),
            GNodeArg(GNA_IN, "S231_Mul_shift", 0),
            GNodeArg(GNA_IN, "S231_Infos", 0)
        )
    );
    // Node S234_Conv2d_256x1x1x480_Relu6 inq -1.37<(i8-0.00)*0.01072699<1.36 forced weightsq chan<(i8-0.00)*chan<chan outq -0.53<(i8-0.00)*0.00412179<0.52 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S234_Conv2d_256x1x1x480_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S231_Output", 0),
            GNodeArg(GNA_IN, "_backbone_extra_0_extra_0_0_extra_0_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1525", 0),
            GNodeArg(GNA_OUT, "S234_Output", 0),
            GNodeArg(GNA_IN, "S234_Mul_scale", 0),
            GNodeArg(GNA_IN, "S234_Mul_shift", 0),
            GNodeArg(GNA_IN, "S234_Infos", 0)
        )
    );
    // Node S237_Conv2d_256x3x3x1_Relu6 inq -0.53<(i8-0.00)*0.00412179<0.52 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S237_Conv2d_256x3x3x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S234_Output", 0),
            GNodeArg(GNA_IN, "_backbone_extra_0_extra_0_1_extra_0_1_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1528", 0),
            GNodeArg(GNA_OUT, "S237_Output", 0),
            GNodeArg(GNA_IN, "S237_Mul_scale", 0),
            GNodeArg(GNA_IN, "S237_Mul_shift", 0),
            GNodeArg(GNA_IN, "S237_Infos", 0)
        )
    );
    // Node S240_Conv2d_512x1x1x256_Relu6 inq 0.00<(i8--128.00)*0.02352941<6.00 weightsq chan<(i8-0.00)*chan<chan outq -0.88<(i8-0.00)*0.00689553<0.88 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S240_Conv2d_512x1x1x256_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S237_Output", 0),
            GNodeArg(GNA_IN, "_backbone_extra_0_extra_0_2_extra_0_2_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1531", 0),
            GNodeArg(GNA_OUT, "S240_Output", 0),
            GNodeArg(GNA_IN, "S240_Mul_scale", 0),
            GNodeArg(GNA_IN, "S240_Mul_shift", 0),
            GNodeArg(GNA_IN, "S240_Infos", 0)
        )
    );
    // Node S243_Conv2d_128x1x1x512_Relu6 inq -0.88<(i8-0.00)*0.00689553<0.88 forced weightsq chan<(i8-0.00)*chan<chan outq -0.47<(i8-0.00)*0.00367925<0.47 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S243_Conv2d_128x1x1x512_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S240_Output", 0),
            GNodeArg(GNA_IN, "_backbone_extra_1_extra_1_0_extra_1_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1534", 0),
            GNodeArg(GNA_OUT, "S243_Output", 0),
            GNodeArg(GNA_IN, "S243_Mul_scale", 0),
            GNodeArg(GNA_IN, "S243_Mul_shift", 0),
            GNodeArg(GNA_IN, "S243_Infos", 0)
        )
    );
    // Node S246_Conv2d_128x3x3x1_Relu6 inq -0.47<(i8-0.00)*0.00367925<0.47 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S246_Conv2d_128x3x3x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S243_Output", 0),
            GNodeArg(GNA_IN, "_backbone_extra_1_extra_1_1_extra_1_1_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1537", 0),
            GNodeArg(GNA_OUT, "S246_Output", 0),
            GNodeArg(GNA_IN, "S246_Mul_scale", 0),
            GNodeArg(GNA_IN, "S246_Mul_shift", 0),
            GNodeArg(GNA_IN, "S246_Infos", 0)
        )
    );
    // Node S249_Conv2d_256x1x1x128_Relu6 inq 0.00<(i8--128.00)*0.02352941<6.00 weightsq chan<(i8-0.00)*chan<chan outq -1.05<(i8-0.00)*0.00818595<1.04 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S249_Conv2d_256x1x1x128_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S246_Output", 0),
            GNodeArg(GNA_IN, "_backbone_extra_1_extra_1_2_extra_1_2_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1540", 0),
            GNodeArg(GNA_OUT, "S249_Output", 0),
            GNodeArg(GNA_IN, "S249_Mul_scale", 0),
            GNodeArg(GNA_IN, "S249_Mul_shift", 0),
            GNodeArg(GNA_IN, "S249_Infos", 0)
        )
    );
    // Node S252_Conv2d_128x1x1x256_Relu6 inq -1.05<(i8-0.00)*0.00818595<1.04 forced weightsq chan<(i8-0.00)*chan<chan outq -0.55<(i8-0.00)*0.00427978<0.54 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S252_Conv2d_128x1x1x256_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S249_Output", 0),
            GNodeArg(GNA_IN, "_backbone_extra_2_extra_2_0_extra_2_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1543", 0),
            GNodeArg(GNA_OUT, "S252_Output", 0),
            GNodeArg(GNA_IN, "S252_Mul_scale", 0),
            GNodeArg(GNA_IN, "S252_Mul_shift", 0),
            GNodeArg(GNA_IN, "S252_Infos", 0)
        )
    );
    // Node S255_Conv2d_128x3x3x1_Relu6 inq -0.55<(i8-0.00)*0.00427978<0.54 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S255_Conv2d_128x3x3x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S252_Output", 0),
            GNodeArg(GNA_IN, "_backbone_extra_2_extra_2_1_extra_2_1_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1546", 0),
            GNodeArg(GNA_OUT, "S255_Output", 0),
            GNodeArg(GNA_IN, "S255_Mul_scale", 0),
            GNodeArg(GNA_IN, "S255_Mul_shift", 0),
            GNodeArg(GNA_IN, "S255_Infos", 0)
        )
    );
    // Node S258_Conv2d_256x1x1x128_Relu6 inq 0.00<(i8--128.00)*0.02352941<6.00 weightsq chan<(i8-0.00)*chan<chan outq -0.60<(i8-0.00)*0.00470736<0.60 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S258_Conv2d_256x1x1x128_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S255_Output", 0),
            GNodeArg(GNA_IN, "_backbone_extra_2_extra_2_2_extra_2_2_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1549", 0),
            GNodeArg(GNA_OUT, "S258_Output", 0),
            GNodeArg(GNA_IN, "S258_Mul_scale", 0),
            GNodeArg(GNA_IN, "S258_Mul_shift", 0),
            GNodeArg(GNA_IN, "S258_Infos", 0)
        )
    );
    // Node S261_Conv2d_64x1x1x256_Relu6 inq -0.60<(i8-0.00)*0.00470736<0.60 forced weightsq chan<(i8-0.00)*chan<chan outq -0.39<(i8-0.00)*0.00308153<0.39 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S261_Conv2d_64x1x1x256_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S258_Output", 0),
            GNodeArg(GNA_IN, "_backbone_extra_3_extra_3_0_extra_3_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1552", 0),
            GNodeArg(GNA_OUT, "S261_Output", 0),
            GNodeArg(GNA_IN, "S261_Mul_scale", 0),
            GNodeArg(GNA_IN, "S261_Mul_shift", 0),
            GNodeArg(GNA_IN, "S261_Infos", 0)
        )
    );
    // Node S264_Conv2d_64x3x3x1_Relu6 inq -0.39<(i8-0.00)*0.00308153<0.39 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S264_Conv2d_64x3x3x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S261_Output", 0),
            GNodeArg(GNA_IN, "_backbone_extra_3_extra_3_1_extra_3_1_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1555", 0),
            GNodeArg(GNA_OUT, "S264_Output", 0),
            GNodeArg(GNA_IN, "S264_Mul_scale", 0),
            GNodeArg(GNA_IN, "S264_Mul_shift", 0),
            GNodeArg(GNA_IN, "S264_Infos", 0)
        )
    );
    // Node S267_Conv2d_128x1x1x64_Relu6 inq 0.00<(i8--128.00)*0.02352941<6.00 weightsq chan<(i8-0.00)*chan<chan outq -1.12<(i8-0.00)*0.00874556<1.11 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S267_Conv2d_128x1x1x64_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S264_Output", 0),
            GNodeArg(GNA_IN, "_backbone_extra_3_extra_3_2_extra_3_2_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1558", 0),
            GNodeArg(GNA_OUT, "S267_Output", 0),
            GNodeArg(GNA_IN, "S267_Mul_scale", 0),
            GNodeArg(GNA_IN, "S267_Mul_shift", 0),
            GNodeArg(GNA_IN, "S267_Infos", 0)
        )
    );
    // Node S270_Conv2d_672x3x3x1_Relu6 inq -2.56<(i8-0.00)*0.01998149<2.54 forced weightsq chan<(i8-0.00)*chan<chan outq -6.05<(i8-0.00)*0.04724409<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S270_Conv2d_672x3x3x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S171_Output", 0),
            GNodeArg(GNA_IN, "_head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1561", 0),
            GNodeArg(GNA_OUT, "S270_Output", 0),
            GNodeArg(GNA_IN, "S270_Mul_scale", 0),
            GNodeArg(GNA_IN, "S270_Mul_shift", 0),
            GNodeArg(GNA_IN, "S270_Infos", 0)
        )
    );
    // Node S273_Conv2d_24x1x1x672_Custom inq -6.05<(i8-0.00)*0.04724409<6.00 weightsq chan<(i8-0.00)*chan<chan outq -0.00<(i8-0.00)*0.00002717<0.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S273_Conv2d_24x1x1x672_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S270_Output", 0),
            GNodeArg(GNA_IN, "_head_regression_head_module_list_0_module_list_0_1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_head_regression_head_module_list_0_1_bias", 0),
            GNodeArg(GNA_OUT, "S273_Output", 0),
            GNodeArg(GNA_IN, "S273_Mul_scale", 0),
            GNodeArg(GNA_IN, "S273_Mul_shift", 0),
            GNodeArg(GNA_IN, "S273_Infos", 0)
        )
    );
    // Node S277_Conv2d_480x3x3x1_Relu6 inq -1.37<(i8-0.00)*0.01072699<1.36 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S277_Conv2d_480x3x3x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S231_Output", 0),
            GNodeArg(GNA_IN, "_head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1564", 0),
            GNodeArg(GNA_OUT, "S277_Output", 0),
            GNodeArg(GNA_IN, "S277_Mul_scale", 0),
            GNodeArg(GNA_IN, "S277_Mul_shift", 0),
            GNodeArg(GNA_IN, "S277_Infos", 0)
        )
    );
    // Node S281_Conv2d_512x3x3x1_Relu6 inq -0.88<(i8-0.00)*0.00689553<0.88 forced weightsq chan<(i8-0.00)*chan<chan outq -6.05<(i8-0.00)*0.04724409<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S281_Conv2d_512x3x3x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S240_Output", 0),
            GNodeArg(GNA_IN, "_head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1567", 0),
            GNodeArg(GNA_OUT, "S281_Output", 0),
            GNodeArg(GNA_IN, "S281_Mul_scale", 0),
            GNodeArg(GNA_IN, "S281_Mul_shift", 0),
            GNodeArg(GNA_IN, "S281_Infos", 0)
        )
    );
    // Node S285_Conv2d_256x3x3x1_Relu6 inq -1.05<(i8-0.00)*0.00818595<1.04 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S285_Conv2d_256x3x3x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S249_Output", 0),
            GNodeArg(GNA_IN, "_head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1570", 0),
            GNodeArg(GNA_OUT, "S285_Output", 0),
            GNodeArg(GNA_IN, "S285_Mul_scale", 0),
            GNodeArg(GNA_IN, "S285_Mul_shift", 0),
            GNodeArg(GNA_IN, "S285_Infos", 0)
        )
    );
    // Node S291_Conv2d_128x3x3x1_Relu6 inq -1.12<(i8-0.00)*0.00874556<1.11 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S291_Conv2d_128x3x3x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S267_Output", 0),
            GNodeArg(GNA_IN, "_head_regression_head_module_list_5_module_list_5_0_module_list_5_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1576", 0),
            GNodeArg(GNA_OUT, "S291_Output", 0),
            GNodeArg(GNA_IN, "S291_Mul_scale", 0),
            GNodeArg(GNA_IN, "S291_Mul_shift", 0),
            GNodeArg(GNA_IN, "S291_Infos", 0)
        )
    );
    // Node S311_Conv2d_672x3x3x1_Relu6 inq -2.56<(i8-0.00)*0.01998149<2.54 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S311_Conv2d_672x3x3x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S171_Output", 0),
            GNodeArg(GNA_IN, "_head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1561_1", 0),
            GNodeArg(GNA_OUT, "S311_Output", 0),
            GNodeArg(GNA_IN, "S311_Mul_scale", 0),
            GNodeArg(GNA_IN, "S311_Mul_shift", 0),
            GNodeArg(GNA_IN, "S311_Infos", 0)
        )
    );
    // Node S312_Conv2d_186x1x1x672 inq 0.00<(i8--128.00)*0.02352941<6.00 weightsq chan<(i8-0.00)*chan<chan outq -0.06<(i8-0.00)*0.00048828<0.06 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S312_Conv2d_186x1x1x672",
        Bindings(7,
            GNodeArg(GNA_IN, "S311_Output", 0),
            GNodeArg(GNA_IN, "_head_classification_head_module_list_0_module_list_0_1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_head_classification_head_module_list_0_1_bias", 0),
            GNodeArg(GNA_OUT, "S312_Output", 0),
            GNodeArg(GNA_IN, "S312_Mul_scale", 0),
            GNodeArg(GNA_IN, "S312_Mul_shift", 0),
            GNodeArg(GNA_IN, "S312_Infos", 0)
        )
    );
    // Node S315_Conv2d_24x1x1x480_Custom inq 0.00<(i8--128.00)*0.02352941<6.00 weightsq chan<(i8-0.00)*chan<chan outq -0.00<(i8-0.00)*0.00002717<0.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S315_Conv2d_24x1x1x480_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S277_Output", 0),
            GNodeArg(GNA_IN, "_head_regression_head_module_list_1_module_list_1_1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_head_regression_head_module_list_0_1_bias_1", 0),
            GNodeArg(GNA_OUT, "S315_Output", 0),
            GNodeArg(GNA_IN, "S315_Mul_scale", 0),
            GNodeArg(GNA_IN, "S315_Mul_shift", 0),
            GNodeArg(GNA_IN, "S315_Infos", 0)
        )
    );
    // Node S318_Conv2d_24x1x1x512_Custom inq -6.05<(i8-0.00)*0.04724409<6.00 weightsq chan<(i8-0.00)*chan<chan outq -0.00<(i8-0.00)*0.00002717<0.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S318_Conv2d_24x1x1x512_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S281_Output", 0),
            GNodeArg(GNA_IN, "_head_regression_head_module_list_2_module_list_2_1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_head_regression_head_module_list_0_1_bias_2", 0),
            GNodeArg(GNA_OUT, "S318_Output", 0),
            GNodeArg(GNA_IN, "S318_Mul_scale", 0),
            GNodeArg(GNA_IN, "S318_Mul_shift", 0),
            GNodeArg(GNA_IN, "S318_Infos", 0)
        )
    );
    // Node S321_Conv2d_24x1x1x256_Custom inq 0.00<(i8--128.00)*0.02352941<6.00 weightsq chan<(i8-0.00)*chan<chan outq -0.00<(i8-0.00)*0.00002717<0.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S321_Conv2d_24x1x1x256_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S285_Output", 0),
            GNodeArg(GNA_IN, "_head_regression_head_module_list_3_module_list_3_1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_head_regression_head_module_list_0_1_bias_3", 0),
            GNodeArg(GNA_OUT, "S321_Output", 0),
            GNodeArg(GNA_IN, "S321_Mul_scale", 0),
            GNodeArg(GNA_IN, "S321_Mul_shift", 0),
            GNodeArg(GNA_IN, "S321_Infos", 0)
        )
    );
    // Node S325_Conv2d_24x1x1x128_Custom inq 0.00<(i8--128.00)*0.02352941<6.00 weightsq chan<(i8-0.00)*chan<chan outq -0.00<(i8-0.00)*0.00002717<0.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S325_Conv2d_24x1x1x128_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S291_Output", 0),
            GNodeArg(GNA_IN, "_head_regression_head_module_list_5_module_list_5_1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_head_regression_head_module_list_0_1_bias_5", 0),
            GNodeArg(GNA_OUT, "S325_Output", 0),
            GNodeArg(GNA_IN, "S325_Mul_scale", 0),
            GNodeArg(GNA_IN, "S325_Mul_shift", 0),
            GNodeArg(GNA_IN, "S325_Infos", 0)
        )
    );
    // Node S328_Conv2d_480x3x3x1_Relu6 inq -1.37<(i8-0.00)*0.01072699<1.36 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S328_Conv2d_480x3x3x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S231_Output", 0),
            GNodeArg(GNA_IN, "_head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1564_1", 0),
            GNodeArg(GNA_OUT, "S328_Output", 0),
            GNodeArg(GNA_IN, "S328_Mul_scale", 0),
            GNodeArg(GNA_IN, "S328_Mul_shift", 0),
            GNodeArg(GNA_IN, "S328_Infos", 0)
        )
    );
    // Node S330_Conv2d_512x3x3x1_Relu6 inq -0.88<(i8-0.00)*0.00689553<0.88 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S330_Conv2d_512x3x3x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S240_Output", 0),
            GNodeArg(GNA_IN, "_head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1567_1", 0),
            GNodeArg(GNA_OUT, "S330_Output", 0),
            GNodeArg(GNA_IN, "S330_Mul_scale", 0),
            GNodeArg(GNA_IN, "S330_Mul_shift", 0),
            GNodeArg(GNA_IN, "S330_Infos", 0)
        )
    );
    // Node S332_Conv2d_256x3x3x1_Relu6 inq -0.60<(i8-0.00)*0.00470736<0.60 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S332_Conv2d_256x3x3x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S258_Output", 0),
            GNodeArg(GNA_IN, "_head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1570_1", 0),
            GNodeArg(GNA_OUT, "S332_Output", 0),
            GNodeArg(GNA_IN, "S332_Mul_scale", 0),
            GNodeArg(GNA_IN, "S332_Mul_shift", 0),
            GNodeArg(GNA_IN, "S332_Infos", 0)
        )
    );
    // Node S333_Conv2d_24x1x1x256_Custom inq 0.00<(i8--128.00)*0.02352941<6.00 weightsq chan<(i8-0.00)*chan<chan outq -0.00<(i8-0.00)*0.00002717<0.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S333_Conv2d_24x1x1x256_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S332_Output", 0),
            GNodeArg(GNA_IN, "_head_regression_head_module_list_4_module_list_4_1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_head_regression_head_module_list_0_1_bias_4", 0),
            GNodeArg(GNA_OUT, "S333_Output", 0),
            GNodeArg(GNA_IN, "S333_Mul_scale", 0),
            GNodeArg(GNA_IN, "S333_Mul_shift", 0),
            GNodeArg(GNA_IN, "S333_Infos", 0)
        )
    );
    // Node _Split inq ['-0.00<(i8-0.00)*0.00002717<0.00'] outq ['-0.00<(i8-0.00)*0.00002717<0.00', '-0.00<(i8-0.00)*0.00002717<0.00', '-0.00<(i8-0.00)*0.00002717<0.00', '-0.00<(i8-0.00)*0.00002717<0.00']
    AddNode("S336_Op__Split",
        Bindings(5,
            GNodeArg(GNA_IN, "S335_Output", 0),
            GNodeArg(GNA_OUT, "S336_Output_0", 0),
            GNodeArg(GNA_OUT, "S336_Output_1", 0),
            GNodeArg(GNA_OUT, "S336_Output_2", 0),
            GNodeArg(GNA_OUT, "S336_Output_3", 0)
        )
    );
    // Node S342_Conv2d_256x3x3x1_Relu6 inq -1.05<(i8-0.00)*0.00818595<1.04 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S342_Conv2d_256x3x3x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S249_Output", 0),
            GNodeArg(GNA_IN, "_head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1570_2", 0),
            GNodeArg(GNA_OUT, "S342_Output", 0),
            GNodeArg(GNA_IN, "S342_Mul_scale", 0),
            GNodeArg(GNA_IN, "S342_Mul_shift", 0),
            GNodeArg(GNA_IN, "S342_Infos", 0)
        )
    );
    // Node S344_Conv2d_256x3x3x1_Relu6 inq -0.60<(i8-0.00)*0.00470736<0.60 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S344_Conv2d_256x3x3x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S258_Output", 0),
            GNodeArg(GNA_IN, "_head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1570_3", 0),
            GNodeArg(GNA_OUT, "S344_Output", 0),
            GNodeArg(GNA_IN, "S344_Mul_scale", 0),
            GNodeArg(GNA_IN, "S344_Mul_shift", 0),
            GNodeArg(GNA_IN, "S344_Infos", 0)
        )
    );
    // Node S346_Conv2d_128x3x3x1_Relu6 inq -1.12<(i8-0.00)*0.00874556<1.11 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S346_Conv2d_128x3x3x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S267_Output", 0),
            GNodeArg(GNA_IN, "_head_classification_head_module_list_5_module_list_5_0_module_list_5_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1576_1", 0),
            GNodeArg(GNA_OUT, "S346_Output", 0),
            GNodeArg(GNA_IN, "S346_Mul_scale", 0),
            GNodeArg(GNA_IN, "S346_Mul_shift", 0),
            GNodeArg(GNA_IN, "S346_Infos", 0)
        )
    );
    // Node S348_Conv2d_186x1x1x480 inq 0.00<(i8--128.00)*0.02352941<6.00 weightsq chan<(i8-0.00)*chan<chan outq -0.06<(i8-0.00)*0.00048828<0.06 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S348_Conv2d_186x1x1x480",
        Bindings(7,
            GNodeArg(GNA_IN, "S328_Output", 0),
            GNodeArg(GNA_IN, "_head_classification_head_module_list_1_module_list_1_1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_head_classification_head_module_list_0_1_bias_1", 0),
            GNodeArg(GNA_OUT, "S348_Output", 0),
            GNodeArg(GNA_IN, "S348_Mul_scale", 0),
            GNodeArg(GNA_IN, "S348_Mul_shift", 0),
            GNodeArg(GNA_IN, "S348_Infos", 0)
        )
    );
    // Node S351_Conv2d_186x1x1x512 inq 0.00<(i8--128.00)*0.02352941<6.00 weightsq chan<(i8-0.00)*chan<chan outq -0.06<(i8-0.00)*0.00048828<0.06 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S351_Conv2d_186x1x1x512",
        Bindings(7,
            GNodeArg(GNA_IN, "S330_Output", 0),
            GNodeArg(GNA_IN, "_head_classification_head_module_list_2_module_list_2_1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_head_classification_head_module_list_0_1_bias_2", 0),
            GNodeArg(GNA_OUT, "S351_Output", 0),
            GNodeArg(GNA_IN, "S351_Mul_scale", 0),
            GNodeArg(GNA_IN, "S351_Mul_shift", 0),
            GNodeArg(GNA_IN, "S351_Infos", 0)
        )
    );
    // Node S354_Conv2d_186x1x1x256 inq 0.00<(i8--128.00)*0.02352941<6.00 weightsq chan<(i8-0.00)*chan<chan outq -0.06<(i8-0.00)*0.00048828<0.06 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S354_Conv2d_186x1x1x256",
        Bindings(7,
            GNodeArg(GNA_IN, "S342_Output", 0),
            GNodeArg(GNA_IN, "_head_classification_head_module_list_3_module_list_3_1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_head_classification_head_module_list_0_1_bias_3", 0),
            GNodeArg(GNA_OUT, "S354_Output", 0),
            GNodeArg(GNA_IN, "S354_Mul_scale", 0),
            GNodeArg(GNA_IN, "S354_Mul_shift", 0),
            GNodeArg(GNA_IN, "S354_Infos", 0)
        )
    );
    // Node S357_Conv2d_186x1x1x256 inq 0.00<(i8--128.00)*0.02352941<6.00 weightsq chan<(i8-0.00)*chan<chan outq -0.06<(i8-0.00)*0.00048828<0.06 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S357_Conv2d_186x1x1x256",
        Bindings(7,
            GNodeArg(GNA_IN, "S344_Output", 0),
            GNodeArg(GNA_IN, "_head_classification_head_module_list_4_module_list_4_1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_head_classification_head_module_list_0_1_bias_4", 0),
            GNodeArg(GNA_OUT, "S357_Output", 0),
            GNodeArg(GNA_IN, "S357_Mul_scale", 0),
            GNodeArg(GNA_IN, "S357_Mul_shift", 0),
            GNodeArg(GNA_IN, "S357_Infos", 0)
        )
    );
    // Node S360_Conv2d_186x1x1x128 inq 0.00<(i8--128.00)*0.02352941<6.00 weightsq chan<(i8-0.00)*chan<chan outq -0.06<(i8-0.00)*0.00048828<0.06 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S360_Conv2d_186x1x1x128",
        Bindings(7,
            GNodeArg(GNA_IN, "S346_Output", 0),
            GNodeArg(GNA_IN, "_head_classification_head_module_list_5_module_list_5_1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_head_classification_head_module_list_0_1_bias_5", 0),
            GNodeArg(GNA_OUT, "S360_Output", 0),
            GNodeArg(GNA_IN, "S360_Mul_scale", 0),
            GNodeArg(GNA_IN, "S360_Mul_shift", 0),
            GNodeArg(GNA_IN, "S360_Infos", 0)
        )
    );
    // Node _Softmax inq -0.06<(i8-0.00)*0.00048828<0.06 forced outq -1.00<(i16-0.00)*0.00003052<1.00
    AddNode("S363_SoftMax",
        Bindings(3,
            GNodeArg(GNA_IN, "S362_Output", 0),
            GNodeArg(GNA_OUT, "Output_5", 0),
            GNodeArg(GNA_IN, "S363_Infos", 0)
        )
    );
    // Node expr_1 in_qs [-0.00<(i8-0.00)*0.00002717<0.00,-258.02<(i8-0.00)*2.01574803<256.00,-249.95<(i8-0.00)*1.95275591<248.00,-258.02<(i8-0.00)*2.01574803<256.00,-0.00<(i8-0.00)*0.00002717<0.00] out_qs [-235.08<(i8-0.00)*1.83654165<233.24,-337.52<(i8-0.00)*2.63685942<334.88]
    AddNode("S366_Op_expr_1",
        Bindings(7,
            GNodeArg(GNA_IN, "S336_Output_1", 0),
            GNodeArg(GNA_IN, "_sub_1", 0),
            GNodeArg(GNA_IN, "_add_1", 0),
            GNodeArg(GNA_IN, "_sub_1_1", 0),
            GNodeArg(GNA_IN, "S336_Output_3", 0),
            GNodeArg(GNA_OUT, "Output_2", 0),
            GNodeArg(GNA_OUT, "Output_4", 0)
        )
    );
    // Node expr_0 in_qs [-0.00<(i8-0.00)*0.00002717<0.00,-322.52<(i8-0.00)*2.51968504<320.00,-314.46<(i8-0.00)*2.45669291<312.00,-258.02<(i8-0.00)*2.01574803<256.00,-0.00<(i8-0.00)*0.00002717<0.00] out_qs [-299.66<(i8-0.00)*2.34111857<297.32,-402.06<(i8-0.00)*3.14109969<398.92]
    AddNode("S372_Op_expr_0",
        Bindings(7,
            GNodeArg(GNA_IN, "S336_Output_0", 0),
            GNodeArg(GNA_IN, "_sub", 0),
            GNodeArg(GNA_IN, "_add", 0),
            GNodeArg(GNA_IN, "_sub_1_2", 0),
            GNodeArg(GNA_IN, "S336_Output_2", 0),
            GNodeArg(GNA_OUT, "Output_1", 0),
            GNodeArg(GNA_OUT, "Output_3", 0)
        )
    );
    CloseGraph();
#endif
}

int main(int argc, char **argv)

{
    if (TilerParseOptions(argc, argv)) {
            printf("Failed to initialize or incorrect output arguments directory.\n"); return 1;
    }
    Mobv3SSDLiteModel(128000, 1150000, 8000000, 64*1024*1024);
    GenerateTilingCode();
    return 0;
}
