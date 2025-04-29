#include <stdint.h>
#include <stdio.h>
#include "AutoTilerLib.h"
#include "CNN_Generators_SQ8.h"
#include "CNN_Generators_NE16.h"

#include "CNN_Copy_Generators.h"

void load_expressions_kernels() {
    LibKernelTemplate(
        "s382_kernel_args_t",
        CArgs(8,
            TCArg("unsigned int", "I0"),
            TCArg("unsigned char *__restrict__ ", "expr_0_in_0"),
            TCArg("signed char *__restrict__ ", "expr_0_in_1"),
            TCArg("signed char *__restrict__ ", "expr_0_in_2"),
            TCArg("signed char *__restrict__ ", "expr_0_in_3"),
            TCArg("unsigned char *__restrict__ ", "expr_0_in_4"),
            TCArg("unsigned char *__restrict__ ", "expr_0_out_0"),
            TCArg("unsigned char *__restrict__ ", "expr_0_out_1")
        )
    );
    
    LibKernel(
        "s382_kernel",
        CALL_PARALLEL,
        0,
        "s382_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s376_kernel_args_t",
        CArgs(8,
            TCArg("unsigned int", "I0"),
            TCArg("unsigned char *__restrict__ ", "expr_1_in_0"),
            TCArg("signed char *__restrict__ ", "expr_1_in_1"),
            TCArg("signed char *__restrict__ ", "expr_1_in_2"),
            TCArg("signed char *__restrict__ ", "expr_1_in_3"),
            TCArg("unsigned char *__restrict__ ", "expr_1_in_4"),
            TCArg("unsigned char *__restrict__ ", "expr_1_out_0"),
            TCArg("unsigned char *__restrict__ ", "expr_1_out_1")
        )
    );
    
    LibKernel(
        "s376_kernel",
        CALL_PARALLEL,
        0,
        "s376_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s150_kernel_args_t",
        CArgs(5,
            TCArg("unsigned int", "I0"),
            TCArg("unsigned int", "I1"),
            TCArg("signed char *__restrict__ ", "expr_11_in_0"),
            TCArg("signed char *__restrict__ ", "expr_11_in_1"),
            TCArg("unsigned char *__restrict__ ", "expr_11_out_0")
        )
    );
    
    LibKernel(
        "s150_kernel",
        CALL_PARALLEL,
        0,
        "s150_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s170_kernel_args_t",
        CArgs(5,
            TCArg("unsigned int", "I0"),
            TCArg("unsigned int", "I1"),
            TCArg("signed char *__restrict__ ", "expr_14_in_0"),
            TCArg("signed char *__restrict__ ", "expr_14_in_1"),
            TCArg("unsigned char *__restrict__ ", "expr_14_out_0")
        )
    );
    
    LibKernel(
        "s170_kernel",
        CALL_PARALLEL,
        0,
        "s170_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s47_kernel_args_t",
        CArgs(5,
            TCArg("unsigned int", "I0"),
            TCArg("unsigned int", "I1"),
            TCArg("signed char *__restrict__ ", "expr_16_in_0"),
            TCArg("signed char *__restrict__ ", "expr_16_in_1"),
            TCArg("unsigned char *__restrict__ ", "expr_16_out_0")
        )
    );
    
    LibKernel(
        "s47_kernel",
        CALL_PARALLEL,
        0,
        "s47_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s68_kernel_args_t",
        CArgs(5,
            TCArg("unsigned int", "I0"),
            TCArg("unsigned int", "I1"),
            TCArg("signed char *__restrict__ ", "expr_17_in_0"),
            TCArg("signed char *__restrict__ ", "expr_17_in_1"),
            TCArg("unsigned char *__restrict__ ", "expr_17_out_0")
        )
    );
    
    LibKernel(
        "s68_kernel",
        CALL_PARALLEL,
        0,
        "s68_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s90_kernel_args_t",
        CArgs(5,
            TCArg("unsigned int", "I0"),
            TCArg("unsigned int", "I1"),
            TCArg("signed char *__restrict__ ", "expr_18_in_0"),
            TCArg("signed char *__restrict__ ", "expr_18_in_1"),
            TCArg("unsigned char *__restrict__ ", "expr_18_out_0")
        )
    );
    
    LibKernel(
        "s90_kernel",
        CALL_PARALLEL,
        0,
        "s90_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s193_kernel_args_t",
        CArgs(5,
            TCArg("unsigned int", "I0"),
            TCArg("unsigned int", "I1"),
            TCArg("signed char *__restrict__ ", "expr_26_in_0"),
            TCArg("signed char *__restrict__ ", "expr_26_in_1"),
            TCArg("unsigned char *__restrict__ ", "expr_26_out_0")
        )
    );
    
    LibKernel(
        "s193_kernel",
        CALL_PARALLEL,
        0,
        "s193_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s213_kernel_args_t",
        CArgs(5,
            TCArg("unsigned int", "I0"),
            TCArg("unsigned int", "I1"),
            TCArg("signed char *__restrict__ ", "expr_29_in_0"),
            TCArg("signed char *__restrict__ ", "expr_29_in_1"),
            TCArg("unsigned char *__restrict__ ", "expr_29_out_0")
        )
    );
    
    LibKernel(
        "s213_kernel",
        CALL_PARALLEL,
        0,
        "s213_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s234_kernel_args_t",
        CArgs(5,
            TCArg("unsigned int", "I0"),
            TCArg("unsigned int", "I1"),
            TCArg("signed char *__restrict__ ", "expr_32_in_0"),
            TCArg("signed char *__restrict__ ", "expr_32_in_1"),
            TCArg("unsigned char *__restrict__ ", "expr_32_out_0")
        )
    );
    
    LibKernel(
        "s234_kernel",
        CALL_PARALLEL,
        0,
        "s234_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "expr_6_args_t",
        CArgs(5,
            TCArg("unsigned char *__restrict__ ", "expr_6_in_0"),
            TCArg("unsigned char *__restrict__ ", "expr_6_out_0"),
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
            TCArg("unsigned char *__restrict__ ", "expr_19_in_0"),
            TCArg("unsigned char *__restrict__ ", "expr_19_out_0"),
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
            TCArg("unsigned char *__restrict__ ", "expr_20_in_0"),
            TCArg("unsigned char *__restrict__ ", "expr_20_out_0"),
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
            TCArg("unsigned char *__restrict__ ", "expr_21_in_0"),
            TCArg("unsigned char *__restrict__ ", "expr_21_out_0"),
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
            TCArg("unsigned char *__restrict__ ", "expr_22_in_0"),
            TCArg("unsigned char *__restrict__ ", "expr_22_out_0"),
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
            TCArg("unsigned char *__restrict__ ", "expr_23_in_0"),
            TCArg("unsigned char *__restrict__ ", "expr_23_out_0"),
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
            TCArg("unsigned char *__restrict__ ", "expr_24_in_0"),
            TCArg("unsigned char *__restrict__ ", "expr_24_out_0"),
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
            TCArg("unsigned char *__restrict__ ", "expr_7_in_0"),
            TCArg("unsigned char *__restrict__ ", "expr_7_out_0"),
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
            TCArg("unsigned char *__restrict__ ", "expr_8_in_0"),
            TCArg("unsigned char *__restrict__ ", "expr_8_out_0"),
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
            TCArg("unsigned char *__restrict__ ", "expr_9_in_0"),
            TCArg("unsigned char *__restrict__ ", "expr_9_out_0"),
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
            TCArg("unsigned char *__restrict__ ", "expr_10_in_0"),
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
            TCArg("unsigned char *__restrict__ ", "expr_12_in_0"),
            TCArg("unsigned char *__restrict__ ", "expr_12_out_0"),
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
            TCArg("unsigned char *__restrict__ ", "expr_13_in_0"),
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
            TCArg("unsigned char *__restrict__ ", "expr_15_in_0"),
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
            TCArg("unsigned char *__restrict__ ", "expr_27_in_0"),
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
            TCArg("unsigned char *__restrict__ ", "expr_30_in_0"),
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
            TCArg("unsigned char *__restrict__ ", "expr_33_in_0"),
            TCArg("unsigned char *__restrict__ ", "expr_33_out_0"),
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
            TCArg("unsigned char *__restrict__ ", "expr_2_in_0"),
            TCArg("unsigned char *__restrict__ ", "expr_2_out_0"),
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
            TCArg("unsigned char *__restrict__ ", "expr_2_in_0"),
            TCArg("unsigned char *__restrict__ ", "expr_2_out_0"),
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
            TCArg("unsigned char *__restrict__ ", "expr_2_in_0"),
            TCArg("unsigned char *__restrict__ ", "expr_2_out_0"),
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
            TCArg("unsigned char *__restrict__ ", "expr_2_in_0"),
            TCArg("unsigned char *__restrict__ ", "expr_2_out_0"),
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
            TCArg("unsigned char *__restrict__ ", "expr_2_in_0"),
            TCArg("unsigned char *__restrict__ ", "expr_2_out_0"),
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
            TCArg("unsigned char *__restrict__ ", "expr_2_in_0"),
            TCArg("unsigned char *__restrict__ ", "expr_2_out_0"),
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



int s382_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (1, 1, 2574) spaces: ((2,),) 
        // parametric_spaces: ((2,),) 
        // exterior_shape: (2574.0,) 
        KernelIterSpace(2, IterParSpace(KER_ITER_D0, 2574, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(7,
            TCArg(CNN_ArgDataTypeUns(1, 1, 1), "expr_0_in_0"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_0_in_1"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_0_in_2"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_0_in_3"),
            TCArg(CNN_ArgDataTypeUns(1, 1, 1), "expr_0_in_4"),
            TCArg(CNN_ArgDataTypeUns(1, 1, 1), "expr_0_out_0"),
            TCArg(CNN_ArgDataTypeUns(1, 1, 1), "expr_0_out_1")
        ),
        Calls(1,
            Call("s382_kernel", LOC_D0,
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
        AddKernelInfos(Name, AT_KERINFO_OPER, 97812, 0);
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
int s376_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (1, 1, 2574) spaces: ((2,),) 
        // parametric_spaces: ((2,),) 
        // exterior_shape: (2574.0,) 
        KernelIterSpace(2, IterParSpace(KER_ITER_D0, 2574, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(7,
            TCArg(CNN_ArgDataTypeUns(1, 1, 1), "expr_1_in_0"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_1_in_1"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_1_in_2"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_1_in_3"),
            TCArg(CNN_ArgDataTypeUns(1, 1, 1), "expr_1_in_4"),
            TCArg(CNN_ArgDataTypeUns(1, 1, 1), "expr_1_out_0"),
            TCArg(CNN_ArgDataTypeUns(1, 1, 1), "expr_1_out_1")
        ),
        Calls(1,
            Call("s376_kernel", LOC_D0,
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
        AddKernelInfos(Name, AT_KERINFO_OPER, 97812, 0);
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
int s150_kernel_gen(char *Name) {
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
            TCArg(CNN_ArgDataTypeUns(1, 1, 1), "expr_11_out_0")
        ),
        Calls(1,
            Call("s150_kernel", LOC_D1,
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
        AddKernelInfos(Name, AT_KERINFO_OPER, 1228800, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 307680, 0);
        AddKernelArgDim(Name, "expr_11_in_0",  2, 480,         1);
        AddKernelArgDim(Name, "expr_11_in_1",  4, 16, 20, 480, 1);
        AddKernelArgDim(Name, "expr_11_out_0", 4, 16, 20, 480, 1);
    }
    return (Kernel!=0);
}
int s170_kernel_gen(char *Name) {
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
            TCArg(CNN_ArgDataTypeUns(1, 1, 1), "expr_14_out_0")
        ),
        Calls(1,
            Call("s170_kernel", LOC_D1,
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
        AddKernelInfos(Name, AT_KERINFO_OPER, 1720320, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 430752, 0);
        AddKernelArgDim(Name, "expr_14_in_0",  2, 672,         1);
        AddKernelArgDim(Name, "expr_14_in_1",  4, 16, 20, 672, 1);
        AddKernelArgDim(Name, "expr_14_out_0", 4, 16, 20, 672, 1);
    }
    return (Kernel!=0);
}
int s47_kernel_gen(char *Name) {
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
            TCArg(CNN_ArgDataTypeUns(1, 1, 1), "expr_16_out_0")
        ),
        Calls(1,
            Call("s47_kernel", LOC_D1,
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
int s68_kernel_gen(char *Name) {
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
            TCArg(CNN_ArgDataTypeUns(1, 1, 1), "expr_17_out_0")
        ),
        Calls(1,
            Call("s68_kernel", LOC_D1,
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
int s90_kernel_gen(char *Name) {
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
            TCArg(CNN_ArgDataTypeUns(1, 1, 1), "expr_18_out_0")
        ),
        Calls(1,
            Call("s90_kernel", LOC_D1,
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
int s193_kernel_gen(char *Name) {
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
            TCArg(CNN_ArgDataTypeUns(1, 1, 1), "expr_26_out_0")
        ),
        Calls(1,
            Call("s193_kernel", LOC_D1,
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
        AddKernelInfos(Name, AT_KERINFO_OPER, 430080, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 108192, 0);
        AddKernelArgDim(Name, "expr_26_in_0",  2, 672,        1);
        AddKernelArgDim(Name, "expr_26_in_1",  4, 8, 10, 672, 1);
        AddKernelArgDim(Name, "expr_26_out_0", 4, 8, 10, 672, 1);
    }
    return (Kernel!=0);
}
int s213_kernel_gen(char *Name) {
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
            TCArg(CNN_ArgDataTypeUns(1, 1, 1), "expr_29_out_0")
        ),
        Calls(1,
            Call("s213_kernel", LOC_D1,
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
        AddKernelInfos(Name, AT_KERINFO_OPER, 307200, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 77280, 0);
        AddKernelArgDim(Name, "expr_29_in_0",  2, 480,        1);
        AddKernelArgDim(Name, "expr_29_in_1",  4, 8, 10, 480, 1);
        AddKernelArgDim(Name, "expr_29_out_0", 4, 8, 10, 480, 1);
    }
    return (Kernel!=0);
}
int s234_kernel_gen(char *Name) {
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
            TCArg(CNN_ArgDataTypeUns(1, 1, 1), "expr_32_out_0")
        ),
        Calls(1,
            Call("s234_kernel", LOC_D1,
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
        AddKernelInfos(Name, AT_KERINFO_OPER, 307200, 0);
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

    SetUsedFilesNames(0, 6, "Gap.h", "Mobv3SSDLite.h", "CNN_BasicKernels_SQ8.h", "CNN_BasicKernels_NE16.h", "CNN_BasicKernels_SQ8.h", "Expression_Kernels.h");
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
    LoadCNN_NE16_SQ8_Library();
    LoadCNN_Copy_Library();
    load_expressions_kernels();

    CNN_GenControl_T gen_ctrl_S3_Conv2d_16x3x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S3_Conv2d_16x3x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S3_Conv2d_16x3x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_6");
    // generator for _backbone_features_0_features_0_0_features_0_0_0_Conv_fusion
    CNN_ConvolutionNE16("S3_Conv2d_16x3x3x3_Custom", &gen_ctrl_S3_Conv2d_16x3x3x3_Custom,
                        -1, -1, 4, 1, 8,
                        3, 16, 320, 256,
                        KOP_CONV, 3, 3, 1, 1, 2, 2, 1, 255,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_CUSTOM);
    
    // generator for _backbone_features_0_features_0_1_block_block_0_block_0_0_Conv_fusion
    CNN_ConvolutionNE16("S6_Conv2d_16x1x3x3_Relu", 0,
                        -1, -1, 4, 1, 8,
                        16, 16, 160, 128,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 128,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    // generator for _backbone_features_0_features_0_1_block_block_1_block_1_0_Conv
    CNN_ConvolutionNE16("S9_Conv2d_16x16x1x1", 0,
                        -1, -1, 4, 1, 8,
                        16, 16, 160, 128,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S10_MatAdd_128x160x16;
    CNN_InitGenCtrl(&gen_ctrl_S10_MatAdd_128x160x16);
    CNN_SetGenCtrl(&gen_ctrl_S10_MatAdd_128x160x16, "OUTPUT_DATASIZE", AT_OPT_VAL(-1));
    CNN_SetGenCtrl(&gen_ctrl_S10_MatAdd_128x160x16, "INPUT_DATASIZE", AT_OPT_VAL(-1));
    
    // generator for _backbone_features_0_features_0_1_Add
    CNN_MatAddAct_SQ8("S10_MatAdd_128x160x16", &gen_ctrl_S10_MatAdd_128x160x16, 128, 160, 16, KOP_MATADD, KOP_NONE);
    
    // generator for _backbone_features_0_features_0_2_block_block_0_block_0_0_Conv_fusion
    CNN_ConvolutionNE16("S13_Conv2d_64x16x1x1_Relu", 0,
                        -1, -1, 4, 1, 8,
                        16, 64, 160, 128,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 60,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    // generator for _backbone_features_0_features_0_2_block_block_1_block_1_0_Conv_fusion
    CNN_ConvolutionNE16("S16_Conv2d_64x1x3x3_Relu", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 160, 128,
                        KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    // generator for _backbone_features_0_features_0_2_block_block_2_block_2_0_Conv
    CNN_ConvolutionNE16("S19_Conv2d_24x64x1x1", 0,
                        -1, -1, 4, 1, 8,
                        64, 24, 80, 64,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    // generator for _backbone_features_0_features_0_3_block_block_0_block_0_0_Conv_fusion
    CNN_ConvolutionNE16("S22_Conv2d_72x24x1x1_Relu", 0,
                        -1, -1, 4, 1, 8,
                        24, 72, 80, 64,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 137,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    // generator for _backbone_features_0_features_0_3_block_block_1_block_1_0_Conv_fusion
    CNN_ConvolutionNE16("S25_Conv2d_72x1x3x3_Relu", 0,
                        -1, -1, 4, 1, 8,
                        72, 72, 80, 64,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    // generator for _backbone_features_0_features_0_3_block_block_2_block_2_0_Conv
    CNN_ConvolutionNE16("S28_Conv2d_24x72x1x1", 0,
                        -1, -1, 4, 1, 8,
                        72, 24, 80, 64,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S29_MatAdd_64x80x24;
    CNN_InitGenCtrl(&gen_ctrl_S29_MatAdd_64x80x24);
    CNN_SetGenCtrl(&gen_ctrl_S29_MatAdd_64x80x24, "OUTPUT_DATASIZE", AT_OPT_VAL(-1));
    CNN_SetGenCtrl(&gen_ctrl_S29_MatAdd_64x80x24, "INPUT_DATASIZE", AT_OPT_VAL(-1));
    
    // generator for _backbone_features_0_features_0_3_Add
    CNN_MatAddAct_SQ8("S29_MatAdd_64x80x24", &gen_ctrl_S29_MatAdd_64x80x24, 64, 80, 24, KOP_MATADD, KOP_NONE);
    
    // generator for _backbone_features_0_features_0_4_block_block_0_block_0_0_Conv_fusion
    CNN_ConvolutionNE16("S32_Conv2d_72x24x1x1_Relu", 0,
                        -1, -1, 4, 1, 8,
                        24, 72, 80, 64,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 161,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    
    // generator for _backbone_features_0_features_0_4_block_block_1_block_1_0_Conv_fusion_qin0
    CNN_Convert("S33_Op__backbone_features_0_features_0_4_block_block_1_block_1_0_Conv_fusion_qin0", -1, 1, 368640, KOP_CONVERT_FP_FP_SCALE);
    
    CNN_GenControl_T gen_ctrl_S36_Conv2d_72x5x5x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S36_Conv2d_72x5x5x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S36_Conv2d_72x5x5x1_Relu, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_4_block_block_1_block_1_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S36_Conv2d_72x5x5x1_Relu", &gen_ctrl_S36_Conv2d_72x5x5x1_Relu,
                               4, 1,
                               72, 72, 80, 64,
                               KOP_CONV_DW, 5, 5, 1, 1, 2, 2, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    
    // generator for _backbone_features_0_features_0_4_block_block_2_avgpool_GlobalAveragePool_trans Transpose 32x40x72 -> 72x32x40 ((1, 0))
    CNN_MatTranspose("S37_Op__backbone_features_0_features_0_4_block_block_2_avgpool_GlobalAveragePool_trans", 0, 1,
                      1, 72, 1280, KOP_MATTRANSP);
    
    // generator for _backbone_features_0_features_0_4_block_block_2_avgpool_GlobalAveragePool
    CNN_GlobalPoolAct_SQ8("S38_Op__backbone_features_0_features_0_4_block_block_2_avgpool_GlobalAveragePool", 0,
                          72, 16, 80,
                          KOP_GLOBAL_AVGPOOL, KOP_NONE);
    
    
    // generator for _backbone_features_0_features_0_4_block_block_2_fc1_Conv_fusion_qin0
    CNN_Convert("S40_Op__backbone_features_0_features_0_4_block_block_2_fc1_Conv_fusion_qin0", 1, -1, 72, KOP_CONVERT_FP_FP_SCALE);
    
    // generator for _backbone_features_0_features_0_4_block_block_2_fc1_Conv_fusion
    CNN_ConvolutionNE16("S43_Conv2d_24x72x1x1_Relu", 0,
                        -1, -1, 4, 1, 8,
                        72, 24, 1, 1,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    // generator for _backbone_features_0_features_0_4_block_block_2_fc2_Conv_fusion
    CNN_ConvolutionNE16("S46_Conv2d_72x24x1x1_Hsigmoid", 0,
                        -1, 1, 4, 1, 8,
                        24, 72, 1, 1,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_HSIGMOID);
    
    
    // generator for expr_16
    s47_kernel_gen("S47_Op_expr_16");
    
    // generator for _backbone_features_0_features_0_4_block_block_3_block_3_0_Conv
    CNN_ConvolutionNE16("S50_Conv2d_40x72x1x1", 0,
                        -1, -1, 4, 1, 8,
                        72, 40, 40, 32,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    // generator for _backbone_features_0_features_0_5_block_block_0_block_0_0_Conv_fusion
    CNN_ConvolutionNE16("S53_Conv2d_120x40x1x1_Relu", 0,
                        -1, -1, 4, 1, 8,
                        40, 120, 40, 32,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 162,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    
    // generator for _backbone_features_0_features_0_5_block_block_1_block_1_0_Conv_fusion_qin0
    CNN_Convert("S54_Op__backbone_features_0_features_0_5_block_block_1_block_1_0_Conv_fusion_qin0", -1, 1, 153600, KOP_CONVERT_FP_FP_SCALE);
    
    CNN_GenControl_T gen_ctrl_S57_Conv2d_120x5x5x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S57_Conv2d_120x5x5x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S57_Conv2d_120x5x5x1_Relu, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_5_block_block_1_block_1_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S57_Conv2d_120x5x5x1_Relu", &gen_ctrl_S57_Conv2d_120x5x5x1_Relu,
                               4, 1,
                               120, 120, 40, 32,
                               KOP_CONV_DW, 5, 5, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    
    // generator for _backbone_features_0_features_0_5_block_block_2_avgpool_GlobalAveragePool_trans Transpose 32x40x120 -> 120x32x40 ((1, 0))
    CNN_MatTranspose("S58_Op__backbone_features_0_features_0_5_block_block_2_avgpool_GlobalAveragePool_trans", 0, 1,
                      1, 120, 1280, KOP_MATTRANSP);
    
    // generator for _backbone_features_0_features_0_5_block_block_2_avgpool_GlobalAveragePool
    CNN_GlobalPoolAct_SQ8("S59_Op__backbone_features_0_features_0_5_block_block_2_avgpool_GlobalAveragePool", 0,
                          120, 16, 80,
                          KOP_GLOBAL_AVGPOOL, KOP_NONE);
    
    
    // generator for _backbone_features_0_features_0_5_block_block_2_fc1_Conv_fusion_qin0
    CNN_Convert("S61_Op__backbone_features_0_features_0_5_block_block_2_fc1_Conv_fusion_qin0", 1, -1, 120, KOP_CONVERT_FP_FP_SCALE);
    
    // generator for _backbone_features_0_features_0_5_block_block_2_fc1_Conv_fusion
    CNN_ConvolutionNE16("S64_Conv2d_32x120x1x1_Relu", 0,
                        -1, -1, 4, 1, 8,
                        120, 32, 1, 1,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    // generator for _backbone_features_0_features_0_5_block_block_2_fc2_Conv_fusion
    CNN_ConvolutionNE16("S67_Conv2d_120x32x1x1_Hsigmoid", 0,
                        -1, 1, 4, 1, 8,
                        32, 120, 1, 1,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_HSIGMOID);
    
    
    // generator for expr_17
    s68_kernel_gen("S68_Op_expr_17");
    
    // generator for _backbone_features_0_features_0_5_block_block_3_block_3_0_Conv
    CNN_ConvolutionNE16("S71_Conv2d_40x120x1x1", 0,
                        -1, -1, 4, 1, 8,
                        120, 40, 40, 32,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S72_MatAdd_32x40x40;
    CNN_InitGenCtrl(&gen_ctrl_S72_MatAdd_32x40x40);
    CNN_SetGenCtrl(&gen_ctrl_S72_MatAdd_32x40x40, "OUTPUT_DATASIZE", AT_OPT_VAL(-1));
    CNN_SetGenCtrl(&gen_ctrl_S72_MatAdd_32x40x40, "INPUT_DATASIZE", AT_OPT_VAL(-1));
    
    // generator for _backbone_features_0_features_0_5_Add
    CNN_MatAddAct_SQ8("S72_MatAdd_32x40x40", &gen_ctrl_S72_MatAdd_32x40x40, 32, 40, 40, KOP_MATADD, KOP_NONE);
    
    // generator for _backbone_features_0_features_0_6_block_block_0_block_0_0_Conv_fusion
    CNN_ConvolutionNE16("S75_Conv2d_120x40x1x1_Relu", 0,
                        -1, -1, 4, 1, 8,
                        40, 120, 40, 32,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 132,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    
    // generator for _backbone_features_0_features_0_6_block_block_1_block_1_0_Conv_fusion_qin0
    CNN_Convert("S76_Op__backbone_features_0_features_0_6_block_block_1_block_1_0_Conv_fusion_qin0", -1, 1, 153600, KOP_CONVERT_FP_FP_SCALE);
    
    CNN_GenControl_T gen_ctrl_S79_Conv2d_120x5x5x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S79_Conv2d_120x5x5x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S79_Conv2d_120x5x5x1_Relu, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_features_0_features_0_6_block_block_1_block_1_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S79_Conv2d_120x5x5x1_Relu", &gen_ctrl_S79_Conv2d_120x5x5x1_Relu,
                               4, 1,
                               120, 120, 40, 32,
                               KOP_CONV_DW, 5, 5, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    
    // generator for _backbone_features_0_features_0_6_block_block_2_avgpool_GlobalAveragePool_trans Transpose 32x40x120 -> 120x32x40 ((1, 0))
    CNN_MatTranspose("S80_Op__backbone_features_0_features_0_6_block_block_2_avgpool_GlobalAveragePool_trans", 0, 1,
                      1, 120, 1280, KOP_MATTRANSP);
    
    // generator for _backbone_features_0_features_0_6_block_block_2_avgpool_GlobalAveragePool
    CNN_GlobalPoolAct_SQ8("S81_Op__backbone_features_0_features_0_6_block_block_2_avgpool_GlobalAveragePool", 0,
                          120, 16, 80,
                          KOP_GLOBAL_AVGPOOL, KOP_NONE);
    
    
    // generator for _backbone_features_0_features_0_6_block_block_2_fc1_Conv_fusion_qin0
    CNN_Convert("S83_Op__backbone_features_0_features_0_6_block_block_2_fc1_Conv_fusion_qin0", 1, -1, 120, KOP_CONVERT_FP_FP_SCALE);
    
    // generator for _backbone_features_0_features_0_6_block_block_2_fc1_Conv_fusion
    CNN_ConvolutionNE16("S86_Conv2d_32x120x1x1_Relu", 0,
                        -1, -1, 4, 1, 8,
                        120, 32, 1, 1,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    // generator for _backbone_features_0_features_0_6_block_block_2_fc2_Conv_fusion
    CNN_ConvolutionNE16("S89_Conv2d_120x32x1x1_Hsigmoid", 0,
                        -1, 1, 4, 1, 8,
                        32, 120, 1, 1,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_HSIGMOID);
    
    
    // generator for expr_18
    s90_kernel_gen("S90_Op_expr_18");
    
    // generator for _backbone_features_0_features_0_6_block_block_3_block_3_0_Conv
    CNN_ConvolutionNE16("S93_Conv2d_40x120x1x1", 0,
                        -1, -1, 4, 1, 8,
                        120, 40, 40, 32,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S94_MatAdd_32x40x40;
    CNN_InitGenCtrl(&gen_ctrl_S94_MatAdd_32x40x40);
    CNN_SetGenCtrl(&gen_ctrl_S94_MatAdd_32x40x40, "OUTPUT_DATASIZE", AT_OPT_VAL(-1));
    CNN_SetGenCtrl(&gen_ctrl_S94_MatAdd_32x40x40, "INPUT_DATASIZE", AT_OPT_VAL(-1));
    
    // generator for _backbone_features_0_features_0_6_Add
    CNN_MatAddAct_SQ8("S94_MatAdd_32x40x40", &gen_ctrl_S94_MatAdd_32x40x40, 32, 40, 40, KOP_MATADD, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S97_Conv2d_240x40x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S97_Conv2d_240x40x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S97_Conv2d_240x40x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_19");
    // generator for _backbone_features_0_features_0_7_block_block_0_block_0_0_Conv_fusion
    CNN_ConvolutionNE16("S97_Conv2d_240x40x1x1_Custom", &gen_ctrl_S97_Conv2d_240x40x1x1_Custom,
                        -1, -1, 4, 1, 8,
                        40, 240, 40, 32,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 115,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S100_Conv2d_240x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S100_Conv2d_240x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S100_Conv2d_240x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_20");
    // generator for _backbone_features_0_features_0_7_block_block_1_block_1_0_Conv_fusion
    CNN_ConvolutionNE16("S100_Conv2d_240x1x3x3_Custom", &gen_ctrl_S100_Conv2d_240x1x3x3_Custom,
                        -1, -1, 4, 1, 8,
                        240, 240, 40, 32,
                        KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1, 128,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_CUSTOM);
    
    // generator for _backbone_features_0_features_0_7_block_block_2_block_2_0_Conv
    CNN_ConvolutionNE16("S103_Conv2d_80x240x1x1", 0,
                        -1, -1, 4, 1, 8,
                        240, 80, 20, 16,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 128,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S106_Conv2d_200x80x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S106_Conv2d_200x80x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S106_Conv2d_200x80x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_21");
    // generator for _backbone_features_0_features_0_8_block_block_0_block_0_0_Conv_fusion
    CNN_ConvolutionNE16("S106_Conv2d_200x80x1x1_Custom", &gen_ctrl_S106_Conv2d_200x80x1x1_Custom,
                        -1, -1, 4, 1, 8,
                        80, 200, 20, 16,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 146,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S109_Conv2d_200x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S109_Conv2d_200x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S109_Conv2d_200x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_22");
    // generator for _backbone_features_0_features_0_8_block_block_1_block_1_0_Conv_fusion
    CNN_ConvolutionNE16("S109_Conv2d_200x1x3x3_Custom", &gen_ctrl_S109_Conv2d_200x1x3x3_Custom,
                        -1, -1, 4, 1, 8,
                        200, 200, 20, 16,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 128,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_CUSTOM);
    
    // generator for _backbone_features_0_features_0_8_block_block_2_block_2_0_Conv
    CNN_ConvolutionNE16("S112_Conv2d_80x200x1x1", 0,
                        -1, -1, 4, 1, 8,
                        200, 80, 20, 16,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 128,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S113_MatAdd_16x20x80;
    CNN_InitGenCtrl(&gen_ctrl_S113_MatAdd_16x20x80);
    CNN_SetGenCtrl(&gen_ctrl_S113_MatAdd_16x20x80, "OUTPUT_DATASIZE", AT_OPT_VAL(-1));
    CNN_SetGenCtrl(&gen_ctrl_S113_MatAdd_16x20x80, "INPUT_DATASIZE", AT_OPT_VAL(-1));
    
    // generator for _backbone_features_0_features_0_8_Add
    CNN_MatAddAct_SQ8("S113_MatAdd_16x20x80", &gen_ctrl_S113_MatAdd_16x20x80, 16, 20, 80, KOP_MATADD, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S116_Conv2d_184x80x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S116_Conv2d_184x80x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S116_Conv2d_184x80x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_23");
    // generator for _backbone_features_0_features_0_9_block_block_0_block_0_0_Conv_fusion
    CNN_ConvolutionNE16("S116_Conv2d_184x80x1x1_Custom", &gen_ctrl_S116_Conv2d_184x80x1x1_Custom,
                        -1, -1, 4, 1, 8,
                        80, 184, 20, 16,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 140,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S119_Conv2d_184x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S119_Conv2d_184x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S119_Conv2d_184x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_24");
    // generator for _backbone_features_0_features_0_9_block_block_1_block_1_0_Conv_fusion
    CNN_ConvolutionNE16("S119_Conv2d_184x1x3x3_Custom", &gen_ctrl_S119_Conv2d_184x1x3x3_Custom,
                        -1, -1, 4, 1, 8,
                        184, 184, 20, 16,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 128,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_CUSTOM);
    
    // generator for _backbone_features_0_features_0_9_block_block_2_block_2_0_Conv
    CNN_ConvolutionNE16("S122_Conv2d_80x184x1x1", 0,
                        -1, -1, 4, 1, 8,
                        184, 80, 20, 16,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 128,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S123_MatAdd_16x20x80;
    CNN_InitGenCtrl(&gen_ctrl_S123_MatAdd_16x20x80);
    CNN_SetGenCtrl(&gen_ctrl_S123_MatAdd_16x20x80, "OUTPUT_DATASIZE", AT_OPT_VAL(-1));
    CNN_SetGenCtrl(&gen_ctrl_S123_MatAdd_16x20x80, "INPUT_DATASIZE", AT_OPT_VAL(-1));
    
    // generator for _backbone_features_0_features_0_9_Add
    CNN_MatAddAct_SQ8("S123_MatAdd_16x20x80", &gen_ctrl_S123_MatAdd_16x20x80, 16, 20, 80, KOP_MATADD, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S126_Conv2d_184x80x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S126_Conv2d_184x80x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S126_Conv2d_184x80x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_7");
    // generator for _backbone_features_0_features_0_10_block_block_0_block_0_0_Conv_fusion
    CNN_ConvolutionNE16("S126_Conv2d_184x80x1x1_Custom", &gen_ctrl_S126_Conv2d_184x80x1x1_Custom,
                        -1, -1, 4, 1, 8,
                        80, 184, 20, 16,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 140,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S129_Conv2d_184x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S129_Conv2d_184x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S129_Conv2d_184x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_8");
    // generator for _backbone_features_0_features_0_10_block_block_1_block_1_0_Conv_fusion
    CNN_ConvolutionNE16("S129_Conv2d_184x1x3x3_Custom", &gen_ctrl_S129_Conv2d_184x1x3x3_Custom,
                        -1, -1, 4, 1, 8,
                        184, 184, 20, 16,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 128,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_CUSTOM);
    
    // generator for _backbone_features_0_features_0_10_block_block_2_block_2_0_Conv
    CNN_ConvolutionNE16("S132_Conv2d_80x184x1x1", 0,
                        -1, -1, 4, 1, 8,
                        184, 80, 20, 16,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 128,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S133_MatAdd_16x20x80;
    CNN_InitGenCtrl(&gen_ctrl_S133_MatAdd_16x20x80);
    CNN_SetGenCtrl(&gen_ctrl_S133_MatAdd_16x20x80, "OUTPUT_DATASIZE", AT_OPT_VAL(-1));
    CNN_SetGenCtrl(&gen_ctrl_S133_MatAdd_16x20x80, "INPUT_DATASIZE", AT_OPT_VAL(-1));
    
    // generator for _backbone_features_0_features_0_10_Add
    CNN_MatAddAct_SQ8("S133_MatAdd_16x20x80", &gen_ctrl_S133_MatAdd_16x20x80, 16, 20, 80, KOP_MATADD, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S136_Conv2d_480x80x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S136_Conv2d_480x80x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S136_Conv2d_480x80x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_9");
    // generator for _backbone_features_0_features_0_11_block_block_0_block_0_0_Conv_fusion
    CNN_ConvolutionNE16("S136_Conv2d_480x80x1x1_Custom", &gen_ctrl_S136_Conv2d_480x80x1x1_Custom,
                        -1, -1, 4, 1, 8,
                        80, 480, 20, 16,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 135,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S139_Conv2d_480x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S139_Conv2d_480x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S139_Conv2d_480x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_10");
    // generator for _backbone_features_0_features_0_11_block_block_1_block_1_0_Conv_fusion
    CNN_ConvolutionNE16("S139_Conv2d_480x1x3x3_Custom", &gen_ctrl_S139_Conv2d_480x1x3x3_Custom,
                        -1, 1, 4, 1, 8,
                        480, 480, 20, 16,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 128,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_CUSTOM);
    
    
    // generator for _backbone_features_0_features_0_11_block_block_2_avgpool_GlobalAveragePool_trans Transpose 16x20x480 -> 480x16x20 ((1, 0))
    CNN_MatTranspose("S140_Op__backbone_features_0_features_0_11_block_block_2_avgpool_GlobalAveragePool_trans", 0, 1,
                      1, 480, 320, KOP_MATTRANSP);
    
    // generator for _backbone_features_0_features_0_11_block_block_2_avgpool_GlobalAveragePool
    CNN_GlobalPoolAct_SQ8("S141_Op__backbone_features_0_features_0_11_block_block_2_avgpool_GlobalAveragePool", 0,
                          480, 8, 40,
                          KOP_GLOBAL_AVGPOOL, KOP_NONE);
    
    
    // generator for _backbone_features_0_features_0_11_block_block_2_fc1_Conv_fusion_qin0
    CNN_Convert("S143_Op__backbone_features_0_features_0_11_block_block_2_fc1_Conv_fusion_qin0", 1, -1, 480, KOP_CONVERT_FP_FP_SCALE);
    
    // generator for _backbone_features_0_features_0_11_block_block_2_fc1_Conv_fusion
    CNN_ConvolutionNE16("S146_Conv2d_120x480x1x1_Relu", 0,
                        -1, -1, 4, 1, 8,
                        480, 120, 1, 1,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 70,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    // generator for _backbone_features_0_features_0_11_block_block_2_fc2_Conv_fusion
    CNN_ConvolutionNE16("S149_Conv2d_480x120x1x1_Hsigmoid", 0,
                        -1, 1, 4, 1, 8,
                        120, 480, 1, 1,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_HSIGMOID);
    
    
    // generator for expr_11
    s150_kernel_gen("S150_Op_expr_11");
    
    // generator for _backbone_features_0_features_0_11_block_block_3_block_3_0_Conv
    CNN_ConvolutionNE16("S153_Conv2d_112x480x1x1", 0,
                        -1, -1, 4, 1, 8,
                        480, 112, 20, 16,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 45,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S156_Conv2d_672x112x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S156_Conv2d_672x112x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S156_Conv2d_672x112x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_12");
    // generator for _backbone_features_0_features_0_12_block_block_0_block_0_0_Conv_fusion
    CNN_ConvolutionNE16("S156_Conv2d_672x112x1x1_Custom", &gen_ctrl_S156_Conv2d_672x112x1x1_Custom,
                        -1, -1, 4, 1, 8,
                        112, 672, 20, 16,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 123,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S159_Conv2d_672x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S159_Conv2d_672x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S159_Conv2d_672x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_13");
    // generator for _backbone_features_0_features_0_12_block_block_1_block_1_0_Conv_fusion
    CNN_ConvolutionNE16("S159_Conv2d_672x1x3x3_Custom", &gen_ctrl_S159_Conv2d_672x1x3x3_Custom,
                        -1, 1, 4, 1, 8,
                        672, 672, 20, 16,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 128,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_CUSTOM);
    
    
    // generator for _backbone_features_0_features_0_12_block_block_2_avgpool_GlobalAveragePool_trans Transpose 16x20x672 -> 672x16x20 ((1, 0))
    CNN_MatTranspose("S160_Op__backbone_features_0_features_0_12_block_block_2_avgpool_GlobalAveragePool_trans", 0, 1,
                      1, 672, 320, KOP_MATTRANSP);
    
    // generator for _backbone_features_0_features_0_12_block_block_2_avgpool_GlobalAveragePool
    CNN_GlobalPoolAct_SQ8("S161_Op__backbone_features_0_features_0_12_block_block_2_avgpool_GlobalAveragePool", 0,
                          672, 8, 40,
                          KOP_GLOBAL_AVGPOOL, KOP_NONE);
    
    
    // generator for _backbone_features_0_features_0_12_block_block_2_fc1_Conv_fusion_qin0
    CNN_Convert("S163_Op__backbone_features_0_features_0_12_block_block_2_fc1_Conv_fusion_qin0", 1, -1, 672, KOP_CONVERT_FP_FP_SCALE);
    
    // generator for _backbone_features_0_features_0_12_block_block_2_fc1_Conv_fusion
    CNN_ConvolutionNE16("S166_Conv2d_168x672x1x1_Relu", 0,
                        -1, -1, 4, 1, 8,
                        672, 168, 1, 1,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 75,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    // generator for _backbone_features_0_features_0_12_block_block_2_fc2_Conv_fusion
    CNN_ConvolutionNE16("S169_Conv2d_672x168x1x1_Hsigmoid", 0,
                        -1, 1, 4, 1, 8,
                        168, 672, 1, 1,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_HSIGMOID);
    
    
    // generator for expr_14
    s170_kernel_gen("S170_Op_expr_14");
    
    // generator for _backbone_features_0_features_0_12_block_block_3_block_3_0_Conv
    CNN_ConvolutionNE16("S173_Conv2d_112x672x1x1", 0,
                        -1, -1, 4, 1, 8,
                        672, 112, 20, 16,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 105,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S174_MatAdd_16x20x112;
    CNN_InitGenCtrl(&gen_ctrl_S174_MatAdd_16x20x112);
    CNN_SetGenCtrl(&gen_ctrl_S174_MatAdd_16x20x112, "OUTPUT_DATASIZE", AT_OPT_VAL(-1));
    CNN_SetGenCtrl(&gen_ctrl_S174_MatAdd_16x20x112, "INPUT_DATASIZE", AT_OPT_VAL(-1));
    
    // generator for _backbone_features_0_features_0_12_Add
    CNN_MatAddAct_SQ8("S174_MatAdd_16x20x112", &gen_ctrl_S174_MatAdd_16x20x112, 16, 20, 112, KOP_MATADD, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S177_Conv2d_672x112x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S177_Conv2d_672x112x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S177_Conv2d_672x112x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_15");
    // generator for _backbone_features_0_features_0_13_features_0_13_0_Conv_fusion
    CNN_ConvolutionNE16("S177_Conv2d_672x112x1x1_Custom", &gen_ctrl_S177_Conv2d_672x112x1x1_Custom,
                        -1, 1, 4, 1, 8,
                        112, 672, 20, 16,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 127,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_CUSTOM);
    
    
    // generator for _head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_fusion_qin0
    CNN_Convert("S178_Op__head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_fusion_qin0", 1, -1, 215040, KOP_CONVERT_FP_FP_ZEROPOINT);
    
    
    // generator for _head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_fusion_qin0
    CNN_Convert("S179_Op__head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_fusion_qin0", 1, -1, 215040, KOP_CONVERT_FP_FP_ZEROPOINT);
    
    CNN_GenControl_T gen_ctrl_S182_Conv2d_672x5x5x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S182_Conv2d_672x5x5x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S182_Conv2d_672x5x5x1_Custom, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S182_Conv2d_672x5x5x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_25");
    // generator for _backbone_features_1_features_1_0_features_1_0_1_features_1_0_1_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S182_Conv2d_672x5x5x1_Custom", &gen_ctrl_S182_Conv2d_672x5x5x1_Custom,
                               4, 1,
                               672, 672, 20, 16,
                               KOP_CONV_DW, 5, 5, 1, 1, 2, 2, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    
    // generator for _backbone_features_1_features_1_0_features_1_0_2_avgpool_GlobalAveragePool_trans Transpose 8x10x672 -> 672x8x10 ((1, 0))
    CNN_MatTranspose("S183_Op__backbone_features_1_features_1_0_features_1_0_2_avgpool_GlobalAveragePool_trans", 0, 1,
                      1, 672, 80, KOP_MATTRANSP);
    
    // generator for _backbone_features_1_features_1_0_features_1_0_2_avgpool_GlobalAveragePool
    CNN_GlobalPoolAct_SQ8("S184_Op__backbone_features_1_features_1_0_features_1_0_2_avgpool_GlobalAveragePool", 0,
                          672, 4, 20,
                          KOP_GLOBAL_AVGPOOL, KOP_NONE);
    
    
    // generator for _backbone_features_1_features_1_0_features_1_0_2_fc1_Conv_fusion_qin0
    CNN_Convert("S186_Op__backbone_features_1_features_1_0_features_1_0_2_fc1_Conv_fusion_qin0", 1, -1, 672, KOP_CONVERT_FP_FP_SCALE);
    
    // generator for _backbone_features_1_features_1_0_features_1_0_2_fc1_Conv_fusion
    CNN_ConvolutionNE16("S189_Conv2d_168x672x1x1_Relu", 0,
                        -1, -1, 4, 1, 8,
                        672, 168, 1, 1,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 75,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    // generator for _backbone_features_1_features_1_0_features_1_0_2_fc2_Conv_fusion
    CNN_ConvolutionNE16("S192_Conv2d_672x168x1x1_Hsigmoid", 0,
                        -1, 1, 4, 1, 8,
                        168, 672, 1, 1,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_HSIGMOID);
    
    
    // generator for expr_26
    s193_kernel_gen("S193_Op_expr_26");
    
    // generator for _backbone_features_1_features_1_0_features_1_0_3_features_1_0_3_0_Conv
    CNN_ConvolutionNE16("S196_Conv2d_80x672x1x1", 0,
                        -1, -1, 4, 1, 8,
                        672, 80, 10, 8,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 72,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S199_Conv2d_480x80x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S199_Conv2d_480x80x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S199_Conv2d_480x80x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_27");
    // generator for _backbone_features_1_features_1_1_block_block_0_block_0_0_Conv_fusion
    CNN_ConvolutionNE16("S199_Conv2d_480x80x1x1_Custom", &gen_ctrl_S199_Conv2d_480x80x1x1_Custom,
                        -1, 1, 4, 1, 8,
                        80, 480, 10, 8,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 128,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S202_Conv2d_480x5x5x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S202_Conv2d_480x5x5x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S202_Conv2d_480x5x5x1_Custom, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S202_Conv2d_480x5x5x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_28");
    // generator for _backbone_features_1_features_1_1_block_block_1_block_1_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S202_Conv2d_480x5x5x1_Custom", &gen_ctrl_S202_Conv2d_480x5x5x1_Custom,
                               4, 1,
                               480, 480, 10, 8,
                               KOP_CONV_DW, 5, 5, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    
    // generator for _backbone_features_1_features_1_1_block_block_2_avgpool_GlobalAveragePool_trans Transpose 8x10x480 -> 480x8x10 ((1, 0))
    CNN_MatTranspose("S203_Op__backbone_features_1_features_1_1_block_block_2_avgpool_GlobalAveragePool_trans", 0, 1,
                      1, 480, 80, KOP_MATTRANSP);
    
    // generator for _backbone_features_1_features_1_1_block_block_2_avgpool_GlobalAveragePool
    CNN_GlobalPoolAct_SQ8("S204_Op__backbone_features_1_features_1_1_block_block_2_avgpool_GlobalAveragePool", 0,
                          480, 4, 20,
                          KOP_GLOBAL_AVGPOOL, KOP_NONE);
    
    
    // generator for _backbone_features_1_features_1_1_block_block_2_fc1_Conv_fusion_qin0
    CNN_Convert("S206_Op__backbone_features_1_features_1_1_block_block_2_fc1_Conv_fusion_qin0", 1, -1, 480, KOP_CONVERT_FP_FP_SCALE);
    
    // generator for _backbone_features_1_features_1_1_block_block_2_fc1_Conv_fusion
    CNN_ConvolutionNE16("S209_Conv2d_120x480x1x1_Relu", 0,
                        -1, -1, 4, 1, 8,
                        480, 120, 1, 1,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 183,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    // generator for _backbone_features_1_features_1_1_block_block_2_fc2_Conv_fusion
    CNN_ConvolutionNE16("S212_Conv2d_480x120x1x1_Hsigmoid", 0,
                        -1, 1, 4, 1, 8,
                        120, 480, 1, 1,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_HSIGMOID);
    
    
    // generator for expr_29
    s213_kernel_gen("S213_Op_expr_29");
    
    // generator for _backbone_features_1_features_1_1_block_block_3_block_3_0_Conv
    CNN_ConvolutionNE16("S216_Conv2d_80x480x1x1", 0,
                        -1, -1, 4, 1, 8,
                        480, 80, 10, 8,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 82,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S217_MatAdd_8x10x80;
    CNN_InitGenCtrl(&gen_ctrl_S217_MatAdd_8x10x80);
    CNN_SetGenCtrl(&gen_ctrl_S217_MatAdd_8x10x80, "OUTPUT_DATASIZE", AT_OPT_VAL(-1));
    CNN_SetGenCtrl(&gen_ctrl_S217_MatAdd_8x10x80, "INPUT_DATASIZE", AT_OPT_VAL(-1));
    
    // generator for _backbone_features_1_features_1_1_Add
    CNN_MatAddAct_SQ8("S217_MatAdd_8x10x80", &gen_ctrl_S217_MatAdd_8x10x80, 8, 10, 80, KOP_MATADD, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S220_Conv2d_480x80x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S220_Conv2d_480x80x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S220_Conv2d_480x80x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_30");
    // generator for _backbone_features_1_features_1_2_block_block_0_block_0_0_Conv_fusion
    CNN_ConvolutionNE16("S220_Conv2d_480x80x1x1_Custom", &gen_ctrl_S220_Conv2d_480x80x1x1_Custom,
                        -1, 1, 4, 1, 8,
                        80, 480, 10, 8,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 124,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S223_Conv2d_480x5x5x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S223_Conv2d_480x5x5x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S223_Conv2d_480x5x5x1_Custom, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S223_Conv2d_480x5x5x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_31");
    // generator for _backbone_features_1_features_1_2_block_block_1_block_1_0_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("S223_Conv2d_480x5x5x1_Custom", &gen_ctrl_S223_Conv2d_480x5x5x1_Custom,
                               4, 1,
                               480, 480, 10, 8,
                               KOP_CONV_DW, 5, 5, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    
    // generator for _backbone_features_1_features_1_2_block_block_2_avgpool_GlobalAveragePool_trans Transpose 8x10x480 -> 480x8x10 ((1, 0))
    CNN_MatTranspose("S224_Op__backbone_features_1_features_1_2_block_block_2_avgpool_GlobalAveragePool_trans", 0, 1,
                      1, 480, 80, KOP_MATTRANSP);
    
    // generator for _backbone_features_1_features_1_2_block_block_2_avgpool_GlobalAveragePool
    CNN_GlobalPoolAct_SQ8("S225_Op__backbone_features_1_features_1_2_block_block_2_avgpool_GlobalAveragePool", 0,
                          480, 4, 20,
                          KOP_GLOBAL_AVGPOOL, KOP_NONE);
    
    
    // generator for _backbone_features_1_features_1_2_block_block_2_fc1_Conv_fusion_qin0
    CNN_Convert("S227_Op__backbone_features_1_features_1_2_block_block_2_fc1_Conv_fusion_qin0", 1, -1, 480, KOP_CONVERT_FP_FP_SCALE);
    
    // generator for _backbone_features_1_features_1_2_block_block_2_fc1_Conv_fusion
    CNN_ConvolutionNE16("S230_Conv2d_120x480x1x1_Relu", 0,
                        -1, -1, 4, 1, 8,
                        480, 120, 1, 1,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 161,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    // generator for _backbone_features_1_features_1_2_block_block_2_fc2_Conv_fusion
    CNN_ConvolutionNE16("S233_Conv2d_480x120x1x1_Hsigmoid", 0,
                        -1, 1, 4, 1, 8,
                        120, 480, 1, 1,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_HSIGMOID);
    
    
    // generator for expr_32
    s234_kernel_gen("S234_Op_expr_32");
    
    // generator for _backbone_features_1_features_1_2_block_block_3_block_3_0_Conv
    CNN_ConvolutionNE16("S237_Conv2d_80x480x1x1", 0,
                        -1, -1, 4, 1, 8,
                        480, 80, 10, 8,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 96,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S238_MatAdd_8x10x80;
    CNN_InitGenCtrl(&gen_ctrl_S238_MatAdd_8x10x80);
    CNN_SetGenCtrl(&gen_ctrl_S238_MatAdd_8x10x80, "OUTPUT_DATASIZE", AT_OPT_VAL(-1));
    CNN_SetGenCtrl(&gen_ctrl_S238_MatAdd_8x10x80, "INPUT_DATASIZE", AT_OPT_VAL(-1));
    
    // generator for _backbone_features_1_features_1_2_Add
    CNN_MatAddAct_SQ8("S238_MatAdd_8x10x80", &gen_ctrl_S238_MatAdd_8x10x80, 8, 10, 80, KOP_MATADD, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S241_Conv2d_480x80x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S241_Conv2d_480x80x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S241_Conv2d_480x80x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_33");
    // generator for _backbone_features_1_features_1_3_features_1_3_0_Conv_fusion
    CNN_ConvolutionNE16("S241_Conv2d_480x80x1x1_Custom", &gen_ctrl_S241_Conv2d_480x80x1x1_Custom,
                        -1, -1, 4, 1, 8,
                        80, 480, 10, 8,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 123,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_CUSTOM);
    
    // generator for _backbone_extra_0_extra_0_0_extra_0_0_0_Conv_fusion
    CNN_ConvolutionNE16("S244_Conv2d_256x480x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        480, 256, 10, 8,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 128,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for _backbone_extra_0_extra_0_1_extra_0_1_0_Conv_fusion
    CNN_ConvolutionNE16("S247_Conv2d_256x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        256, 256, 10, 8,
                        KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for _backbone_extra_0_extra_0_2_extra_0_2_0_Conv_fusion
    CNN_ConvolutionNE16("S250_Conv2d_512x256x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        256, 512, 5, 4,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for _backbone_extra_1_extra_1_0_extra_1_0_0_Conv_fusion
    CNN_ConvolutionNE16("S253_Conv2d_128x512x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        512, 128, 5, 4,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for _backbone_extra_1_extra_1_1_extra_1_1_0_Conv_fusion
    CNN_ConvolutionNE16("S256_Conv2d_128x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 128, 5, 4,
                        KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for _backbone_extra_1_extra_1_2_extra_1_2_0_Conv_fusion
    CNN_ConvolutionNE16("S259_Conv2d_256x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 256, 3, 2,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for _backbone_extra_2_extra_2_0_extra_2_0_0_Conv_fusion
    CNN_ConvolutionNE16("S262_Conv2d_128x256x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        256, 128, 3, 2,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for _backbone_extra_2_extra_2_1_extra_2_1_0_Conv_fusion
    CNN_ConvolutionNE16("S265_Conv2d_128x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 128, 3, 2,
                        KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for _backbone_extra_2_extra_2_2_extra_2_2_0_Conv_fusion
    CNN_ConvolutionNE16("S268_Conv2d_256x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 256, 2, 1,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for _backbone_extra_3_extra_3_0_extra_3_0_0_Conv_fusion
    CNN_ConvolutionNE16("S271_Conv2d_64x256x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        256, 64, 2, 1,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for _backbone_extra_3_extra_3_1_extra_3_1_0_Conv_fusion
    CNN_ConvolutionNE16("S274_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 2, 1,
                        KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for _backbone_extra_3_extra_3_2_extra_3_2_0_Conv_fusion
    CNN_ConvolutionNE16("S277_Conv2d_128x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 128, 1, 1,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for _head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_fusion
    CNN_ConvolutionNE16("S280_Conv2d_672x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        672, 672, 20, 16,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 128,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S283_Conv2d_24x672x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S283_Conv2d_24x672x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S283_Conv2d_24x672x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_2");
    // generator for _head_regression_head_module_list_0_module_list_0_1_Conv_fusion
    CNN_ConvolutionNE16("S283_Conv2d_24x672x1x1_Custom", &gen_ctrl_S283_Conv2d_24x672x1x1_Custom,
                        -1, -1, 4, 1, 8,
                        672, 24, 20, 16,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_CUSTOM);
    
    // generator for _head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_fusion
    CNN_ConvolutionNE16("S287_Conv2d_480x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        480, 480, 10, 8,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 128,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for _head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_fusion
    CNN_ConvolutionNE16("S291_Conv2d_512x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        512, 512, 5, 4,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for _head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_fusion
    CNN_ConvolutionNE16("S295_Conv2d_256x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        256, 256, 3, 2,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for _head_regression_head_module_list_5_module_list_5_0_module_list_5_0_0_Conv_fusion
    CNN_ConvolutionNE16("S301_Conv2d_128x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 128, 1, 1,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for _head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_fusion
    CNN_ConvolutionNE16("S321_Conv2d_672x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        672, 672, 20, 16,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 128,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for _head_classification_head_module_list_0_module_list_0_1_Conv
    CNN_ConvolutionNE16("S322_Conv2d_186x672x1x1", 0,
                        -1, 1, 4, 1, 8,
                        672, 186, 20, 16,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S325_Conv2d_24x480x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S325_Conv2d_24x480x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S325_Conv2d_24x480x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_2_1");
    // generator for _head_regression_head_module_list_1_module_list_1_1_Conv_fusion
    CNN_ConvolutionNE16("S325_Conv2d_24x480x1x1_Custom", &gen_ctrl_S325_Conv2d_24x480x1x1_Custom,
                        -1, -1, 4, 1, 8,
                        480, 24, 10, 8,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S328_Conv2d_24x512x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S328_Conv2d_24x512x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S328_Conv2d_24x512x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_2_2");
    // generator for _head_regression_head_module_list_2_module_list_2_1_Conv_fusion
    CNN_ConvolutionNE16("S328_Conv2d_24x512x1x1_Custom", &gen_ctrl_S328_Conv2d_24x512x1x1_Custom,
                        -1, -1, 4, 1, 8,
                        512, 24, 5, 4,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S331_Conv2d_24x256x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S331_Conv2d_24x256x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S331_Conv2d_24x256x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_2_3");
    // generator for _head_regression_head_module_list_3_module_list_3_1_Conv_fusion
    CNN_ConvolutionNE16("S331_Conv2d_24x256x1x1_Custom", &gen_ctrl_S331_Conv2d_24x256x1x1_Custom,
                        -1, -1, 4, 1, 8,
                        256, 24, 3, 2,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S335_Conv2d_24x128x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S335_Conv2d_24x128x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S335_Conv2d_24x128x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_2_5");
    // generator for _head_regression_head_module_list_5_module_list_5_1_Conv_fusion
    CNN_ConvolutionNE16("S335_Conv2d_24x128x1x1_Custom", &gen_ctrl_S335_Conv2d_24x128x1x1_Custom,
                        -1, -1, 4, 1, 8,
                        128, 24, 1, 1,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_CUSTOM);
    
    // generator for _head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_fusion
    CNN_ConvolutionNE16("S338_Conv2d_480x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        480, 480, 10, 8,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 128,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for _head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_fusion
    CNN_ConvolutionNE16("S340_Conv2d_512x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        512, 512, 5, 4,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for _head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_fusion
    CNN_ConvolutionNE16("S342_Conv2d_256x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        256, 256, 2, 1,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S343_Conv2d_24x256x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S343_Conv2d_24x256x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S343_Conv2d_24x256x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_2_4");
    // generator for _head_regression_head_module_list_4_module_list_4_1_Conv_fusion
    CNN_ConvolutionNE16("S343_Conv2d_24x256x1x1_Custom", &gen_ctrl_S343_Conv2d_24x256x1x1_Custom,
                        -1, -1, 4, 1, 8,
                        256, 24, 2, 1,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_CUSTOM);
    
    
    // generator for _Split
    CNN_SplitLastAxis_Generator("S346_Op__Split", 0, -1, 2574, 1, 1, 1, 1, KOP_SPLIT);
    
    // generator for _head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_fusion
    CNN_ConvolutionNE16("S352_Conv2d_256x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        256, 256, 3, 2,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for _head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_fusion
    CNN_ConvolutionNE16("S354_Conv2d_256x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        256, 256, 2, 1,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for _head_classification_head_module_list_5_module_list_5_0_module_list_5_0_0_Conv_fusion
    CNN_ConvolutionNE16("S356_Conv2d_128x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 128, 1, 1,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for _head_classification_head_module_list_1_module_list_1_1_Conv
    CNN_ConvolutionNE16("S358_Conv2d_186x480x1x1", 0,
                        -1, 1, 4, 1, 8,
                        480, 186, 10, 8,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    // generator for _head_classification_head_module_list_2_module_list_2_1_Conv
    CNN_ConvolutionNE16("S361_Conv2d_186x512x1x1", 0,
                        -1, 1, 4, 1, 8,
                        512, 186, 5, 4,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    // generator for _head_classification_head_module_list_3_module_list_3_1_Conv
    CNN_ConvolutionNE16("S364_Conv2d_186x256x1x1", 0,
                        -1, 1, 4, 1, 8,
                        256, 186, 3, 2,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    // generator for _head_classification_head_module_list_4_module_list_4_1_Conv
    CNN_ConvolutionNE16("S367_Conv2d_186x256x1x1", 0,
                        -1, 1, 4, 1, 8,
                        256, 186, 2, 1,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    // generator for _head_classification_head_module_list_5_module_list_5_1_Conv
    CNN_ConvolutionNE16("S370_Conv2d_186x128x1x1", 0,
                        -1, 1, 4, 1, 8,
                        128, 186, 1, 1,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    // generator for _Softmax
    CNN_SoftMax2D_SQ8("S373_SoftMax", 0, 2574, 31, KOP_SOFTMAX);
    
    
    // generator for expr_1
    s376_kernel_gen("S376_Op_expr_1");
    
    
    // generator for expr_0
    s382_kernel_gen("S382_Op_expr_0");
    

#define GRAPH
#ifdef GRAPH
    CreateGraph("Mobv3SSDLiteCNN",
        /* Arguments either passed or globals */
            CArgs(534,
                TCArgInfo("unsigned char * __restrict__", "Input_1", ARG_SCOPE_ARG_ALLOC, ARG_DIR_IN, AT_MEM_L2, AT_MEM_L2, 0),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_0_features_0_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_0_features_0_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1387", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1387.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S3_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S3_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S3_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S3_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.08766 out: 0.08433  BIASN: 0 PRENORM: 0 NE16_PADVAL: [255] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S3_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S3_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_1_block_block_0_block_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_1_block_block_0_block_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1390", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1390.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S6_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S6_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S6_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S6_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01608 out: 0.01608  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [128] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S6_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S6_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_1_block_block_1_block_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_1_block_block_1_block_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1393", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1393.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S9_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S9_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S9_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S9_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S9_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S9_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [129] IN1SCALEN: [6] OUTSCALE: [85] OUTSCALEN: [7] ADD_BIAS: [-192]
                TCArgInfo("signed char * __restrict__", "S10_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S10_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_2_block_block_0_block_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_2_block_block_0_block_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1396", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1396.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S13_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S13_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S13_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S13_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00994 out: 0.00994  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [60] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S13_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S13_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_2_block_block_1_block_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_2_block_block_1_block_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1399", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1399.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S16_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S16_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S16_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S16_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01776 out: 0.01776  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S16_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S16_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_2_block_block_2_block_2_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_2_block_block_2_block_2_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1402", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1402.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S19_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S19_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S19_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S19_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S19_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S19_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_3_block_block_0_block_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_3_block_block_0_block_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1405", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1405.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S22_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S22_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S22_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S22_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00664 out: 0.00664  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [137] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S22_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S22_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_3_block_block_1_block_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_3_block_block_1_block_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1408", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1408.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S25_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S25_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S25_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S25_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01167 out: 0.01167  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S25_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S25_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_3_block_block_2_block_2_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_3_block_block_2_block_2_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1411", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1411.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S28_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S28_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S28_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S28_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S28_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S28_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [65] IN1SCALEN: [6] OUTSCALE: [41] OUTSCALEN: [6] ADD_BIAS: [-3]
                TCArgInfo("signed char * __restrict__", "S29_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S29_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_4_block_block_0_block_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_4_block_block_0_block_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1414", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1414.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S32_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S32_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S32_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S32_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00988 out: 0.00988  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [161] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S32_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S32_Infos.tensor", 1, 1, 8, 0)),
                // in q: 0.00<(u8-0.00)*0.00987938<2.52 out_q: -2.54<(i8-0.00)*0.01983655<2.52 forced
                TCArgInfo("signed char * __restrict__", "S33_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S33_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_4_block_block_1_block_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_4_block_block_1_block_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1417", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1417.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S36_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S36_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S36_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S36_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.09209 out: 0.09209  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S36_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S36_Infos.tensor", 1, 1, 8, 0)),
                // no activation ACTSCALE: 0 ACTSCALEN: 0 GLOBAL_SUM_SCALE: [21] GLOBAL_SUM_SCALEN: [10]
                TCArgInfo("signed char * __restrict__", "S38_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S38_Infos.tensor", 1, 1, 8, 0)),
                // in q: -4.50<(i8-0.00)*0.03516809<4.47 out_q: 0.00<(u8-0.00)*0.01751509<4.47
                TCArgInfo("signed char * __restrict__", "S40_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S40_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_4_block_block_2_fc1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_4_block_block_2_fc1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_backbone_features_0_4_block_2_fc1_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_backbone_features_0_4_block_2_fc1_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S43_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S43_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S43_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S43_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00801 out: 0.00801  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S43_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S43_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_4_block_block_2_fc2_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_4_block_block_2_fc2_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_backbone_features_0_4_block_2_fc2_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_backbone_features_0_4_block_2_fc2_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S46_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S46_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S46_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S46_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.00787  actscale: [1] actscalen: [0] a0: [127] b0: [64] c0: [0] BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S46_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S46_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_4_block_block_3_block_3_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_4_block_block_3_block_3_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1420", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1420.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S50_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S50_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S50_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S50_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S50_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S50_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_5_block_block_0_block_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_5_block_block_0_block_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1423", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1423.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S53_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S53_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S53_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S53_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00522 out: 0.00522  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [162] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S53_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S53_Infos.tensor", 1, 1, 8, 0)),
                // in q: 0.00<(u8-0.00)*0.00521777<1.33 out_q: -1.34<(i8-0.00)*0.01047662<1.33 forced
                TCArgInfo("signed char * __restrict__", "S54_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S54_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_5_block_block_1_block_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_5_block_block_1_block_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1426", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1426.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S57_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S57_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S57_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S57_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02622 out: 0.02622  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S57_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S57_Infos.tensor", 1, 1, 8, 0)),
                // no activation ACTSCALE: 0 ACTSCALEN: 0 GLOBAL_SUM_SCALE: [9] GLOBAL_SUM_SCALEN: [9]
                TCArgInfo("signed char * __restrict__", "S59_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S59_Infos.tensor", 1, 1, 8, 0)),
                // in q: -1.49<(i8-0.00)*0.01167131<1.48 out_q: 0.00<(u8-0.00)*0.00581277<1.48
                TCArgInfo("signed char * __restrict__", "S61_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S61_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_5_block_block_2_fc1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_5_block_block_2_fc1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_backbone_features_0_5_block_2_fc1_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_backbone_features_0_5_block_2_fc1_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S64_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S64_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S64_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S64_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00692 out: 0.00692  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S64_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S64_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_5_block_block_2_fc2_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_5_block_block_2_fc2_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_backbone_features_0_5_block_2_fc2_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_backbone_features_0_5_block_2_fc2_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S67_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S67_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S67_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S67_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.00787  actscale: [1] actscalen: [0] a0: [127] b0: [64] c0: [0] BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S67_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S67_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_5_block_block_3_block_3_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_5_block_block_3_block_3_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1429", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1429.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S71_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S71_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S71_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S71_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S71_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S71_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [179] IN1SCALEN: [7] OUTSCALE: [169] OUTSCALEN: [8] ADD_BIAS: [-110]
                TCArgInfo("signed char * __restrict__", "S72_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S72_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_6_block_block_0_block_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_6_block_block_0_block_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1432", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1432.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S75_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S75_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S75_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S75_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00446 out: 0.00446  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [132] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S75_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S75_Infos.tensor", 1, 1, 8, 0)),
                // in q: 0.00<(u8-0.00)*0.00445911<1.14 out_q: -1.15<(i8-0.00)*0.00895333<1.14 forced
                TCArgInfo("signed char * __restrict__", "S76_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S76_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_0_features_0_6_block_block_1_block_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_6_block_block_1_block_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1435", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1435.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S79_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S79_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S79_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S79_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03002 out: 0.03002  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S79_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S79_Infos.tensor", 1, 1, 8, 0)),
                // no activation ACTSCALE: 0 ACTSCALEN: 0 GLOBAL_SUM_SCALE: [119] GLOBAL_SUM_SCALEN: [13]
                TCArgInfo("signed char * __restrict__", "S81_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S81_Infos.tensor", 1, 1, 8, 0)),
                // in q: -2.07<(i8-0.00)*0.01617177<2.05 out_q: 0.00<(u8-0.00)*0.00805418<2.05
                TCArgInfo("signed char * __restrict__", "S83_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S83_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_6_block_block_2_fc1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_6_block_block_2_fc1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_backbone_features_0_6_block_2_fc1_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_backbone_features_0_6_block_2_fc1_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S86_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S86_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S86_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S86_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00809 out: 0.00809  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S86_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S86_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_6_block_block_2_fc2_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_6_block_block_2_fc2_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_backbone_features_0_6_block_2_fc2_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_backbone_features_0_6_block_2_fc2_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S89_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S89_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S89_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S89_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.00787  actscale: [1] actscalen: [0] a0: [127] b0: [64] c0: [0] BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S89_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S89_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_6_block_block_3_block_3_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_6_block_block_3_block_3_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1438", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1438.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S93_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S93_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S93_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S93_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S93_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S93_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [47] IN1SCALEN: [5] OUTSCALE: [139] OUTSCALEN: [8] ADD_BIAS: [-58]
                TCArgInfo("signed char * __restrict__", "S94_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S94_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_7_block_block_0_block_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_7_block_block_0_block_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1441", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1441.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S97_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S97_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S97_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S97_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04929 out: 0.04143  BIASN: 0 PRENORM: 0 NE16_PADVAL: [115] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S97_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S97_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_7_block_block_1_block_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_7_block_block_1_block_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1444", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1444.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S100_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S100_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S100_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S100_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03240 out: 0.03843  BIASN: 0 PRENORM: 0 NE16_PADVAL: [128] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S100_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S100_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_7_block_block_2_block_2_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_7_block_block_2_block_2_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1447", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1447.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S103_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S103_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S103_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S103_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0 NE16_PADVAL: [128] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S103_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S103_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_8_block_block_0_block_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_8_block_block_0_block_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1450", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1450.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S106_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S106_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S106_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S106_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03914 out: 0.05710  BIASN: 0 PRENORM: 0 NE16_PADVAL: [146] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S106_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S106_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_8_block_block_1_block_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_8_block_block_1_block_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1453", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1453.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S109_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S109_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S109_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S109_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02393 out: 0.01573  BIASN: 0 PRENORM: 0 NE16_PADVAL: [128] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S109_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S109_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_8_block_block_2_block_2_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_8_block_block_2_block_2_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1456", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1456.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S112_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S112_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S112_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S112_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0 NE16_PADVAL: [128] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S112_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S112_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [101] IN1SCALEN: [6] OUTSCALE: [147] OUTSCALEN: [8] ADD_BIAS: [-63]
                TCArgInfo("signed char * __restrict__", "S113_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S113_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_9_block_block_0_block_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_9_block_block_0_block_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1459", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1459.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S116_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S116_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S116_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S116_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02830 out: 0.03408  BIASN: 0 PRENORM: 0 NE16_PADVAL: [140] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S116_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S116_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_9_block_block_1_block_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_9_block_block_1_block_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1462", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1462.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S119_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S119_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S119_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S119_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02322 out: 0.02536  BIASN: 0 PRENORM: 0 NE16_PADVAL: [128] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S119_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S119_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_9_block_block_2_block_2_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_9_block_block_2_block_2_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1465", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1465.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S122_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S122_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S122_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S122_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0 NE16_PADVAL: [128] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S122_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S122_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [89] IN1SCALEN: [6] OUTSCALE: [85] OUTSCALEN: [7] ADD_BIAS: [-83]
                TCArgInfo("signed char * __restrict__", "S123_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S123_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_10_block_block_0_block_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_10_block_block_0_block_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1468", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1468.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S126_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S126_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S126_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S126_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02411 out: 0.02832  BIASN: 0 PRENORM: 0 NE16_PADVAL: [140] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S126_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S126_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_10_block_block_1_block_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_10_block_block_1_block_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1471", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1471.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S129_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S129_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S129_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S129_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02058 out: 0.02171  BIASN: 0 PRENORM: 0 NE16_PADVAL: [128] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S129_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S129_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_10_block_block_2_block_2_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_10_block_block_2_block_2_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1474", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1474.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S132_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S132_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S132_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S132_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0 NE16_PADVAL: [128] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S132_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S132_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [103] IN1SCALEN: [6] OUTSCALE: [153] OUTSCALEN: [8] ADD_BIAS: [-68]
                TCArgInfo("signed char * __restrict__", "S133_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S133_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_11_block_block_0_block_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_11_block_block_0_block_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1477", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1477.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S136_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S136_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S136_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S136_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02830 out: 0.03097  BIASN: 0 PRENORM: 0 NE16_PADVAL: [135] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S136_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S136_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_11_block_block_1_block_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_11_block_block_1_block_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1480", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1480.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S139_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S139_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S139_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S139_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04110 out: 0.03031  BIASN: 0 PRENORM: 0 NE16_PADVAL: [128] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S139_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S139_Infos.tensor", 1, 1, 8, 0)),
                // no activation ACTSCALE: 0 ACTSCALEN: 0 GLOBAL_SUM_SCALE: [129] GLOBAL_SUM_SCALEN: [12]
                TCArgInfo("signed char * __restrict__", "S141_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S141_Infos.tensor", 1, 1, 8, 0)),
                // in q: -0.96<(i8-0.00)*0.00752969<0.96 out_q: -0.36<(u8-70.00)*0.00515881<0.95
                TCArgInfo("signed char * __restrict__", "S143_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S143_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_11_block_block_2_fc1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_11_block_block_2_fc1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_backbone_features_0_11_block_2_fc1_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_backbone_features_0_11_block_2_fc1_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S146_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S146_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S146_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S146_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00959 out: 0.00959  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [70] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S146_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S146_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_11_block_block_2_fc2_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_11_block_block_2_fc2_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_backbone_features_0_11_block_2_fc2_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_backbone_features_0_11_block_2_fc2_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S149_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S149_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S149_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S149_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.00787  actscale: [1] actscalen: [0] a0: [127] b0: [64] c0: [0] BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S149_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S149_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_11_block_block_3_block_3_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_11_block_block_3_block_3_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1483", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1483.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S153_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S153_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S153_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S153_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0 NE16_PADVAL: [45] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S153_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S153_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_12_block_block_0_block_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_12_block_block_0_block_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1486", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1486.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S156_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S156_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S156_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S156_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01972 out: 0.01741  BIASN: 0 PRENORM: 0 NE16_PADVAL: [123] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S156_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S156_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_12_block_block_1_block_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_12_block_block_1_block_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1489", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1489.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S159_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S159_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S159_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S159_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02586 out: 0.02654  BIASN: 0 PRENORM: 0 NE16_PADVAL: [128] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S159_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S159_Infos.tensor", 1, 1, 8, 0)),
                // no activation ACTSCALE: 0 ACTSCALEN: 0 GLOBAL_SUM_SCALE: [239] GLOBAL_SUM_SCALEN: [13]
                TCArgInfo("signed char * __restrict__", "S161_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S161_Infos.tensor", 1, 1, 8, 0)),
                // in q: -0.91<(i8-0.00)*0.00710956<0.90 out_q: -0.38<(u8-75.00)*0.00500570<0.90
                TCArgInfo("signed char * __restrict__", "S163_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S163_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_12_block_block_2_fc1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_12_block_block_2_fc1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_backbone_features_0_12_block_2_fc1_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_backbone_features_0_12_block_2_fc1_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S166_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S166_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S166_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S166_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00879 out: 0.00879  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [75] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S166_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S166_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_12_block_block_2_fc2_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_12_block_block_2_fc2_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_backbone_features_0_12_block_2_fc2_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_backbone_features_0_12_block_2_fc2_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S169_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S169_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S169_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S169_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.00787  actscale: [1] actscalen: [0] a0: [127] b0: [64] c0: [0] BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S169_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S169_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_12_block_block_3_block_3_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_12_block_block_3_block_3_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1492", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1492.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S173_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S173_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S173_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S173_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0 NE16_PADVAL: [105] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S173_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S173_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [49] IN1SCALEN: [5] OUTSCALE: [77] OUTSCALEN: [7] ADD_BIAS: [-69]
                TCArgInfo("signed char * __restrict__", "S174_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S174_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_0_features_0_13_features_0_13_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_0_features_0_13_features_0_13_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1495", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1495.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S177_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S177_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S177_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S177_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01812 out: 0.01999  BIASN: 0 PRENORM: 0 NE16_PADVAL: [127] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S177_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S177_Infos.tensor", 1, 1, 8, 0)),
                // in q: -2.56<(i8-0.00)*0.01998533<2.54 forced out_q: -2.56<(u8-128.00)*0.01998533<2.54
                TCArgInfo("signed char * __restrict__", "S178_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S178_Infos.tensor", 1, 1, 8, 0)),
                // in q: -2.56<(i8-0.00)*0.01998533<2.54 forced out_q: -2.56<(u8-128.00)*0.01998533<2.54
                TCArgInfo("signed char * __restrict__", "S179_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S179_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_1_features_1_0_features_1_0_1_features_1_0_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_1_features_1_0_features_1_0_1_features_1_0_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1498", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1498.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S182_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S182_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S182_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S182_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01849 out: 0.02070  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S182_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S182_Infos.tensor", 1, 1, 8, 0)),
                // no activation ACTSCALE: 0 ACTSCALEN: 0 GLOBAL_SUM_SCALE: [191] GLOBAL_SUM_SCALEN: [13]
                TCArgInfo("signed char * __restrict__", "S184_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S184_Infos.tensor", 1, 1, 8, 0)),
                // in q: -0.89<(i8-0.00)*0.00694834<0.88 out_q: -0.37<(u8-75.00)*0.00490833<0.88
                TCArgInfo("signed char * __restrict__", "S186_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S186_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_1_features_1_0_features_1_0_2_fc1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_1_features_1_0_features_1_0_2_fc1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_backbone_features_1_0_2_fc1_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_backbone_features_1_0_2_fc1_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S189_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S189_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S189_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S189_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00469 out: 0.00469  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [75] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S189_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S189_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_1_features_1_0_features_1_0_2_fc2_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_1_features_1_0_features_1_0_2_fc2_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_backbone_features_1_0_2_fc2_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_backbone_features_1_0_2_fc2_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S192_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S192_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S192_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S192_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.00787  actscale: [1] actscalen: [0] a0: [127] b0: [64] c0: [0] BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S192_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S192_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_1_features_1_0_features_1_0_3_features_1_0_3_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_1_features_1_0_features_1_0_3_features_1_0_3_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1501", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1501.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S196_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S196_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S196_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S196_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0 NE16_PADVAL: [72] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S196_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S196_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_1_features_1_1_block_block_0_block_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_1_features_1_1_block_block_0_block_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1504", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1504.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S199_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S199_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S199_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S199_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01129 out: 0.00718  BIASN: 0 PRENORM: 0 NE16_PADVAL: [128] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S199_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S199_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_1_features_1_1_block_block_1_block_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_1_features_1_1_block_block_1_block_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1507", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1507.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S202_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S202_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S202_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S202_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01382 out: 0.00638  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S202_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S202_Infos.tensor", 1, 1, 8, 0)),
                // no activation ACTSCALE: 0 ACTSCALEN: 0 GLOBAL_SUM_SCALE: [145] GLOBAL_SUM_SCALEN: [13]
                TCArgInfo("signed char * __restrict__", "S204_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S204_Infos.tensor", 1, 1, 8, 0)),
                // in q: -0.36<(i8-0.00)*0.00281069<0.36 out_q: -0.36<(u8-183.00)*0.00196114<0.14
                TCArgInfo("signed char * __restrict__", "S206_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S206_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_1_features_1_1_block_block_2_fc1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_1_features_1_1_block_block_2_fc1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_backbone_features_1_1_block_2_fc1_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_backbone_features_1_1_block_2_fc1_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S209_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S209_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S209_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S209_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00202 out: 0.00202  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [183] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S209_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S209_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_1_features_1_1_block_block_2_fc2_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_1_features_1_1_block_block_2_fc2_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_backbone_features_1_1_block_2_fc2_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_backbone_features_1_1_block_2_fc2_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S212_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S212_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S212_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S212_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.00787  actscale: [1] actscalen: [0] a0: [127] b0: [64] c0: [0] BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S212_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S212_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_1_features_1_1_block_block_3_block_3_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_1_features_1_1_block_block_3_block_3_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1510", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1510.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S216_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S216_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S216_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S216_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0 NE16_PADVAL: [82] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S216_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S216_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [143] IN1SCALEN: [7] OUTSCALE: [47] OUTSCALEN: [6] ADD_BIAS: [-59]
                TCArgInfo("signed char * __restrict__", "S217_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S217_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_1_features_1_2_block_block_0_block_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_1_features_1_2_block_block_0_block_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1513", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1513.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S220_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S220_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S220_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S220_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01236 out: 0.00710  BIASN: 0 PRENORM: 0 NE16_PADVAL: [124] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S220_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S220_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_backbone_features_1_features_1_2_block_block_1_block_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_1_features_1_2_block_block_1_block_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1516", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1516.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S223_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S223_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S223_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S223_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01221 out: 0.00530  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S223_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S223_Infos.tensor", 1, 1, 8, 0)),
                // no activation ACTSCALE: 0 ACTSCALEN: 0 GLOBAL_SUM_SCALE: [67] GLOBAL_SUM_SCALEN: [12]
                TCArgInfo("signed char * __restrict__", "S225_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S225_Infos.tensor", 1, 1, 8, 0)),
                // in q: -0.32<(i8-0.00)*0.00252946<0.32 out_q: -0.32<(u8-161.00)*0.00201196<0.19
                TCArgInfo("signed char * __restrict__", "S227_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S227_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_1_features_1_2_block_block_2_fc1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_1_features_1_2_block_block_2_fc1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_backbone_features_1_2_block_2_fc1_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_backbone_features_1_2_block_2_fc1_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S230_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S230_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S230_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S230_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00316 out: 0.00316  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [161] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S230_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S230_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_1_features_1_2_block_block_2_fc2_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_1_features_1_2_block_block_2_fc2_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_backbone_features_1_2_block_2_fc2_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_backbone_features_1_2_block_2_fc2_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S233_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S233_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S233_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S233_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.00787  actscale: [1] actscalen: [0] a0: [127] b0: [64] c0: [0] BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S233_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S233_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_1_features_1_2_block_block_3_block_3_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_1_features_1_2_block_block_3_block_3_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1519", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1519.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S237_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S237_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S237_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S237_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0 NE16_PADVAL: [96] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S237_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S237_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [133] IN1SCALEN: [7] OUTSCALE: [187] OUTSCALEN: [8] ADD_BIAS: [-65]
                TCArgInfo("signed char * __restrict__", "S238_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S238_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_features_1_features_1_3_features_1_3_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_features_1_features_1_3_features_1_3_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1522", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1522.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S241_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S241_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S241_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S241_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01307 out: 0.01072  BIASN: 0 PRENORM: 0 NE16_PADVAL: [123] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S241_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S241_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_extra_0_extra_0_0_extra_0_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_extra_0_extra_0_0_extra_0_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1525", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1525.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S244_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S244_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S244_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S244_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [128] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S244_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S244_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_extra_0_extra_0_1_extra_0_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_extra_0_extra_0_1_extra_0_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1528", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1528.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S247_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S247_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S247_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S247_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S247_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S247_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_extra_0_extra_0_2_extra_0_2_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_extra_0_extra_0_2_extra_0_2_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1531", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1531.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S250_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S250_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S250_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S250_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S250_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S250_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_extra_1_extra_1_0_extra_1_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_extra_1_extra_1_0_extra_1_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1534", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1534.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S253_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S253_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S253_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S253_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S253_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S253_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_extra_1_extra_1_1_extra_1_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_extra_1_extra_1_1_extra_1_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1537", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1537.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S256_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S256_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S256_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S256_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S256_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S256_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_extra_1_extra_1_2_extra_1_2_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_extra_1_extra_1_2_extra_1_2_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1540", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1540.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S259_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S259_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S259_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S259_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S259_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S259_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_extra_2_extra_2_0_extra_2_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_extra_2_extra_2_0_extra_2_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1543", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1543.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S262_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S262_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S262_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S262_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S262_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S262_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_extra_2_extra_2_1_extra_2_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_extra_2_extra_2_1_extra_2_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1546", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1546.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S265_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S265_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S265_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S265_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S265_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S265_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_extra_2_extra_2_2_extra_2_2_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_extra_2_extra_2_2_extra_2_2_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1549", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1549.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S268_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S268_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S268_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S268_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S268_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S268_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_extra_3_extra_3_0_extra_3_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_extra_3_extra_3_0_extra_3_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1552", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1552.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S271_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S271_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S271_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S271_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S271_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S271_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_extra_3_extra_3_1_extra_3_1_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_extra_3_extra_3_1_extra_3_1_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1555", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1555.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S274_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S274_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S274_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S274_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S274_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S274_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_backbone_extra_3_extra_3_2_extra_3_2_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_extra_3_extra_3_2_extra_3_2_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1558", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1558.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S277_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S277_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S277_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S277_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S277_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S277_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1561", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1561.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S280_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S280_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S280_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S280_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [128] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S280_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S280_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_head_regression_head_module_list_0_module_list_0_1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_regression_head_module_list_0_module_list_0_1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_head_regression_head_module_list_0_1_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_head_regression_head_module_list_0_1_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S283_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S283_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S283_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S283_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00025 out: 0.00003  BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S283_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S283_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1564", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1564.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S287_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S287_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S287_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S287_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [128] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S287_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S287_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_head_regression_head_module_list_1_module_list_1_1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_regression_head_module_list_1_module_list_1_1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1567", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1567.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S291_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S291_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S291_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S291_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S291_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S291_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_head_regression_head_module_list_2_module_list_2_1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_regression_head_module_list_2_module_list_2_1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1570", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1570.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S295_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S295_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S295_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S295_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S295_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S295_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_head_regression_head_module_list_3_module_list_3_1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_regression_head_module_list_3_module_list_3_1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_head_regression_head_module_list_4_module_list_4_1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_regression_head_module_list_4_module_list_4_1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_head_regression_head_module_list_5_module_list_5_0_module_list_5_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_regression_head_module_list_5_module_list_5_0_module_list_5_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1576", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1576.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S301_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S301_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S301_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S301_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S301_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S301_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_head_regression_head_module_list_5_module_list_5_1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_regression_head_module_list_5_module_list_5_1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_head_classification_head_module_list_0_module_list_0_1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_classification_head_module_list_0_module_list_0_1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_head_classification_head_module_list_0_1_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_head_classification_head_module_list_0_1_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "_head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_head_classification_head_module_list_1_module_list_1_1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_classification_head_module_list_1_module_list_1_1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_head_classification_head_module_list_2_module_list_2_1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_classification_head_module_list_2_module_list_2_1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_head_classification_head_module_list_3_module_list_3_1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_classification_head_module_list_3_module_list_3_1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_head_classification_head_module_list_4_module_list_4_1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_classification_head_module_list_4_module_list_4_1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_head_classification_head_module_list_5_module_list_5_0_module_list_5_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_classification_head_module_list_5_module_list_5_0_module_list_5_0_0_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "_head_classification_head_module_list_5_module_list_5_1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_classification_head_module_list_5_module_list_5_1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_sub", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_sub.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_add", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_add.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_sub_1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_sub_1.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_add_1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_add_1.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1561_1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1561_1.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S321_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S321_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S321_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S321_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [128] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S321_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S321_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S322_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S322_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S322_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S322_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S322_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S322_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_head_regression_head_module_list_0_1_bias_1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_head_regression_head_module_list_0_1_bias_1.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S325_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S325_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S325_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S325_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00013 out: 0.00003  BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S325_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S325_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_head_regression_head_module_list_0_1_bias_2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_head_regression_head_module_list_0_1_bias_2.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S328_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S328_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S328_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S328_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00011 out: 0.00003  BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S328_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S328_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_head_regression_head_module_list_0_1_bias_3", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_head_regression_head_module_list_0_1_bias_3.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S331_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S331_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S331_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S331_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00012 out: 0.00003  BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S331_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S331_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_head_regression_head_module_list_0_1_bias_4", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_head_regression_head_module_list_0_1_bias_4.tensor", 1, 1, 32, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_head_regression_head_module_list_0_1_bias_5", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_head_regression_head_module_list_0_1_bias_5.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S335_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S335_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S335_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S335_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00005 out: 0.00003  BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S335_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S335_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1564_1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1564_1.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S338_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S338_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S338_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S338_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [128] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S338_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S338_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1567_1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1567_1.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S340_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S340_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S340_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S340_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S340_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S340_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1570_1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1570_1.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S342_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S342_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S342_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S342_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S342_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S342_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "S343_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S343_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S343_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S343_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00005 out: 0.00003  BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S343_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S343_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1570_2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1570_2.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S352_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S352_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S352_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S352_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S352_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S352_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1570_3", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1570_3.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S354_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S354_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S354_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S354_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S354_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S354_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_1576_1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1576_1.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S356_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S356_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S356_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S356_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S356_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S356_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_head_classification_head_module_list_0_1_bias_1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_head_classification_head_module_list_0_1_bias_1.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S358_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S358_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S358_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S358_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S358_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S358_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_head_classification_head_module_list_0_1_bias_2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_head_classification_head_module_list_0_1_bias_2.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S361_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S361_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S361_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S361_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S361_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S361_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_head_classification_head_module_list_0_1_bias_3", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_head_classification_head_module_list_0_1_bias_3.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S364_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S364_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S364_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S364_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S364_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S364_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_head_classification_head_module_list_0_1_bias_4", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_head_classification_head_module_list_0_1_bias_4.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S367_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S367_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S367_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S367_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S367_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S367_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_model_head_classification_head_module_list_0_1_bias_5", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_model_head_classification_head_module_list_0_1_bias_5.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S370_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S370_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S370_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S370_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S370_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S370_Infos.tensor", 1, 1, 8, 0)),
                // in: 0.00049 out: 0.00003  BIASL_SM: [4]
                TCArgInfo("signed char * __restrict__", "S373_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S373_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_sub_1_1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_sub_1_1.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_sub_1_2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_sub_1_2.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Output_1", ARG_SCOPE_ARG_ALLOC, ARG_DIR_OUT, AT_MEM_L2, AT_MEM_L2, 0),
                TCArgInfo("unsigned char * __restrict__", "Output_2", ARG_SCOPE_ARG_ALLOC, ARG_DIR_OUT, AT_MEM_L2, AT_MEM_L2, 0),
                TCArgInfo("unsigned char * __restrict__", "Output_3", ARG_SCOPE_ARG_ALLOC, ARG_DIR_OUT, AT_MEM_L2, AT_MEM_L2, 0),
                TCArgInfo("unsigned char * __restrict__", "Output_4", ARG_SCOPE_ARG_ALLOC, ARG_DIR_OUT, AT_MEM_L2, AT_MEM_L2, 0),
                TCArgInfo("signed short * __restrict__", "Output_5", ARG_SCOPE_ARG_ALLOC, ARG_DIR_OUT, AT_MEM_L2, AT_MEM_L2, 0)
            ),
        /* Locals, allocated dynamically */
        CArgs(139,
            TCArgInfo("unsigned char * __restrict__", "S3_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S6_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S9_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S10_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S13_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S16_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S19_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S22_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S25_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S28_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S29_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S32_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S33_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S36_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S37_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S38_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S40_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S43_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S46_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S47_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S50_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S53_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S54_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S57_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S58_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S59_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S61_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S64_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S67_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S68_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S71_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S72_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S75_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S76_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S79_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S80_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S81_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S83_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S86_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S89_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S90_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S93_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S94_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S97_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S100_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S103_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S106_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S109_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S112_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S113_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S116_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S119_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S122_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S123_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S126_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S129_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S132_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S133_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S136_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S139_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S140_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S141_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S143_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S146_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S149_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S150_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S153_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S156_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S159_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S160_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S161_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S163_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S166_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S169_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S170_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S173_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S174_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S177_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S178_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S179_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S182_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S183_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S184_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S186_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S189_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S192_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S193_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S196_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S199_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S202_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S203_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S204_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S206_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S209_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S212_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S213_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S216_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S217_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S220_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S223_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S224_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S225_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S227_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S230_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S233_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S234_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S237_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S238_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S241_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S244_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S247_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S250_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S253_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S256_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S259_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S262_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S265_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S268_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S271_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S274_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S277_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S280_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S287_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S291_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S295_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S301_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S321_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S338_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S340_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S342_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S345_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S346_Output_1", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S346_Output_0", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S346_Output_3", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S346_Output_2", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S352_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S354_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S356_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S372_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0)
        )
    );

    // Stacked tensors for concats and splits
    AddStackedTensors("S345_Output", 6, "S283_Output", "S325_Output", "S328_Output", "S331_Output", "S343_Output", "S335_Output");
    AddStackedTensors("S372_Output", 6, "S322_Output", "S358_Output", "S361_Output", "S364_Output", "S367_Output", "S370_Output");

    // Node S3_Conv2d_16x3x3x3_Custom inq -1.00<(u8-255.00)*0.00392157<0.00 weightsq chan<(u8-128.00)*chan<chan outq -10.79<(u8-128.00)*0.08432654<10.71 forced biasesq chan<(i32-0.00)*chan<chan
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
    // Node S6_Conv2d_16x1x3x3_Relu inq -10.79<(u8-128.00)*0.08432654<10.71 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.01608291<4.10 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S6_Conv2d_16x1x3x3_Relu",
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
    // Node S9_Conv2d_16x16x1x1 inq 0.00<(u8-0.00)*0.01608291<4.10 weightsq chan<(u8-128.00)*chan<chan outq -5.09<(u8-122.00)*0.04171812<5.55 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S9_Conv2d_16x16x1x1",
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
    // Node S10_MatAdd_128x160x16 in1q -10.79<(u8-128.00)*0.08432654<10.71 forced
    //   in2q -5.09<(u8-122.00)*0.04171812<5.55 forced
    //   outq -3.77<(u8-60.00)*0.06279226<12.24 forced scaled input 0 is node input 1
    AddNode("S10_MatAdd_128x160x16",
        Bindings(4,
            GNodeArg(GNA_IN, "S3_Output", 0),
            GNodeArg(GNA_IN, "S9_Output", 0),
            GNodeArg(GNA_OUT, "S10_Output", 0),
            GNodeArg(GNA_IN, "S10_Infos", 0)
        )
    );
    // Node S13_Conv2d_64x16x1x1_Relu inq -3.77<(u8-60.00)*0.06279226<12.24 forced weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.00993937<2.53 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S13_Conv2d_64x16x1x1_Relu",
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
    // Node S16_Conv2d_64x1x3x3_Relu inq 0.00<(u8-0.00)*0.00993937<2.53 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.01776073<4.53 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S16_Conv2d_64x1x3x3_Relu",
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
    // Node S19_Conv2d_24x64x1x1 inq 0.00<(u8-0.00)*0.01776073<4.53 weightsq chan<(u8-128.00)*chan<chan outq -3.76<(u8-137.00)*0.02745290<3.24 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S19_Conv2d_24x64x1x1",
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
    // Node S22_Conv2d_72x24x1x1_Relu inq -3.76<(u8-137.00)*0.02745290<3.24 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.00663882<1.69 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S22_Conv2d_72x24x1x1_Relu",
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
    // Node S25_Conv2d_72x1x3x3_Relu inq 0.00<(u8-0.00)*0.00663882<1.69 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.01166781<2.98 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S25_Conv2d_72x1x3x3_Relu",
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
    // Node S28_Conv2d_24x72x1x1 inq 0.00<(u8-0.00)*0.01166781<2.98 weightsq chan<(u8-128.00)*chan<chan outq -3.17<(u8-117.00)*0.02710584<3.74 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S28_Conv2d_24x72x1x1",
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
    // Node S29_MatAdd_64x80x24 in1q -3.76<(u8-137.00)*0.02745290<3.24 forced
    //   in2q -3.17<(u8-117.00)*0.02710584<3.74 forced
    //   outq -6.81<(u8-161.00)*0.04230706<3.98 forced scaled input 0 is node input 1
    AddNode("S29_MatAdd_64x80x24",
        Bindings(4,
            GNodeArg(GNA_IN, "S19_Output", 0),
            GNodeArg(GNA_IN, "S28_Output", 0),
            GNodeArg(GNA_OUT, "S29_Output", 0),
            GNodeArg(GNA_IN, "S29_Infos", 0)
        )
    );
    // Node S32_Conv2d_72x24x1x1_Relu inq -6.81<(u8-161.00)*0.04230706<3.98 forced weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.00987938<2.52 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S32_Conv2d_72x24x1x1_Relu",
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
    // Node _backbone_features_0_features_0_4_block_block_1_block_1_0_Conv_fusion_qin0 inq 0.00<(u8-0.00)*0.00987938<2.52 outq -2.54<(i8-0.00)*0.01983655<2.52 forced
    AddNode("S33_Op__backbone_features_0_features_0_4_block_block_1_block_1_0_Conv_fusion_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S32_Output", 0),
            GNodeArg(GNA_OUT, "S33_Output", 0),
            GNodeArg(GNA_IN, "S33_Infos", 0)
        )
    );
    // Node S36_Conv2d_72x5x5x1_Relu inq -2.54<(i8-0.00)*0.01983655<2.52 forced weightsq chan<(i8-0.00)*chan<chan outq -11.79<(i8-0.00)*0.09209285<11.70 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S36_Conv2d_72x5x5x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S33_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_4_block_block_1_block_1_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1417", 0),
            GNodeArg(GNA_OUT, "S36_Output", 0),
            GNodeArg(GNA_IN, "S36_Mul_scale", 0),
            GNodeArg(GNA_IN, "S36_Mul_shift", 0),
            GNodeArg(GNA_IN, "S36_Infos", 0)
        )
    );
    // Node _backbone_features_0_features_0_4_block_block_2_avgpool_GlobalAveragePool_trans inq -11.79<(i8-0.00)*0.09209285<11.70 forced outq -11.79<(i8-0.00)*0.09209285<11.70 forced
    AddNode("S37_Op__backbone_features_0_features_0_4_block_block_2_avgpool_GlobalAveragePool_trans",
        Bindings(2,
            GNodeArg(GNA_IN, "S36_Output", 0),
            GNodeArg(GNA_OUT, "S37_Output", 0)
        )
    );
    // Node _backbone_features_0_features_0_4_block_block_2_avgpool_GlobalAveragePool inq -11.79<(i8-0.00)*0.09209285<11.70 forced outq -4.50<(i8-0.00)*0.03516809<4.47
    AddNode("S38_Op__backbone_features_0_features_0_4_block_block_2_avgpool_GlobalAveragePool",
        Bindings(3,
            GNodeArg(GNA_IN, "S37_Output", 0),
            GNodeArg(GNA_OUT, "S38_Output", 0),
            GNodeArg(GNA_IN, "S38_Infos", 0)
        )
    );
    // Node _backbone_features_0_features_0_4_block_block_2_fc1_Conv_fusion_qin0 inq -4.50<(i8-0.00)*0.03516809<4.47 outq 0.00<(u8-0.00)*0.01751509<4.47
    AddNode("S40_Op__backbone_features_0_features_0_4_block_block_2_fc1_Conv_fusion_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S38_Output", 0),
            GNodeArg(GNA_OUT, "S40_Output", 0),
            GNodeArg(GNA_IN, "S40_Infos", 0)
        )
    );
    // Node S43_Conv2d_24x72x1x1_Relu inq 0.00<(u8-0.00)*0.01751509<4.47 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.00800914<2.04 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S43_Conv2d_24x72x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S40_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_4_block_block_2_fc1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_backbone_features_0_4_block_2_fc1_bias", 0),
            GNodeArg(GNA_OUT, "S43_Output", 0),
            GNodeArg(GNA_IN, "S43_Mul_scale", 0),
            GNodeArg(GNA_IN, "S43_Mul_shift", 0),
            GNodeArg(GNA_IN, "S43_Infos", 0)
        )
    );
    // Node S46_Conv2d_72x24x1x1_Hsigmoid inq 0.00<(u8-0.00)*0.00800914<2.04 weightsq chan<(u8-128.00)*chan<chan outq -1.01<(i8-0.00)*0.00787402<1.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S46_Conv2d_72x24x1x1_Hsigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S43_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_4_block_block_2_fc2_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_backbone_features_0_4_block_2_fc2_bias", 0),
            GNodeArg(GNA_OUT, "S46_Output", 0),
            GNodeArg(GNA_IN, "S46_Mul_scale", 0),
            GNodeArg(GNA_IN, "S46_Mul_shift", 0),
            GNodeArg(GNA_IN, "S46_Infos", 0)
        )
    );
    // Node expr_16 in_qs [-11.79<(i8-0.00)*0.09209285<11.70 forced,-1.01<(i8-0.00)*0.00787402<1.00] out_qs [0.00<(u8-0.00)*0.00578475<1.48]
    AddNode("S47_Op_expr_16",
        Bindings(3,
            GNodeArg(GNA_IN, "S36_Output", 0),
            GNodeArg(GNA_IN, "S46_Output", 0),
            GNodeArg(GNA_OUT, "S47_Output", 0)
        )
    );
    // Node S50_Conv2d_40x72x1x1 inq 0.00<(u8-0.00)*0.00578475<1.48 weightsq chan<(u8-128.00)*chan<chan outq -3.03<(u8-162.00)*0.01872035<1.74 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S50_Conv2d_40x72x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S47_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_4_block_block_3_block_3_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1420", 0),
            GNodeArg(GNA_OUT, "S50_Output", 0),
            GNodeArg(GNA_IN, "S50_Mul_scale", 0),
            GNodeArg(GNA_IN, "S50_Mul_shift", 0),
            GNodeArg(GNA_IN, "S50_Infos", 0)
        )
    );
    // Node S53_Conv2d_120x40x1x1_Relu inq -3.03<(u8-162.00)*0.01872035<1.74 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.00521777<1.33 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S53_Conv2d_120x40x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S50_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_5_block_block_0_block_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1423", 0),
            GNodeArg(GNA_OUT, "S53_Output", 0),
            GNodeArg(GNA_IN, "S53_Mul_scale", 0),
            GNodeArg(GNA_IN, "S53_Mul_shift", 0),
            GNodeArg(GNA_IN, "S53_Infos", 0)
        )
    );
    // Node _backbone_features_0_features_0_5_block_block_1_block_1_0_Conv_fusion_qin0 inq 0.00<(u8-0.00)*0.00521777<1.33 outq -1.34<(i8-0.00)*0.01047662<1.33 forced
    AddNode("S54_Op__backbone_features_0_features_0_5_block_block_1_block_1_0_Conv_fusion_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S53_Output", 0),
            GNodeArg(GNA_OUT, "S54_Output", 0),
            GNodeArg(GNA_IN, "S54_Infos", 0)
        )
    );
    // Node S57_Conv2d_120x5x5x1_Relu inq -1.34<(i8-0.00)*0.01047662<1.33 forced weightsq chan<(i8-0.00)*chan<chan outq -3.36<(i8-0.00)*0.02622083<3.33 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S57_Conv2d_120x5x5x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S54_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_5_block_block_1_block_1_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1426", 0),
            GNodeArg(GNA_OUT, "S57_Output", 0),
            GNodeArg(GNA_IN, "S57_Mul_scale", 0),
            GNodeArg(GNA_IN, "S57_Mul_shift", 0),
            GNodeArg(GNA_IN, "S57_Infos", 0)
        )
    );
    // Node _backbone_features_0_features_0_5_block_block_2_avgpool_GlobalAveragePool_trans inq -3.36<(i8-0.00)*0.02622083<3.33 forced outq -3.36<(i8-0.00)*0.02622083<3.33 forced
    AddNode("S58_Op__backbone_features_0_features_0_5_block_block_2_avgpool_GlobalAveragePool_trans",
        Bindings(2,
            GNodeArg(GNA_IN, "S57_Output", 0),
            GNodeArg(GNA_OUT, "S58_Output", 0)
        )
    );
    // Node _backbone_features_0_features_0_5_block_block_2_avgpool_GlobalAveragePool inq -3.36<(i8-0.00)*0.02622083<3.33 forced outq -1.49<(i8-0.00)*0.01167131<1.48
    AddNode("S59_Op__backbone_features_0_features_0_5_block_block_2_avgpool_GlobalAveragePool",
        Bindings(3,
            GNodeArg(GNA_IN, "S58_Output", 0),
            GNodeArg(GNA_OUT, "S59_Output", 0),
            GNodeArg(GNA_IN, "S59_Infos", 0)
        )
    );
    // Node _backbone_features_0_features_0_5_block_block_2_fc1_Conv_fusion_qin0 inq -1.49<(i8-0.00)*0.01167131<1.48 outq 0.00<(u8-0.00)*0.00581277<1.48
    AddNode("S61_Op__backbone_features_0_features_0_5_block_block_2_fc1_Conv_fusion_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S59_Output", 0),
            GNodeArg(GNA_OUT, "S61_Output", 0),
            GNodeArg(GNA_IN, "S61_Infos", 0)
        )
    );
    // Node S64_Conv2d_32x120x1x1_Relu inq 0.00<(u8-0.00)*0.00581277<1.48 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.00691863<1.76 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S64_Conv2d_32x120x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S61_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_5_block_block_2_fc1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_backbone_features_0_5_block_2_fc1_bias", 0),
            GNodeArg(GNA_OUT, "S64_Output", 0),
            GNodeArg(GNA_IN, "S64_Mul_scale", 0),
            GNodeArg(GNA_IN, "S64_Mul_shift", 0),
            GNodeArg(GNA_IN, "S64_Infos", 0)
        )
    );
    // Node S67_Conv2d_120x32x1x1_Hsigmoid inq 0.00<(u8-0.00)*0.00691863<1.76 weightsq chan<(u8-128.00)*chan<chan outq -1.01<(i8-0.00)*0.00787402<1.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S67_Conv2d_120x32x1x1_Hsigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S64_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_5_block_block_2_fc2_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_backbone_features_0_5_block_2_fc2_bias", 0),
            GNodeArg(GNA_OUT, "S67_Output", 0),
            GNodeArg(GNA_IN, "S67_Mul_scale", 0),
            GNodeArg(GNA_IN, "S67_Mul_shift", 0),
            GNodeArg(GNA_IN, "S67_Infos", 0)
        )
    );
    // Node expr_17 in_qs [-3.36<(i8-0.00)*0.02622083<3.33 forced,-1.01<(i8-0.00)*0.00787402<1.00] out_qs [0.00<(u8-0.00)*0.00390377<1.00]
    AddNode("S68_Op_expr_17",
        Bindings(3,
            GNodeArg(GNA_IN, "S57_Output", 0),
            GNodeArg(GNA_IN, "S67_Output", 0),
            GNodeArg(GNA_OUT, "S68_Output", 0)
        )
    );
    // Node S71_Conv2d_40x120x1x1 inq 0.00<(u8-0.00)*0.00390377<1.00 weightsq chan<(u8-128.00)*chan<chan outq -3.82<(u8-146.00)*0.02617160<2.85 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S71_Conv2d_40x120x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S68_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_5_block_block_3_block_3_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1429", 0),
            GNodeArg(GNA_OUT, "S71_Output", 0),
            GNodeArg(GNA_IN, "S71_Mul_scale", 0),
            GNodeArg(GNA_IN, "S71_Mul_shift", 0),
            GNodeArg(GNA_IN, "S71_Infos", 0)
        )
    );
    // Node S72_MatAdd_32x40x40 in1q -3.82<(u8-146.00)*0.02617160<2.85 forced
    //   in2q -3.03<(u8-162.00)*0.01872035<1.74 forced
    //   outq -3.74<(u8-132.00)*0.02834549<3.49 forced scaled input 0 is node input 0
    AddNode("S72_MatAdd_32x40x40",
        Bindings(4,
            GNodeArg(GNA_IN, "S71_Output", 0),
            GNodeArg(GNA_IN, "S50_Output", 0),
            GNodeArg(GNA_OUT, "S72_Output", 0),
            GNodeArg(GNA_IN, "S72_Infos", 0)
        )
    );
    // Node S75_Conv2d_120x40x1x1_Relu inq -3.74<(u8-132.00)*0.02834549<3.49 forced weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.00445911<1.14 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S75_Conv2d_120x40x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S72_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_6_block_block_0_block_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1432", 0),
            GNodeArg(GNA_OUT, "S75_Output", 0),
            GNodeArg(GNA_IN, "S75_Mul_scale", 0),
            GNodeArg(GNA_IN, "S75_Mul_shift", 0),
            GNodeArg(GNA_IN, "S75_Infos", 0)
        )
    );
    // Node _backbone_features_0_features_0_6_block_block_1_block_1_0_Conv_fusion_qin0 inq 0.00<(u8-0.00)*0.00445911<1.14 outq -1.15<(i8-0.00)*0.00895333<1.14 forced
    AddNode("S76_Op__backbone_features_0_features_0_6_block_block_1_block_1_0_Conv_fusion_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S75_Output", 0),
            GNodeArg(GNA_OUT, "S76_Output", 0),
            GNodeArg(GNA_IN, "S76_Infos", 0)
        )
    );
    // Node S79_Conv2d_120x5x5x1_Relu inq -1.15<(i8-0.00)*0.00895333<1.14 forced weightsq chan<(i8-0.00)*chan<chan outq -3.84<(i8-0.00)*0.03002487<3.81 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S79_Conv2d_120x5x5x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S76_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_6_block_block_1_block_1_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1435", 0),
            GNodeArg(GNA_OUT, "S79_Output", 0),
            GNodeArg(GNA_IN, "S79_Mul_scale", 0),
            GNodeArg(GNA_IN, "S79_Mul_shift", 0),
            GNodeArg(GNA_IN, "S79_Infos", 0)
        )
    );
    // Node _backbone_features_0_features_0_6_block_block_2_avgpool_GlobalAveragePool_trans inq -3.84<(i8-0.00)*0.03002487<3.81 forced outq -3.84<(i8-0.00)*0.03002487<3.81 forced
    AddNode("S80_Op__backbone_features_0_features_0_6_block_block_2_avgpool_GlobalAveragePool_trans",
        Bindings(2,
            GNodeArg(GNA_IN, "S79_Output", 0),
            GNodeArg(GNA_OUT, "S80_Output", 0)
        )
    );
    // Node _backbone_features_0_features_0_6_block_block_2_avgpool_GlobalAveragePool inq -3.84<(i8-0.00)*0.03002487<3.81 forced outq -2.07<(i8-0.00)*0.01617177<2.05
    AddNode("S81_Op__backbone_features_0_features_0_6_block_block_2_avgpool_GlobalAveragePool",
        Bindings(3,
            GNodeArg(GNA_IN, "S80_Output", 0),
            GNodeArg(GNA_OUT, "S81_Output", 0),
            GNodeArg(GNA_IN, "S81_Infos", 0)
        )
    );
    // Node _backbone_features_0_features_0_6_block_block_2_fc1_Conv_fusion_qin0 inq -2.07<(i8-0.00)*0.01617177<2.05 outq 0.00<(u8-0.00)*0.00805418<2.05
    AddNode("S83_Op__backbone_features_0_features_0_6_block_block_2_fc1_Conv_fusion_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S81_Output", 0),
            GNodeArg(GNA_OUT, "S83_Output", 0),
            GNodeArg(GNA_IN, "S83_Infos", 0)
        )
    );
    // Node S86_Conv2d_32x120x1x1_Relu inq 0.00<(u8-0.00)*0.00805418<2.05 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.00809449<2.06 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S86_Conv2d_32x120x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S83_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_6_block_block_2_fc1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_backbone_features_0_6_block_2_fc1_bias", 0),
            GNodeArg(GNA_OUT, "S86_Output", 0),
            GNodeArg(GNA_IN, "S86_Mul_scale", 0),
            GNodeArg(GNA_IN, "S86_Mul_shift", 0),
            GNodeArg(GNA_IN, "S86_Infos", 0)
        )
    );
    // Node S89_Conv2d_120x32x1x1_Hsigmoid inq 0.00<(u8-0.00)*0.00809449<2.06 weightsq chan<(u8-128.00)*chan<chan outq -1.01<(i8-0.00)*0.00787402<1.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S89_Conv2d_120x32x1x1_Hsigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S86_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_6_block_block_2_fc2_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_backbone_features_0_6_block_2_fc2_bias", 0),
            GNodeArg(GNA_OUT, "S89_Output", 0),
            GNodeArg(GNA_IN, "S89_Mul_scale", 0),
            GNodeArg(GNA_IN, "S89_Mul_shift", 0),
            GNodeArg(GNA_IN, "S89_Infos", 0)
        )
    );
    // Node expr_18 in_qs [-3.84<(i8-0.00)*0.03002487<3.81 forced,-1.01<(i8-0.00)*0.00787402<1.00] out_qs [0.00<(u8-0.00)*0.00342750<0.87]
    AddNode("S90_Op_expr_18",
        Bindings(3,
            GNodeArg(GNA_IN, "S79_Output", 0),
            GNodeArg(GNA_IN, "S89_Output", 0),
            GNodeArg(GNA_OUT, "S90_Output", 0)
        )
    );
    // Node S93_Conv2d_40x120x1x1 inq 0.00<(u8-0.00)*0.00342750<0.87 weightsq chan<(u8-128.00)*chan<chan outq -2.40<(u8-124.00)*0.01931882<2.53 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S93_Conv2d_40x120x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S90_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_6_block_block_3_block_3_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1438", 0),
            GNodeArg(GNA_OUT, "S93_Output", 0),
            GNodeArg(GNA_IN, "S93_Mul_scale", 0),
            GNodeArg(GNA_IN, "S93_Mul_shift", 0),
            GNodeArg(GNA_IN, "S93_Infos", 0)
        )
    );
    // Node S94_MatAdd_32x40x40 in1q -3.74<(u8-132.00)*0.02834549<3.49 forced
    //   in2q -2.40<(u8-124.00)*0.01931882<2.53 forced
    //   outq -4.08<(u8-115.00)*0.03546171<4.96 forced scaled input 0 is node input 1
    AddNode("S94_MatAdd_32x40x40",
        Bindings(4,
            GNodeArg(GNA_IN, "S72_Output", 0),
            GNodeArg(GNA_IN, "S93_Output", 0),
            GNodeArg(GNA_OUT, "S94_Output", 0),
            GNodeArg(GNA_IN, "S94_Infos", 0)
        )
    );
    // Node S97_Conv2d_240x40x1x1_Custom inq -4.08<(u8-115.00)*0.03546171<4.96 forced weightsq chan<(u8-128.00)*chan<chan outq -5.30<(u8-128.00)*0.04143073<5.26 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S97_Conv2d_240x40x1x1_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S94_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_7_block_block_0_block_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1441", 0),
            GNodeArg(GNA_OUT, "S97_Output", 0),
            GNodeArg(GNA_IN, "S97_Mul_scale", 0),
            GNodeArg(GNA_IN, "S97_Mul_shift", 0),
            GNodeArg(GNA_IN, "S97_Infos", 0)
        )
    );
    // Node S100_Conv2d_240x1x3x3_Custom inq -5.30<(u8-128.00)*0.04143073<5.26 weightsq chan<(u8-128.00)*chan<chan outq -4.92<(u8-128.00)*0.03842762<4.88 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S100_Conv2d_240x1x3x3_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S97_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_7_block_block_1_block_1_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1444", 0),
            GNodeArg(GNA_OUT, "S100_Output", 0),
            GNodeArg(GNA_IN, "S100_Mul_scale", 0),
            GNodeArg(GNA_IN, "S100_Mul_shift", 0),
            GNodeArg(GNA_IN, "S100_Infos", 0)
        )
    );
    // Node S103_Conv2d_80x240x1x1 inq -4.92<(u8-128.00)*0.03842762<4.88 weightsq chan<(u8-128.00)*chan<chan outq -2.50<(u8-146.00)*0.01711651<1.87 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S103_Conv2d_80x240x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S100_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_7_block_block_2_block_2_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1447", 0),
            GNodeArg(GNA_OUT, "S103_Output", 0),
            GNodeArg(GNA_IN, "S103_Mul_scale", 0),
            GNodeArg(GNA_IN, "S103_Mul_shift", 0),
            GNodeArg(GNA_IN, "S103_Infos", 0)
        )
    );
    // Node S106_Conv2d_200x80x1x1_Custom inq -2.50<(u8-146.00)*0.01711651<1.87 weightsq chan<(u8-128.00)*chan<chan outq -7.31<(u8-128.00)*0.05709939<7.25 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S106_Conv2d_200x80x1x1_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S103_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_8_block_block_0_block_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1450", 0),
            GNodeArg(GNA_OUT, "S106_Output", 0),
            GNodeArg(GNA_IN, "S106_Mul_scale", 0),
            GNodeArg(GNA_IN, "S106_Mul_shift", 0),
            GNodeArg(GNA_IN, "S106_Infos", 0)
        )
    );
    // Node S109_Conv2d_200x1x3x3_Custom inq -7.31<(u8-128.00)*0.05709939<7.25 weightsq chan<(u8-128.00)*chan<chan outq -2.01<(u8-128.00)*0.01572623<2.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S109_Conv2d_200x1x3x3_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S106_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_8_block_block_1_block_1_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1453", 0),
            GNodeArg(GNA_OUT, "S109_Output", 0),
            GNodeArg(GNA_IN, "S109_Mul_scale", 0),
            GNodeArg(GNA_IN, "S109_Mul_shift", 0),
            GNodeArg(GNA_IN, "S109_Infos", 0)
        )
    );
    // Node S112_Conv2d_80x200x1x1 inq -2.01<(u8-128.00)*0.01572623<2.00 weightsq chan<(u8-128.00)*chan<chan outq -1.33<(u8-123.00)*0.01082555<1.43 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S112_Conv2d_80x200x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S109_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_8_block_block_2_block_2_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1456", 0),
            GNodeArg(GNA_OUT, "S112_Output", 0),
            GNodeArg(GNA_IN, "S112_Mul_scale", 0),
            GNodeArg(GNA_IN, "S112_Mul_shift", 0),
            GNodeArg(GNA_IN, "S112_Infos", 0)
        )
    );
    // Node S113_MatAdd_16x20x80 in1q -2.50<(u8-146.00)*0.01711651<1.87 forced
    //   in2q -1.33<(u8-123.00)*0.01082555<1.43 forced
    //   outq -2.64<(u8-140.00)*0.01883096<2.17 forced scaled input 0 is node input 1
    AddNode("S113_MatAdd_16x20x80",
        Bindings(4,
            GNodeArg(GNA_IN, "S103_Output", 0),
            GNodeArg(GNA_IN, "S112_Output", 0),
            GNodeArg(GNA_OUT, "S113_Output", 0),
            GNodeArg(GNA_IN, "S113_Infos", 0)
        )
    );
    // Node S116_Conv2d_184x80x1x1_Custom inq -2.64<(u8-140.00)*0.01883096<2.17 forced weightsq chan<(u8-128.00)*chan<chan outq -4.36<(u8-128.00)*0.03408024<4.33 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S116_Conv2d_184x80x1x1_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S113_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_9_block_block_0_block_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1459", 0),
            GNodeArg(GNA_OUT, "S116_Output", 0),
            GNodeArg(GNA_IN, "S116_Mul_scale", 0),
            GNodeArg(GNA_IN, "S116_Mul_shift", 0),
            GNodeArg(GNA_IN, "S116_Infos", 0)
        )
    );
    // Node S119_Conv2d_184x1x3x3_Custom inq -4.36<(u8-128.00)*0.03408024<4.33 weightsq chan<(u8-128.00)*chan<chan outq -3.25<(u8-128.00)*0.02536117<3.22 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S119_Conv2d_184x1x3x3_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S116_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_9_block_block_1_block_1_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1462", 0),
            GNodeArg(GNA_OUT, "S119_Output", 0),
            GNodeArg(GNA_IN, "S119_Mul_scale", 0),
            GNodeArg(GNA_IN, "S119_Mul_shift", 0),
            GNodeArg(GNA_IN, "S119_Infos", 0)
        )
    );
    // Node S122_Conv2d_80x184x1x1 inq -3.25<(u8-128.00)*0.02536117<3.22 weightsq chan<(u8-128.00)*chan<chan outq -1.91<(u8-141.00)*0.01354603<1.54 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S122_Conv2d_80x184x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S119_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_9_block_block_2_block_2_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1465", 0),
            GNodeArg(GNA_OUT, "S122_Output", 0),
            GNodeArg(GNA_IN, "S122_Mul_scale", 0),
            GNodeArg(GNA_IN, "S122_Mul_shift", 0),
            GNodeArg(GNA_IN, "S122_Infos", 0)
        )
    );
    // Node S123_MatAdd_16x20x80 in1q -2.64<(u8-140.00)*0.01883096<2.17 forced
    //   in2q -1.91<(u8-141.00)*0.01354603<1.54 forced
    //   outq -2.86<(u8-140.00)*0.02040668<2.35 forced scaled input 0 is node input 1
    AddNode("S123_MatAdd_16x20x80",
        Bindings(4,
            GNodeArg(GNA_IN, "S113_Output", 0),
            GNodeArg(GNA_IN, "S122_Output", 0),
            GNodeArg(GNA_OUT, "S123_Output", 0),
            GNodeArg(GNA_IN, "S123_Infos", 0)
        )
    );
    // Node S126_Conv2d_184x80x1x1_Custom inq -2.86<(u8-140.00)*0.02040668<2.35 forced weightsq chan<(u8-128.00)*chan<chan outq -3.62<(u8-128.00)*0.02831604<3.60 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S126_Conv2d_184x80x1x1_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S123_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_10_block_block_0_block_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1468", 0),
            GNodeArg(GNA_OUT, "S126_Output", 0),
            GNodeArg(GNA_IN, "S126_Mul_scale", 0),
            GNodeArg(GNA_IN, "S126_Mul_shift", 0),
            GNodeArg(GNA_IN, "S126_Infos", 0)
        )
    );
    // Node S129_Conv2d_184x1x3x3_Custom inq -3.62<(u8-128.00)*0.02831604<3.60 weightsq chan<(u8-128.00)*chan<chan outq -2.78<(u8-128.00)*0.02171489<2.76 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S129_Conv2d_184x1x3x3_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S126_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_10_block_block_1_block_1_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1471", 0),
            GNodeArg(GNA_OUT, "S129_Output", 0),
            GNodeArg(GNA_IN, "S129_Mul_scale", 0),
            GNodeArg(GNA_IN, "S129_Mul_shift", 0),
            GNodeArg(GNA_IN, "S129_Infos", 0)
        )
    );
    // Node S132_Conv2d_80x184x1x1 inq -2.78<(u8-128.00)*0.02171489<2.76 weightsq chan<(u8-128.00)*chan<chan outq -1.46<(u8-115.00)*0.01269088<1.78 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S132_Conv2d_80x184x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S129_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_10_block_block_2_block_2_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1474", 0),
            GNodeArg(GNA_OUT, "S132_Output", 0),
            GNodeArg(GNA_IN, "S132_Mul_scale", 0),
            GNodeArg(GNA_IN, "S132_Mul_shift", 0),
            GNodeArg(GNA_IN, "S132_Infos", 0)
        )
    );
    // Node S133_MatAdd_16x20x80 in1q -2.86<(u8-140.00)*0.02040668<2.35 forced
    //   in2q -1.46<(u8-115.00)*0.01269088<1.78 forced
    //   outq -2.86<(u8-135.00)*0.02118537<2.54 forced scaled input 0 is node input 1
    AddNode("S133_MatAdd_16x20x80",
        Bindings(4,
            GNodeArg(GNA_IN, "S123_Output", 0),
            GNodeArg(GNA_IN, "S132_Output", 0),
            GNodeArg(GNA_OUT, "S133_Output", 0),
            GNodeArg(GNA_IN, "S133_Infos", 0)
        )
    );
    // Node S136_Conv2d_480x80x1x1_Custom inq -2.86<(u8-135.00)*0.02118537<2.54 forced weightsq chan<(u8-128.00)*chan<chan outq -3.96<(u8-128.00)*0.03097155<3.93 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S136_Conv2d_480x80x1x1_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S133_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_11_block_block_0_block_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1477", 0),
            GNodeArg(GNA_OUT, "S136_Output", 0),
            GNodeArg(GNA_IN, "S136_Mul_scale", 0),
            GNodeArg(GNA_IN, "S136_Mul_shift", 0),
            GNodeArg(GNA_IN, "S136_Infos", 0)
        )
    );
    // Node S139_Conv2d_480x1x3x3_Custom inq -3.96<(u8-128.00)*0.03097155<3.93 weightsq chan<(u8-128.00)*chan<chan outq -3.88<(i8-0.00)*0.03031310<3.85 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S139_Conv2d_480x1x3x3_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S136_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_11_block_block_1_block_1_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1480", 0),
            GNodeArg(GNA_OUT, "S139_Output", 0),
            GNodeArg(GNA_IN, "S139_Mul_scale", 0),
            GNodeArg(GNA_IN, "S139_Mul_shift", 0),
            GNodeArg(GNA_IN, "S139_Infos", 0)
        )
    );
    // Node _backbone_features_0_features_0_11_block_block_2_avgpool_GlobalAveragePool_trans inq -3.88<(i8-0.00)*0.03031310<3.85 outq -3.88<(i8-0.00)*0.03031310<3.85
    AddNode("S140_Op__backbone_features_0_features_0_11_block_block_2_avgpool_GlobalAveragePool_trans",
        Bindings(2,
            GNodeArg(GNA_IN, "S139_Output", 0),
            GNodeArg(GNA_OUT, "S140_Output", 0)
        )
    );
    // Node _backbone_features_0_features_0_11_block_block_2_avgpool_GlobalAveragePool inq -3.88<(i8-0.00)*0.03031310<3.85 outq -0.96<(i8-0.00)*0.00752969<0.96
    AddNode("S141_Op__backbone_features_0_features_0_11_block_block_2_avgpool_GlobalAveragePool",
        Bindings(3,
            GNodeArg(GNA_IN, "S140_Output", 0),
            GNodeArg(GNA_OUT, "S141_Output", 0),
            GNodeArg(GNA_IN, "S141_Infos", 0)
        )
    );
    // Node _backbone_features_0_features_0_11_block_block_2_fc1_Conv_fusion_qin0 inq -0.96<(i8-0.00)*0.00752969<0.96 outq -0.36<(u8-70.00)*0.00515881<0.95
    AddNode("S143_Op__backbone_features_0_features_0_11_block_block_2_fc1_Conv_fusion_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S141_Output", 0),
            GNodeArg(GNA_OUT, "S143_Output", 0),
            GNodeArg(GNA_IN, "S143_Infos", 0)
        )
    );
    // Node S146_Conv2d_120x480x1x1_Relu inq -0.36<(u8-70.00)*0.00515881<0.95 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.00958566<2.44 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S146_Conv2d_120x480x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S143_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_11_block_block_2_fc1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_backbone_features_0_11_block_2_fc1_bias", 0),
            GNodeArg(GNA_OUT, "S146_Output", 0),
            GNodeArg(GNA_IN, "S146_Mul_scale", 0),
            GNodeArg(GNA_IN, "S146_Mul_shift", 0),
            GNodeArg(GNA_IN, "S146_Infos", 0)
        )
    );
    // Node S149_Conv2d_480x120x1x1_Hsigmoid inq 0.00<(u8-0.00)*0.00958566<2.44 weightsq chan<(u8-128.00)*chan<chan outq -1.01<(i8-0.00)*0.00787402<1.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S149_Conv2d_480x120x1x1_Hsigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S146_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_11_block_block_2_fc2_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_backbone_features_0_11_block_2_fc2_bias", 0),
            GNodeArg(GNA_OUT, "S149_Output", 0),
            GNodeArg(GNA_IN, "S149_Mul_scale", 0),
            GNodeArg(GNA_IN, "S149_Mul_shift", 0),
            GNodeArg(GNA_IN, "S149_Infos", 0)
        )
    );
    // Node expr_11 in_qs [-1.01<(i8-0.00)*0.00787402<1.00,-3.88<(i8-0.00)*0.03031310<3.85] out_qs [-0.10<(u8-45.00)*0.00229723<0.48]
    AddNode("S150_Op_expr_11",
        Bindings(3,
            GNodeArg(GNA_IN, "S149_Output", 0),
            GNodeArg(GNA_IN, "S139_Output", 0),
            GNodeArg(GNA_OUT, "S150_Output", 0)
        )
    );
    // Node S153_Conv2d_112x480x1x1 inq -0.10<(u8-45.00)*0.00229723<0.48 weightsq chan<(u8-128.00)*chan<chan outq -0.78<(u8-123.00)*0.00637318<0.84 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S153_Conv2d_112x480x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S150_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_11_block_block_3_block_3_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1483", 0),
            GNodeArg(GNA_OUT, "S153_Output", 0),
            GNodeArg(GNA_IN, "S153_Mul_scale", 0),
            GNodeArg(GNA_IN, "S153_Mul_shift", 0),
            GNodeArg(GNA_IN, "S153_Infos", 0)
        )
    );
    // Node S156_Conv2d_672x112x1x1_Custom inq -0.78<(u8-123.00)*0.00637318<0.84 weightsq chan<(u8-128.00)*chan<chan outq -2.23<(u8-128.00)*0.01741075<2.21 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S156_Conv2d_672x112x1x1_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S153_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_12_block_block_0_block_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1486", 0),
            GNodeArg(GNA_OUT, "S156_Output", 0),
            GNodeArg(GNA_IN, "S156_Mul_scale", 0),
            GNodeArg(GNA_IN, "S156_Mul_shift", 0),
            GNodeArg(GNA_IN, "S156_Infos", 0)
        )
    );
    // Node S159_Conv2d_672x1x3x3_Custom inq -2.23<(u8-128.00)*0.01741075<2.21 weightsq chan<(u8-128.00)*chan<chan outq -3.40<(i8-0.00)*0.02653582<3.37 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S159_Conv2d_672x1x3x3_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S156_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_12_block_block_1_block_1_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1489", 0),
            GNodeArg(GNA_OUT, "S159_Output", 0),
            GNodeArg(GNA_IN, "S159_Mul_scale", 0),
            GNodeArg(GNA_IN, "S159_Mul_shift", 0),
            GNodeArg(GNA_IN, "S159_Infos", 0)
        )
    );
    // Node _backbone_features_0_features_0_12_block_block_2_avgpool_GlobalAveragePool_trans inq -3.40<(i8-0.00)*0.02653582<3.37 outq -3.40<(i8-0.00)*0.02653582<3.37
    AddNode("S160_Op__backbone_features_0_features_0_12_block_block_2_avgpool_GlobalAveragePool_trans",
        Bindings(2,
            GNodeArg(GNA_IN, "S159_Output", 0),
            GNodeArg(GNA_OUT, "S160_Output", 0)
        )
    );
    // Node _backbone_features_0_features_0_12_block_block_2_avgpool_GlobalAveragePool inq -3.40<(i8-0.00)*0.02653582<3.37 outq -0.91<(i8-0.00)*0.00710956<0.90
    AddNode("S161_Op__backbone_features_0_features_0_12_block_block_2_avgpool_GlobalAveragePool",
        Bindings(3,
            GNodeArg(GNA_IN, "S160_Output", 0),
            GNodeArg(GNA_OUT, "S161_Output", 0),
            GNodeArg(GNA_IN, "S161_Infos", 0)
        )
    );
    // Node _backbone_features_0_features_0_12_block_block_2_fc1_Conv_fusion_qin0 inq -0.91<(i8-0.00)*0.00710956<0.90 outq -0.38<(u8-75.00)*0.00500570<0.90
    AddNode("S163_Op__backbone_features_0_features_0_12_block_block_2_fc1_Conv_fusion_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S161_Output", 0),
            GNodeArg(GNA_OUT, "S163_Output", 0),
            GNodeArg(GNA_IN, "S163_Infos", 0)
        )
    );
    // Node S166_Conv2d_168x672x1x1_Relu inq -0.38<(u8-75.00)*0.00500570<0.90 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.00878996<2.24 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S166_Conv2d_168x672x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S163_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_12_block_block_2_fc1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_backbone_features_0_12_block_2_fc1_bias", 0),
            GNodeArg(GNA_OUT, "S166_Output", 0),
            GNodeArg(GNA_IN, "S166_Mul_scale", 0),
            GNodeArg(GNA_IN, "S166_Mul_shift", 0),
            GNodeArg(GNA_IN, "S166_Infos", 0)
        )
    );
    // Node S169_Conv2d_672x168x1x1_Hsigmoid inq 0.00<(u8-0.00)*0.00878996<2.24 weightsq chan<(u8-128.00)*chan<chan outq -1.01<(i8-0.00)*0.00787402<1.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S169_Conv2d_672x168x1x1_Hsigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S166_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_12_block_block_2_fc2_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_backbone_features_0_12_block_2_fc2_bias", 0),
            GNodeArg(GNA_OUT, "S169_Output", 0),
            GNodeArg(GNA_IN, "S169_Mul_scale", 0),
            GNodeArg(GNA_IN, "S169_Mul_shift", 0),
            GNodeArg(GNA_IN, "S169_Infos", 0)
        )
    );
    // Node expr_14 in_qs [-1.01<(i8-0.00)*0.00787402<1.00,-3.40<(i8-0.00)*0.02653582<3.37] out_qs [-0.24<(u8-105.00)*0.00227582<0.34]
    AddNode("S170_Op_expr_14",
        Bindings(3,
            GNodeArg(GNA_IN, "S169_Output", 0),
            GNodeArg(GNA_IN, "S159_Output", 0),
            GNodeArg(GNA_OUT, "S170_Output", 0)
        )
    );
    // Node S173_Conv2d_112x672x1x1 inq -0.24<(u8-105.00)*0.00227582<0.34 weightsq chan<(u8-128.00)*chan<chan outq -1.29<(u8-132.00)*0.00974556<1.20 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S173_Conv2d_112x672x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S170_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_12_block_block_3_block_3_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1492", 0),
            GNodeArg(GNA_OUT, "S173_Output", 0),
            GNodeArg(GNA_IN, "S173_Mul_scale", 0),
            GNodeArg(GNA_IN, "S173_Mul_shift", 0),
            GNodeArg(GNA_IN, "S173_Infos", 0)
        )
    );
    // Node S174_MatAdd_16x20x112 in1q -1.29<(u8-132.00)*0.00974556<1.20 forced
    //   in2q -0.78<(u8-123.00)*0.00637318<0.84 forced
    //   outq -1.35<(u8-127.00)*0.01062452<1.36 forced scaled input 0 is node input 0
    AddNode("S174_MatAdd_16x20x112",
        Bindings(4,
            GNodeArg(GNA_IN, "S173_Output", 0),
            GNodeArg(GNA_IN, "S153_Output", 0),
            GNodeArg(GNA_OUT, "S174_Output", 0),
            GNodeArg(GNA_IN, "S174_Infos", 0)
        )
    );
    // Node S177_Conv2d_672x112x1x1_Custom inq -1.35<(u8-127.00)*0.01062452<1.36 forced weightsq chan<(u8-128.00)*chan<chan outq -2.56<(i8-0.00)*0.01998533<2.54 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S177_Conv2d_672x112x1x1_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S174_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_0_features_0_13_features_0_13_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1495", 0),
            GNodeArg(GNA_OUT, "S177_Output", 0),
            GNodeArg(GNA_IN, "S177_Mul_scale", 0),
            GNodeArg(GNA_IN, "S177_Mul_shift", 0),
            GNodeArg(GNA_IN, "S177_Infos", 0)
        )
    );
    // Node _head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_fusion_qin0 inq -2.56<(i8-0.00)*0.01998533<2.54 forced outq -2.56<(u8-128.00)*0.01998533<2.54
    AddNode("S178_Op__head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_fusion_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S177_Output", 0),
            GNodeArg(GNA_OUT, "S178_Output", 0),
            GNodeArg(GNA_IN, "S178_Infos", 0)
        )
    );
    // Node _head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_fusion_qin0 inq -2.56<(i8-0.00)*0.01998533<2.54 forced outq -2.56<(u8-128.00)*0.01998533<2.54
    AddNode("S179_Op__head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_fusion_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S177_Output", 0),
            GNodeArg(GNA_OUT, "S179_Output", 0),
            GNodeArg(GNA_IN, "S179_Infos", 0)
        )
    );
    // Node S182_Conv2d_672x5x5x1_Custom inq -2.56<(i8-0.00)*0.01998533<2.54 forced weightsq chan<(i8-0.00)*chan<chan outq -2.65<(i8-0.00)*0.02069671<2.63 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S182_Conv2d_672x5x5x1_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S177_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_1_features_1_0_features_1_0_1_features_1_0_1_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1498", 0),
            GNodeArg(GNA_OUT, "S182_Output", 0),
            GNodeArg(GNA_IN, "S182_Mul_scale", 0),
            GNodeArg(GNA_IN, "S182_Mul_shift", 0),
            GNodeArg(GNA_IN, "S182_Infos", 0)
        )
    );
    // Node _backbone_features_1_features_1_0_features_1_0_2_avgpool_GlobalAveragePool_trans inq -2.65<(i8-0.00)*0.02069671<2.63 outq -2.65<(i8-0.00)*0.02069671<2.63
    AddNode("S183_Op__backbone_features_1_features_1_0_features_1_0_2_avgpool_GlobalAveragePool_trans",
        Bindings(2,
            GNodeArg(GNA_IN, "S182_Output", 0),
            GNodeArg(GNA_OUT, "S183_Output", 0)
        )
    );
    // Node _backbone_features_1_features_1_0_features_1_0_2_avgpool_GlobalAveragePool inq -2.65<(i8-0.00)*0.02069671<2.63 outq -0.89<(i8-0.00)*0.00694834<0.88
    AddNode("S184_Op__backbone_features_1_features_1_0_features_1_0_2_avgpool_GlobalAveragePool",
        Bindings(3,
            GNodeArg(GNA_IN, "S183_Output", 0),
            GNodeArg(GNA_OUT, "S184_Output", 0),
            GNodeArg(GNA_IN, "S184_Infos", 0)
        )
    );
    // Node _backbone_features_1_features_1_0_features_1_0_2_fc1_Conv_fusion_qin0 inq -0.89<(i8-0.00)*0.00694834<0.88 outq -0.37<(u8-75.00)*0.00490833<0.88
    AddNode("S186_Op__backbone_features_1_features_1_0_features_1_0_2_fc1_Conv_fusion_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S184_Output", 0),
            GNodeArg(GNA_OUT, "S186_Output", 0),
            GNodeArg(GNA_IN, "S186_Infos", 0)
        )
    );
    // Node S189_Conv2d_168x672x1x1_Relu inq -0.37<(u8-75.00)*0.00490833<0.88 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.00468539<1.19 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S189_Conv2d_168x672x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S186_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_1_features_1_0_features_1_0_2_fc1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_backbone_features_1_0_2_fc1_bias", 0),
            GNodeArg(GNA_OUT, "S189_Output", 0),
            GNodeArg(GNA_IN, "S189_Mul_scale", 0),
            GNodeArg(GNA_IN, "S189_Mul_shift", 0),
            GNodeArg(GNA_IN, "S189_Infos", 0)
        )
    );
    // Node S192_Conv2d_672x168x1x1_Hsigmoid inq 0.00<(u8-0.00)*0.00468539<1.19 weightsq chan<(u8-128.00)*chan<chan outq -1.01<(i8-0.00)*0.00787402<1.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S192_Conv2d_672x168x1x1_Hsigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S189_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_1_features_1_0_features_1_0_2_fc2_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_backbone_features_1_0_2_fc2_bias", 0),
            GNodeArg(GNA_OUT, "S192_Output", 0),
            GNodeArg(GNA_IN, "S192_Mul_scale", 0),
            GNodeArg(GNA_IN, "S192_Mul_shift", 0),
            GNodeArg(GNA_IN, "S192_Infos", 0)
        )
    );
    // Node expr_26 in_qs [-1.01<(i8-0.00)*0.00787402<1.00,-2.65<(i8-0.00)*0.02069671<2.63] out_qs [-0.19<(u8-72.00)*0.00265136<0.49]
    AddNode("S193_Op_expr_26",
        Bindings(3,
            GNodeArg(GNA_IN, "S192_Output", 0),
            GNodeArg(GNA_IN, "S182_Output", 0),
            GNodeArg(GNA_OUT, "S193_Output", 0)
        )
    );
    // Node S196_Conv2d_80x672x1x1 inq -0.19<(u8-72.00)*0.00265136<0.49 weightsq chan<(u8-128.00)*chan<chan outq -0.69<(u8-128.00)*0.00537080<0.68 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S196_Conv2d_80x672x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S193_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_1_features_1_0_features_1_0_3_features_1_0_3_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1501", 0),
            GNodeArg(GNA_OUT, "S196_Output", 0),
            GNodeArg(GNA_IN, "S196_Mul_scale", 0),
            GNodeArg(GNA_IN, "S196_Mul_shift", 0),
            GNodeArg(GNA_IN, "S196_Infos", 0)
        )
    );
    // Node S199_Conv2d_480x80x1x1_Custom inq -0.69<(u8-128.00)*0.00537080<0.68 weightsq chan<(u8-128.00)*chan<chan outq -0.92<(i8-0.00)*0.00718334<0.91 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S199_Conv2d_480x80x1x1_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S196_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_1_features_1_1_block_block_0_block_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1504", 0),
            GNodeArg(GNA_OUT, "S199_Output", 0),
            GNodeArg(GNA_IN, "S199_Mul_scale", 0),
            GNodeArg(GNA_IN, "S199_Mul_shift", 0),
            GNodeArg(GNA_IN, "S199_Infos", 0)
        )
    );
    // Node S202_Conv2d_480x5x5x1_Custom inq -0.92<(i8-0.00)*0.00718334<0.91 forced weightsq chan<(i8-0.00)*chan<chan outq -0.82<(i8-0.00)*0.00638288<0.81 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S202_Conv2d_480x5x5x1_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S199_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_1_features_1_1_block_block_1_block_1_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1507", 0),
            GNodeArg(GNA_OUT, "S202_Output", 0),
            GNodeArg(GNA_IN, "S202_Mul_scale", 0),
            GNodeArg(GNA_IN, "S202_Mul_shift", 0),
            GNodeArg(GNA_IN, "S202_Infos", 0)
        )
    );
    // Node _backbone_features_1_features_1_1_block_block_2_avgpool_GlobalAveragePool_trans inq -0.82<(i8-0.00)*0.00638288<0.81 outq -0.82<(i8-0.00)*0.00638288<0.81
    AddNode("S203_Op__backbone_features_1_features_1_1_block_block_2_avgpool_GlobalAveragePool_trans",
        Bindings(2,
            GNodeArg(GNA_IN, "S202_Output", 0),
            GNodeArg(GNA_OUT, "S203_Output", 0)
        )
    );
    // Node _backbone_features_1_features_1_1_block_block_2_avgpool_GlobalAveragePool inq -0.82<(i8-0.00)*0.00638288<0.81 outq -0.36<(i8-0.00)*0.00281069<0.36
    AddNode("S204_Op__backbone_features_1_features_1_1_block_block_2_avgpool_GlobalAveragePool",
        Bindings(3,
            GNodeArg(GNA_IN, "S203_Output", 0),
            GNodeArg(GNA_OUT, "S204_Output", 0),
            GNodeArg(GNA_IN, "S204_Infos", 0)
        )
    );
    // Node _backbone_features_1_features_1_1_block_block_2_fc1_Conv_fusion_qin0 inq -0.36<(i8-0.00)*0.00281069<0.36 outq -0.36<(u8-183.00)*0.00196114<0.14
    AddNode("S206_Op__backbone_features_1_features_1_1_block_block_2_fc1_Conv_fusion_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S204_Output", 0),
            GNodeArg(GNA_OUT, "S206_Output", 0),
            GNodeArg(GNA_IN, "S206_Infos", 0)
        )
    );
    // Node S209_Conv2d_120x480x1x1_Relu inq -0.36<(u8-183.00)*0.00196114<0.14 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.00201594<0.51 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S209_Conv2d_120x480x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S206_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_1_features_1_1_block_block_2_fc1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_backbone_features_1_1_block_2_fc1_bias", 0),
            GNodeArg(GNA_OUT, "S209_Output", 0),
            GNodeArg(GNA_IN, "S209_Mul_scale", 0),
            GNodeArg(GNA_IN, "S209_Mul_shift", 0),
            GNodeArg(GNA_IN, "S209_Infos", 0)
        )
    );
    // Node S212_Conv2d_480x120x1x1_Hsigmoid inq 0.00<(u8-0.00)*0.00201594<0.51 weightsq chan<(u8-128.00)*chan<chan outq -1.01<(i8-0.00)*0.00787402<1.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S212_Conv2d_480x120x1x1_Hsigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S209_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_1_features_1_1_block_block_2_fc2_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_backbone_features_1_1_block_2_fc2_bias", 0),
            GNodeArg(GNA_OUT, "S212_Output", 0),
            GNodeArg(GNA_IN, "S212_Mul_scale", 0),
            GNodeArg(GNA_IN, "S212_Mul_shift", 0),
            GNodeArg(GNA_IN, "S212_Infos", 0)
        )
    );
    // Node expr_29 in_qs [-1.01<(i8-0.00)*0.00787402<1.00,-0.82<(i8-0.00)*0.00638288<0.81] out_qs [-0.20<(u8-82.00)*0.00248038<0.43]
    AddNode("S213_Op_expr_29",
        Bindings(3,
            GNodeArg(GNA_IN, "S212_Output", 0),
            GNodeArg(GNA_IN, "S202_Output", 0),
            GNodeArg(GNA_OUT, "S213_Output", 0)
        )
    );
    // Node S216_Conv2d_80x480x1x1 inq -0.20<(u8-82.00)*0.00248038<0.43 weightsq chan<(u8-128.00)*chan<chan outq -0.51<(u8-106.00)*0.00482429<0.72 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S216_Conv2d_80x480x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S213_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_1_features_1_1_block_block_3_block_3_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1510", 0),
            GNodeArg(GNA_OUT, "S216_Output", 0),
            GNodeArg(GNA_IN, "S216_Mul_scale", 0),
            GNodeArg(GNA_IN, "S216_Mul_shift", 0),
            GNodeArg(GNA_IN, "S216_Infos", 0)
        )
    );
    // Node S217_MatAdd_8x10x80 in1q -0.69<(u8-128.00)*0.00537080<0.68 forced
    //   in2q -0.51<(u8-106.00)*0.00482429<0.72 forced
    //   outq -0.82<(u8-124.00)*0.00657713<0.86 forced scaled input 0 is node input 1
    AddNode("S217_MatAdd_8x10x80",
        Bindings(4,
            GNodeArg(GNA_IN, "S196_Output", 0),
            GNodeArg(GNA_IN, "S216_Output", 0),
            GNodeArg(GNA_OUT, "S217_Output", 0),
            GNodeArg(GNA_IN, "S217_Infos", 0)
        )
    );
    // Node S220_Conv2d_480x80x1x1_Custom inq -0.82<(u8-124.00)*0.00657713<0.86 forced weightsq chan<(u8-128.00)*chan<chan outq -0.91<(i8-0.00)*0.00709636<0.90 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S220_Conv2d_480x80x1x1_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S217_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_1_features_1_2_block_block_0_block_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1513", 0),
            GNodeArg(GNA_OUT, "S220_Output", 0),
            GNodeArg(GNA_IN, "S220_Mul_scale", 0),
            GNodeArg(GNA_IN, "S220_Mul_shift", 0),
            GNodeArg(GNA_IN, "S220_Infos", 0)
        )
    );
    // Node S223_Conv2d_480x5x5x1_Custom inq -0.91<(i8-0.00)*0.00709636<0.90 forced weightsq chan<(i8-0.00)*chan<chan outq -0.68<(i8-0.00)*0.00530376<0.67 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S223_Conv2d_480x5x5x1_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S220_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_1_features_1_2_block_block_1_block_1_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1516", 0),
            GNodeArg(GNA_OUT, "S223_Output", 0),
            GNodeArg(GNA_IN, "S223_Mul_scale", 0),
            GNodeArg(GNA_IN, "S223_Mul_shift", 0),
            GNodeArg(GNA_IN, "S223_Infos", 0)
        )
    );
    // Node _backbone_features_1_features_1_2_block_block_2_avgpool_GlobalAveragePool_trans inq -0.68<(i8-0.00)*0.00530376<0.67 outq -0.68<(i8-0.00)*0.00530376<0.67
    AddNode("S224_Op__backbone_features_1_features_1_2_block_block_2_avgpool_GlobalAveragePool_trans",
        Bindings(2,
            GNodeArg(GNA_IN, "S223_Output", 0),
            GNodeArg(GNA_OUT, "S224_Output", 0)
        )
    );
    // Node _backbone_features_1_features_1_2_block_block_2_avgpool_GlobalAveragePool inq -0.68<(i8-0.00)*0.00530376<0.67 outq -0.32<(i8-0.00)*0.00252946<0.32
    AddNode("S225_Op__backbone_features_1_features_1_2_block_block_2_avgpool_GlobalAveragePool",
        Bindings(3,
            GNodeArg(GNA_IN, "S224_Output", 0),
            GNodeArg(GNA_OUT, "S225_Output", 0),
            GNodeArg(GNA_IN, "S225_Infos", 0)
        )
    );
    // Node _backbone_features_1_features_1_2_block_block_2_fc1_Conv_fusion_qin0 inq -0.32<(i8-0.00)*0.00252946<0.32 outq -0.32<(u8-161.00)*0.00201196<0.19
    AddNode("S227_Op__backbone_features_1_features_1_2_block_block_2_fc1_Conv_fusion_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S225_Output", 0),
            GNodeArg(GNA_OUT, "S227_Output", 0),
            GNodeArg(GNA_IN, "S227_Infos", 0)
        )
    );
    // Node S230_Conv2d_120x480x1x1_Relu inq -0.32<(u8-161.00)*0.00201196<0.19 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.00316438<0.81 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S230_Conv2d_120x480x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S227_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_1_features_1_2_block_block_2_fc1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_backbone_features_1_2_block_2_fc1_bias", 0),
            GNodeArg(GNA_OUT, "S230_Output", 0),
            GNodeArg(GNA_IN, "S230_Mul_scale", 0),
            GNodeArg(GNA_IN, "S230_Mul_shift", 0),
            GNodeArg(GNA_IN, "S230_Infos", 0)
        )
    );
    // Node S233_Conv2d_480x120x1x1_Hsigmoid inq 0.00<(u8-0.00)*0.00316438<0.81 weightsq chan<(u8-128.00)*chan<chan outq -1.01<(i8-0.00)*0.00787402<1.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S233_Conv2d_480x120x1x1_Hsigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S230_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_1_features_1_2_block_block_2_fc2_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_backbone_features_1_2_block_2_fc2_bias", 0),
            GNodeArg(GNA_OUT, "S233_Output", 0),
            GNodeArg(GNA_IN, "S233_Mul_scale", 0),
            GNodeArg(GNA_IN, "S233_Mul_shift", 0),
            GNodeArg(GNA_IN, "S233_Infos", 0)
        )
    );
    // Node expr_32 in_qs [-1.01<(i8-0.00)*0.00787402<1.00,-0.68<(i8-0.00)*0.00530376<0.67] out_qs [-0.21<(u8-96.00)*0.00214075<0.34]
    AddNode("S234_Op_expr_32",
        Bindings(3,
            GNodeArg(GNA_IN, "S233_Output", 0),
            GNodeArg(GNA_IN, "S223_Output", 0),
            GNodeArg(GNA_OUT, "S234_Output", 0)
        )
    );
    // Node S237_Conv2d_80x480x1x1 inq -0.21<(u8-96.00)*0.00214075<0.34 weightsq chan<(u8-128.00)*chan<chan outq -0.88<(u8-129.00)*0.00683760<0.86 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S237_Conv2d_80x480x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S234_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_1_features_1_2_block_block_3_block_3_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1519", 0),
            GNodeArg(GNA_OUT, "S237_Output", 0),
            GNodeArg(GNA_IN, "S237_Mul_scale", 0),
            GNodeArg(GNA_IN, "S237_Mul_shift", 0),
            GNodeArg(GNA_IN, "S237_Infos", 0)
        )
    );
    // Node S238_MatAdd_8x10x80 in1q -0.88<(u8-129.00)*0.00683760<0.86 forced
    //   in2q -0.82<(u8-124.00)*0.00657713<0.86 forced
    //   outq -1.11<(u8-123.00)*0.00900292<1.19 forced scaled input 0 is node input 0
    AddNode("S238_MatAdd_8x10x80",
        Bindings(4,
            GNodeArg(GNA_IN, "S237_Output", 0),
            GNodeArg(GNA_IN, "S217_Output", 0),
            GNodeArg(GNA_OUT, "S238_Output", 0),
            GNodeArg(GNA_IN, "S238_Infos", 0)
        )
    );
    // Node S241_Conv2d_480x80x1x1_Custom inq -1.11<(u8-123.00)*0.00900292<1.19 forced weightsq chan<(u8-128.00)*chan<chan outq -1.37<(u8-128.00)*0.01072368<1.36 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S241_Conv2d_480x80x1x1_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S238_Output", 0),
            GNodeArg(GNA_IN, "_backbone_features_1_features_1_3_features_1_3_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1522", 0),
            GNodeArg(GNA_OUT, "S241_Output", 0),
            GNodeArg(GNA_IN, "S241_Mul_scale", 0),
            GNodeArg(GNA_IN, "S241_Mul_shift", 0),
            GNodeArg(GNA_IN, "S241_Infos", 0)
        )
    );
    // Node S244_Conv2d_256x480x1x1_Relu6 inq -1.37<(u8-128.00)*0.01072368<1.36 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S244_Conv2d_256x480x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S241_Output", 0),
            GNodeArg(GNA_IN, "_backbone_extra_0_extra_0_0_extra_0_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1525", 0),
            GNodeArg(GNA_OUT, "S244_Output", 0),
            GNodeArg(GNA_IN, "S244_Mul_scale", 0),
            GNodeArg(GNA_IN, "S244_Mul_shift", 0),
            GNodeArg(GNA_IN, "S244_Infos", 0)
        )
    );
    // Node S247_Conv2d_256x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S247_Conv2d_256x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S244_Output", 0),
            GNodeArg(GNA_IN, "_backbone_extra_0_extra_0_1_extra_0_1_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1528", 0),
            GNodeArg(GNA_OUT, "S247_Output", 0),
            GNodeArg(GNA_IN, "S247_Mul_scale", 0),
            GNodeArg(GNA_IN, "S247_Mul_shift", 0),
            GNodeArg(GNA_IN, "S247_Infos", 0)
        )
    );
    // Node S250_Conv2d_512x256x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S250_Conv2d_512x256x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S247_Output", 0),
            GNodeArg(GNA_IN, "_backbone_extra_0_extra_0_2_extra_0_2_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1531", 0),
            GNodeArg(GNA_OUT, "S250_Output", 0),
            GNodeArg(GNA_IN, "S250_Mul_scale", 0),
            GNodeArg(GNA_IN, "S250_Mul_shift", 0),
            GNodeArg(GNA_IN, "S250_Infos", 0)
        )
    );
    // Node S253_Conv2d_128x512x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S253_Conv2d_128x512x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S250_Output", 0),
            GNodeArg(GNA_IN, "_backbone_extra_1_extra_1_0_extra_1_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1534", 0),
            GNodeArg(GNA_OUT, "S253_Output", 0),
            GNodeArg(GNA_IN, "S253_Mul_scale", 0),
            GNodeArg(GNA_IN, "S253_Mul_shift", 0),
            GNodeArg(GNA_IN, "S253_Infos", 0)
        )
    );
    // Node S256_Conv2d_128x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S256_Conv2d_128x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S253_Output", 0),
            GNodeArg(GNA_IN, "_backbone_extra_1_extra_1_1_extra_1_1_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1537", 0),
            GNodeArg(GNA_OUT, "S256_Output", 0),
            GNodeArg(GNA_IN, "S256_Mul_scale", 0),
            GNodeArg(GNA_IN, "S256_Mul_shift", 0),
            GNodeArg(GNA_IN, "S256_Infos", 0)
        )
    );
    // Node S259_Conv2d_256x128x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S259_Conv2d_256x128x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S256_Output", 0),
            GNodeArg(GNA_IN, "_backbone_extra_1_extra_1_2_extra_1_2_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1540", 0),
            GNodeArg(GNA_OUT, "S259_Output", 0),
            GNodeArg(GNA_IN, "S259_Mul_scale", 0),
            GNodeArg(GNA_IN, "S259_Mul_shift", 0),
            GNodeArg(GNA_IN, "S259_Infos", 0)
        )
    );
    // Node S262_Conv2d_128x256x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S262_Conv2d_128x256x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S259_Output", 0),
            GNodeArg(GNA_IN, "_backbone_extra_2_extra_2_0_extra_2_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1543", 0),
            GNodeArg(GNA_OUT, "S262_Output", 0),
            GNodeArg(GNA_IN, "S262_Mul_scale", 0),
            GNodeArg(GNA_IN, "S262_Mul_shift", 0),
            GNodeArg(GNA_IN, "S262_Infos", 0)
        )
    );
    // Node S265_Conv2d_128x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S265_Conv2d_128x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S262_Output", 0),
            GNodeArg(GNA_IN, "_backbone_extra_2_extra_2_1_extra_2_1_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1546", 0),
            GNodeArg(GNA_OUT, "S265_Output", 0),
            GNodeArg(GNA_IN, "S265_Mul_scale", 0),
            GNodeArg(GNA_IN, "S265_Mul_shift", 0),
            GNodeArg(GNA_IN, "S265_Infos", 0)
        )
    );
    // Node S268_Conv2d_256x128x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S268_Conv2d_256x128x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S265_Output", 0),
            GNodeArg(GNA_IN, "_backbone_extra_2_extra_2_2_extra_2_2_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1549", 0),
            GNodeArg(GNA_OUT, "S268_Output", 0),
            GNodeArg(GNA_IN, "S268_Mul_scale", 0),
            GNodeArg(GNA_IN, "S268_Mul_shift", 0),
            GNodeArg(GNA_IN, "S268_Infos", 0)
        )
    );
    // Node S271_Conv2d_64x256x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S271_Conv2d_64x256x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S268_Output", 0),
            GNodeArg(GNA_IN, "_backbone_extra_3_extra_3_0_extra_3_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1552", 0),
            GNodeArg(GNA_OUT, "S271_Output", 0),
            GNodeArg(GNA_IN, "S271_Mul_scale", 0),
            GNodeArg(GNA_IN, "S271_Mul_shift", 0),
            GNodeArg(GNA_IN, "S271_Infos", 0)
        )
    );
    // Node S274_Conv2d_64x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S274_Conv2d_64x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S271_Output", 0),
            GNodeArg(GNA_IN, "_backbone_extra_3_extra_3_1_extra_3_1_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1555", 0),
            GNodeArg(GNA_OUT, "S274_Output", 0),
            GNodeArg(GNA_IN, "S274_Mul_scale", 0),
            GNodeArg(GNA_IN, "S274_Mul_shift", 0),
            GNodeArg(GNA_IN, "S274_Infos", 0)
        )
    );
    // Node S277_Conv2d_128x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S277_Conv2d_128x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S274_Output", 0),
            GNodeArg(GNA_IN, "_backbone_extra_3_extra_3_2_extra_3_2_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1558", 0),
            GNodeArg(GNA_OUT, "S277_Output", 0),
            GNodeArg(GNA_IN, "S277_Mul_scale", 0),
            GNodeArg(GNA_IN, "S277_Mul_shift", 0),
            GNodeArg(GNA_IN, "S277_Infos", 0)
        )
    );
    // Node S280_Conv2d_672x1x3x3_Relu6 inq -2.56<(u8-128.00)*0.01998533<2.54 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S280_Conv2d_672x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S178_Output", 0),
            GNodeArg(GNA_IN, "_head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1561", 0),
            GNodeArg(GNA_OUT, "S280_Output", 0),
            GNodeArg(GNA_IN, "S280_Mul_scale", 0),
            GNodeArg(GNA_IN, "S280_Mul_shift", 0),
            GNodeArg(GNA_IN, "S280_Infos", 0)
        )
    );
    // Node S283_Conv2d_24x672x1x1_Custom inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq -0.00<(u8-128.00)*0.00002718<0.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S283_Conv2d_24x672x1x1_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S280_Output", 0),
            GNodeArg(GNA_IN, "_head_regression_head_module_list_0_module_list_0_1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_head_regression_head_module_list_0_1_bias", 0),
            GNodeArg(GNA_OUT, "S283_Output", 0),
            GNodeArg(GNA_IN, "S283_Mul_scale", 0),
            GNodeArg(GNA_IN, "S283_Mul_shift", 0),
            GNodeArg(GNA_IN, "S283_Infos", 0)
        )
    );
    // Node S287_Conv2d_480x1x3x3_Relu6 inq -1.37<(u8-128.00)*0.01072368<1.36 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S287_Conv2d_480x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S241_Output", 0),
            GNodeArg(GNA_IN, "_head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1564", 0),
            GNodeArg(GNA_OUT, "S287_Output", 0),
            GNodeArg(GNA_IN, "S287_Mul_scale", 0),
            GNodeArg(GNA_IN, "S287_Mul_shift", 0),
            GNodeArg(GNA_IN, "S287_Infos", 0)
        )
    );
    // Node S291_Conv2d_512x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S291_Conv2d_512x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S250_Output", 0),
            GNodeArg(GNA_IN, "_head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1567", 0),
            GNodeArg(GNA_OUT, "S291_Output", 0),
            GNodeArg(GNA_IN, "S291_Mul_scale", 0),
            GNodeArg(GNA_IN, "S291_Mul_shift", 0),
            GNodeArg(GNA_IN, "S291_Infos", 0)
        )
    );
    // Node S295_Conv2d_256x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S295_Conv2d_256x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S259_Output", 0),
            GNodeArg(GNA_IN, "_head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1570", 0),
            GNodeArg(GNA_OUT, "S295_Output", 0),
            GNodeArg(GNA_IN, "S295_Mul_scale", 0),
            GNodeArg(GNA_IN, "S295_Mul_shift", 0),
            GNodeArg(GNA_IN, "S295_Infos", 0)
        )
    );
    // Node S301_Conv2d_128x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S301_Conv2d_128x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S277_Output", 0),
            GNodeArg(GNA_IN, "_head_regression_head_module_list_5_module_list_5_0_module_list_5_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1576", 0),
            GNodeArg(GNA_OUT, "S301_Output", 0),
            GNodeArg(GNA_IN, "S301_Mul_scale", 0),
            GNodeArg(GNA_IN, "S301_Mul_shift", 0),
            GNodeArg(GNA_IN, "S301_Infos", 0)
        )
    );
    // Node S321_Conv2d_672x1x3x3_Relu6 inq -2.56<(u8-128.00)*0.01998533<2.54 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S321_Conv2d_672x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S179_Output", 0),
            GNodeArg(GNA_IN, "_head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1561_1", 0),
            GNodeArg(GNA_OUT, "S321_Output", 0),
            GNodeArg(GNA_IN, "S321_Mul_scale", 0),
            GNodeArg(GNA_IN, "S321_Mul_shift", 0),
            GNodeArg(GNA_IN, "S321_Infos", 0)
        )
    );
    // Node S322_Conv2d_186x672x1x1 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq -0.06<(i8-0.00)*0.00048828<0.06 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S322_Conv2d_186x672x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S321_Output", 0),
            GNodeArg(GNA_IN, "_head_classification_head_module_list_0_module_list_0_1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_head_classification_head_module_list_0_1_bias", 0),
            GNodeArg(GNA_OUT, "S322_Output", 0),
            GNodeArg(GNA_IN, "S322_Mul_scale", 0),
            GNodeArg(GNA_IN, "S322_Mul_shift", 0),
            GNodeArg(GNA_IN, "S322_Infos", 0)
        )
    );
    // Node S325_Conv2d_24x480x1x1_Custom inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq -0.00<(u8-128.00)*0.00002718<0.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S325_Conv2d_24x480x1x1_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S287_Output", 0),
            GNodeArg(GNA_IN, "_head_regression_head_module_list_1_module_list_1_1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_head_regression_head_module_list_0_1_bias_1", 0),
            GNodeArg(GNA_OUT, "S325_Output", 0),
            GNodeArg(GNA_IN, "S325_Mul_scale", 0),
            GNodeArg(GNA_IN, "S325_Mul_shift", 0),
            GNodeArg(GNA_IN, "S325_Infos", 0)
        )
    );
    // Node S328_Conv2d_24x512x1x1_Custom inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq -0.00<(u8-128.00)*0.00002718<0.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S328_Conv2d_24x512x1x1_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S291_Output", 0),
            GNodeArg(GNA_IN, "_head_regression_head_module_list_2_module_list_2_1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_head_regression_head_module_list_0_1_bias_2", 0),
            GNodeArg(GNA_OUT, "S328_Output", 0),
            GNodeArg(GNA_IN, "S328_Mul_scale", 0),
            GNodeArg(GNA_IN, "S328_Mul_shift", 0),
            GNodeArg(GNA_IN, "S328_Infos", 0)
        )
    );
    // Node S331_Conv2d_24x256x1x1_Custom inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq -0.00<(u8-128.00)*0.00002718<0.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S331_Conv2d_24x256x1x1_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S295_Output", 0),
            GNodeArg(GNA_IN, "_head_regression_head_module_list_3_module_list_3_1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_head_regression_head_module_list_0_1_bias_3", 0),
            GNodeArg(GNA_OUT, "S331_Output", 0),
            GNodeArg(GNA_IN, "S331_Mul_scale", 0),
            GNodeArg(GNA_IN, "S331_Mul_shift", 0),
            GNodeArg(GNA_IN, "S331_Infos", 0)
        )
    );
    // Node S335_Conv2d_24x128x1x1_Custom inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq -0.00<(u8-128.00)*0.00002718<0.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S335_Conv2d_24x128x1x1_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S301_Output", 0),
            GNodeArg(GNA_IN, "_head_regression_head_module_list_5_module_list_5_1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_head_regression_head_module_list_0_1_bias_5", 0),
            GNodeArg(GNA_OUT, "S335_Output", 0),
            GNodeArg(GNA_IN, "S335_Mul_scale", 0),
            GNodeArg(GNA_IN, "S335_Mul_shift", 0),
            GNodeArg(GNA_IN, "S335_Infos", 0)
        )
    );
    // Node S338_Conv2d_480x1x3x3_Relu6 inq -1.37<(u8-128.00)*0.01072368<1.36 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S338_Conv2d_480x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S241_Output", 0),
            GNodeArg(GNA_IN, "_head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1564_1", 0),
            GNodeArg(GNA_OUT, "S338_Output", 0),
            GNodeArg(GNA_IN, "S338_Mul_scale", 0),
            GNodeArg(GNA_IN, "S338_Mul_shift", 0),
            GNodeArg(GNA_IN, "S338_Infos", 0)
        )
    );
    // Node S340_Conv2d_512x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S340_Conv2d_512x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S250_Output", 0),
            GNodeArg(GNA_IN, "_head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1567_1", 0),
            GNodeArg(GNA_OUT, "S340_Output", 0),
            GNodeArg(GNA_IN, "S340_Mul_scale", 0),
            GNodeArg(GNA_IN, "S340_Mul_shift", 0),
            GNodeArg(GNA_IN, "S340_Infos", 0)
        )
    );
    // Node S342_Conv2d_256x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S342_Conv2d_256x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S268_Output", 0),
            GNodeArg(GNA_IN, "_head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1570_1", 0),
            GNodeArg(GNA_OUT, "S342_Output", 0),
            GNodeArg(GNA_IN, "S342_Mul_scale", 0),
            GNodeArg(GNA_IN, "S342_Mul_shift", 0),
            GNodeArg(GNA_IN, "S342_Infos", 0)
        )
    );
    // Node S343_Conv2d_24x256x1x1_Custom inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq -0.00<(u8-128.00)*0.00002718<0.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S343_Conv2d_24x256x1x1_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S342_Output", 0),
            GNodeArg(GNA_IN, "_head_regression_head_module_list_4_module_list_4_1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_head_regression_head_module_list_0_1_bias_4", 0),
            GNodeArg(GNA_OUT, "S343_Output", 0),
            GNodeArg(GNA_IN, "S343_Mul_scale", 0),
            GNodeArg(GNA_IN, "S343_Mul_shift", 0),
            GNodeArg(GNA_IN, "S343_Infos", 0)
        )
    );
    // Node _Split inq ['-0.00<(u8-128.00)*0.00002718<0.00'] outq ['-0.00<(u8-128.00)*0.00002718<0.00', '-0.00<(u8-128.00)*0.00002718<0.00', '-0.00<(u8-128.00)*0.00002718<0.00', '-0.00<(u8-128.00)*0.00002718<0.00']
    AddNode("S346_Op__Split",
        Bindings(5,
            GNodeArg(GNA_IN, "S345_Output", 0),
            GNodeArg(GNA_OUT, "S346_Output_0", 0),
            GNodeArg(GNA_OUT, "S346_Output_1", 0),
            GNodeArg(GNA_OUT, "S346_Output_2", 0),
            GNodeArg(GNA_OUT, "S346_Output_3", 0)
        )
    );
    // Node S352_Conv2d_256x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S352_Conv2d_256x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S259_Output", 0),
            GNodeArg(GNA_IN, "_head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1570_2", 0),
            GNodeArg(GNA_OUT, "S352_Output", 0),
            GNodeArg(GNA_IN, "S352_Mul_scale", 0),
            GNodeArg(GNA_IN, "S352_Mul_shift", 0),
            GNodeArg(GNA_IN, "S352_Infos", 0)
        )
    );
    // Node S354_Conv2d_256x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S354_Conv2d_256x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S268_Output", 0),
            GNodeArg(GNA_IN, "_head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1570_3", 0),
            GNodeArg(GNA_OUT, "S354_Output", 0),
            GNodeArg(GNA_IN, "S354_Mul_scale", 0),
            GNodeArg(GNA_IN, "S354_Mul_shift", 0),
            GNodeArg(GNA_IN, "S354_Infos", 0)
        )
    );
    // Node S356_Conv2d_128x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S356_Conv2d_128x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S277_Output", 0),
            GNodeArg(GNA_IN, "_head_classification_head_module_list_5_module_list_5_0_module_list_5_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1576_1", 0),
            GNodeArg(GNA_OUT, "S356_Output", 0),
            GNodeArg(GNA_IN, "S356_Mul_scale", 0),
            GNodeArg(GNA_IN, "S356_Mul_shift", 0),
            GNodeArg(GNA_IN, "S356_Infos", 0)
        )
    );
    // Node S358_Conv2d_186x480x1x1 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq -0.06<(i8-0.00)*0.00048828<0.06 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S358_Conv2d_186x480x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S338_Output", 0),
            GNodeArg(GNA_IN, "_head_classification_head_module_list_1_module_list_1_1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_head_classification_head_module_list_0_1_bias_1", 0),
            GNodeArg(GNA_OUT, "S358_Output", 0),
            GNodeArg(GNA_IN, "S358_Mul_scale", 0),
            GNodeArg(GNA_IN, "S358_Mul_shift", 0),
            GNodeArg(GNA_IN, "S358_Infos", 0)
        )
    );
    // Node S361_Conv2d_186x512x1x1 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq -0.06<(i8-0.00)*0.00048828<0.06 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S361_Conv2d_186x512x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S340_Output", 0),
            GNodeArg(GNA_IN, "_head_classification_head_module_list_2_module_list_2_1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_head_classification_head_module_list_0_1_bias_2", 0),
            GNodeArg(GNA_OUT, "S361_Output", 0),
            GNodeArg(GNA_IN, "S361_Mul_scale", 0),
            GNodeArg(GNA_IN, "S361_Mul_shift", 0),
            GNodeArg(GNA_IN, "S361_Infos", 0)
        )
    );
    // Node S364_Conv2d_186x256x1x1 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq -0.06<(i8-0.00)*0.00048828<0.06 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S364_Conv2d_186x256x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S352_Output", 0),
            GNodeArg(GNA_IN, "_head_classification_head_module_list_3_module_list_3_1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_head_classification_head_module_list_0_1_bias_3", 0),
            GNodeArg(GNA_OUT, "S364_Output", 0),
            GNodeArg(GNA_IN, "S364_Mul_scale", 0),
            GNodeArg(GNA_IN, "S364_Mul_shift", 0),
            GNodeArg(GNA_IN, "S364_Infos", 0)
        )
    );
    // Node S367_Conv2d_186x256x1x1 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq -0.06<(i8-0.00)*0.00048828<0.06 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S367_Conv2d_186x256x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S354_Output", 0),
            GNodeArg(GNA_IN, "_head_classification_head_module_list_4_module_list_4_1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_head_classification_head_module_list_0_1_bias_4", 0),
            GNodeArg(GNA_OUT, "S367_Output", 0),
            GNodeArg(GNA_IN, "S367_Mul_scale", 0),
            GNodeArg(GNA_IN, "S367_Mul_shift", 0),
            GNodeArg(GNA_IN, "S367_Infos", 0)
        )
    );
    // Node S370_Conv2d_186x128x1x1 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq -0.06<(i8-0.00)*0.00048828<0.06 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S370_Conv2d_186x128x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S356_Output", 0),
            GNodeArg(GNA_IN, "_head_classification_head_module_list_5_module_list_5_1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_model_head_classification_head_module_list_0_1_bias_5", 0),
            GNodeArg(GNA_OUT, "S370_Output", 0),
            GNodeArg(GNA_IN, "S370_Mul_scale", 0),
            GNodeArg(GNA_IN, "S370_Mul_shift", 0),
            GNodeArg(GNA_IN, "S370_Infos", 0)
        )
    );
    // Node _Softmax inq -0.06<(i8-0.00)*0.00048828<0.06 forced outq -1.00<(i16-0.00)*0.00003052<1.00
    AddNode("S373_SoftMax",
        Bindings(3,
            GNodeArg(GNA_IN, "S372_Output", 0),
            GNodeArg(GNA_OUT, "Output_5", 0),
            GNodeArg(GNA_IN, "S373_Infos", 0)
        )
    );
    // Node expr_1 in_qs [-0.00<(u8-128.00)*0.00002718<0.00,-258.02<(i8-0.00)*2.01574803<256.00,-249.95<(i8-0.00)*1.95275591<248.00,-258.02<(i8-0.00)*2.01574803<256.00,-0.00<(u8-128.00)*0.00002718<0.00] out_qs [-235.08<(u8-128.00)*1.83653915<233.24,-337.52<(u8-128.00)*2.63686562<334.88]
    AddNode("S376_Op_expr_1",
        Bindings(7,
            GNodeArg(GNA_IN, "S346_Output_1", 0),
            GNodeArg(GNA_IN, "_sub_1", 0),
            GNodeArg(GNA_IN, "_add_1", 0),
            GNodeArg(GNA_IN, "_sub_1_1", 0),
            GNodeArg(GNA_IN, "S346_Output_3", 0),
            GNodeArg(GNA_OUT, "Output_2", 0),
            GNodeArg(GNA_OUT, "Output_4", 0)
        )
    );
    // Node expr_0 in_qs [-0.00<(u8-128.00)*0.00002718<0.00,-322.52<(i8-0.00)*2.51968504<320.00,-314.46<(i8-0.00)*2.45669291<312.00,-258.02<(i8-0.00)*2.01574803<256.00,-0.00<(u8-128.00)*0.00002718<0.00] out_qs [-299.66<(u8-128.00)*2.34112048<297.32,-402.06<(u8-128.00)*3.14109659<398.92]
    AddNode("S382_Op_expr_0",
        Bindings(7,
            GNodeArg(GNA_IN, "S346_Output_0", 0),
            GNodeArg(GNA_IN, "_sub", 0),
            GNodeArg(GNA_IN, "_add", 0),
            GNodeArg(GNA_IN, "_sub_1_2", 0),
            GNodeArg(GNA_IN, "S346_Output_2", 0),
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
