#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wpointer-sign"
#pragma GCC diagnostic ignored "-Wsign-compare"
#include "Expression_Kernels.h"

static int CoreCountDynamic = 1;
static int ActiveCore = gap_ncore();

static inline unsigned int __attribute__((always_inline)) ChunkSize(unsigned int X)

{
	unsigned int NCore;
	unsigned int Log2Core;
	unsigned int Chunk;

	if (CoreCountDynamic) NCore = ActiveCore; else NCore = gap_ncore();
	Log2Core = gap_fl1(NCore);
	Chunk = (X>>Log2Core) + ((X&(NCore-1))!=0);
	return Chunk;
}

#ifndef AT_NORM
#define AT_NORM(x, n)   gap_roundnorm_reg((x), (n))
#endif
#define ATLShift(x, n)  ((x) << (n))

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void s382_kernel(s382_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    unsigned char *__restrict__  expr_0_in_0 = Args->expr_0_in_0; // (1, 1, 2574) uint8 0.007 Q8
    signed char *__restrict__  expr_0_in_1 = Args->expr_0_in_1; // (1, 1, 2574) int8 322.520 Q7
    signed char *__restrict__  expr_0_in_2 = Args->expr_0_in_2; // (1, 1, 2574) int8 314.457 Q7
    signed char *__restrict__  expr_0_in_3 = Args->expr_0_in_3; // (1, 1, 2574) int8 258.016 Q7
    unsigned char *__restrict__  expr_0_in_4 = Args->expr_0_in_4; // (1, 1, 2574) uint8 0.007 Q8
    unsigned char *__restrict__  expr_0_out_0 = Args->expr_0_out_0; // (1, 1, 2574) uint8 599.327 Q8
    unsigned char *__restrict__  expr_0_out_1 = Args->expr_0_out_1; // (1, 1, 2574) uint8 804.121 Q8
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (1, 1, 2574) var shapes:
    // inter__Add_2: (1, 1, 2574) expr_0_in_0: (1, 1, 2574) expr_0_in_1: (1, 1,
    // 2574) expr_0_in_2: (1, 1, 2574) inter__Mul_7: (1, 1, 2574) expr_0_in_4:
    // (1, 1, 2574) expr_0_in_3: (1, 1, 2574) expr_0_out_0: (1, 1, 2574)
    // expr_0_out_1: (1, 1, 2574)
    // Iteration reduced to spaces ((2,),)
    // Fixed spaces ()
    // Parameteric spaces ((2,),)
    // Paralelized space (2,)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_0_in_0: uint8 0.007 Q8 expr_0_in_1: int8 322.520 Q7
        // expr_0_in_2: int8 314.457 Q7
        // inter__Add_2 = Add(Norm(Mul(Mul(Sub(Cast(expr_0_in_0, int32), [128]), Cast(expr_0_in_1, int32)), [234]), [23]), Cast(expr_0_in_2, int32))
        int inter__Add_2 = (gap_roundnorm_reg((((((int)expr_0_in_0[i0])-(128))*((int)expr_0_in_1[i0]))*(234)), (23))+((int)expr_0_in_2[i0]));
        // inputs expr_0_in_4: uint8 0.007 Q8 expr_0_in_3: int8 258.016 Q7
        // inter__Mul_7 = Mul([1], Norm(Mul(Norm(Mul(Cast(Exp1715(Norm(Mul(GapMin([152576], GapMax([-36700160], Sub(Cast(expr_0_in_4, int32), [128]))), [228]), [8])), int32), [255]), [8]), Cast(expr_0_in_3, int32)), [7]))
        // Min22 scale arg 0 to 1 - int32 4.135 Q0 -> int32 0.007 Q8
        int _SYMBOL_Constant3896 = (152576);
        // Max22 scale arg 0 to 1 - int32 -1000.000 Q0 -> int32 0.007 Q8
        int _SYMBOL_Constant3894 = (-36700160);
        int inter__Mul_7 = ((1)*gap_roundnorm_reg((gap_roundnorm_reg((((int)exp_17_15(gap_roundnorm_reg((gap_min(_SYMBOL_Constant3896, gap_max(_SYMBOL_Constant3894, (((int)expr_0_in_4[i0])-(128))))*(228)), (8))))*(255)), (8))*((int)expr_0_in_3[i0])), (7)));
        // inputs inter__Add_2: int32 314.457 Q7 inter__Mul_7: int32 129.291
        // Q15
        // expr_0_out_0 = Cast(Clip(Add(Norm(Mul(Sub(inter__Add_2, Norm(Mul(inter__Mul_7, [211]), [17])), [134]), [7]), [128]), 0, 255), uint8)
        expr_0_out_0[i0] = ((unsigned char)gap_clipu(((gap_roundnorm_reg(((inter__Add_2-gap_roundnorm_reg((inter__Mul_7*(211)), (17)))*(134)), (7))+(128))), ((8))));
        // inputs inter__Add_2: int32 314.457 Q7 inter__Mul_7: int32 129.291
        // Q15
        // expr_0_out_1 = Cast(Clip(Add(Norm(Mul(Add(inter__Add_2, Norm(Mul(inter__Mul_7, [211]), [17])), [200]), [8]), [128]), 0, 255), uint8)
        expr_0_out_1[i0] = ((unsigned char)gap_clipu(((gap_roundnorm_reg(((inter__Add_2+gap_roundnorm_reg((inter__Mul_7*(211)), (17)))*(200)), (8))+(128))), ((8))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void s376_kernel(s376_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    unsigned char *__restrict__  expr_1_in_0 = Args->expr_1_in_0; // (1, 1, 2574) uint8 0.007 Q8
    signed char *__restrict__  expr_1_in_1 = Args->expr_1_in_1; // (1, 1, 2574) int8 258.016 Q7
    signed char *__restrict__  expr_1_in_2 = Args->expr_1_in_2; // (1, 1, 2574) int8 249.953 Q7
    signed char *__restrict__  expr_1_in_3 = Args->expr_1_in_3; // (1, 1, 2574) int8 258.016 Q7
    unsigned char *__restrict__  expr_1_in_4 = Args->expr_1_in_4; // (1, 1, 2574) uint8 0.007 Q8
    unsigned char *__restrict__  expr_1_out_0 = Args->expr_1_out_0; // (1, 1, 2574) uint8 470.154 Q8
    unsigned char *__restrict__  expr_1_out_1 = Args->expr_1_out_1; // (1, 1, 2574) uint8 675.038 Q8
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (1, 1, 2574) var shapes:
    // inter__Add_3: (1, 1, 2574) expr_1_in_0: (1, 1, 2574) expr_1_in_1: (1, 1,
    // 2574) expr_1_in_2: (1, 1, 2574) inter__Mul_6: (1, 1, 2574) expr_1_in_4:
    // (1, 1, 2574) expr_1_in_3: (1, 1, 2574) expr_1_out_0: (1, 1, 2574)
    // expr_1_out_1: (1, 1, 2574)
    // Iteration reduced to spaces ((2,),)
    // Fixed spaces ()
    // Parameteric spaces ((2,),)
    // Paralelized space (2,)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_1_in_0: uint8 0.007 Q8 expr_1_in_1: int8 258.016 Q7
        // expr_1_in_2: int8 249.953 Q7
        // inter__Add_3 = Add(Norm(Mul(Mul(Sub(Cast(expr_1_in_0, int32), [128]), Cast(expr_1_in_1, int32)), [235]), [23]), Cast(expr_1_in_2, int32))
        int inter__Add_3 = (gap_roundnorm_reg((((((int)expr_1_in_0[i0])-(128))*((int)expr_1_in_1[i0]))*(235)), (23))+((int)expr_1_in_2[i0]));
        // inputs expr_1_in_4: uint8 0.007 Q8 expr_1_in_3: int8 258.016 Q7
        // inter__Mul_6 = Mul([1], Norm(Mul(Norm(Mul(Cast(Exp1715(Norm(Mul(GapMin([152576], GapMax([-36700160], Sub(Cast(expr_1_in_4, int32), [128]))), [228]), [8])), int32), [256]), [8]), Cast(expr_1_in_3, int32)), [7]))
        // Min23 scale arg 0 to 1 - int32 4.135 Q0 -> int32 0.007 Q8
        int _SYMBOL_Constant3892 = (152576);
        // Max23 scale arg 0 to 1 - int32 -1000.000 Q0 -> int32 0.007 Q8
        int _SYMBOL_Constant3890 = (-36700160);
        int inter__Mul_6 = ((1)*gap_roundnorm_reg((gap_roundnorm_reg((((int)exp_17_15(gap_roundnorm_reg((gap_min(_SYMBOL_Constant3892, gap_max(_SYMBOL_Constant3890, (((int)expr_1_in_4[i0])-(128))))*(228)), (8))))*(256)), (8))*((int)expr_1_in_3[i0])), (7)));
        // inputs inter__Add_3: int32 249.953 Q7 inter__Mul_6: int32 129.197
        // Q15
        // expr_1_out_0 = Cast(Clip(Add(Norm(Mul(Sub(inter__Add_3, Norm(Mul(inter__Mul_6, [132]), [16])), [136]), [7]), [128]), 0, 255), uint8)
        expr_1_out_0[i0] = ((unsigned char)gap_clipu(((gap_roundnorm_reg(((inter__Add_3-gap_roundnorm_reg((inter__Mul_6*(132)), (16)))*(136)), (7))+(128))), ((8))));
        // inputs inter__Add_3: int32 249.953 Q7 inter__Mul_6: int32 129.197
        // Q15
        // expr_1_out_1 = Cast(Clip(Add(Norm(Mul(Add(inter__Add_3, Norm(Mul(inter__Mul_6, [132]), [16])), [190]), [8]), [128]), 0, 255), uint8)
        expr_1_out_1[i0] = ((unsigned char)gap_clipu(((gap_roundnorm_reg(((inter__Add_3+gap_roundnorm_reg((inter__Mul_6*(132)), (16)))*(190)), (8))+(128))), ((8))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 2 external iteration spaces
void s150_kernel(s150_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    unsigned int I1 = Args->I1;
    signed char *__restrict__  expr_11_in_0 = Args->expr_11_in_0; // (1, 1, 480)   int8 1.008 Q7
    signed char *__restrict__  expr_11_in_1 = Args->expr_11_in_1; // (16, 20, 480) int8 3.880 Q7
    unsigned char *__restrict__  expr_11_out_0 = Args->expr_11_out_0; // (16, 20, 480) uint8 0.588 Q8
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I1);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I1);
    // Max shape: (16, 20, 480) var shapes:
    // expr_11_out_0: (16, 20, 480) expr_11_in_0: (1, 1, 480) expr_11_in_1:
    // (16, 20, 480)
    // Iteration reduced to spaces ((0, 1), (2,))
    // Fixed spaces ()
    // Parameteric spaces ((0, 1), (2,))
    // Paralelized space (2,)
    // Interior spaces ()
    for (int i0=0; i0<I0; i0++) {
        for (int i1=First; i1<Last; i1++) {
            // inputs expr_11_in_0: int8 1.008 Q7 expr_11_in_1: int8 3.880 Q7
            // expr_11_out_0 = Cast(Clip(Add(Norm(Mul(Mul(Cast(expr_11_in_0, int32), Cast(expr_11_in_1, int32)), [213]), [11]), [45]), 0, 255), uint8)
            expr_11_out_0[(i0*I1)+i1] = ((unsigned char)gap_clipu(((gap_roundnorm_reg(((((int)expr_11_in_0[i1])*((int)expr_11_in_1[(i0*I1)+i1]))*(213)), (11))+(45))), ((8))));
        }
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 2 external iteration spaces
void s170_kernel(s170_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    unsigned int I1 = Args->I1;
    signed char *__restrict__  expr_14_in_0 = Args->expr_14_in_0; // (1, 1, 672)   int8 1.008 Q7
    signed char *__restrict__  expr_14_in_1 = Args->expr_14_in_1; // (16, 20, 672) int8 3.397 Q7
    unsigned char *__restrict__  expr_14_out_0 = Args->expr_14_out_0; // (16, 20, 672) uint8 0.583 Q8
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I1);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I1);
    // Max shape: (16, 20, 672) var shapes:
    // expr_14_out_0: (16, 20, 672) expr_14_in_0: (1, 1, 672) expr_14_in_1:
    // (16, 20, 672)
    // Iteration reduced to spaces ((0, 1), (2,))
    // Fixed spaces ()
    // Parameteric spaces ((0, 1), (2,))
    // Paralelized space (2,)
    // Interior spaces ()
    for (int i0=0; i0<I0; i0++) {
        for (int i1=First; i1<Last; i1++) {
            // inputs expr_14_in_0: int8 1.008 Q7 expr_14_in_1: int8 3.397 Q7
            // expr_14_out_0 = Cast(Clip(Add(Norm(Mul(Mul(Cast(expr_14_in_0, int32), Cast(expr_14_in_1, int32)), [188]), [11]), [105]), 0, 255), uint8)
            expr_14_out_0[(i0*I1)+i1] = ((unsigned char)gap_clipu(((gap_roundnorm_reg(((((int)expr_14_in_0[i1])*((int)expr_14_in_1[(i0*I1)+i1]))*(188)), (11))+(105))), ((8))));
        }
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 2 external iteration spaces
void s47_kernel(s47_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    unsigned int I1 = Args->I1;
    signed char *__restrict__  expr_16_in_0 = Args->expr_16_in_0; // (32, 40, 72) int8 11.788 Q7
    signed char *__restrict__  expr_16_in_1 = Args->expr_16_in_1; // (1, 1, 72)   int8 1.008 Q7
    unsigned char *__restrict__  expr_16_out_0 = Args->expr_16_out_0; // (32, 40, 72) uint8 1.481 Q8
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I1);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I1);
    // Max shape: (32, 40, 72) var shapes:
    // expr_16_out_0: (32, 40, 72) expr_16_in_1: (1, 1, 72) expr_16_in_0: (32,
    // 40, 72)
    // Iteration reduced to spaces ((0, 1), (2,))
    // Fixed spaces ()
    // Parameteric spaces ((0, 1), (2,))
    // Paralelized space (2,)
    // Interior spaces ()
    for (int i0=0; i0<I0; i0++) {
        for (int i1=First; i1<Last; i1++) {
            // inputs expr_16_in_1: int8 1.008 Q7 expr_16_in_0: int8 11.788 Q7
            // expr_16_out_0 = Cast(Clip(Norm(Mul(Mul(Cast(expr_16_in_1, int32), Cast(expr_16_in_0, int32)), [128]), [10]), 0, 255), uint8)
            expr_16_out_0[(i0*I1)+i1] = ((unsigned char)gap_clipu((gap_roundnorm_reg(((((int)expr_16_in_1[i1])*((int)expr_16_in_0[(i0*I1)+i1]))*(128)), (10))), ((8))));
        }
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 2 external iteration spaces
void s68_kernel(s68_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    unsigned int I1 = Args->I1;
    signed char *__restrict__  expr_17_in_0 = Args->expr_17_in_0; // (32, 40, 120) int8 3.356 Q7
    signed char *__restrict__  expr_17_in_1 = Args->expr_17_in_1; // (1, 1, 120)   int8 1.008 Q7
    unsigned char *__restrict__  expr_17_out_0 = Args->expr_17_out_0; // (32, 40, 120) uint8 0.999 Q8
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I1);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I1);
    // Max shape: (32, 40, 120) var shapes:
    // expr_17_out_0: (32, 40, 120) expr_17_in_1: (1, 1, 120) expr_17_in_0:
    // (32, 40, 120)
    // Iteration reduced to spaces ((0, 1), (2,))
    // Fixed spaces ()
    // Parameteric spaces ((0, 1), (2,))
    // Paralelized space (2,)
    // Interior spaces ()
    for (int i0=0; i0<I0; i0++) {
        for (int i1=First; i1<Last; i1++) {
            // inputs expr_17_in_1: int8 1.008 Q7 expr_17_in_0: int8 3.356 Q7
            // expr_17_out_0 = Cast(Clip(Norm(Mul(Mul(Cast(expr_17_in_1, int32), Cast(expr_17_in_0, int32)), [217]), [12]), 0, 255), uint8)
            expr_17_out_0[(i0*I1)+i1] = ((unsigned char)gap_clipu((gap_roundnorm_reg(((((int)expr_17_in_1[i1])*((int)expr_17_in_0[(i0*I1)+i1]))*(217)), (12))), ((8))));
        }
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 2 external iteration spaces
void s90_kernel(s90_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    unsigned int I1 = Args->I1;
    signed char *__restrict__  expr_18_in_0 = Args->expr_18_in_0; // (32, 40, 120) int8 3.843 Q7
    signed char *__restrict__  expr_18_in_1 = Args->expr_18_in_1; // (1, 1, 120)   int8 1.008 Q7
    unsigned char *__restrict__  expr_18_out_0 = Args->expr_18_out_0; // (32, 40, 120) uint8 0.877 Q8
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I1);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I1);
    // Max shape: (32, 40, 120) var shapes:
    // expr_18_out_0: (32, 40, 120) expr_18_in_1: (1, 1, 120) expr_18_in_0:
    // (32, 40, 120)
    // Iteration reduced to spaces ((0, 1), (2,))
    // Fixed spaces ()
    // Parameteric spaces ((0, 1), (2,))
    // Paralelized space (2,)
    // Interior spaces ()
    for (int i0=0; i0<I0; i0++) {
        for (int i1=First; i1<Last; i1++) {
            // inputs expr_18_in_1: int8 1.008 Q7 expr_18_in_0: int8 3.843 Q7
            // expr_18_out_0 = Cast(Clip(Norm(Mul(Mul(Cast(expr_18_in_1, int32), Cast(expr_18_in_0, int32)), [141]), [11]), 0, 255), uint8)
            expr_18_out_0[(i0*I1)+i1] = ((unsigned char)gap_clipu((gap_roundnorm_reg(((((int)expr_18_in_1[i1])*((int)expr_18_in_0[(i0*I1)+i1]))*(141)), (11))), ((8))));
        }
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 2 external iteration spaces
void s193_kernel(s193_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    unsigned int I1 = Args->I1;
    signed char *__restrict__  expr_26_in_0 = Args->expr_26_in_0; // (1, 1, 672)  int8 1.008 Q7
    signed char *__restrict__  expr_26_in_1 = Args->expr_26_in_1; // (8, 10, 672) int8 2.649 Q7
    unsigned char *__restrict__  expr_26_out_0 = Args->expr_26_out_0; // (8, 10, 672) uint8 0.679 Q8
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I1);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I1);
    // Max shape: (8, 10, 672) var shapes:
    // expr_26_out_0: (8, 10, 672) expr_26_in_0: (1, 1, 672) expr_26_in_1: (8,
    // 10, 672)
    // Iteration reduced to spaces ((0, 1), (2,))
    // Fixed spaces ()
    // Parameteric spaces ((0, 1), (2,))
    // Paralelized space (2,)
    // Interior spaces ()
    for (int i0=0; i0<I0; i0++) {
        for (int i1=First; i1<Last; i1++) {
            // inputs expr_26_in_0: int8 1.008 Q7 expr_26_in_1: int8 2.649 Q7
            // expr_26_out_0 = Cast(Clip(Add(Norm(Mul(Mul(Cast(expr_26_in_0, int32), Cast(expr_26_in_1, int32)), [252]), [12]), [72]), 0, 255), uint8)
            expr_26_out_0[(i0*I1)+i1] = ((unsigned char)gap_clipu(((gap_roundnorm_reg(((((int)expr_26_in_0[i1])*((int)expr_26_in_1[(i0*I1)+i1]))*(252)), (12))+(72))), ((8))));
        }
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 2 external iteration spaces
void s213_kernel(s213_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    unsigned int I1 = Args->I1;
    signed char *__restrict__  expr_29_in_0 = Args->expr_29_in_0; // (1, 1, 480)  int8 1.008 Q7
    signed char *__restrict__  expr_29_in_1 = Args->expr_29_in_1; // (8, 10, 480) int8 0.817 Q7
    unsigned char *__restrict__  expr_29_out_0 = Args->expr_29_out_0; // (8, 10, 480) uint8 0.635 Q8
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I1);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I1);
    // Max shape: (8, 10, 480) var shapes:
    // expr_29_out_0: (8, 10, 480) expr_29_in_0: (1, 1, 480) expr_29_in_1: (8,
    // 10, 480)
    // Iteration reduced to spaces ((0, 1), (2,))
    // Fixed spaces ()
    // Parameteric spaces ((0, 1), (2,))
    // Paralelized space (2,)
    // Interior spaces ()
    for (int i0=0; i0<I0; i0++) {
        for (int i1=First; i1<Last; i1++) {
            // inputs expr_29_in_0: int8 1.008 Q7 expr_29_in_1: int8 0.817 Q7
            // expr_29_out_0 = Cast(Clip(Add(Norm(Mul(Mul(Cast(expr_29_in_0, int32), Cast(expr_29_in_1, int32)), [166]), [13]), [82]), 0, 255), uint8)
            expr_29_out_0[(i0*I1)+i1] = ((unsigned char)gap_clipu(((gap_roundnorm_reg(((((int)expr_29_in_0[i1])*((int)expr_29_in_1[(i0*I1)+i1]))*(166)), (13))+(82))), ((8))));
        }
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 2 external iteration spaces
void s234_kernel(s234_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    unsigned int I1 = Args->I1;
    signed char *__restrict__  expr_32_in_0 = Args->expr_32_in_0; // (1, 1, 480)  int8 1.008 Q7
    signed char *__restrict__  expr_32_in_1 = Args->expr_32_in_1; // (8, 10, 480) int8 0.679 Q7
    unsigned char *__restrict__  expr_32_out_0 = Args->expr_32_out_0; // (8, 10, 480) uint8 0.548 Q8
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I1);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I1);
    // Max shape: (8, 10, 480) var shapes:
    // expr_32_out_0: (8, 10, 480) expr_32_in_0: (1, 1, 480) expr_32_in_1: (8,
    // 10, 480)
    // Iteration reduced to spaces ((0, 1), (2,))
    // Fixed spaces ()
    // Parameteric spaces ((0, 1), (2,))
    // Paralelized space (2,)
    // Interior spaces ()
    for (int i0=0; i0<I0; i0++) {
        for (int i1=First; i1<Last; i1++) {
            // inputs expr_32_in_0: int8 1.008 Q7 expr_32_in_1: int8 0.679 Q7
            // expr_32_out_0 = Cast(Clip(Add(Norm(Mul(Mul(Cast(expr_32_in_0, int32), Cast(expr_32_in_1, int32)), [160]), [13]), [96]), 0, 255), uint8)
            expr_32_out_0[(i0*I1)+i1] = ((unsigned char)gap_clipu(((gap_roundnorm_reg(((((int)expr_32_in_0[i1])*((int)expr_32_in_1[(i0*I1)+i1]))*(160)), (13))+(96))), ((8))));
        }
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_6(expr_6_args_t *Args) {
    unsigned char *__restrict__  expr_6_in_0 = Args->expr_6_in_0;
    unsigned char *__restrict__  expr_6_out_0 = Args->expr_6_out_0; // (128, 160, 16) uint8 22.440 Q8
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (128, 160, 16) var shapes:
    // expr_6_out_0: (128, 160, 16) expr_6_in_0: (128, 160, 16)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_6_in_0: uint8 22.440 Q8
        // expr_6_out_0 = Cast(Clip(Add(Norm(Mul(Norm(Mul(Sub(Cast(expr_6_in_0, int32), [133]), Mul(GapMin([69], GapMax([0], Add(Sub(Cast(expr_6_in_0, int32), [133]), [34]))), [1])), [1]), [249]), [13]), [128]), 0, 255), uint8)
        // Min24 scale arg 0 to 1 - int32 6.000 Q0 -> int32 22.440 Q8
        int _SYMBOL_Constant3785 = (69);
        // Max24 scale arg 0 to 1 - int32 1.000 Q15 -> int32 22.440 Q8
        int _SYMBOL_QuantizedConstant1458 = (0);
        // Add152 scale arg 1 to 0 - int32 3.000 Q0 -> int32 22.440 Q8
        int _SYMBOL_Constant3781 = (34);
        expr_6_out_0[i0] = ((unsigned char)gap_clipu(((gap_roundnorm_reg((gap_roundnorm_reg(((((int)expr_6_in_0[i0])-(133))*(gap_min(_SYMBOL_Constant3785, gap_max(_SYMBOL_QuantizedConstant1458, ((((int)expr_6_in_0[i0])-(133))+_SYMBOL_Constant3781)))*(1))), (1))*(249)), (13))+(128))), ((8))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_19(expr_19_args_t *Args) {
    unsigned char *__restrict__  expr_19_in_0 = Args->expr_19_in_0;
    unsigned char *__restrict__  expr_19_out_0 = Args->expr_19_out_0; // (32, 40, 240) uint8 12.618 Q8
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 40, 240) var shapes:
    // expr_19_out_0: (32, 40, 240) expr_19_in_0: (32, 40, 240)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_19_in_0: uint8 12.618 Q8
        // expr_19_out_0 = Cast(Clip(Add(Norm(Mul(Norm(Mul(Sub(Cast(expr_19_in_0, int32), [148]), Mul(GapMin([122], GapMax([0], Add(Sub(Cast(expr_19_in_0, int32), [148]), [61]))), [1])), [1]), [160]), [13]), [128]), 0, 255), uint8)
        // Min32 scale arg 0 to 1 - int32 6.000 Q0 -> int32 12.618 Q8
        int _SYMBOL_Constant3791 = (122);
        // Max32 scale arg 0 to 1 - int32 1.000 Q15 -> int32 12.618 Q8
        int _SYMBOL_QuantizedConstant1496 = (0);
        // Add160 scale arg 1 to 0 - int32 3.000 Q0 -> int32 12.618 Q8
        int _SYMBOL_Constant3787 = (61);
        expr_19_out_0[i0] = ((unsigned char)gap_clipu(((gap_roundnorm_reg((gap_roundnorm_reg(((((int)expr_19_in_0[i0])-(148))*(gap_min(_SYMBOL_Constant3791, gap_max(_SYMBOL_QuantizedConstant1496, ((((int)expr_19_in_0[i0])-(148))+_SYMBOL_Constant3787)))*(1))), (1))*(160)), (13))+(128))), ((8))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_20(expr_20_args_t *Args) {
    unsigned char *__restrict__  expr_20_in_0 = Args->expr_20_in_0;
    unsigned char *__restrict__  expr_20_out_0 = Args->expr_20_out_0; // (16, 20, 240) uint8 8.296 Q8
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 20, 240) var shapes:
    // expr_20_out_0: (16, 20, 240) expr_20_in_0: (16, 20, 240)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_20_in_0: uint8 8.296 Q8
        // expr_20_out_0 = Cast(Clip(Add(Norm(Mul(Norm(Mul(Sub(Cast(expr_20_in_0, int32), [104]), Mul(GapMin([185], GapMax([0], Add(Sub(Cast(expr_20_in_0, int32), [104]), [93]))), [1])), [1]), [149]), [14]), [128]), 0, 255), uint8)
        // Min33 scale arg 0 to 1 - int32 6.000 Q0 -> int32 8.296 Q8
        int _SYMBOL_Constant3796 = (185);
        // Max33 scale arg 0 to 1 - int32 1.000 Q15 -> int32 8.296 Q8
        int _SYMBOL_QuantizedConstant1515 = (0);
        // Add161 scale arg 1 to 0 - int32 3.000 Q0 -> int32 8.296 Q8
        int _SYMBOL_Constant3793 = (93);
        expr_20_out_0[i0] = ((unsigned char)gap_clipu(((gap_roundnorm_reg((gap_roundnorm_reg(((((int)expr_20_in_0[i0])-(104))*(gap_min(_SYMBOL_Constant3796, gap_max(_SYMBOL_QuantizedConstant1515, ((((int)expr_20_in_0[i0])-(104))+_SYMBOL_Constant3793)))*(1))), (1))*(149)), (14))+(128))), ((8))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_21(expr_21_args_t *Args) {
    unsigned char *__restrict__  expr_21_in_0 = Args->expr_21_in_0;
    unsigned char *__restrict__  expr_21_out_0 = Args->expr_21_out_0; // (16, 20, 200) uint8 10.021 Q8
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 20, 200) var shapes:
    // expr_21_out_0: (16, 20, 200) expr_21_in_0: (16, 20, 200)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_21_in_0: uint8 10.021 Q8
        // expr_21_out_0 = Cast(Clip(Add(Norm(Mul(Norm(Mul(Sub(Cast(expr_21_in_0, int32), [70]), Mul(GapMin([153], GapMax([0], Add(Sub(Cast(expr_21_in_0, int32), [70]), [77]))), [1])), [1]), [147]), [14]), [128]), 0, 255), uint8)
        // Min34 scale arg 0 to 1 - int32 6.000 Q0 -> int32 10.021 Q8
        int _SYMBOL_Constant3801 = (153);
        // Max34 scale arg 0 to 1 - int32 1.000 Q15 -> int32 10.021 Q8
        int _SYMBOL_QuantizedConstant1534 = (0);
        // Add162 scale arg 1 to 0 - int32 3.000 Q0 -> int32 10.021 Q8
        int _SYMBOL_Constant3798 = (77);
        expr_21_out_0[i0] = ((unsigned char)gap_clipu(((gap_roundnorm_reg((gap_roundnorm_reg(((((int)expr_21_in_0[i0])-(70))*(gap_min(_SYMBOL_Constant3801, gap_max(_SYMBOL_QuantizedConstant1534, ((((int)expr_21_in_0[i0])-(70))+_SYMBOL_Constant3798)))*(1))), (1))*(147)), (14))+(128))), ((8))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_22(expr_22_args_t *Args) {
    unsigned char *__restrict__  expr_22_in_0 = Args->expr_22_in_0;
    unsigned char *__restrict__  expr_22_out_0 = Args->expr_22_out_0; // (16, 20, 200) uint8 6.125 Q8
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 20, 200) var shapes:
    // expr_22_out_0: (16, 20, 200) expr_22_in_0: (16, 20, 200)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_22_in_0: uint8 6.125 Q8
        // expr_22_out_0 = Cast(Clip(Add(Norm(Mul(Norm(Mul(Sub(Cast(expr_22_in_0, int32), [160]), Mul(GapMin([251], GapMax([0], Add(Sub(Cast(expr_22_in_0, int32), [160]), [126]))), [1])), [1]), [199]), [14]), [128]), 0, 255), uint8)
        // Min35 scale arg 0 to 1 - int32 6.000 Q0 -> int32 6.125 Q8
        int _SYMBOL_Constant3806 = (251);
        // Max35 scale arg 0 to 1 - int32 1.000 Q15 -> int32 6.125 Q8
        int _SYMBOL_QuantizedConstant1553 = (0);
        // Add163 scale arg 1 to 0 - int32 3.000 Q0 -> int32 6.125 Q8
        int _SYMBOL_Constant3803 = (126);
        expr_22_out_0[i0] = ((unsigned char)gap_clipu(((gap_roundnorm_reg((gap_roundnorm_reg(((((int)expr_22_in_0[i0])-(160))*(gap_min(_SYMBOL_Constant3806, gap_max(_SYMBOL_QuantizedConstant1553, ((((int)expr_22_in_0[i0])-(160))+_SYMBOL_Constant3803)))*(1))), (1))*(199)), (14))+(128))), ((8))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_23(expr_23_args_t *Args) {
    unsigned char *__restrict__  expr_23_in_0 = Args->expr_23_in_0;
    unsigned char *__restrict__  expr_23_out_0 = Args->expr_23_out_0; // (16, 20, 184) uint8 7.244 Q8
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 20, 184) var shapes:
    // expr_23_out_0: (16, 20, 184) expr_23_in_0: (16, 20, 184)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_23_in_0: uint8 7.244 Q8
        // expr_23_out_0 = Cast(Clip(Add(Norm(Mul(Norm(Mul(Sub(Cast(expr_23_in_0, int32), [102]), Mul(GapMin([212], GapMax([0], Add(Sub(Cast(expr_23_in_0, int32), [102]), [106]))), [1])), [1]), [128]), [14]), [128]), 0, 255), uint8)
        // Min36 scale arg 0 to 1 - int32 6.000 Q0 -> int32 7.244 Q8
        int _SYMBOL_Constant3811 = (212);
        // Max36 scale arg 0 to 1 - int32 1.000 Q15 -> int32 7.244 Q8
        int _SYMBOL_QuantizedConstant1572 = (0);
        // Add164 scale arg 1 to 0 - int32 3.000 Q0 -> int32 7.244 Q8
        int _SYMBOL_Constant3808 = (106);
        expr_23_out_0[i0] = ((unsigned char)gap_clipu(((gap_roundnorm_reg((gap_roundnorm_reg(((((int)expr_23_in_0[i0])-(102))*(gap_min(_SYMBOL_Constant3811, gap_max(_SYMBOL_QuantizedConstant1572, ((((int)expr_23_in_0[i0])-(102))+_SYMBOL_Constant3808)))*(1))), (1))*(128)), (14))+(128))), ((8))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_24(expr_24_args_t *Args) {
    unsigned char *__restrict__  expr_24_in_0 = Args->expr_24_in_0;
    unsigned char *__restrict__  expr_24_out_0 = Args->expr_24_out_0; // (16, 20, 184) uint8 5.945 Q8
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 20, 184) var shapes:
    // expr_24_out_0: (16, 20, 184) expr_24_in_0: (16, 20, 184)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_24_in_0: uint8 5.945 Q8
        // expr_24_out_0 = Cast(Clip(Add(Norm(Mul(Norm(Mul(Sub(Cast(expr_24_in_0, int32), [116]), Mul(GapMin([258], GapMax([0], Add(Sub(Cast(expr_24_in_0, int32), [116]), [129]))), [1])), [1]), [232]), [15]), [128]), 0, 255), uint8)
        // Min37 scale arg 0 to 1 - int32 6.000 Q0 -> int32 5.945 Q8
        int _SYMBOL_Constant3816 = (258);
        // Max37 scale arg 0 to 1 - int32 1.000 Q15 -> int32 5.945 Q8
        int _SYMBOL_QuantizedConstant1591 = (0);
        // Add165 scale arg 1 to 0 - int32 3.000 Q0 -> int32 5.945 Q8
        int _SYMBOL_Constant3812 = (129);
        expr_24_out_0[i0] = ((unsigned char)gap_clipu(((gap_roundnorm_reg((gap_roundnorm_reg(((((int)expr_24_in_0[i0])-(116))*(gap_min(_SYMBOL_Constant3816, gap_max(_SYMBOL_QuantizedConstant1591, ((((int)expr_24_in_0[i0])-(116))+_SYMBOL_Constant3812)))*(1))), (1))*(232)), (15))+(128))), ((8))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_7(expr_7_args_t *Args) {
    unsigned char *__restrict__  expr_7_in_0 = Args->expr_7_in_0;
    unsigned char *__restrict__  expr_7_out_0 = Args->expr_7_out_0; // (16, 20, 184) uint8 6.171 Q8
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 20, 184) var shapes:
    // expr_7_out_0: (16, 20, 184) expr_7_in_0: (16, 20, 184)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_7_in_0: uint8 6.171 Q8
        // expr_7_out_0 = Cast(Clip(Add(Norm(Mul(Norm(Mul(Sub(Cast(expr_7_in_0, int32), [106]), Mul(GapMin([249], GapMax([0], Add(Sub(Cast(expr_7_in_0, int32), [106]), [125]))), [1])), [1]), [224]), [15]), [128]), 0, 255), uint8)
        // Min25 scale arg 0 to 1 - int32 6.000 Q0 -> int32 6.171 Q8
        int _SYMBOL_Constant3821 = (249);
        // Max25 scale arg 0 to 1 - int32 1.000 Q15 -> int32 6.171 Q8
        int _SYMBOL_QuantizedConstant1610 = (0);
        // Add153 scale arg 1 to 0 - int32 3.000 Q0 -> int32 6.171 Q8
        int _SYMBOL_Constant3818 = (125);
        expr_7_out_0[i0] = ((unsigned char)gap_clipu(((gap_roundnorm_reg((gap_roundnorm_reg(((((int)expr_7_in_0[i0])-(106))*(gap_min(_SYMBOL_Constant3821, gap_max(_SYMBOL_QuantizedConstant1610, ((((int)expr_7_in_0[i0])-(106))+_SYMBOL_Constant3818)))*(1))), (1))*(224)), (15))+(128))), ((8))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_8(expr_8_args_t *Args) {
    unsigned char *__restrict__  expr_8_in_0 = Args->expr_8_in_0;
    unsigned char *__restrict__  expr_8_out_0 = Args->expr_8_out_0; // (16, 20, 184) uint8 5.267 Q8
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 20, 184) var shapes:
    // expr_8_out_0: (16, 20, 184) expr_8_in_0: (16, 20, 184)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_8_in_0: uint8 5.267 Q8
        // expr_8_out_0 = Cast(Clip(Add(Norm(Mul(Norm(Mul(Sub(Cast(expr_8_in_0, int32), [117]), Mul(GapMin([292], GapMax([0], Add(Sub(Cast(expr_8_in_0, int32), [117]), [146]))), [1])), [1]), [213]), [15]), [128]), 0, 255), uint8)
        // Min26 scale arg 0 to 1 - int32 6.000 Q0 -> int32 5.267 Q8
        int _SYMBOL_Constant3826 = (292);
        // Max26 scale arg 0 to 1 - int32 1.000 Q15 -> int32 5.267 Q8
        int _SYMBOL_QuantizedConstant1629 = (0);
        // Add154 scale arg 1 to 0 - int32 3.000 Q0 -> int32 5.267 Q8
        int _SYMBOL_Constant3822 = (146);
        expr_8_out_0[i0] = ((unsigned char)gap_clipu(((gap_roundnorm_reg((gap_roundnorm_reg(((((int)expr_8_in_0[i0])-(117))*(gap_min(_SYMBOL_Constant3826, gap_max(_SYMBOL_QuantizedConstant1629, ((((int)expr_8_in_0[i0])-(117))+_SYMBOL_Constant3822)))*(1))), (1))*(213)), (15))+(128))), ((8))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_9(expr_9_args_t *Args) {
    unsigned char *__restrict__  expr_9_in_0 = Args->expr_9_in_0;
    unsigned char *__restrict__  expr_9_out_0 = Args->expr_9_out_0; // (16, 20, 480) uint8 7.246 Q8
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 20, 480) var shapes:
    // expr_9_out_0: (16, 20, 480) expr_9_in_0: (16, 20, 480)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_9_in_0: uint8 7.246 Q8
        // expr_9_out_0 = Cast(Clip(Add(Norm(Mul(Norm(Mul(Sub(Cast(expr_9_in_0, int32), [116]), Mul(GapMin([212], GapMax([0], Add(Sub(Cast(expr_9_in_0, int32), [116]), [106]))), [1])), [1]), [141]), [14]), [128]), 0, 255), uint8)
        // Min27 scale arg 0 to 1 - int32 6.000 Q0 -> int32 7.246 Q8
        int _SYMBOL_Constant3831 = (212);
        // Max27 scale arg 0 to 1 - int32 1.000 Q15 -> int32 7.246 Q8
        int _SYMBOL_QuantizedConstant1648 = (0);
        // Add155 scale arg 1 to 0 - int32 3.000 Q0 -> int32 7.246 Q8
        int _SYMBOL_Constant3828 = (106);
        expr_9_out_0[i0] = ((unsigned char)gap_clipu(((gap_roundnorm_reg((gap_roundnorm_reg(((((int)expr_9_in_0[i0])-(116))*(gap_min(_SYMBOL_Constant3831, gap_max(_SYMBOL_QuantizedConstant1648, ((((int)expr_9_in_0[i0])-(116))+_SYMBOL_Constant3828)))*(1))), (1))*(141)), (14))+(128))), ((8))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_10(expr_10_args_t *Args) {
    unsigned char *__restrict__  expr_10_in_0 = Args->expr_10_in_0;
    signed char *__restrict__  expr_10_out_0 = Args->expr_10_out_0; // (16, 20, 480) uint8 10.521 Q8
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 20, 480) var shapes:
    // expr_10_out_0: (16, 20, 480) expr_10_in_0: (16, 20, 480)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_10_in_0: uint8 10.521 Q8
        // expr_10_out_0 = Cast(Clip(Norm(Mul(Norm(Mul(Sub(Cast(expr_10_in_0, int32), [161]), Mul(GapMin([146], GapMax([0], Add(Sub(Cast(expr_10_in_0, int32), [161]), [73]))), [1])), [1]), [152]), [13]), -128, 127), int8)
        // Min28 scale arg 0 to 1 - int32 6.000 Q0 -> int32 10.521 Q8
        int _SYMBOL_Constant3901 = (146);
        // Max28 scale arg 0 to 1 - int32 1.000 Q15 -> int32 10.521 Q8
        int _SYMBOL_QuantizedConstant2011 = (0);
        // Add156 scale arg 1 to 0 - int32 3.000 Q0 -> int32 10.521 Q8
        int _SYMBOL_Constant3898 = (73);
        expr_10_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((gap_roundnorm_reg(((((int)expr_10_in_0[i0])-(161))*(gap_min(_SYMBOL_Constant3901, gap_max(_SYMBOL_QuantizedConstant2011, ((((int)expr_10_in_0[i0])-(161))+_SYMBOL_Constant3898)))*(1))), (1))*(152)), (13))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_12(expr_12_args_t *Args) {
    unsigned char *__restrict__  expr_12_in_0 = Args->expr_12_in_0;
    unsigned char *__restrict__  expr_12_out_0 = Args->expr_12_out_0; // (16, 20, 672) uint8 5.049 Q8
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 20, 672) var shapes:
    // expr_12_out_0: (16, 20, 672) expr_12_in_0: (16, 20, 672)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_12_in_0: uint8 5.049 Q8
        // expr_12_out_0 = Cast(Clip(Add(Norm(Mul(Norm(Mul(Sub(Cast(expr_12_in_0, int32), [131]), Mul(GapMin([304], GapMax([0], Add(Sub(Cast(expr_12_in_0, int32), [131]), [152]))), [1])), [1]), [244]), [15]), [128]), 0, 255), uint8)
        // Min29 scale arg 0 to 1 - int32 6.000 Q0 -> int32 5.049 Q8
        int _SYMBOL_Constant3841 = (304);
        // Max29 scale arg 0 to 1 - int32 1.000 Q15 -> int32 5.049 Q8
        int _SYMBOL_QuantizedConstant1692 = (0);
        // Add157 scale arg 1 to 0 - int32 3.000 Q0 -> int32 5.049 Q8
        int _SYMBOL_Constant3837 = (152);
        expr_12_out_0[i0] = ((unsigned char)gap_clipu(((gap_roundnorm_reg((gap_roundnorm_reg(((((int)expr_12_in_0[i0])-(131))*(gap_min(_SYMBOL_Constant3841, gap_max(_SYMBOL_QuantizedConstant1692, ((((int)expr_12_in_0[i0])-(131))+_SYMBOL_Constant3837)))*(1))), (1))*(244)), (15))+(128))), ((8))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_13(expr_13_args_t *Args) {
    unsigned char *__restrict__  expr_13_in_0 = Args->expr_13_in_0;
    signed char *__restrict__  expr_13_out_0 = Args->expr_13_out_0; // (16, 20, 672) uint8 6.619 Q8
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 20, 672) var shapes:
    // expr_13_out_0: (16, 20, 672) expr_13_in_0: (16, 20, 672)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_13_in_0: uint8 6.619 Q8
        // expr_13_out_0 = Cast(Clip(Norm(Mul(Norm(Mul(Sub(Cast(expr_13_in_0, int32), [125]), Mul(GapMin([232], GapMax([0], Add(Sub(Cast(expr_13_in_0, int32), [125]), [116]))), [1])), [1]), [138]), [14]), -128, 127), int8)
        // Min30 scale arg 0 to 1 - int32 6.000 Q0 -> int32 6.619 Q8
        int _SYMBOL_Constant3906 = (232);
        // Max30 scale arg 0 to 1 - int32 1.000 Q15 -> int32 6.619 Q8
        int _SYMBOL_QuantizedConstant2035 = (0);
        // Add158 scale arg 1 to 0 - int32 3.000 Q0 -> int32 6.619 Q8
        int _SYMBOL_Constant3903 = (116);
        expr_13_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((gap_roundnorm_reg(((((int)expr_13_in_0[i0])-(125))*(gap_min(_SYMBOL_Constant3906, gap_max(_SYMBOL_QuantizedConstant2035, ((((int)expr_13_in_0[i0])-(125))+_SYMBOL_Constant3903)))*(1))), (1))*(138)), (14))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_15(expr_15_args_t *Args) {
    unsigned char *__restrict__  expr_15_in_0 = Args->expr_15_in_0;
    signed char *__restrict__  expr_15_out_0 = Args->expr_15_out_0; // (16, 20, 672) uint8 4.638 Q8
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 20, 672) var shapes:
    // expr_15_out_0: (16, 20, 672) expr_15_in_0: (16, 20, 672)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_15_in_0: uint8 4.638 Q8
        // expr_15_out_0 = Cast(Clip(Norm(Mul(Norm(Mul(Sub(Cast(expr_15_in_0, int32), [107]), Mul(GapMin([332], GapMax([0], Add(Sub(Cast(expr_15_in_0, int32), [107]), [166]))), [1])), [1]), [179]), [15]), -128, 127), int8)
        // Min31 scale arg 0 to 1 - int32 6.000 Q0 -> int32 4.638 Q8
        int _SYMBOL_Constant3911 = (332);
        // Max31 scale arg 0 to 1 - int32 1.000 Q15 -> int32 4.638 Q8
        int _SYMBOL_QuantizedConstant2059 = (0);
        // Add159 scale arg 1 to 0 - int32 3.000 Q0 -> int32 4.638 Q8
        int _SYMBOL_Constant3907 = (166);
        expr_15_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((gap_roundnorm_reg(((((int)expr_15_in_0[i0])-(107))*(gap_min(_SYMBOL_Constant3911, gap_max(_SYMBOL_QuantizedConstant2059, ((((int)expr_15_in_0[i0])-(107))+_SYMBOL_Constant3907)))*(1))), (1))*(179)), (15))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_25(expr_25_args_t *Args) {
    signed char *__restrict__  expr_25_in_0 = Args->expr_25_in_0;
    signed char *__restrict__  expr_25_out_0 = Args->expr_25_out_0; // (8, 10, 672) int8 2.367 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (8, 10, 672) var shapes:
    // expr_25_out_0: (8, 10, 672) expr_25_in_0: (8, 10, 672)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_25_in_0: int8 2.367 Q7
        // expr_25_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_25_in_0, int32), [-21]), Mul(GapMin([324], GapMax([0], Add(Sub(Cast(expr_25_in_0, int32), [-21]), [162]))), [1])), [180]), [16]), -128, 127), int8)
        // Min38 scale arg 0 to 1 - int32 6.000 Q0 -> int32 2.367 Q7
        int _SYMBOL_Constant3856 = (324);
        // Max38 scale arg 0 to 1 - int32 1.000 Q15 -> int32 2.367 Q7
        int _SYMBOL_QuantizedConstant1753 = (0);
        // Add166 scale arg 1 to 0 - int32 3.000 Q0 -> int32 2.367 Q7
        int _SYMBOL_Constant3852 = (162);
        expr_25_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_25_in_0[i0])-(-21))*(gap_min(_SYMBOL_Constant3856, gap_max(_SYMBOL_QuantizedConstant1753, ((((int)expr_25_in_0[i0])-(-21))+_SYMBOL_Constant3852)))*(1)))*(180)), (16))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_27(expr_27_args_t *Args) {
    unsigned char *__restrict__  expr_27_in_0 = Args->expr_27_in_0;
    signed char *__restrict__  expr_27_out_0 = Args->expr_27_out_0; // (8, 10, 480) uint8 2.891 Q8
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (8, 10, 480) var shapes:
    // expr_27_out_0: (8, 10, 480) expr_27_in_0: (8, 10, 480)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_27_in_0: uint8 2.891 Q8
        // expr_27_out_0 = Cast(Clip(Norm(Mul(Norm(Mul(Sub(Cast(expr_27_in_0, int32), [142]), Mul(GapMin([532], GapMax([0], Add(Sub(Cast(expr_27_in_0, int32), [142]), [266]))), [1])), [1]), [194]), [15]), -128, 127), int8)
        // Min39 scale arg 0 to 1 - int32 6.000 Q0 -> int32 2.891 Q8
        int _SYMBOL_Constant3917 = (532);
        // Max39 scale arg 0 to 1 - int32 1.000 Q15 -> int32 2.891 Q8
        int _SYMBOL_QuantizedConstant2084 = (0);
        // Add167 scale arg 1 to 0 - int32 3.000 Q0 -> int32 2.891 Q8
        int _SYMBOL_Constant3913 = (266);
        expr_27_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((gap_roundnorm_reg(((((int)expr_27_in_0[i0])-(142))*(gap_min(_SYMBOL_Constant3917, gap_max(_SYMBOL_QuantizedConstant2084, ((((int)expr_27_in_0[i0])-(142))+_SYMBOL_Constant3913)))*(1))), (1))*(194)), (15))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_28(expr_28_args_t *Args) {
    signed char *__restrict__  expr_28_in_0 = Args->expr_28_in_0;
    signed char *__restrict__  expr_28_out_0 = Args->expr_28_out_0; // (8, 10, 480) int8 1.769 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (8, 10, 480) var shapes:
    // expr_28_out_0: (8, 10, 480) expr_28_in_0: (8, 10, 480)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_28_in_0: int8 1.769 Q7
        // expr_28_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_28_in_0, int32), [43]), Mul(GapMin([434], GapMax([0], Add(Sub(Cast(expr_28_in_0, int32), [43]), [217]))), [1])), [163]), [15]), -128, 127), int8)
        // Min40 scale arg 0 to 1 - int32 6.000 Q0 -> int32 1.769 Q7
        int _SYMBOL_Constant3867 = (434);
        // Max40 scale arg 0 to 1 - int32 1.000 Q15 -> int32 1.769 Q7
        int _SYMBOL_QuantizedConstant1795 = (0);
        // Add168 scale arg 1 to 0 - int32 3.000 Q0 -> int32 1.769 Q7
        int _SYMBOL_Constant3863 = (217);
        expr_28_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_28_in_0[i0])-(43))*(gap_min(_SYMBOL_Constant3867, gap_max(_SYMBOL_QuantizedConstant1795, ((((int)expr_28_in_0[i0])-(43))+_SYMBOL_Constant3863)))*(1)))*(163)), (15))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_30(expr_30_args_t *Args) {
    unsigned char *__restrict__  expr_30_in_0 = Args->expr_30_in_0;
    signed char *__restrict__  expr_30_out_0 = Args->expr_30_out_0; // (8, 10, 480) uint8 3.164 Q8
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (8, 10, 480) var shapes:
    // expr_30_out_0: (8, 10, 480) expr_30_in_0: (8, 10, 480)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_30_in_0: uint8 3.164 Q8
        // expr_30_out_0 = Cast(Clip(Norm(Mul(Norm(Mul(Sub(Cast(expr_30_in_0, int32), [152]), Mul(GapMin([486], GapMax([0], Add(Sub(Cast(expr_30_in_0, int32), [152]), [243]))), [1])), [1]), [235]), [15]), -128, 127), int8)
        // Min41 scale arg 0 to 1 - int32 6.000 Q0 -> int32 3.164 Q8
        int _SYMBOL_Constant3922 = (486);
        // Max41 scale arg 0 to 1 - int32 1.000 Q15 -> int32 3.164 Q8
        int _SYMBOL_QuantizedConstant2108 = (0);
        // Add169 scale arg 1 to 0 - int32 3.000 Q0 -> int32 3.164 Q8
        int _SYMBOL_Constant3918 = (243);
        expr_30_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((gap_roundnorm_reg(((((int)expr_30_in_0[i0])-(152))*(gap_min(_SYMBOL_Constant3922, gap_max(_SYMBOL_QuantizedConstant2108, ((((int)expr_30_in_0[i0])-(152))+_SYMBOL_Constant3918)))*(1))), (1))*(235)), (15))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_31(expr_31_args_t *Args) {
    signed char *__restrict__  expr_31_in_0 = Args->expr_31_in_0;
    signed char *__restrict__  expr_31_out_0 = Args->expr_31_out_0; // (8, 10, 480) int8 1.563 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (8, 10, 480) var shapes:
    // expr_31_out_0: (8, 10, 480) expr_31_in_0: (8, 10, 480)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_31_in_0: int8 1.563 Q7
        // expr_31_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_31_in_0, int32), [44]), Mul(GapMin([492], GapMax([0], Add(Sub(Cast(expr_31_in_0, int32), [44]), [246]))), [1])), [154]), [15]), -128, 127), int8)
        // Min42 scale arg 0 to 1 - int32 6.000 Q0 -> int32 1.563 Q7
        int _SYMBOL_Constant3877 = (492);
        // Max42 scale arg 0 to 1 - int32 1.000 Q15 -> int32 1.563 Q7
        int _SYMBOL_QuantizedConstant1836 = (0);
        // Add170 scale arg 1 to 0 - int32 3.000 Q0 -> int32 1.563 Q7
        int _SYMBOL_Constant3873 = (246);
        expr_31_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_31_in_0[i0])-(44))*(gap_min(_SYMBOL_Constant3877, gap_max(_SYMBOL_QuantizedConstant1836, ((((int)expr_31_in_0[i0])-(44))+_SYMBOL_Constant3873)))*(1)))*(154)), (15))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_33(expr_33_args_t *Args) {
    unsigned char *__restrict__  expr_33_in_0 = Args->expr_33_in_0;
    unsigned char *__restrict__  expr_33_out_0 = Args->expr_33_out_0; // (8, 10, 480) uint8 3.346 Q8
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (8, 10, 480) var shapes:
    // expr_33_out_0: (8, 10, 480) expr_33_in_0: (8, 10, 480)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_33_in_0: uint8 3.346 Q8
        // expr_33_out_0 = Cast(Clip(Add(Norm(Mul(Norm(Mul(Sub(Cast(expr_33_in_0, int32), [123]), Mul(GapMin([460], GapMax([0], Add(Sub(Cast(expr_33_in_0, int32), [123]), [230]))), [1])), [1]), [174]), [15]), [128]), 0, 255), uint8)
        // Min43 scale arg 0 to 1 - int32 6.000 Q0 -> int32 3.346 Q8
        int _SYMBOL_Constant3882 = (460);
        // Max43 scale arg 0 to 1 - int32 1.000 Q15 -> int32 3.346 Q8
        int _SYMBOL_QuantizedConstant1860 = (0);
        // Add171 scale arg 1 to 0 - int32 3.000 Q0 -> int32 3.346 Q8
        int _SYMBOL_Constant3878 = (230);
        expr_33_out_0[i0] = ((unsigned char)gap_clipu(((gap_roundnorm_reg((gap_roundnorm_reg(((((int)expr_33_in_0[i0])-(123))*(gap_min(_SYMBOL_Constant3882, gap_max(_SYMBOL_QuantizedConstant1860, ((((int)expr_33_in_0[i0])-(123))+_SYMBOL_Constant3878)))*(1))), (1))*(174)), (15))+(128))), ((8))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_2(expr_2_args_t *Args) {
    unsigned char *__restrict__  expr_2_in_0 = Args->expr_2_in_0;
    unsigned char *__restrict__  expr_2_out_0 = Args->expr_2_out_0; // (16, 20, 24) uint8 0.064 Q8
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 20, 24) var shapes:
    // expr_2_out_0: (16, 20, 24) expr_2_in_0: (16, 20, 24)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_2_in_0: uint8 0.064 Q8
        // expr_2_out_0 = Cast(Clip(Add(Norm(Mul(Mul(Sub(Cast(expr_2_in_0, int32), [117]), [1]), [235]), [8]), [128]), 0, 255), uint8)
        expr_2_out_0[i0] = ((unsigned char)gap_clipu(((gap_roundnorm_reg((((((int)expr_2_in_0[i0])-(117))*(1))*(235)), (8))+(128))), ((8))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_2_1(expr_2_1_args_t *Args) {
    unsigned char *__restrict__  expr_2_in_0 = Args->expr_2_in_0;
    unsigned char *__restrict__  expr_2_out_0 = Args->expr_2_out_0; // (8, 10, 24) uint8 0.033 Q8
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (8, 10, 24) var shapes:
    // expr_2_out_0: (8, 10, 24) expr_2_in_0: (8, 10, 24)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_2_in_0: uint8 0.033 Q8
        // expr_2_out_0 = Cast(Clip(Add(Norm(Mul(Mul(Sub(Cast(expr_2_in_0, int32), [149]), [1]), [245]), [9]), [128]), 0, 255), uint8)
        expr_2_out_0[i0] = ((unsigned char)gap_clipu(((gap_roundnorm_reg((((((int)expr_2_in_0[i0])-(149))*(1))*(245)), (9))+(128))), ((8))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_2_2(expr_2_2_args_t *Args) {
    unsigned char *__restrict__  expr_2_in_0 = Args->expr_2_in_0;
    unsigned char *__restrict__  expr_2_out_0 = Args->expr_2_out_0; // (4, 5, 24) uint8 0.029 Q8
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (4, 5, 24) var shapes:
    // expr_2_out_0: (4, 5, 24) expr_2_in_0: (4, 5, 24)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_2_in_0: uint8 0.029 Q8
        // expr_2_out_0 = Cast(Clip(Add(Norm(Mul(Mul(Sub(Cast(expr_2_in_0, int32), [105]), [1]), [210]), [9]), [128]), 0, 255), uint8)
        expr_2_out_0[i0] = ((unsigned char)gap_clipu(((gap_roundnorm_reg((((((int)expr_2_in_0[i0])-(105))*(1))*(210)), (9))+(128))), ((8))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_2_3(expr_2_3_args_t *Args) {
    unsigned char *__restrict__  expr_2_in_0 = Args->expr_2_in_0;
    unsigned char *__restrict__  expr_2_out_0 = Args->expr_2_out_0; // (2, 3, 24) uint8 0.031 Q8
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (2, 3, 24) var shapes:
    // expr_2_out_0: (2, 3, 24) expr_2_in_0: (2, 3, 24)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_2_in_0: uint8 0.031 Q8
        // expr_2_out_0 = Cast(Clip(Add(Norm(Mul(Mul(Sub(Cast(expr_2_in_0, int32), [132]), [1]), [226]), [9]), [128]), 0, 255), uint8)
        expr_2_out_0[i0] = ((unsigned char)gap_clipu(((gap_roundnorm_reg((((((int)expr_2_in_0[i0])-(132))*(1))*(226)), (9))+(128))), ((8))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_2_4(expr_2_4_args_t *Args) {
    unsigned char *__restrict__  expr_2_in_0 = Args->expr_2_in_0;
    unsigned char *__restrict__  expr_2_out_0 = Args->expr_2_out_0; // (1, 2, 24) uint8 0.014 Q8
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (1, 2, 24) var shapes:
    // expr_2_out_0: (1, 2, 24) expr_2_in_0: (1, 2, 24)
    // Iteration reduced to spaces ((1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((1, 2),)
    // Paralelized space (1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_2_in_0: uint8 0.014 Q8
        // expr_2_out_0 = Cast(Clip(Add(Norm(Mul(Mul(Sub(Cast(expr_2_in_0, int32), [114]), [1]), [204]), [10]), [128]), 0, 255), uint8)
        expr_2_out_0[i0] = ((unsigned char)gap_clipu(((gap_roundnorm_reg((((((int)expr_2_in_0[i0])-(114))*(1))*(204)), (10))+(128))), ((8))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_2_5(expr_2_5_args_t *Args) {
    unsigned char *__restrict__  expr_2_in_0 = Args->expr_2_in_0;
    unsigned char *__restrict__  expr_2_out_0 = Args->expr_2_out_0; // (1, 1, 24) uint8 0.012 Q8
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (1, 1, 24) var shapes:
    // expr_2_out_0: (1, 1, 24) expr_2_in_0: (1, 1, 24)
    // Iteration reduced to spaces ((2,),)
    // Fixed spaces ()
    // Parameteric spaces ((2,),)
    // Paralelized space (2,)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_2_in_0: uint8 0.012 Q8
        // expr_2_out_0 = Cast(Clip(Add(Norm(Mul(Mul(Sub(Cast(expr_2_in_0, int32), [140]), [1]), [172]), [10]), [128]), 0, 255), uint8)
        expr_2_out_0[i0] = ((unsigned char)gap_clipu(((gap_roundnorm_reg((((((int)expr_2_in_0[i0])-(140))*(1))*(172)), (10))+(128))), ((8))));
    }
    gap_waitbarrier(0);
}


#pragma GCC diagnostic pop