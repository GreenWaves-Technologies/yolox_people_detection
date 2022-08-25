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
void s24_kernel(s24_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    signed char *__restrict__  expr_0_in_0 = Args->expr_0_in_0; // (16, 64, 80) int8 13.773 Q7
    signed char *__restrict__  expr_0_in_1 = Args->expr_0_in_1; // (16, 64, 80) int8 15.796 Q7
    signed char *__restrict__  expr_0_out_0 = Args->expr_0_out_0; // (16, 64, 80) int8 15.796 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 64, 80) var shapes:
    // expr_0_out_0: (16, 64, 80) expr_0_in_0: (16, 64, 80) expr_0_in_1: (16,
    // 64, 80)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_0_in_0: int8 13.773 Q7 expr_0_in_1: int8 15.796 Q7
        // expr_0_out_0 = Cast(Clip(Add(Norm(Mul(Norm(Mul(Sub(Cast(expr_0_in_0, int32), [8]), SigmoidLUT(LShift(Mul(Sub(Cast(expr_0_in_0, int32), [8]), [220]), [1]))), [7]), [223]), [16]), Cast(expr_0_in_1, int32)), -128, 127), int8)
        expr_0_out_0[i0] = ((signed char)gap_clip(((gap_roundnorm_reg((gap_roundnorm_reg(((((int)expr_0_in_0[i0])-(8))*Sigmoid((((((int)expr_0_in_0[i0])-(8))*(220))<<(1)))), (7))*(223)), (16))+((int)expr_0_in_1[i0]))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void s50_kernel(s50_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    signed char *__restrict__  expr_1_in_0 = Args->expr_1_in_0; // (32, 32, 40) int8 3.255 Q7
    signed char *__restrict__  expr_1_in_1 = Args->expr_1_in_1; // (32, 32, 40) int8 6.388 Q7
    signed char *__restrict__  expr_1_out_0 = Args->expr_1_out_0; // (32, 32, 40) int8 6.188 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 32, 40) var shapes:
    // expr_1_out_0: (32, 32, 40) expr_1_in_0: (32, 32, 40) expr_1_in_1: (32,
    // 32, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_1_in_0: int8 3.255 Q7 expr_1_in_1: int8 6.388 Q7
        // expr_1_out_0 = Cast(Clip(Norm(Mul(Add(Norm(Mul(Norm(Mul(Sub(Cast(expr_1_in_0, int32), [-11]), SigmoidLUT(Norm(Mul(Sub(Cast(expr_1_in_0, int32), [-11]), [208]), [1]))), [7]), [130]), [16]), Cast(expr_1_in_1, int32)), [132]), [7]), -128, 127), int8)
        expr_1_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg(((gap_roundnorm_reg((gap_roundnorm_reg(((((int)expr_1_in_0[i0])-(-11))*Sigmoid(gap_roundnorm_reg(((((int)expr_1_in_0[i0])-(-11))*(208)), (1)))), (7))*(130)), (16))+((int)expr_1_in_1[i0]))*(132)), (7))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void s96_kernel(s96_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    signed char *__restrict__  expr_2_in_0 = Args->expr_2_in_0; // (64, 16, 20) int8 2.875 Q7
    signed char *__restrict__  expr_2_in_1 = Args->expr_2_in_1; // (64, 16, 20) int8 7.993 Q7
    signed char *__restrict__  expr_2_out_0 = Args->expr_2_out_0; // (64, 16, 20) int8 4.829 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 16, 20) var shapes:
    // expr_2_out_0: (64, 16, 20) expr_2_in_0: (64, 16, 20) expr_2_in_1: (64,
    // 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_2_in_0: int8 2.875 Q7 expr_2_in_1: int8 7.993 Q7
        // expr_2_out_0 = Cast(Clip(Norm(Mul(Add(Norm(Mul(Norm(Mul(Sub(Cast(expr_2_in_0, int32), [-8]), SigmoidLUT(Norm(Mul(Sub(Cast(expr_2_in_0, int32), [-8]), [184]), [1]))), [7]), [184]), [17]), Cast(expr_2_in_1, int32)), [212]), [7]), -128, 127), int8)
        expr_2_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg(((gap_roundnorm_reg((gap_roundnorm_reg(((((int)expr_2_in_0[i0])-(-8))*Sigmoid(gap_roundnorm_reg(((((int)expr_2_in_0[i0])-(-8))*(184)), (1)))), (7))*(184)), (17))+((int)expr_2_in_1[i0]))*(212)), (7))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void s271_kernel(s271_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    signed char *__restrict__  expr_57_in_0 = Args->expr_57_in_0; // (64, 32, 40) int8 3.281 Q7
    signed char *__restrict__  expr_57_out_0 = Args->expr_57_out_0; // (64, 32, 40) int8 3.455 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 32, 40) var shapes:
    // expr_57_out_0: (64, 32, 40) expr_57_in_0: (64, 32, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_57_in_0: int8 3.281 Q7
        // expr_57_out_0 = Cast(Clip(Norm(Mul(Norm(Mul(Sub(Cast(expr_57_in_0, int32), [-11]), SigmoidLUT(Norm(Mul(Sub(Cast(expr_57_in_0, int32), [-11]), [210]), [1]))), [7]), [243]), [16]), -128, 127), int8)
        expr_57_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((gap_roundnorm_reg(((((int)expr_57_in_0[i0])-(-11))*Sigmoid(gap_roundnorm_reg(((((int)expr_57_in_0[i0])-(-11))*(210)), (1)))), (7))*(243)), (16))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void s310_kernel(s310_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    signed char *__restrict__  expr_68_in_0 = Args->expr_68_in_0; // (64, 16, 20) int8 1.383 Q7
    signed char *__restrict__  expr_68_out_0 = Args->expr_68_out_0; // (64, 16, 20) int8 1.425 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 16, 20) var shapes:
    // expr_68_out_0: (64, 16, 20) expr_68_in_0: (64, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_68_in_0: int8 1.383 Q7
        // expr_68_out_0 = Cast(Clip(Norm(Mul(Norm(Mul(Sub(Cast(expr_68_in_0, int32), [-28]), SigmoidLUT(Norm(Mul(Sub(Cast(expr_68_in_0, int32), [-28]), [177]), [2]))), [7]), [248]), [16]), -128, 127), int8)
        expr_68_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((gap_roundnorm_reg(((((int)expr_68_in_0[i0])-(-28))*Sigmoid(gap_roundnorm_reg(((((int)expr_68_in_0[i0])-(-28))*(177)), (2)))), (7))*(248)), (16))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void s349_kernel(s349_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    signed char *__restrict__  expr_78_in_0 = Args->expr_78_in_0; // (64, 8, 10) int8 1.390 Q7
    signed char *__restrict__  expr_78_out_0 = Args->expr_78_out_0; // (64, 8, 10) int8 1.829 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 8, 10) var shapes:
    // expr_78_out_0: (64, 8, 10) expr_78_in_0: (64, 8, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_78_in_0: int8 1.390 Q7
        // expr_78_out_0 = Cast(Clip(Norm(Mul(Norm(Mul(Sub(Cast(expr_78_in_0, int32), [-62]), SigmoidLUT(Norm(Mul(Sub(Cast(expr_78_in_0, int32), [-62]), [178]), [2]))), [7]), [195]), [16]), -128, 127), int8)
        expr_78_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((gap_roundnorm_reg(((((int)expr_78_in_0[i0])-(-62))*Sigmoid(gap_roundnorm_reg(((((int)expr_78_in_0[i0])-(-62))*(178)), (2)))), (7))*(195)), (16))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void custom_0(custom_0_args_t *Args) {
    signed char *__restrict__  expr_66_in_0 = Args->expr_66_in_0;
    signed char *__restrict__  expr_66_out_0 = Args->expr_66_out_0;
    signed char * __restrict__ Infos = Args->Infos;
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 128, 160) var shapes:
    // expr_66_out_0: (16, 128, 160) expr_66_in_0: (16, 128, 160)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_66_in_0: int8 12.468 Q7
        // expr_66_out_0 = Cast(Clip(Norm(Mul(Norm(Mul(Sub(Cast(expr_66_in_0, int32), InfosRef(0, 0)), SigmoidLUT(LShift(Mul(Sub(Cast(expr_66_in_0, int32), InfosRef(0, 1)), InfosRef(0, 2)), [1]))), [7]), InfosRef(0, 3)), InfosRef(0, 4)), -128, 127), int8)
        expr_66_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((gap_roundnorm_reg(((((int)expr_66_in_0[i0])-(*((signed char *)&Infos[0])))*Sigmoid((((((int)expr_66_in_0[i0])-(*((signed char *)&Infos[1])))*(*((int *)&Infos[4])))<<(1)))), (7))*(*((int *)&Infos[8]))), (*((signed char *)&Infos[12])))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void custom_1(custom_1_args_t *Args) {
    signed char *__restrict__  expr_91_in_0 = Args->expr_91_in_0;
    signed char *__restrict__  expr_91_out_0 = Args->expr_91_out_0;
    signed char * __restrict__ Infos = Args->Infos;
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 64, 80) var shapes:
    // expr_91_out_0: (16, 64, 80) expr_91_in_0: (16, 64, 80)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_91_in_0: int8 16.634 Q7
        // expr_91_out_0 = Cast(Clip(Norm(Mul(Norm(Mul(Sub(Cast(expr_91_in_0, int32), InfosRef(0, 0)), SigmoidLUT(Clip(LShift(Mul(Sub(Cast(expr_91_in_0, int32), InfosRef(0, 1)), InfosRef(0, 2)), InfosRef(0, 3)), -65536, 65535))), [7]), InfosRef(0, 4)), [15]), -128, 127), int8)
        expr_91_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((gap_roundnorm_reg(((((int)expr_91_in_0[i0])-(*((signed char *)&Infos[0])))*Sigmoid(gap_clip(((((((int)expr_91_in_0[i0])-(*((signed char *)&Infos[1])))*(*((int *)&Infos[4])))<<(*((signed char *)&Infos[8])))), ((16))))), (7))*(*((int *)&Infos[12]))), (15))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void custom_2(custom_2_args_t *Args) {
    signed char *__restrict__  expr_27_in_0 = Args->expr_27_in_0;
    signed char *__restrict__  expr_27_out_0 = Args->expr_27_out_0;
    signed char * __restrict__ Infos = Args->Infos;
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 64, 80) var shapes:
    // expr_27_out_0: (16, 64, 80) expr_27_in_0: (16, 64, 80)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_27_in_0: int8 7.685 Q7
        // expr_27_out_0 = Cast(Clip(Norm(Mul(Norm(Mul(Sub(Cast(expr_27_in_0, int32), InfosRef(0, 0)), SigmoidLUT(Mul(Sub(Cast(expr_27_in_0, int32), InfosRef(0, 1)), InfosRef(0, 2)))), [7]), InfosRef(0, 3)), InfosRef(0, 4)), -128, 127), int8)
        expr_27_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((gap_roundnorm_reg(((((int)expr_27_in_0[i0])-(*((signed char *)&Infos[0])))*Sigmoid(((((int)expr_27_in_0[i0])-(*((signed char *)&Infos[1])))*(*((int *)&Infos[4]))))), (7))*(*((int *)&Infos[8]))), (*((signed char *)&Infos[12])))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void custom_3(custom_3_args_t *Args) {
    signed char *__restrict__  expr_86_in_0 = Args->expr_86_in_0;
    signed char *__restrict__  expr_86_out_0 = Args->expr_86_out_0;
    signed char * __restrict__ Infos = Args->Infos;
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 32, 40) var shapes:
    // expr_86_out_0: (32, 32, 40) expr_86_in_0: (32, 32, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_86_in_0: int8 3.888 Q7
        // expr_86_out_0 = Cast(Clip(Norm(Mul(Norm(Mul(Sub(Cast(expr_86_in_0, int32), InfosRef(0, 0)), SigmoidLUT(Norm(Mul(Sub(Cast(expr_86_in_0, int32), InfosRef(0, 1)), InfosRef(0, 2)), InfosRef(0, 3)))), [7]), InfosRef(0, 4)), InfosRef(0, 5)), -128, 127), int8)
        expr_86_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((gap_roundnorm_reg(((((int)expr_86_in_0[i0])-(*((signed char *)&Infos[0])))*Sigmoid(gap_roundnorm_reg(((((int)expr_86_in_0[i0])-(*((signed char *)&Infos[1])))*(*((int *)&Infos[4]))), (*((signed char *)&Infos[8]))))), (7))*(*((int *)&Infos[12]))), (*((signed char *)&Infos[16])))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_39(expr_39_args_t *Args) {
    signed char *__restrict__  expr_39_in_0 = Args->expr_39_in_0;
    signed char *__restrict__  expr_39_out_0 = Args->expr_39_out_0; // (128, 16, 20) int8 4.770 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (128, 16, 20) var shapes:
    // expr_39_out_0: (128, 16, 20) expr_39_in_0: (128, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_39_in_0: int8 4.770 Q7
        // expr_39_out_0 = Cast(Clip(Norm(Mul(Norm(Mul(Cast(expr_39_in_0, int32), SigmoidLUT(Mul(Cast(expr_39_in_0, int32), [153]))), [7]), [137]), [15]), -128, 127), int8)
        expr_39_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((gap_roundnorm_reg((((int)expr_39_in_0[i0])*Sigmoid((((int)expr_39_in_0[i0])*(153)))), (7))*(137)), (15))), ((7))));
    }
    gap_waitbarrier(0);
}


#pragma GCC diagnostic pop