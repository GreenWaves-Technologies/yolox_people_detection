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
void s25_kernel(s25_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    F16 *__restrict__  expr_0_in_0 = Args->expr_0_in_0; // (16, 64, 80) f16
    F16 *__restrict__  expr_0_in_1 = Args->expr_0_in_1; // (16, 64, 80) f16
    F16 *__restrict__  expr_0_out_0 = Args->expr_0_out_0; // (16, 64, 80) f16
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
        // inputs expr_0_in_0: f16 expr_0_in_1: f16
        // expr_0_out_0 = Add(Mul(expr_0_in_0, FastFloatSigmoid(expr_0_in_0)), expr_0_in_1)
        expr_0_out_0[i0] = ((expr_0_in_0[i0]*FastSigmoidF16(expr_0_in_0[i0]))+expr_0_in_1[i0]);
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void s51_kernel(s51_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    F16 *__restrict__  expr_1_in_0 = Args->expr_1_in_0; // (32, 32, 40) f16
    F16 *__restrict__  expr_1_in_1 = Args->expr_1_in_1; // (32, 32, 40) f16
    F16 *__restrict__  expr_1_out_0 = Args->expr_1_out_0; // (32, 32, 40) f16
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
        // inputs expr_1_in_0: f16 expr_1_in_1: f16
        // expr_1_out_0 = Add(Mul(expr_1_in_0, FastFloatSigmoid(expr_1_in_0)), expr_1_in_1)
        expr_1_out_0[i0] = ((expr_1_in_0[i0]*FastSigmoidF16(expr_1_in_0[i0]))+expr_1_in_1[i0]);
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void s97_kernel(s97_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    F16 *__restrict__  expr_2_in_0 = Args->expr_2_in_0; // (64, 16, 20) f16
    F16 *__restrict__  expr_2_in_1 = Args->expr_2_in_1; // (64, 16, 20) f16
    F16 *__restrict__  expr_2_out_0 = Args->expr_2_out_0; // (64, 16, 20) f16
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
        // inputs expr_2_in_0: f16 expr_2_in_1: f16
        // expr_2_out_0 = Add(Mul(expr_2_in_0, FastFloatSigmoid(expr_2_in_0)), expr_2_in_1)
        expr_2_out_0[i0] = ((expr_2_in_0[i0]*FastSigmoidF16(expr_2_in_0[i0]))+expr_2_in_1[i0]);
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_66(expr_66_args_t *Args) {
    F16 *__restrict__  expr_66_in_0 = Args->expr_66_in_0;
    F16 *__restrict__  expr_66_out_0 = Args->expr_66_out_0; // (16, 128, 160) f16
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
        // inputs expr_66_in_0: f16
        // expr_66_out_0 = Mul(expr_66_in_0, FastFloatSigmoid(expr_66_in_0))
        expr_66_out_0[i0] = (expr_66_in_0[i0]*FastSigmoidF16(expr_66_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_91(expr_91_args_t *Args) {
    F16 *__restrict__  expr_91_in_0 = Args->expr_91_in_0;
    F16 *__restrict__  expr_91_out_0 = Args->expr_91_out_0; // (16, 64, 80) f16
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
        // inputs expr_91_in_0: f16
        // expr_91_out_0 = Mul(expr_91_in_0, FastFloatSigmoid(expr_91_in_0))
        expr_91_out_0[i0] = (expr_91_in_0[i0]*FastSigmoidF16(expr_91_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_101(expr_101_args_t *Args) {
    F16 *__restrict__  expr_101_in_0 = Args->expr_101_in_0;
    F16 *__restrict__  expr_101_out_0 = Args->expr_101_out_0; // (32, 64, 80) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 64, 80) var shapes:
    // expr_101_out_0: (32, 64, 80) expr_101_in_0: (32, 64, 80)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_101_in_0: f16
        // expr_101_out_0 = Mul(expr_101_in_0, FastFloatSigmoid(expr_101_in_0))
        expr_101_out_0[i0] = (expr_101_in_0[i0]*FastSigmoidF16(expr_101_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_8(expr_8_args_t *Args) {
    F16 *__restrict__  expr_8_in_0 = Args->expr_8_in_0;
    F16 *__restrict__  expr_8_out_0 = Args->expr_8_out_0; // (16, 64, 80) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 64, 80) var shapes:
    // expr_8_out_0: (16, 64, 80) expr_8_in_0: (16, 64, 80)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_8_in_0: f16
        // expr_8_out_0 = Mul(expr_8_in_0, FastFloatSigmoid(expr_8_in_0))
        expr_8_out_0[i0] = (expr_8_in_0[i0]*FastSigmoidF16(expr_8_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_18(expr_18_args_t *Args) {
    F16 *__restrict__  expr_18_in_0 = Args->expr_18_in_0;
    F16 *__restrict__  expr_18_out_0 = Args->expr_18_out_0; // (16, 64, 80) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 64, 80) var shapes:
    // expr_18_out_0: (16, 64, 80) expr_18_in_0: (16, 64, 80)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_18_in_0: f16
        // expr_18_out_0 = Mul(expr_18_in_0, FastFloatSigmoid(expr_18_in_0))
        expr_18_out_0[i0] = (expr_18_in_0[i0]*FastSigmoidF16(expr_18_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_27(expr_27_args_t *Args) {
    F16 *__restrict__  expr_27_in_0 = Args->expr_27_in_0;
    F16 *__restrict__  expr_27_out_0 = Args->expr_27_out_0; // (16, 64, 80) f16
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
        // inputs expr_27_in_0: f16
        // expr_27_out_0 = Mul(expr_27_in_0, FastFloatSigmoid(expr_27_in_0))
        expr_27_out_0[i0] = (expr_27_in_0[i0]*FastSigmoidF16(expr_27_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_37(expr_37_args_t *Args) {
    F16 *__restrict__  expr_37_in_0 = Args->expr_37_in_0;
    F16 *__restrict__  expr_37_out_0 = Args->expr_37_out_0; // (16, 64, 80) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 64, 80) var shapes:
    // expr_37_out_0: (16, 64, 80) expr_37_in_0: (16, 64, 80)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_37_in_0: f16
        // expr_37_out_0 = Mul(expr_37_in_0, FastFloatSigmoid(expr_37_in_0))
        expr_37_out_0[i0] = (expr_37_in_0[i0]*FastSigmoidF16(expr_37_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_62(expr_62_args_t *Args) {
    F16 *__restrict__  expr_62_in_0 = Args->expr_62_in_0;
    F16 *__restrict__  expr_62_out_0 = Args->expr_62_out_0; // (32, 64, 80) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 64, 80) var shapes:
    // expr_62_out_0: (32, 64, 80) expr_62_in_0: (32, 64, 80)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_62_in_0: f16
        // expr_62_out_0 = Mul(expr_62_in_0, FastFloatSigmoid(expr_62_in_0))
        expr_62_out_0[i0] = (expr_62_in_0[i0]*FastSigmoidF16(expr_62_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_73(expr_73_args_t *Args) {
    F16 *__restrict__  expr_73_in_0 = Args->expr_73_in_0;
    F16 *__restrict__  expr_73_out_0 = Args->expr_73_out_0; // (32, 32, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 32, 40) var shapes:
    // expr_73_out_0: (32, 32, 40) expr_73_in_0: (32, 32, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_73_in_0: f16
        // expr_73_out_0 = Mul(expr_73_in_0, FastFloatSigmoid(expr_73_in_0))
        expr_73_out_0[i0] = (expr_73_in_0[i0]*FastSigmoidF16(expr_73_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_82(expr_82_args_t *Args) {
    F16 *__restrict__  expr_82_in_0 = Args->expr_82_in_0;
    F16 *__restrict__  expr_82_out_0 = Args->expr_82_out_0; // (64, 32, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 32, 40) var shapes:
    // expr_82_out_0: (64, 32, 40) expr_82_in_0: (64, 32, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_82_in_0: f16
        // expr_82_out_0 = Mul(expr_82_in_0, FastFloatSigmoid(expr_82_in_0))
        expr_82_out_0[i0] = (expr_82_in_0[i0]*FastSigmoidF16(expr_82_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_84(expr_84_args_t *Args) {
    F16 *__restrict__  expr_84_in_0 = Args->expr_84_in_0;
    F16 *__restrict__  expr_84_out_0 = Args->expr_84_out_0; // (32, 32, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 32, 40) var shapes:
    // expr_84_out_0: (32, 32, 40) expr_84_in_0: (32, 32, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_84_in_0: f16
        // expr_84_out_0 = Mul(expr_84_in_0, FastFloatSigmoid(expr_84_in_0))
        expr_84_out_0[i0] = (expr_84_in_0[i0]*FastSigmoidF16(expr_84_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_85(expr_85_args_t *Args) {
    F16 *__restrict__  expr_85_in_0 = Args->expr_85_in_0;
    F16 *__restrict__  expr_85_out_0 = Args->expr_85_out_0; // (32, 32, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 32, 40) var shapes:
    // expr_85_out_0: (32, 32, 40) expr_85_in_0: (32, 32, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_85_in_0: f16
        // expr_85_out_0 = Mul(expr_85_in_0, FastFloatSigmoid(expr_85_in_0))
        expr_85_out_0[i0] = (expr_85_in_0[i0]*FastSigmoidF16(expr_85_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_86(expr_86_args_t *Args) {
    F16 *__restrict__  expr_86_in_0 = Args->expr_86_in_0;
    F16 *__restrict__  expr_86_out_0 = Args->expr_86_out_0; // (32, 32, 40) f16
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
        // inputs expr_86_in_0: f16
        // expr_86_out_0 = Mul(expr_86_in_0, FastFloatSigmoid(expr_86_in_0))
        expr_86_out_0[i0] = (expr_86_in_0[i0]*FastSigmoidF16(expr_86_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_87(expr_87_args_t *Args) {
    F16 *__restrict__  expr_87_in_0 = Args->expr_87_in_0;
    F16 *__restrict__  expr_87_out_0 = Args->expr_87_out_0; // (32, 32, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 32, 40) var shapes:
    // expr_87_out_0: (32, 32, 40) expr_87_in_0: (32, 32, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_87_in_0: f16
        // expr_87_out_0 = Mul(expr_87_in_0, FastFloatSigmoid(expr_87_in_0))
        expr_87_out_0[i0] = (expr_87_in_0[i0]*FastSigmoidF16(expr_87_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_88(expr_88_args_t *Args) {
    F16 *__restrict__  expr_88_in_0 = Args->expr_88_in_0;
    F16 *__restrict__  expr_88_out_0 = Args->expr_88_out_0; // (32, 32, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 32, 40) var shapes:
    // expr_88_out_0: (32, 32, 40) expr_88_in_0: (32, 32, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_88_in_0: f16
        // expr_88_out_0 = Mul(expr_88_in_0, FastFloatSigmoid(expr_88_in_0))
        expr_88_out_0[i0] = (expr_88_in_0[i0]*FastSigmoidF16(expr_88_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_89(expr_89_args_t *Args) {
    F16 *__restrict__  expr_89_in_0 = Args->expr_89_in_0;
    F16 *__restrict__  expr_89_out_0 = Args->expr_89_out_0; // (32, 32, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 32, 40) var shapes:
    // expr_89_out_0: (32, 32, 40) expr_89_in_0: (32, 32, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_89_in_0: f16
        // expr_89_out_0 = Mul(expr_89_in_0, FastFloatSigmoid(expr_89_in_0))
        expr_89_out_0[i0] = (expr_89_in_0[i0]*FastSigmoidF16(expr_89_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_90(expr_90_args_t *Args) {
    F16 *__restrict__  expr_90_in_0 = Args->expr_90_in_0;
    F16 *__restrict__  expr_90_out_0 = Args->expr_90_out_0; // (32, 32, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 32, 40) var shapes:
    // expr_90_out_0: (32, 32, 40) expr_90_in_0: (32, 32, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_90_in_0: f16
        // expr_90_out_0 = Mul(expr_90_in_0, FastFloatSigmoid(expr_90_in_0))
        expr_90_out_0[i0] = (expr_90_in_0[i0]*FastSigmoidF16(expr_90_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_92(expr_92_args_t *Args) {
    F16 *__restrict__  expr_92_in_0 = Args->expr_92_in_0;
    F16 *__restrict__  expr_92_out_0 = Args->expr_92_out_0; // (32, 32, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 32, 40) var shapes:
    // expr_92_out_0: (32, 32, 40) expr_92_in_0: (32, 32, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_92_in_0: f16
        // expr_92_out_0 = Mul(expr_92_in_0, FastFloatSigmoid(expr_92_in_0))
        expr_92_out_0[i0] = (expr_92_in_0[i0]*FastSigmoidF16(expr_92_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_93(expr_93_args_t *Args) {
    F16 *__restrict__  expr_93_in_0 = Args->expr_93_in_0;
    F16 *__restrict__  expr_93_out_0 = Args->expr_93_out_0; // (32, 32, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 32, 40) var shapes:
    // expr_93_out_0: (32, 32, 40) expr_93_in_0: (32, 32, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_93_in_0: f16
        // expr_93_out_0 = Mul(expr_93_in_0, FastFloatSigmoid(expr_93_in_0))
        expr_93_out_0[i0] = (expr_93_in_0[i0]*FastSigmoidF16(expr_93_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_94(expr_94_args_t *Args) {
    F16 *__restrict__  expr_94_in_0 = Args->expr_94_in_0;
    F16 *__restrict__  expr_94_out_0 = Args->expr_94_out_0; // (32, 32, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 32, 40) var shapes:
    // expr_94_out_0: (32, 32, 40) expr_94_in_0: (32, 32, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_94_in_0: f16
        // expr_94_out_0 = Mul(expr_94_in_0, FastFloatSigmoid(expr_94_in_0))
        expr_94_out_0[i0] = (expr_94_in_0[i0]*FastSigmoidF16(expr_94_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_95(expr_95_args_t *Args) {
    F16 *__restrict__  expr_95_in_0 = Args->expr_95_in_0;
    F16 *__restrict__  expr_95_out_0 = Args->expr_95_out_0; // (64, 32, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 32, 40) var shapes:
    // expr_95_out_0: (64, 32, 40) expr_95_in_0: (64, 32, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_95_in_0: f16
        // expr_95_out_0 = Mul(expr_95_in_0, FastFloatSigmoid(expr_95_in_0))
        expr_95_out_0[i0] = (expr_95_in_0[i0]*FastSigmoidF16(expr_95_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_96(expr_96_args_t *Args) {
    F16 *__restrict__  expr_96_in_0 = Args->expr_96_in_0;
    F16 *__restrict__  expr_96_out_0 = Args->expr_96_out_0; // (64, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 16, 20) var shapes:
    // expr_96_out_0: (64, 16, 20) expr_96_in_0: (64, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_96_in_0: f16
        // expr_96_out_0 = Mul(expr_96_in_0, FastFloatSigmoid(expr_96_in_0))
        expr_96_out_0[i0] = (expr_96_in_0[i0]*FastSigmoidF16(expr_96_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_97(expr_97_args_t *Args) {
    F16 *__restrict__  expr_97_in_0 = Args->expr_97_in_0;
    F16 *__restrict__  expr_97_out_0 = Args->expr_97_out_0; // (128, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (128, 16, 20) var shapes:
    // expr_97_out_0: (128, 16, 20) expr_97_in_0: (128, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_97_in_0: f16
        // expr_97_out_0 = Mul(expr_97_in_0, FastFloatSigmoid(expr_97_in_0))
        expr_97_out_0[i0] = (expr_97_in_0[i0]*FastSigmoidF16(expr_97_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_98(expr_98_args_t *Args) {
    F16 *__restrict__  expr_98_in_0 = Args->expr_98_in_0;
    F16 *__restrict__  expr_98_out_0 = Args->expr_98_out_0; // (64, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 16, 20) var shapes:
    // expr_98_out_0: (64, 16, 20) expr_98_in_0: (64, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_98_in_0: f16
        // expr_98_out_0 = Mul(expr_98_in_0, FastFloatSigmoid(expr_98_in_0))
        expr_98_out_0[i0] = (expr_98_in_0[i0]*FastSigmoidF16(expr_98_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_99(expr_99_args_t *Args) {
    F16 *__restrict__  expr_99_in_0 = Args->expr_99_in_0;
    F16 *__restrict__  expr_99_out_0 = Args->expr_99_out_0; // (64, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 16, 20) var shapes:
    // expr_99_out_0: (64, 16, 20) expr_99_in_0: (64, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_99_in_0: f16
        // expr_99_out_0 = Mul(expr_99_in_0, FastFloatSigmoid(expr_99_in_0))
        expr_99_out_0[i0] = (expr_99_in_0[i0]*FastSigmoidF16(expr_99_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_100(expr_100_args_t *Args) {
    F16 *__restrict__  expr_100_in_0 = Args->expr_100_in_0;
    F16 *__restrict__  expr_100_out_0 = Args->expr_100_out_0; // (64, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 16, 20) var shapes:
    // expr_100_out_0: (64, 16, 20) expr_100_in_0: (64, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_100_in_0: f16
        // expr_100_out_0 = Mul(expr_100_in_0, FastFloatSigmoid(expr_100_in_0))
        expr_100_out_0[i0] = (expr_100_in_0[i0]*FastSigmoidF16(expr_100_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_102(expr_102_args_t *Args) {
    F16 *__restrict__  expr_102_in_0 = Args->expr_102_in_0;
    F16 *__restrict__  expr_102_out_0 = Args->expr_102_out_0; // (64, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 16, 20) var shapes:
    // expr_102_out_0: (64, 16, 20) expr_102_in_0: (64, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_102_in_0: f16
        // expr_102_out_0 = Mul(expr_102_in_0, FastFloatSigmoid(expr_102_in_0))
        expr_102_out_0[i0] = (expr_102_in_0[i0]*FastSigmoidF16(expr_102_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_103(expr_103_args_t *Args) {
    F16 *__restrict__  expr_103_in_0 = Args->expr_103_in_0;
    F16 *__restrict__  expr_103_out_0 = Args->expr_103_out_0; // (64, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 16, 20) var shapes:
    // expr_103_out_0: (64, 16, 20) expr_103_in_0: (64, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_103_in_0: f16
        // expr_103_out_0 = Mul(expr_103_in_0, FastFloatSigmoid(expr_103_in_0))
        expr_103_out_0[i0] = (expr_103_in_0[i0]*FastSigmoidF16(expr_103_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_3(expr_3_args_t *Args) {
    F16 *__restrict__  expr_3_in_0 = Args->expr_3_in_0;
    F16 *__restrict__  expr_3_out_0 = Args->expr_3_out_0; // (64, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 16, 20) var shapes:
    // expr_3_out_0: (64, 16, 20) expr_3_in_0: (64, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_3_in_0: f16
        // expr_3_out_0 = Mul(expr_3_in_0, FastFloatSigmoid(expr_3_in_0))
        expr_3_out_0[i0] = (expr_3_in_0[i0]*FastSigmoidF16(expr_3_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_4(expr_4_args_t *Args) {
    F16 *__restrict__  expr_4_in_0 = Args->expr_4_in_0;
    F16 *__restrict__  expr_4_out_0 = Args->expr_4_out_0; // (64, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 16, 20) var shapes:
    // expr_4_out_0: (64, 16, 20) expr_4_in_0: (64, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_4_in_0: f16
        // expr_4_out_0 = Mul(expr_4_in_0, FastFloatSigmoid(expr_4_in_0))
        expr_4_out_0[i0] = (expr_4_in_0[i0]*FastSigmoidF16(expr_4_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_5(expr_5_args_t *Args) {
    F16 *__restrict__  expr_5_in_0 = Args->expr_5_in_0;
    F16 *__restrict__  expr_5_out_0 = Args->expr_5_out_0; // (64, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 16, 20) var shapes:
    // expr_5_out_0: (64, 16, 20) expr_5_in_0: (64, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_5_in_0: f16
        // expr_5_out_0 = Mul(expr_5_in_0, FastFloatSigmoid(expr_5_in_0))
        expr_5_out_0[i0] = (expr_5_in_0[i0]*FastSigmoidF16(expr_5_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_6(expr_6_args_t *Args) {
    F16 *__restrict__  expr_6_in_0 = Args->expr_6_in_0;
    F16 *__restrict__  expr_6_out_0 = Args->expr_6_out_0; // (64, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 16, 20) var shapes:
    // expr_6_out_0: (64, 16, 20) expr_6_in_0: (64, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_6_in_0: f16
        // expr_6_out_0 = Mul(expr_6_in_0, FastFloatSigmoid(expr_6_in_0))
        expr_6_out_0[i0] = (expr_6_in_0[i0]*FastSigmoidF16(expr_6_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_7(expr_7_args_t *Args) {
    F16 *__restrict__  expr_7_in_0 = Args->expr_7_in_0;
    F16 *__restrict__  expr_7_out_0 = Args->expr_7_out_0; // (64, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 16, 20) var shapes:
    // expr_7_out_0: (64, 16, 20) expr_7_in_0: (64, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_7_in_0: f16
        // expr_7_out_0 = Mul(expr_7_in_0, FastFloatSigmoid(expr_7_in_0))
        expr_7_out_0[i0] = (expr_7_in_0[i0]*FastSigmoidF16(expr_7_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_9(expr_9_args_t *Args) {
    F16 *__restrict__  expr_9_in_0 = Args->expr_9_in_0;
    F16 *__restrict__  expr_9_out_0 = Args->expr_9_out_0; // (128, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (128, 16, 20) var shapes:
    // expr_9_out_0: (128, 16, 20) expr_9_in_0: (128, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_9_in_0: f16
        // expr_9_out_0 = Mul(expr_9_in_0, FastFloatSigmoid(expr_9_in_0))
        expr_9_out_0[i0] = (expr_9_in_0[i0]*FastSigmoidF16(expr_9_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_10(expr_10_args_t *Args) {
    F16 *__restrict__  expr_10_in_0 = Args->expr_10_in_0;
    F16 *__restrict__  expr_10_out_0 = Args->expr_10_out_0; // (128, 8, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (128, 8, 10) var shapes:
    // expr_10_out_0: (128, 8, 10) expr_10_in_0: (128, 8, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_10_in_0: f16
        // expr_10_out_0 = Mul(expr_10_in_0, FastFloatSigmoid(expr_10_in_0))
        expr_10_out_0[i0] = (expr_10_in_0[i0]*FastSigmoidF16(expr_10_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_11(expr_11_args_t *Args) {
    F16 *__restrict__  expr_11_in_0 = Args->expr_11_in_0;
    F16 *__restrict__  expr_11_out_0 = Args->expr_11_out_0; // (256, 8, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (256, 8, 10) var shapes:
    // expr_11_out_0: (256, 8, 10) expr_11_in_0: (256, 8, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_11_in_0: f16
        // expr_11_out_0 = Mul(expr_11_in_0, FastFloatSigmoid(expr_11_in_0))
        expr_11_out_0[i0] = (expr_11_in_0[i0]*FastSigmoidF16(expr_11_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_12(expr_12_args_t *Args) {
    F16 *__restrict__  expr_12_in_0 = Args->expr_12_in_0;
    F16 *__restrict__  expr_12_out_0 = Args->expr_12_out_0; // (128, 8, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (128, 8, 10) var shapes:
    // expr_12_out_0: (128, 8, 10) expr_12_in_0: (128, 8, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_12_in_0: f16
        // expr_12_out_0 = Mul(expr_12_in_0, FastFloatSigmoid(expr_12_in_0))
        expr_12_out_0[i0] = (expr_12_in_0[i0]*FastSigmoidF16(expr_12_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_13(expr_13_args_t *Args) {
    F16 *__restrict__  expr_13_in_0 = Args->expr_13_in_0;
    F16 *__restrict__  expr_13_out_0 = Args->expr_13_out_0; // (256, 8, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (256, 8, 10) var shapes:
    // expr_13_out_0: (256, 8, 10) expr_13_in_0: (256, 8, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_13_in_0: f16
        // expr_13_out_0 = Mul(expr_13_in_0, FastFloatSigmoid(expr_13_in_0))
        expr_13_out_0[i0] = (expr_13_in_0[i0]*FastSigmoidF16(expr_13_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_14(expr_14_args_t *Args) {
    F16 *__restrict__  expr_14_in_0 = Args->expr_14_in_0;
    F16 *__restrict__  expr_14_out_0 = Args->expr_14_out_0; // (128, 8, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (128, 8, 10) var shapes:
    // expr_14_out_0: (128, 8, 10) expr_14_in_0: (128, 8, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_14_in_0: f16
        // expr_14_out_0 = Mul(expr_14_in_0, FastFloatSigmoid(expr_14_in_0))
        expr_14_out_0[i0] = (expr_14_in_0[i0]*FastSigmoidF16(expr_14_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_15(expr_15_args_t *Args) {
    F16 *__restrict__  expr_15_in_0 = Args->expr_15_in_0;
    F16 *__restrict__  expr_15_out_0 = Args->expr_15_out_0; // (128, 8, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (128, 8, 10) var shapes:
    // expr_15_out_0: (128, 8, 10) expr_15_in_0: (128, 8, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_15_in_0: f16
        // expr_15_out_0 = Mul(expr_15_in_0, FastFloatSigmoid(expr_15_in_0))
        expr_15_out_0[i0] = (expr_15_in_0[i0]*FastSigmoidF16(expr_15_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_16(expr_16_args_t *Args) {
    F16 *__restrict__  expr_16_in_0 = Args->expr_16_in_0;
    F16 *__restrict__  expr_16_out_0 = Args->expr_16_out_0; // (128, 8, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (128, 8, 10) var shapes:
    // expr_16_out_0: (128, 8, 10) expr_16_in_0: (128, 8, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_16_in_0: f16
        // expr_16_out_0 = Mul(expr_16_in_0, FastFloatSigmoid(expr_16_in_0))
        expr_16_out_0[i0] = (expr_16_in_0[i0]*FastSigmoidF16(expr_16_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_17(expr_17_args_t *Args) {
    F16 *__restrict__  expr_17_in_0 = Args->expr_17_in_0;
    F16 *__restrict__  expr_17_out_0 = Args->expr_17_out_0; // (128, 8, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (128, 8, 10) var shapes:
    // expr_17_out_0: (128, 8, 10) expr_17_in_0: (128, 8, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_17_in_0: f16
        // expr_17_out_0 = Mul(expr_17_in_0, FastFloatSigmoid(expr_17_in_0))
        expr_17_out_0[i0] = (expr_17_in_0[i0]*FastSigmoidF16(expr_17_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_19(expr_19_args_t *Args) {
    F16 *__restrict__  expr_19_in_0 = Args->expr_19_in_0;
    F16 *__restrict__  expr_19_out_0 = Args->expr_19_out_0; // (128, 8, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (128, 8, 10) var shapes:
    // expr_19_out_0: (128, 8, 10) expr_19_in_0: (128, 8, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_19_in_0: f16
        // expr_19_out_0 = Mul(expr_19_in_0, FastFloatSigmoid(expr_19_in_0))
        expr_19_out_0[i0] = (expr_19_in_0[i0]*FastSigmoidF16(expr_19_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_20(expr_20_args_t *Args) {
    F16 *__restrict__  expr_20_in_0 = Args->expr_20_in_0;
    F16 *__restrict__  expr_20_out_0 = Args->expr_20_out_0; // (256, 8, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (256, 8, 10) var shapes:
    // expr_20_out_0: (256, 8, 10) expr_20_in_0: (256, 8, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_20_in_0: f16
        // expr_20_out_0 = Mul(expr_20_in_0, FastFloatSigmoid(expr_20_in_0))
        expr_20_out_0[i0] = (expr_20_in_0[i0]*FastSigmoidF16(expr_20_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_21(expr_21_args_t *Args) {
    F16 *__restrict__  expr_21_in_0 = Args->expr_21_in_0;
    F16 *__restrict__  expr_21_out_0 = Args->expr_21_out_0; // (128, 8, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (128, 8, 10) var shapes:
    // expr_21_out_0: (128, 8, 10) expr_21_in_0: (128, 8, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_21_in_0: f16
        // expr_21_out_0 = Mul(expr_21_in_0, FastFloatSigmoid(expr_21_in_0))
        expr_21_out_0[i0] = (expr_21_in_0[i0]*FastSigmoidF16(expr_21_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_22(expr_22_args_t *Args) {
    F16 *__restrict__  expr_22_in_0 = Args->expr_22_in_0;
    F16 *__restrict__  expr_22_out_0 = Args->expr_22_out_0; // (64, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 16, 20) var shapes:
    // expr_22_out_0: (64, 16, 20) expr_22_in_0: (64, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_22_in_0: f16
        // expr_22_out_0 = Mul(expr_22_in_0, FastFloatSigmoid(expr_22_in_0))
        expr_22_out_0[i0] = (expr_22_in_0[i0]*FastSigmoidF16(expr_22_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_23(expr_23_args_t *Args) {
    F16 *__restrict__  expr_23_in_0 = Args->expr_23_in_0;
    F16 *__restrict__  expr_23_out_0 = Args->expr_23_out_0; // (64, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 16, 20) var shapes:
    // expr_23_out_0: (64, 16, 20) expr_23_in_0: (64, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_23_in_0: f16
        // expr_23_out_0 = Mul(expr_23_in_0, FastFloatSigmoid(expr_23_in_0))
        expr_23_out_0[i0] = (expr_23_in_0[i0]*FastSigmoidF16(expr_23_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_24(expr_24_args_t *Args) {
    F16 *__restrict__  expr_24_in_0 = Args->expr_24_in_0;
    F16 *__restrict__  expr_24_out_0 = Args->expr_24_out_0; // (64, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 16, 20) var shapes:
    // expr_24_out_0: (64, 16, 20) expr_24_in_0: (64, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_24_in_0: f16
        // expr_24_out_0 = Mul(expr_24_in_0, FastFloatSigmoid(expr_24_in_0))
        expr_24_out_0[i0] = (expr_24_in_0[i0]*FastSigmoidF16(expr_24_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_25(expr_25_args_t *Args) {
    F16 *__restrict__  expr_25_in_0 = Args->expr_25_in_0;
    F16 *__restrict__  expr_25_out_0 = Args->expr_25_out_0; // (64, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 16, 20) var shapes:
    // expr_25_out_0: (64, 16, 20) expr_25_in_0: (64, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_25_in_0: f16
        // expr_25_out_0 = Mul(expr_25_in_0, FastFloatSigmoid(expr_25_in_0))
        expr_25_out_0[i0] = (expr_25_in_0[i0]*FastSigmoidF16(expr_25_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_26(expr_26_args_t *Args) {
    F16 *__restrict__  expr_26_in_0 = Args->expr_26_in_0;
    F16 *__restrict__  expr_26_out_0 = Args->expr_26_out_0; // (64, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 16, 20) var shapes:
    // expr_26_out_0: (64, 16, 20) expr_26_in_0: (64, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_26_in_0: f16
        // expr_26_out_0 = Mul(expr_26_in_0, FastFloatSigmoid(expr_26_in_0))
        expr_26_out_0[i0] = (expr_26_in_0[i0]*FastSigmoidF16(expr_26_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_28(expr_28_args_t *Args) {
    F16 *__restrict__  expr_28_in_0 = Args->expr_28_in_0;
    F16 *__restrict__  expr_28_out_0 = Args->expr_28_out_0; // (128, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (128, 16, 20) var shapes:
    // expr_28_out_0: (128, 16, 20) expr_28_in_0: (128, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_28_in_0: f16
        // expr_28_out_0 = Mul(expr_28_in_0, FastFloatSigmoid(expr_28_in_0))
        expr_28_out_0[i0] = (expr_28_in_0[i0]*FastSigmoidF16(expr_28_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_29(expr_29_args_t *Args) {
    F16 *__restrict__  expr_29_in_0 = Args->expr_29_in_0;
    F16 *__restrict__  expr_29_out_0 = Args->expr_29_out_0; // (64, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 16, 20) var shapes:
    // expr_29_out_0: (64, 16, 20) expr_29_in_0: (64, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_29_in_0: f16
        // expr_29_out_0 = Mul(expr_29_in_0, FastFloatSigmoid(expr_29_in_0))
        expr_29_out_0[i0] = (expr_29_in_0[i0]*FastSigmoidF16(expr_29_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_30(expr_30_args_t *Args) {
    F16 *__restrict__  expr_30_in_0 = Args->expr_30_in_0;
    F16 *__restrict__  expr_30_out_0 = Args->expr_30_out_0; // (32, 32, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 32, 40) var shapes:
    // expr_30_out_0: (32, 32, 40) expr_30_in_0: (32, 32, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_30_in_0: f16
        // expr_30_out_0 = Mul(expr_30_in_0, FastFloatSigmoid(expr_30_in_0))
        expr_30_out_0[i0] = (expr_30_in_0[i0]*FastSigmoidF16(expr_30_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_31(expr_31_args_t *Args) {
    F16 *__restrict__  expr_31_in_0 = Args->expr_31_in_0;
    F16 *__restrict__  expr_31_out_0 = Args->expr_31_out_0; // (32, 32, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 32, 40) var shapes:
    // expr_31_out_0: (32, 32, 40) expr_31_in_0: (32, 32, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_31_in_0: f16
        // expr_31_out_0 = Mul(expr_31_in_0, FastFloatSigmoid(expr_31_in_0))
        expr_31_out_0[i0] = (expr_31_in_0[i0]*FastSigmoidF16(expr_31_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_32(expr_32_args_t *Args) {
    F16 *__restrict__  expr_32_in_0 = Args->expr_32_in_0;
    F16 *__restrict__  expr_32_out_0 = Args->expr_32_out_0; // (32, 32, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 32, 40) var shapes:
    // expr_32_out_0: (32, 32, 40) expr_32_in_0: (32, 32, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_32_in_0: f16
        // expr_32_out_0 = Mul(expr_32_in_0, FastFloatSigmoid(expr_32_in_0))
        expr_32_out_0[i0] = (expr_32_in_0[i0]*FastSigmoidF16(expr_32_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_33(expr_33_args_t *Args) {
    F16 *__restrict__  expr_33_in_0 = Args->expr_33_in_0;
    F16 *__restrict__  expr_33_out_0 = Args->expr_33_out_0; // (32, 32, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 32, 40) var shapes:
    // expr_33_out_0: (32, 32, 40) expr_33_in_0: (32, 32, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_33_in_0: f16
        // expr_33_out_0 = Mul(expr_33_in_0, FastFloatSigmoid(expr_33_in_0))
        expr_33_out_0[i0] = (expr_33_in_0[i0]*FastSigmoidF16(expr_33_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_34(expr_34_args_t *Args) {
    F16 *__restrict__  expr_34_in_0 = Args->expr_34_in_0;
    F16 *__restrict__  expr_34_out_0 = Args->expr_34_out_0; // (32, 32, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 32, 40) var shapes:
    // expr_34_out_0: (32, 32, 40) expr_34_in_0: (32, 32, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_34_in_0: f16
        // expr_34_out_0 = Mul(expr_34_in_0, FastFloatSigmoid(expr_34_in_0))
        expr_34_out_0[i0] = (expr_34_in_0[i0]*FastSigmoidF16(expr_34_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_35(expr_35_args_t *Args) {
    F16 *__restrict__  expr_35_in_0 = Args->expr_35_in_0;
    F16 *__restrict__  expr_35_out_0 = Args->expr_35_out_0; // (64, 32, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 32, 40) var shapes:
    // expr_35_out_0: (64, 32, 40) expr_35_in_0: (64, 32, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_35_in_0: f16
        // expr_35_out_0 = Mul(expr_35_in_0, FastFloatSigmoid(expr_35_in_0))
        expr_35_out_0[i0] = (expr_35_in_0[i0]*FastSigmoidF16(expr_35_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_36(expr_36_args_t *Args) {
    F16 *__restrict__  expr_36_in_0 = Args->expr_36_in_0;
    F16 *__restrict__  expr_36_out_0 = Args->expr_36_out_0; // (64, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 16, 20) var shapes:
    // expr_36_out_0: (64, 16, 20) expr_36_in_0: (64, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_36_in_0: f16
        // expr_36_out_0 = Mul(expr_36_in_0, FastFloatSigmoid(expr_36_in_0))
        expr_36_out_0[i0] = (expr_36_in_0[i0]*FastSigmoidF16(expr_36_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_38(expr_38_args_t *Args) {
    F16 *__restrict__  expr_38_in_0 = Args->expr_38_in_0;
    F16 *__restrict__  expr_38_out_0 = Args->expr_38_out_0; // (64, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 16, 20) var shapes:
    // expr_38_out_0: (64, 16, 20) expr_38_in_0: (64, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_38_in_0: f16
        // expr_38_out_0 = Mul(expr_38_in_0, FastFloatSigmoid(expr_38_in_0))
        expr_38_out_0[i0] = (expr_38_in_0[i0]*FastSigmoidF16(expr_38_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_39(expr_39_args_t *Args) {
    F16 *__restrict__  expr_39_in_0 = Args->expr_39_in_0;
    F16 *__restrict__  expr_39_out_0 = Args->expr_39_out_0; // (64, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 16, 20) var shapes:
    // expr_39_out_0: (64, 16, 20) expr_39_in_0: (64, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_39_in_0: f16
        // expr_39_out_0 = Mul(expr_39_in_0, FastFloatSigmoid(expr_39_in_0))
        expr_39_out_0[i0] = (expr_39_in_0[i0]*FastSigmoidF16(expr_39_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_40(expr_40_args_t *Args) {
    F16 *__restrict__  expr_40_in_0 = Args->expr_40_in_0;
    F16 *__restrict__  expr_40_out_0 = Args->expr_40_out_0; // (64, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 16, 20) var shapes:
    // expr_40_out_0: (64, 16, 20) expr_40_in_0: (64, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_40_in_0: f16
        // expr_40_out_0 = Mul(expr_40_in_0, FastFloatSigmoid(expr_40_in_0))
        expr_40_out_0[i0] = (expr_40_in_0[i0]*FastSigmoidF16(expr_40_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_41(expr_41_args_t *Args) {
    F16 *__restrict__  expr_41_in_0 = Args->expr_41_in_0;
    F16 *__restrict__  expr_41_out_0 = Args->expr_41_out_0; // (64, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 16, 20) var shapes:
    // expr_41_out_0: (64, 16, 20) expr_41_in_0: (64, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_41_in_0: f16
        // expr_41_out_0 = Mul(expr_41_in_0, FastFloatSigmoid(expr_41_in_0))
        expr_41_out_0[i0] = (expr_41_in_0[i0]*FastSigmoidF16(expr_41_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_42(expr_42_args_t *Args) {
    F16 *__restrict__  expr_42_in_0 = Args->expr_42_in_0;
    F16 *__restrict__  expr_42_out_0 = Args->expr_42_out_0; // (64, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 16, 20) var shapes:
    // expr_42_out_0: (64, 16, 20) expr_42_in_0: (64, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_42_in_0: f16
        // expr_42_out_0 = Mul(expr_42_in_0, FastFloatSigmoid(expr_42_in_0))
        expr_42_out_0[i0] = (expr_42_in_0[i0]*FastSigmoidF16(expr_42_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_43(expr_43_args_t *Args) {
    F16 *__restrict__  expr_43_in_0 = Args->expr_43_in_0;
    F16 *__restrict__  expr_43_out_0 = Args->expr_43_out_0; // (64, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 16, 20) var shapes:
    // expr_43_out_0: (64, 16, 20) expr_43_in_0: (64, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_43_in_0: f16
        // expr_43_out_0 = Mul(expr_43_in_0, FastFloatSigmoid(expr_43_in_0))
        expr_43_out_0[i0] = (expr_43_in_0[i0]*FastSigmoidF16(expr_43_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_44(expr_44_args_t *Args) {
    F16 *__restrict__  expr_44_in_0 = Args->expr_44_in_0;
    F16 *__restrict__  expr_44_out_0 = Args->expr_44_out_0; // (128, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (128, 16, 20) var shapes:
    // expr_44_out_0: (128, 16, 20) expr_44_in_0: (128, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_44_in_0: f16
        // expr_44_out_0 = Mul(expr_44_in_0, FastFloatSigmoid(expr_44_in_0))
        expr_44_out_0[i0] = (expr_44_in_0[i0]*FastSigmoidF16(expr_44_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_45(expr_45_args_t *Args) {
    F16 *__restrict__  expr_45_in_0 = Args->expr_45_in_0;
    F16 *__restrict__  expr_45_out_0 = Args->expr_45_out_0; // (128, 8, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (128, 8, 10) var shapes:
    // expr_45_out_0: (128, 8, 10) expr_45_in_0: (128, 8, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_45_in_0: f16
        // expr_45_out_0 = Mul(expr_45_in_0, FastFloatSigmoid(expr_45_in_0))
        expr_45_out_0[i0] = (expr_45_in_0[i0]*FastSigmoidF16(expr_45_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_46(expr_46_args_t *Args) {
    F16 *__restrict__  expr_46_in_0 = Args->expr_46_in_0;
    F16 *__restrict__  expr_46_out_0 = Args->expr_46_out_0; // (128, 8, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (128, 8, 10) var shapes:
    // expr_46_out_0: (128, 8, 10) expr_46_in_0: (128, 8, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_46_in_0: f16
        // expr_46_out_0 = Mul(expr_46_in_0, FastFloatSigmoid(expr_46_in_0))
        expr_46_out_0[i0] = (expr_46_in_0[i0]*FastSigmoidF16(expr_46_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_47(expr_47_args_t *Args) {
    F16 *__restrict__  expr_47_in_0 = Args->expr_47_in_0;
    F16 *__restrict__  expr_47_out_0 = Args->expr_47_out_0; // (128, 8, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (128, 8, 10) var shapes:
    // expr_47_out_0: (128, 8, 10) expr_47_in_0: (128, 8, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_47_in_0: f16
        // expr_47_out_0 = Mul(expr_47_in_0, FastFloatSigmoid(expr_47_in_0))
        expr_47_out_0[i0] = (expr_47_in_0[i0]*FastSigmoidF16(expr_47_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_48(expr_48_args_t *Args) {
    F16 *__restrict__  expr_48_in_0 = Args->expr_48_in_0;
    F16 *__restrict__  expr_48_out_0 = Args->expr_48_out_0; // (128, 8, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (128, 8, 10) var shapes:
    // expr_48_out_0: (128, 8, 10) expr_48_in_0: (128, 8, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_48_in_0: f16
        // expr_48_out_0 = Mul(expr_48_in_0, FastFloatSigmoid(expr_48_in_0))
        expr_48_out_0[i0] = (expr_48_in_0[i0]*FastSigmoidF16(expr_48_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_49(expr_49_args_t *Args) {
    F16 *__restrict__  expr_49_in_0 = Args->expr_49_in_0;
    F16 *__restrict__  expr_49_out_0 = Args->expr_49_out_0; // (128, 8, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (128, 8, 10) var shapes:
    // expr_49_out_0: (128, 8, 10) expr_49_in_0: (128, 8, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_49_in_0: f16
        // expr_49_out_0 = Mul(expr_49_in_0, FastFloatSigmoid(expr_49_in_0))
        expr_49_out_0[i0] = (expr_49_in_0[i0]*FastSigmoidF16(expr_49_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_50(expr_50_args_t *Args) {
    F16 *__restrict__  expr_50_in_0 = Args->expr_50_in_0;
    F16 *__restrict__  expr_50_out_0 = Args->expr_50_out_0; // (128, 8, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (128, 8, 10) var shapes:
    // expr_50_out_0: (128, 8, 10) expr_50_in_0: (128, 8, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_50_in_0: f16
        // expr_50_out_0 = Mul(expr_50_in_0, FastFloatSigmoid(expr_50_in_0))
        expr_50_out_0[i0] = (expr_50_in_0[i0]*FastSigmoidF16(expr_50_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_51(expr_51_args_t *Args) {
    F16 *__restrict__  expr_51_in_0 = Args->expr_51_in_0;
    F16 *__restrict__  expr_51_out_0 = Args->expr_51_out_0; // (128, 8, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (128, 8, 10) var shapes:
    // expr_51_out_0: (128, 8, 10) expr_51_in_0: (128, 8, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_51_in_0: f16
        // expr_51_out_0 = Mul(expr_51_in_0, FastFloatSigmoid(expr_51_in_0))
        expr_51_out_0[i0] = (expr_51_in_0[i0]*FastSigmoidF16(expr_51_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_52(expr_52_args_t *Args) {
    F16 *__restrict__  expr_52_in_0 = Args->expr_52_in_0;
    F16 *__restrict__  expr_52_out_0 = Args->expr_52_out_0; // (256, 8, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (256, 8, 10) var shapes:
    // expr_52_out_0: (256, 8, 10) expr_52_in_0: (256, 8, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_52_in_0: f16
        // expr_52_out_0 = Mul(expr_52_in_0, FastFloatSigmoid(expr_52_in_0))
        expr_52_out_0[i0] = (expr_52_in_0[i0]*FastSigmoidF16(expr_52_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_53(expr_53_args_t *Args) {
    F16 *__restrict__  expr_53_in_0 = Args->expr_53_in_0;
    F16 *__restrict__  expr_53_out_0 = Args->expr_53_out_0; // (64, 32, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 32, 40) var shapes:
    // expr_53_out_0: (64, 32, 40) expr_53_in_0: (64, 32, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_53_in_0: f16
        // expr_53_out_0 = Mul(expr_53_in_0, FastFloatSigmoid(expr_53_in_0))
        expr_53_out_0[i0] = (expr_53_in_0[i0]*FastSigmoidF16(expr_53_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_54(expr_54_args_t *Args) {
    F16 *__restrict__  expr_54_in_0 = Args->expr_54_in_0;
    F16 *__restrict__  expr_54_out_0 = Args->expr_54_out_0; // (64, 32, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 32, 40) var shapes:
    // expr_54_out_0: (64, 32, 40) expr_54_in_0: (64, 32, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_54_in_0: f16
        // expr_54_out_0 = Mul(expr_54_in_0, FastFloatSigmoid(expr_54_in_0))
        expr_54_out_0[i0] = (expr_54_in_0[i0]*FastSigmoidF16(expr_54_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_55(expr_55_args_t *Args) {
    F16 *__restrict__  expr_55_in_0 = Args->expr_55_in_0;
    F16 *__restrict__  expr_55_out_0 = Args->expr_55_out_0; // (64, 32, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 32, 40) var shapes:
    // expr_55_out_0: (64, 32, 40) expr_55_in_0: (64, 32, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_55_in_0: f16
        // expr_55_out_0 = Mul(expr_55_in_0, FastFloatSigmoid(expr_55_in_0))
        expr_55_out_0[i0] = (expr_55_in_0[i0]*FastSigmoidF16(expr_55_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_56(expr_56_args_t *Args) {
    F16 *__restrict__  expr_56_in_0 = Args->expr_56_in_0;
    F16 *__restrict__  expr_56_out_0 = Args->expr_56_out_0; // (64, 32, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 32, 40) var shapes:
    // expr_56_out_0: (64, 32, 40) expr_56_in_0: (64, 32, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_56_in_0: f16
        // expr_56_out_0 = Mul(expr_56_in_0, FastFloatSigmoid(expr_56_in_0))
        expr_56_out_0[i0] = (expr_56_in_0[i0]*FastSigmoidF16(expr_56_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_57(expr_57_args_t *Args) {
    F16 *__restrict__  expr_57_in_0 = Args->expr_57_in_0;
    F16 *__restrict__  expr_57_out_0 = Args->expr_57_out_0; // (64, 32, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
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
        // inputs expr_57_in_0: f16
        // expr_57_out_0 = Mul(expr_57_in_0, FastFloatSigmoid(expr_57_in_0))
        expr_57_out_0[i0] = (expr_57_in_0[i0]*FastSigmoidF16(expr_57_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_58(expr_58_args_t *Args) {
    F16 *__restrict__  expr_58_in_0 = Args->expr_58_in_0;
    F16 *__restrict__  expr_58_out_0 = Args->expr_58_out_0; // (64, 32, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 32, 40) var shapes:
    // expr_58_out_0: (64, 32, 40) expr_58_in_0: (64, 32, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_58_in_0: f16
        // expr_58_out_0 = Mul(expr_58_in_0, FastFloatSigmoid(expr_58_in_0))
        expr_58_out_0[i0] = (expr_58_in_0[i0]*FastSigmoidF16(expr_58_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_59(expr_59_args_t *Args) {
    F16 *__restrict__  expr_59_in_0 = Args->expr_59_in_0;
    F16 *__restrict__  expr_59_out_0 = Args->expr_59_out_0; // (64, 32, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 32, 40) var shapes:
    // expr_59_out_0: (64, 32, 40) expr_59_in_0: (64, 32, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_59_in_0: f16
        // expr_59_out_0 = Mul(expr_59_in_0, FastFloatSigmoid(expr_59_in_0))
        expr_59_out_0[i0] = (expr_59_in_0[i0]*FastSigmoidF16(expr_59_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_60(expr_60_args_t *Args) {
    F16 *__restrict__  expr_60_in_0 = Args->expr_60_in_0;
    F16 *__restrict__  expr_60_out_0 = Args->expr_60_out_0; // (64, 32, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 32, 40) var shapes:
    // expr_60_out_0: (64, 32, 40) expr_60_in_0: (64, 32, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_60_in_0: f16
        // expr_60_out_0 = Mul(expr_60_in_0, FastFloatSigmoid(expr_60_in_0))
        expr_60_out_0[i0] = (expr_60_in_0[i0]*FastSigmoidF16(expr_60_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_61(expr_61_args_t *Args) {
    F16 *__restrict__  expr_61_in_0 = Args->expr_61_in_0;
    F16 *__restrict__  expr_61_out_0 = Args->expr_61_out_0; // (64, 32, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 32, 40) var shapes:
    // expr_61_out_0: (64, 32, 40) expr_61_in_0: (64, 32, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_61_in_0: f16
        // expr_61_out_0 = Mul(expr_61_in_0, FastFloatSigmoid(expr_61_in_0))
        expr_61_out_0[i0] = (expr_61_in_0[i0]*FastSigmoidF16(expr_61_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_63(expr_63_args_t *Args) {
    F16 *__restrict__  expr_63_in_0 = Args->expr_63_in_0;
    F16 *__restrict__  expr_63_out_0 = Args->expr_63_out_0; // (64, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 16, 20) var shapes:
    // expr_63_out_0: (64, 16, 20) expr_63_in_0: (64, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_63_in_0: f16
        // expr_63_out_0 = Mul(expr_63_in_0, FastFloatSigmoid(expr_63_in_0))
        expr_63_out_0[i0] = (expr_63_in_0[i0]*FastSigmoidF16(expr_63_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_64(expr_64_args_t *Args) {
    F16 *__restrict__  expr_64_in_0 = Args->expr_64_in_0;
    F16 *__restrict__  expr_64_out_0 = Args->expr_64_out_0; // (64, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 16, 20) var shapes:
    // expr_64_out_0: (64, 16, 20) expr_64_in_0: (64, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_64_in_0: f16
        // expr_64_out_0 = Mul(expr_64_in_0, FastFloatSigmoid(expr_64_in_0))
        expr_64_out_0[i0] = (expr_64_in_0[i0]*FastSigmoidF16(expr_64_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_65(expr_65_args_t *Args) {
    F16 *__restrict__  expr_65_in_0 = Args->expr_65_in_0;
    F16 *__restrict__  expr_65_out_0 = Args->expr_65_out_0; // (64, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 16, 20) var shapes:
    // expr_65_out_0: (64, 16, 20) expr_65_in_0: (64, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_65_in_0: f16
        // expr_65_out_0 = Mul(expr_65_in_0, FastFloatSigmoid(expr_65_in_0))
        expr_65_out_0[i0] = (expr_65_in_0[i0]*FastSigmoidF16(expr_65_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_67(expr_67_args_t *Args) {
    F16 *__restrict__  expr_67_in_0 = Args->expr_67_in_0;
    F16 *__restrict__  expr_67_out_0 = Args->expr_67_out_0; // (64, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 16, 20) var shapes:
    // expr_67_out_0: (64, 16, 20) expr_67_in_0: (64, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_67_in_0: f16
        // expr_67_out_0 = Mul(expr_67_in_0, FastFloatSigmoid(expr_67_in_0))
        expr_67_out_0[i0] = (expr_67_in_0[i0]*FastSigmoidF16(expr_67_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_68(expr_68_args_t *Args) {
    F16 *__restrict__  expr_68_in_0 = Args->expr_68_in_0;
    F16 *__restrict__  expr_68_out_0 = Args->expr_68_out_0; // (64, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
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
        // inputs expr_68_in_0: f16
        // expr_68_out_0 = Mul(expr_68_in_0, FastFloatSigmoid(expr_68_in_0))
        expr_68_out_0[i0] = (expr_68_in_0[i0]*FastSigmoidF16(expr_68_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_69(expr_69_args_t *Args) {
    F16 *__restrict__  expr_69_in_0 = Args->expr_69_in_0;
    F16 *__restrict__  expr_69_out_0 = Args->expr_69_out_0; // (64, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 16, 20) var shapes:
    // expr_69_out_0: (64, 16, 20) expr_69_in_0: (64, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_69_in_0: f16
        // expr_69_out_0 = Mul(expr_69_in_0, FastFloatSigmoid(expr_69_in_0))
        expr_69_out_0[i0] = (expr_69_in_0[i0]*FastSigmoidF16(expr_69_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_70(expr_70_args_t *Args) {
    F16 *__restrict__  expr_70_in_0 = Args->expr_70_in_0;
    F16 *__restrict__  expr_70_out_0 = Args->expr_70_out_0; // (64, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 16, 20) var shapes:
    // expr_70_out_0: (64, 16, 20) expr_70_in_0: (64, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_70_in_0: f16
        // expr_70_out_0 = Mul(expr_70_in_0, FastFloatSigmoid(expr_70_in_0))
        expr_70_out_0[i0] = (expr_70_in_0[i0]*FastSigmoidF16(expr_70_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_71(expr_71_args_t *Args) {
    F16 *__restrict__  expr_71_in_0 = Args->expr_71_in_0;
    F16 *__restrict__  expr_71_out_0 = Args->expr_71_out_0; // (64, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 16, 20) var shapes:
    // expr_71_out_0: (64, 16, 20) expr_71_in_0: (64, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_71_in_0: f16
        // expr_71_out_0 = Mul(expr_71_in_0, FastFloatSigmoid(expr_71_in_0))
        expr_71_out_0[i0] = (expr_71_in_0[i0]*FastSigmoidF16(expr_71_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_72(expr_72_args_t *Args) {
    F16 *__restrict__  expr_72_in_0 = Args->expr_72_in_0;
    F16 *__restrict__  expr_72_out_0 = Args->expr_72_out_0; // (64, 16, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 16, 20) var shapes:
    // expr_72_out_0: (64, 16, 20) expr_72_in_0: (64, 16, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_72_in_0: f16
        // expr_72_out_0 = Mul(expr_72_in_0, FastFloatSigmoid(expr_72_in_0))
        expr_72_out_0[i0] = (expr_72_in_0[i0]*FastSigmoidF16(expr_72_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_74(expr_74_args_t *Args) {
    F16 *__restrict__  expr_74_in_0 = Args->expr_74_in_0;
    F16 *__restrict__  expr_74_out_0 = Args->expr_74_out_0; // (64, 8, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 8, 10) var shapes:
    // expr_74_out_0: (64, 8, 10) expr_74_in_0: (64, 8, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_74_in_0: f16
        // expr_74_out_0 = Mul(expr_74_in_0, FastFloatSigmoid(expr_74_in_0))
        expr_74_out_0[i0] = (expr_74_in_0[i0]*FastSigmoidF16(expr_74_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_75(expr_75_args_t *Args) {
    F16 *__restrict__  expr_75_in_0 = Args->expr_75_in_0;
    F16 *__restrict__  expr_75_out_0 = Args->expr_75_out_0; // (64, 8, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 8, 10) var shapes:
    // expr_75_out_0: (64, 8, 10) expr_75_in_0: (64, 8, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_75_in_0: f16
        // expr_75_out_0 = Mul(expr_75_in_0, FastFloatSigmoid(expr_75_in_0))
        expr_75_out_0[i0] = (expr_75_in_0[i0]*FastSigmoidF16(expr_75_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_76(expr_76_args_t *Args) {
    F16 *__restrict__  expr_76_in_0 = Args->expr_76_in_0;
    F16 *__restrict__  expr_76_out_0 = Args->expr_76_out_0; // (64, 8, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 8, 10) var shapes:
    // expr_76_out_0: (64, 8, 10) expr_76_in_0: (64, 8, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_76_in_0: f16
        // expr_76_out_0 = Mul(expr_76_in_0, FastFloatSigmoid(expr_76_in_0))
        expr_76_out_0[i0] = (expr_76_in_0[i0]*FastSigmoidF16(expr_76_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_77(expr_77_args_t *Args) {
    F16 *__restrict__  expr_77_in_0 = Args->expr_77_in_0;
    F16 *__restrict__  expr_77_out_0 = Args->expr_77_out_0; // (64, 8, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 8, 10) var shapes:
    // expr_77_out_0: (64, 8, 10) expr_77_in_0: (64, 8, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_77_in_0: f16
        // expr_77_out_0 = Mul(expr_77_in_0, FastFloatSigmoid(expr_77_in_0))
        expr_77_out_0[i0] = (expr_77_in_0[i0]*FastSigmoidF16(expr_77_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_78(expr_78_args_t *Args) {
    F16 *__restrict__  expr_78_in_0 = Args->expr_78_in_0;
    F16 *__restrict__  expr_78_out_0 = Args->expr_78_out_0; // (64, 8, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
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
        // inputs expr_78_in_0: f16
        // expr_78_out_0 = Mul(expr_78_in_0, FastFloatSigmoid(expr_78_in_0))
        expr_78_out_0[i0] = (expr_78_in_0[i0]*FastSigmoidF16(expr_78_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_79(expr_79_args_t *Args) {
    F16 *__restrict__  expr_79_in_0 = Args->expr_79_in_0;
    F16 *__restrict__  expr_79_out_0 = Args->expr_79_out_0; // (64, 8, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 8, 10) var shapes:
    // expr_79_out_0: (64, 8, 10) expr_79_in_0: (64, 8, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_79_in_0: f16
        // expr_79_out_0 = Mul(expr_79_in_0, FastFloatSigmoid(expr_79_in_0))
        expr_79_out_0[i0] = (expr_79_in_0[i0]*FastSigmoidF16(expr_79_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_80(expr_80_args_t *Args) {
    F16 *__restrict__  expr_80_in_0 = Args->expr_80_in_0;
    F16 *__restrict__  expr_80_out_0 = Args->expr_80_out_0; // (64, 8, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 8, 10) var shapes:
    // expr_80_out_0: (64, 8, 10) expr_80_in_0: (64, 8, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_80_in_0: f16
        // expr_80_out_0 = Mul(expr_80_in_0, FastFloatSigmoid(expr_80_in_0))
        expr_80_out_0[i0] = (expr_80_in_0[i0]*FastSigmoidF16(expr_80_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_81(expr_81_args_t *Args) {
    F16 *__restrict__  expr_81_in_0 = Args->expr_81_in_0;
    F16 *__restrict__  expr_81_out_0 = Args->expr_81_out_0; // (64, 8, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 8, 10) var shapes:
    // expr_81_out_0: (64, 8, 10) expr_81_in_0: (64, 8, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_81_in_0: f16
        // expr_81_out_0 = Mul(expr_81_in_0, FastFloatSigmoid(expr_81_in_0))
        expr_81_out_0[i0] = (expr_81_in_0[i0]*FastSigmoidF16(expr_81_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_83(expr_83_args_t *Args) {
    F16 *__restrict__  expr_83_in_0 = Args->expr_83_in_0;
    F16 *__restrict__  expr_83_out_0 = Args->expr_83_out_0; // (64, 8, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 8, 10) var shapes:
    // expr_83_out_0: (64, 8, 10) expr_83_in_0: (64, 8, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_83_in_0: f16
        // expr_83_out_0 = Mul(expr_83_in_0, FastFloatSigmoid(expr_83_in_0))
        expr_83_out_0[i0] = (expr_83_in_0[i0]*FastSigmoidF16(expr_83_in_0[i0]));
    }
    gap_waitbarrier(0);
}


#pragma GCC diagnostic pop