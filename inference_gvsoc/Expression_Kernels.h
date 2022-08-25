#ifndef MODEL_BASIC_KERNELS_H
#define MODEL_BASIC_KERNELS_H
#include "Gap.h"
#include "math_funcs.h"
#include "CNN_BasicKernels_SQ8.h"
#include "Gap.h"

typedef struct {
    unsigned int I0;
    signed char *__restrict__  expr_0_in_0;
    signed char *__restrict__  expr_0_in_1;
    signed char *__restrict__  expr_0_out_0;
} s24_kernel_args_t;

typedef struct {
    unsigned int I0;
    signed char *__restrict__  expr_1_in_0;
    signed char *__restrict__  expr_1_in_1;
    signed char *__restrict__  expr_1_out_0;
} s50_kernel_args_t;

typedef struct {
    unsigned int I0;
    signed char *__restrict__  expr_2_in_0;
    signed char *__restrict__  expr_2_in_1;
    signed char *__restrict__  expr_2_out_0;
} s96_kernel_args_t;

typedef struct {
    unsigned int I0;
    signed char *__restrict__  expr_57_in_0;
    signed char *__restrict__  expr_57_out_0;
} s271_kernel_args_t;

typedef struct {
    unsigned int I0;
    signed char *__restrict__  expr_68_in_0;
    signed char *__restrict__  expr_68_out_0;
} s310_kernel_args_t;

typedef struct {
    unsigned int I0;
    signed char *__restrict__  expr_78_in_0;
    signed char *__restrict__  expr_78_out_0;
} s349_kernel_args_t;

typedef struct {
    signed char *__restrict__  expr_66_in_0;
    signed char *__restrict__  expr_66_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
    signed char * __restrict__ Infos;
} custom_0_args_t;

typedef struct {
    signed char *__restrict__  expr_91_in_0;
    signed char *__restrict__  expr_91_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
    signed char * __restrict__ Infos;
} custom_1_args_t;

typedef struct {
    signed char *__restrict__  expr_27_in_0;
    signed char *__restrict__  expr_27_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
    signed char * __restrict__ Infos;
} custom_2_args_t;

typedef struct {
    signed char *__restrict__  expr_86_in_0;
    signed char *__restrict__  expr_86_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
    signed char * __restrict__ Infos;
} custom_3_args_t;

typedef struct {
    signed char *__restrict__  expr_39_in_0;
    signed char *__restrict__  expr_39_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_39_args_t;


void s24_kernel(s24_kernel_args_t *Args);

void s50_kernel(s50_kernel_args_t *Args);

void s96_kernel(s96_kernel_args_t *Args);

void s271_kernel(s271_kernel_args_t *Args);

void s310_kernel(s310_kernel_args_t *Args);

void s349_kernel(s349_kernel_args_t *Args);

void custom_0(custom_0_args_t *Args);

void custom_1(custom_1_args_t *Args);

void custom_2(custom_2_args_t *Args);

void custom_3(custom_3_args_t *Args);

void expr_39(expr_39_args_t *Args);


#endif // MODEL_BASIC_KERNELS_H