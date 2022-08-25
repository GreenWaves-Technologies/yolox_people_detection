#ifndef __MODELKERNEL_H__
#define __MODELKERNEL_H__

#include "AutoTilerLibTypes.h"
#include "Gap.h"
#include "model.h"
#include "CNN_BasicKernels_SQ8.h"
#include "ResizeBasicKernels.h"
#include "Expression_Kernels.h"
#include "CNN_Copy.h"
#define _model_L1_Memory_SIZE 115696
#define _model_L2_Memory_SIZE 426560
#define _model_L2_Memory_Dyn_SIZE 573440
extern char *model_L1_Memory; /* Size given for generation: 115712 bytes, used: 115696 bytes */
extern char *model_L2_Memory; /* Size used for generation (static): 426560 bytes */
extern char *model_L2_Memory_Dyn; /* Size used for generation (dynamic): 573440 bytes */
extern void S3_Conv2d_16x12x3x3_Custom(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S6_Conv2d_16x1x3x3_Custom(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S9_Conv2d_32x16x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S12_Conv2d_32x32x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S14_Op_Conv_9_split_copy(
		signed char * __restrict__ In,
		signed char * __restrict__ Out);
extern void S17_Conv2d_16x16x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S20_Conv2d_16x1x3x3_Custom(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S23_Conv2d_16x16x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S24_Op_expr_0(
		signed char * __restrict__ expr_0_in_0,
		signed char * __restrict__ expr_0_in_1,
		signed char * __restrict__ expr_0_out_0);
extern void S28_Conv2d_32x32x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S31_Conv2d_32x1x3x3_Custom(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S34_Conv2d_64x32x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S37_Conv2d_64x64x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S39_Op_Conv_41_fusion_qin0(
		signed char * __restrict__ In,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S40_Op_Conv_35_split_copy(
		signed char * __restrict__ In,
		signed char * __restrict__ Out);
extern void S43_Conv2d_32x32x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S46_Conv2d_32x1x3x3_Custom(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S49_Conv2d_32x32x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S50_Op_expr_1(
		signed char * __restrict__ expr_1_in_0,
		signed char * __restrict__ expr_1_in_1,
		signed char * __restrict__ expr_1_out_0);
extern void S53_Conv2d_32x32x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S56_Conv2d_32x1x3x3_Custom(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S59_Conv2d_32x32x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S60_MatAdd_32x32x40(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S63_Conv2d_32x32x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S66_Conv2d_32x1x3x3_Custom(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S69_Conv2d_32x32x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S70_MatAdd_32x32x40(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S74_Conv2d_64x64x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S77_Conv2d_64x1x3x3_Custom(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S80_Conv2d_128x64x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S83_Conv2d_128x128x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S85_Op_Conv_87_fusion_qin0(
		signed char * __restrict__ In,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S86_Op_Conv_81_split_copy(
		signed char * __restrict__ In,
		signed char * __restrict__ Out);
extern void S89_Conv2d_64x64x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S92_Conv2d_64x1x3x3_Custom(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S95_Conv2d_64x64x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S96_Op_expr_2(
		signed char * __restrict__ expr_2_in_0,
		signed char * __restrict__ expr_2_in_1,
		signed char * __restrict__ expr_2_out_0);
extern void S99_Conv2d_64x64x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S102_Conv2d_64x1x3x3_Custom(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S105_Conv2d_64x64x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S106_MatAdd_64x16x20(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S109_Conv2d_64x64x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S112_Conv2d_64x1x3x3_Custom(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S115_Conv2d_64x64x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S116_MatAdd_64x16x20(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S120_Conv2d_128x128x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S123_Conv2d_128x1x3x3_Custom(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S126_Conv2d_256x128x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S129_Conv2d_128x256x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S130_MaxPool_13x13(
		signed char * __restrict__ In,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S131_MaxPool_5x5(
		signed char * __restrict__ In,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S132_MaxPool_9x9(
		signed char * __restrict__ In,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S136_Conv2d_256x512x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S139_Conv2d_256x256x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S141_Op_Conv_143_fusion_qin0(
		signed char * __restrict__ In,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S142_Op_Conv_137_split_copy(
		signed char * __restrict__ In,
		signed char * __restrict__ Out);
extern void S145_Conv2d_128x128x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S148_Conv2d_128x1x3x3_Custom(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S151_Conv2d_128x128x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S155_Conv2d_256x256x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S158_Conv2d_128x256x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S159_Op_Resize_160(
		signed char * In,
		signed char * Out);
extern void S163_Conv2d_128x256x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S165_Op_Conv_168_fusion_qin0(
		signed char * __restrict__ In,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S166_Op_Conv_162_split_copy(
		signed char * __restrict__ In,
		signed char * __restrict__ Out);
extern void S169_Conv2d_64x64x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S172_Conv2d_64x1x3x3_Custom(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S175_Conv2d_64x64x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S179_Conv2d_128x128x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S182_Conv2d_64x128x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S183_Op_Resize_185(
		signed char * In,
		signed char * Out);
extern void S184_Op_Concat_186_qin0(
		signed char * __restrict__ In,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S188_Conv2d_64x128x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S190_Op_Conv_193_fusion_qin0(
		signed char * __restrict__ In,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S191_Op_Conv_187_split_copy(
		signed char * __restrict__ In,
		signed char * __restrict__ Out);
extern void S194_Conv2d_32x32x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S197_Conv2d_32x1x3x3_Custom(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S200_Conv2d_32x32x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S204_Conv2d_64x64x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S207_Conv2d_64x1x3x3_Custom(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S210_Conv2d_64x64x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S214_Conv2d_128x128x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S216_Op_Conv_213_split_copy(
		signed char * __restrict__ In,
		signed char * __restrict__ Out);
extern void S219_Conv2d_64x64x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S222_Conv2d_64x1x3x3_Custom(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S225_Conv2d_64x64x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S229_Conv2d_128x128x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S232_Conv2d_128x1x3x3_Custom(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S235_Conv2d_128x128x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S237_Op_Conv_239_fusion_qin0(
		signed char * __restrict__ In,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S240_Conv2d_256x256x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S242_Op_Conv_239_split_copy(
		signed char * __restrict__ In,
		signed char * __restrict__ Out);
extern void S245_Conv2d_128x128x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S248_Conv2d_128x1x3x3_Custom(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S251_Conv2d_128x128x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S255_Conv2d_256x256x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S258_Conv2d_64x64x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S261_Conv2d_64x1x3x3_Custom(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S264_Conv2d_64x64x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S267_Conv2d_64x1x3x3_Custom(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S270_Conv2d_64x64x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S271_Op_expr_57(
		signed char * __restrict__ expr_57_in_0,
		signed char * __restrict__ expr_57_out_0);
extern void S274_Conv2d_1x64x1x1_Sigmoid(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S277_Conv2d_64x1x3x3_Custom(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S280_Conv2d_64x64x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S283_Conv2d_64x1x3x3_Custom(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S286_Conv2d_64x64x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S289_Conv2d_4x64x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S292_Conv2d_1x64x1x1_Sigmoid(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S297_Conv2d_64x128x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S300_Conv2d_64x1x3x3_Custom(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S303_Conv2d_64x64x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S306_Conv2d_64x1x3x3_Custom(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S309_Conv2d_64x64x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S310_Op_expr_68(
		signed char * __restrict__ expr_68_in_0,
		signed char * __restrict__ expr_68_out_0);
extern void S313_Conv2d_1x64x1x1_Sigmoid(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S316_Conv2d_64x1x3x3_Custom(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S319_Conv2d_64x64x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S322_Conv2d_64x1x3x3_Custom(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S325_Conv2d_64x64x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S328_Conv2d_4x64x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S331_Conv2d_1x64x1x1_Sigmoid(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S336_Conv2d_64x256x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S339_Conv2d_64x1x3x3_Custom(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S342_Conv2d_64x64x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S345_Conv2d_64x1x3x3_Custom(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S348_Conv2d_64x64x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S349_Op_expr_78(
		signed char * __restrict__ expr_78_in_0,
		signed char * __restrict__ expr_78_out_0);
extern void S352_Conv2d_1x64x1x1_Sigmoid(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S355_Conv2d_64x1x3x3_Custom(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S358_Conv2d_64x64x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S361_Conv2d_64x1x3x3_Custom(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S364_Conv2d_64x64x1x1_Custom(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos,
		signed char * __restrict__ CustomInfos);
extern void S367_Conv2d_4x64x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S370_Conv2d_1x64x1x1_Sigmoid(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S373_Concat(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ In3,
		signed char * __restrict__ Out);
extern int modelCNN_Construct();
extern int modelCNN_Destruct();
extern int modelCNN_Memory(int Which);
extern signed char * __restrict__ Input_1;
extern int modelCNN(
		signed char * __restrict__ Output_1);
extern unsigned int AT_GraphPerf[137];
extern unsigned int AT_GraphPerf_CNN_Total;
extern char * AT_GraphNodeNames[137];
extern unsigned int AT_GraphOperInfosNames[137];
#endif
