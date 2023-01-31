#include <stdint.h>
#include <stdio.h>
#include "AutoTilerLib.h"
#include "CNN_Generators_SQ8.h"
#include "CNN_Generators_fp16.h"
#include "CNN_Generators_NE16.h"
#include "ResizeGenerator.h"

#include "CNN_Copy_Generators.h"





void mainModel(unsigned int L1Memory, unsigned int L2Memory, unsigned int L3Memory, unsigned int L3Flash)
{
    KernelOper_T Cop = KOP_CONV;

    // SetKernelOpts(KER_OPT_NONE, KER_OPT_BUFFER_PROMOTE);
    SetSymbolDynamics();

    SetUsedFilesNames(0, 6, "Gap.h", "main.h", "CNN_BasicKernels_SQ8.h", "CNN_BasicKernels_fp16.h", "CNN_BasicKernels_NE16.h", "ResizeBasicKernels.h");
    SetGeneratedFilesNames("mainKernels.c", "mainKernels.h");
    AT_SetGraphCtrl(AT_GRAPH_MONITOR_CYCLES, AT_OPT_ON);
    AT_SetGraphCtrl(AT_GRAPH_PRODUCE_NODE_NAMES, AT_OPT_ON);
    AT_SetGraphCtrl(AT_GRAPH_PRODUCE_OPERINFOS, AT_OPT_ON);
    AT_SetGraphCtrl(AT_GRAPH_CONST_EXEC_FROM_FLASH, AT_OPT_ON);
//    AT_SetGraphCtrl(AT_GRAPH_DUMP_TENSOR, AT_OPT_VAL(6));

    SetMemoryDeviceInfos(4,
        AT_MEM_L1, L1Memory, "main_L1_Memory", 0, 0,
        AT_MEM_L2, L2Memory, "main_L2_Memory", 0, 1,
        AT_MEM_L3_DEFAULTRAM, L3Memory, "main_L3_Memory", 0, 0,
        AT_MEM_L3_DEFAULTFLASH, L3Flash, "main_L3_Flash", "main_L3_Flash_Const.dat", 0
    );

    LoadCNN_SQ8_Library();
    LoadCNNLibrary_fp16();
    LoadCNN_NE16_SQ8_Library();
    LoadResizeLibrary();
    LoadCNN_Copy_Library();


    // generator for Conv_0_fusion
    CNN_ConvolutionNE16("S4_Conv2d_8x12x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        12, 8, 240, 180,
                        KOP_CONV, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_4_fusion
    CNN_ConvolutionNE16("S7_Conv2d_8x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        8, 8, 240, 180,
                        KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_8_fusion
    CNN_ConvolutionNE16("S10_Conv2d_8x8x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        8, 8, 120, 90,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_12_fusion
    CNN_ConvolutionNE16("S13_Conv2d_8x8x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        8, 8, 120, 90,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_16_fusion
    CNN_ConvolutionNE16("S16_Conv2d_8x8x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        8, 8, 120, 90,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_20_fusion
    CNN_ConvolutionNE16("S19_Conv2d_8x8x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        8, 8, 120, 90,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_24_fusion
    CNN_ConvolutionNE16("S22_Conv2d_8x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        8, 8, 120, 90,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_28_fusion
    CNN_ConvolutionNE16("S25_Conv2d_8x8x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        8, 8, 120, 90,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S26_MatAdd_90x120x8;
    CNN_InitGenCtrl(&gen_ctrl_S26_MatAdd_90x120x8);
    CNN_SetGenCtrl(&gen_ctrl_S26_MatAdd_90x120x8, "OUTPUT_DATASIZE", AT_OPT_VAL(-1));
    CNN_SetGenCtrl(&gen_ctrl_S26_MatAdd_90x120x8, "INPUT_DATASIZE", AT_OPT_VAL(-1));
    
    // generator for Add_32
    CNN_MatAddAct_SQ8("S26_MatAdd_90x120x8", &gen_ctrl_S26_MatAdd_90x120x8, 90, 120, 8, KOP_MATADD, KOP_NONE);
    
    
    // generator for Concat_33
    CNN_ConcatLastAxis_Generator("S27_Concat", 0, -1, 10800, 8, 8, 0, 0, KOP_CONCAT);
    
    // generator for Conv_34_fusion
    CNN_ConvolutionNE16("S30_Conv2d_16x16x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        16, 16, 120, 90,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_38_fusion
    CNN_ConvolutionNE16("S33_Conv2d_16x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        16, 16, 120, 90,
                        KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_42_fusion
    CNN_ConvolutionNE16("S36_Conv2d_64x16x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        16, 64, 60, 45,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_46_fusion
    CNN_ConvolutionNE16("S39_Conv2d_32x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 32, 60, 45,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_50_fusion
    CNN_ConvolutionNE16("S42_Conv2d_32x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 32, 60, 45,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_54_fusion
    CNN_ConvolutionNE16("S45_Conv2d_32x32x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        32, 32, 60, 45,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_58_fusion
    CNN_ConvolutionNE16("S48_Conv2d_32x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        32, 32, 60, 45,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_62_fusion
    CNN_ConvolutionNE16("S51_Conv2d_32x32x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        32, 32, 60, 45,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S52_MatAdd_45x60x32;
    CNN_InitGenCtrl(&gen_ctrl_S52_MatAdd_45x60x32);
    CNN_SetGenCtrl(&gen_ctrl_S52_MatAdd_45x60x32, "OUTPUT_DATASIZE", AT_OPT_VAL(-1));
    CNN_SetGenCtrl(&gen_ctrl_S52_MatAdd_45x60x32, "INPUT_DATASIZE", AT_OPT_VAL(-1));
    
    // generator for Add_66
    CNN_MatAddAct_SQ8("S52_MatAdd_45x60x32", &gen_ctrl_S52_MatAdd_45x60x32, 45, 60, 32, KOP_MATADD, KOP_NONE);
    
    // generator for Conv_67_fusion
    CNN_ConvolutionNE16("S55_Conv2d_32x32x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        32, 32, 60, 45,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_71_fusion
    CNN_ConvolutionNE16("S58_Conv2d_32x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        32, 32, 60, 45,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_75_fusion
    CNN_ConvolutionNE16("S61_Conv2d_32x32x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        32, 32, 60, 45,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S62_MatAdd_45x60x32;
    CNN_InitGenCtrl(&gen_ctrl_S62_MatAdd_45x60x32);
    CNN_SetGenCtrl(&gen_ctrl_S62_MatAdd_45x60x32, "OUTPUT_DATASIZE", AT_OPT_VAL(-1));
    CNN_SetGenCtrl(&gen_ctrl_S62_MatAdd_45x60x32, "INPUT_DATASIZE", AT_OPT_VAL(-1));
    
    // generator for Add_79
    CNN_MatAddAct_SQ8("S62_MatAdd_45x60x32", &gen_ctrl_S62_MatAdd_45x60x32, 45, 60, 32, KOP_MATADD, KOP_NONE);
    
    // generator for Conv_80_fusion
    CNN_ConvolutionNE16("S65_Conv2d_32x32x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        32, 32, 60, 45,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_84_fusion
    CNN_ConvolutionNE16("S68_Conv2d_32x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        32, 32, 60, 45,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_88_fusion
    CNN_ConvolutionNE16("S71_Conv2d_32x32x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        32, 32, 60, 45,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S72_MatAdd_45x60x32;
    CNN_InitGenCtrl(&gen_ctrl_S72_MatAdd_45x60x32);
    CNN_SetGenCtrl(&gen_ctrl_S72_MatAdd_45x60x32, "OUTPUT_DATASIZE", AT_OPT_VAL(-1));
    CNN_SetGenCtrl(&gen_ctrl_S72_MatAdd_45x60x32, "INPUT_DATASIZE", AT_OPT_VAL(-1));
    
    // generator for Add_92
    CNN_MatAddAct_SQ8("S72_MatAdd_45x60x32", &gen_ctrl_S72_MatAdd_45x60x32, 45, 60, 32, KOP_MATADD, KOP_NONE);
    
    
    // generator for Concat_93
    CNN_ConcatLastAxis_Generator("S73_Concat", 0, -1, 2700, 32, 32, 0, 0, KOP_CONCAT);
    
    // generator for Conv_94_fusion
    CNN_ConvolutionNE16("S76_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 60, 45,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_98_fusion
    CNN_ConvolutionNE16("S79_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 60, 45,
                        KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_102_fusion
    CNN_ConvolutionNE16("S82_Conv2d_128x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 128, 30, 23,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_106_fusion
    CNN_ConvolutionNE16("S85_Conv2d_64x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 64, 30, 23,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_110_fusion
    CNN_ConvolutionNE16("S88_Conv2d_64x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 64, 30, 23,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_114_fusion
    CNN_ConvolutionNE16("S91_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 30, 23,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_118_fusion
    CNN_ConvolutionNE16("S94_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 30, 23,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_122_fusion
    CNN_ConvolutionNE16("S97_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 30, 23,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S98_MatAdd_23x30x64;
    CNN_InitGenCtrl(&gen_ctrl_S98_MatAdd_23x30x64);
    CNN_SetGenCtrl(&gen_ctrl_S98_MatAdd_23x30x64, "OUTPUT_DATASIZE", AT_OPT_VAL(-1));
    CNN_SetGenCtrl(&gen_ctrl_S98_MatAdd_23x30x64, "INPUT_DATASIZE", AT_OPT_VAL(-1));
    
    // generator for Add_126
    CNN_MatAddAct_SQ8("S98_MatAdd_23x30x64", &gen_ctrl_S98_MatAdd_23x30x64, 23, 30, 64, KOP_MATADD, KOP_NONE);
    
    // generator for Conv_127_fusion
    CNN_ConvolutionNE16("S101_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 30, 23,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_131_fusion
    CNN_ConvolutionNE16("S104_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 30, 23,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_135_fusion
    CNN_ConvolutionNE16("S107_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 30, 23,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S108_MatAdd_23x30x64;
    CNN_InitGenCtrl(&gen_ctrl_S108_MatAdd_23x30x64);
    CNN_SetGenCtrl(&gen_ctrl_S108_MatAdd_23x30x64, "OUTPUT_DATASIZE", AT_OPT_VAL(-1));
    CNN_SetGenCtrl(&gen_ctrl_S108_MatAdd_23x30x64, "INPUT_DATASIZE", AT_OPT_VAL(-1));
    
    // generator for Add_139
    CNN_MatAddAct_SQ8("S108_MatAdd_23x30x64", &gen_ctrl_S108_MatAdd_23x30x64, 23, 30, 64, KOP_MATADD, KOP_NONE);
    
    // generator for Conv_140_fusion
    CNN_ConvolutionNE16("S111_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 30, 23,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_144_fusion
    CNN_ConvolutionNE16("S114_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 30, 23,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_148_fusion
    CNN_ConvolutionNE16("S117_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 30, 23,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S118_MatAdd_23x30x64;
    CNN_InitGenCtrl(&gen_ctrl_S118_MatAdd_23x30x64);
    CNN_SetGenCtrl(&gen_ctrl_S118_MatAdd_23x30x64, "OUTPUT_DATASIZE", AT_OPT_VAL(-1));
    CNN_SetGenCtrl(&gen_ctrl_S118_MatAdd_23x30x64, "INPUT_DATASIZE", AT_OPT_VAL(-1));
    
    // generator for Add_152
    CNN_MatAddAct_SQ8("S118_MatAdd_23x30x64", &gen_ctrl_S118_MatAdd_23x30x64, 23, 30, 64, KOP_MATADD, KOP_NONE);
    
    
    // generator for Concat_153
    CNN_ConcatLastAxis_Generator("S119_Concat", 0, -1, 690, 64, 64, 0, 0, KOP_CONCAT);
    
    // generator for Conv_154_fusion
    CNN_ConvolutionNE16("S122_Conv2d_128x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 128, 30, 23,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_158_fusion
    CNN_ConvolutionNE16("S125_Conv2d_128x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 128, 30, 23,
                        KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_162_fusion
    CNN_ConvolutionNE16("S128_Conv2d_256x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 256, 15, 12,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_166_fusion
    CNN_ConvolutionNE16("S131_Conv2d_128x256x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        256, 128, 15, 12,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S132_MaxPool_5x5;
    CNN_InitGenCtrl(&gen_ctrl_S132_MaxPool_5x5);
    CNN_SetGenCtrl(&gen_ctrl_S132_MaxPool_5x5, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S132_MaxPool_5x5, "OUTPUT_DATASIZE", AT_OPT_VAL(-1));
    CNN_SetGenCtrl(&gen_ctrl_S132_MaxPool_5x5, "INPUT_DATASIZE", AT_OPT_VAL(-1));
    // generator for MaxPool_170
    CNN_PoolAct_SQ8("S132_MaxPool_5x5", &gen_ctrl_S132_MaxPool_5x5,
                    128, 15, 12,
                    KOP_MAXPOOL, 5, 5, 1, 1, 1, 1, 1,
                    KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S133_MaxPool_9x9;
    CNN_InitGenCtrl(&gen_ctrl_S133_MaxPool_9x9);
    CNN_SetGenCtrl(&gen_ctrl_S133_MaxPool_9x9, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S133_MaxPool_9x9, "OUTPUT_DATASIZE", AT_OPT_VAL(-1));
    CNN_SetGenCtrl(&gen_ctrl_S133_MaxPool_9x9, "INPUT_DATASIZE", AT_OPT_VAL(-1));
    // generator for MaxPool_171
    CNN_PoolAct_SQ8("S133_MaxPool_9x9", &gen_ctrl_S133_MaxPool_9x9,
                    128, 15, 12,
                    KOP_MAXPOOL, 9, 9, 1, 1, 1, 1, 1,
                    KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S134_MaxPool_13x13;
    CNN_InitGenCtrl(&gen_ctrl_S134_MaxPool_13x13);
    CNN_SetGenCtrl(&gen_ctrl_S134_MaxPool_13x13, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S134_MaxPool_13x13, "OUTPUT_DATASIZE", AT_OPT_VAL(-1));
    CNN_SetGenCtrl(&gen_ctrl_S134_MaxPool_13x13, "INPUT_DATASIZE", AT_OPT_VAL(-1));
    // generator for MaxPool_172
    CNN_PoolAct_SQ8("S134_MaxPool_13x13", &gen_ctrl_S134_MaxPool_13x13,
                    128, 15, 12,
                    KOP_MAXPOOL, 13, 13, 1, 1, 1, 1, 1,
                    KOP_NONE);
    
    
    // generator for Concat_173
    CNN_ConcatLastAxis_Generator("S135_Concat", 0, -1, 180, 128, 128, 128, 128, KOP_CONCAT);
    
    // generator for Conv_174_fusion
    CNN_ConvolutionNE16("S138_Conv2d_256x512x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        512, 256, 15, 12,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_178_fusion
    CNN_ConvolutionNE16("S141_Conv2d_128x256x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        256, 128, 15, 12,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_182_fusion
    CNN_ConvolutionNE16("S144_Conv2d_128x256x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        256, 128, 15, 12,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_186_fusion
    CNN_ConvolutionNE16("S147_Conv2d_128x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 128, 15, 12,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_190_fusion
    CNN_ConvolutionNE16("S150_Conv2d_128x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 128, 15, 12,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_194_fusion
    CNN_ConvolutionNE16("S153_Conv2d_128x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 128, 15, 12,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    
    // generator for Concat_198
    CNN_ConcatLastAxis_Generator("S154_Concat", 0, -1, 180, 128, 128, 0, 0, KOP_CONCAT);
    
    // generator for Conv_199_fusion
    CNN_ConvolutionNE16("S157_Conv2d_256x256x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        256, 256, 15, 12,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_203_fusion
    CNN_ConvolutionNE16("S160_Conv2d_128x256x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        256, 128, 15, 12,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    
    // generator for Resize_208_trans Transpose 12x15x128 -> 128x12x15 ((1, 0))
    CNN_MatTranspose("S161_Op_Resize_208_trans", 0, -1,
                      1, 128, 180, KOP_MATTRANSP);
    
    
    // generator for Resize_208
    GenerateResizeMultiChannel("S162_Op_Resize_208", 15, 12, 30, 24, 128, UNSIGNED_INOUT, KOP_NEAREST_NEIGHBOR_RESIZE);
    
    
    // generator for Slice_213_trans_in0 Transpose 128x24x30 -> 24x128x30 ((1, 0, 2))
    CNN_3DTensorPermute("S163_Op_Slice_213_trans_in0", 0, -1,
                         128, 30, 24, KOP_MATPERM_CHW2HCW);
    
    
    // generator for Slice_213_trans_out0 Transpose 23x128x30 -> 23x30x128 ((0, 2, 1))
    CNN_3DTensorPermute("S165_Op_Slice_213_trans_out0", 0, -1,
                         23, 30, 128, KOP_MATPERM_CHW2CWH);
    
    
    // generator for Concat_214
    CNN_ConcatLastAxis_Generator("S166_Concat", 0, -1, 690, 128, 128, 0, 0, KOP_CONCAT);
    
    // generator for Conv_215_fusion
    CNN_ConvolutionNE16("S170_Conv2d_64x256x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        256, 64, 30, 23,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_219_fusion
    CNN_ConvolutionNE16("S173_Conv2d_64x256x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        256, 64, 30, 23,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_223_fusion
    CNN_ConvolutionNE16("S176_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 30, 23,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_227_fusion
    CNN_ConvolutionNE16("S179_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 30, 23,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_231_fusion
    CNN_ConvolutionNE16("S182_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 30, 23,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    
    // generator for Concat_235
    CNN_ConcatLastAxis_Generator("S183_Concat", 0, -1, 690, 64, 64, 0, 0, KOP_CONCAT);
    
    // generator for Conv_236_fusion
    CNN_ConvolutionNE16("S186_Conv2d_128x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 128, 30, 23,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_240_fusion
    CNN_ConvolutionNE16("S189_Conv2d_64x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 64, 30, 23,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    
    // generator for Resize_245_trans Transpose 23x30x64 -> 64x23x30 ((1, 0))
    CNN_MatTranspose("S190_Op_Resize_245_trans", 0, -1,
                      1, 64, 690, KOP_MATTRANSP);
    
    
    // generator for Resize_245
    GenerateResizeMultiChannel("S191_Op_Resize_245", 30, 23, 60, 46, 64, UNSIGNED_INOUT, KOP_NEAREST_NEIGHBOR_RESIZE);
    
    
    // generator for Slice_250_trans_in0 Transpose 64x46x60 -> 46x64x60 ((1, 0, 2))
    CNN_3DTensorPermute("S192_Op_Slice_250_trans_in0", 0, -1,
                         64, 60, 46, KOP_MATPERM_CHW2HCW);
    
    
    // generator for Slice_250_trans_out0 Transpose 45x64x60 -> 45x60x64 ((0, 2, 1))
    CNN_3DTensorPermute("S194_Op_Slice_250_trans_out0", 0, -1,
                         45, 60, 64, KOP_MATPERM_CHW2CWH);
    
    
    // generator for Concat_251
    CNN_ConcatLastAxis_Generator("S195_Concat", 0, -1, 2700, 64, 64, 0, 0, KOP_CONCAT);
    
    // generator for Conv_252_fusion
    CNN_ConvolutionNE16("S199_Conv2d_32x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 32, 60, 45,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_256_fusion
    CNN_ConvolutionNE16("S202_Conv2d_32x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 32, 60, 45,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_260_fusion
    CNN_ConvolutionNE16("S205_Conv2d_32x32x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        32, 32, 60, 45,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_264_fusion
    CNN_ConvolutionNE16("S208_Conv2d_32x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        32, 32, 60, 45,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_268_fusion
    CNN_ConvolutionNE16("S211_Conv2d_32x32x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        32, 32, 60, 45,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    
    // generator for Concat_272
    CNN_ConcatLastAxis_Generator("S212_Concat", 0, -1, 2700, 32, 32, 0, 0, KOP_CONCAT);
    
    // generator for Conv_273_fusion
    CNN_ConvolutionNE16("S215_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 60, 45,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_277_fusion
    CNN_ConvolutionNE16("S218_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 60, 45,
                        KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_281_fusion
    CNN_ConvolutionNE16("S221_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 30, 23,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    
    // generator for Concat_285
    CNN_ConcatLastAxis_Generator("S222_Concat", 0, -1, 690, 64, 64, 0, 0, KOP_CONCAT);
    
    // generator for Conv_286_fusion
    CNN_ConvolutionNE16("S225_Conv2d_64x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 64, 30, 23,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_290_fusion
    CNN_ConvolutionNE16("S228_Conv2d_64x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 64, 30, 23,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_294_fusion
    CNN_ConvolutionNE16("S231_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 30, 23,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_298_fusion
    CNN_ConvolutionNE16("S234_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 30, 23,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_302_fusion
    CNN_ConvolutionNE16("S237_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 30, 23,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    
    // generator for Concat_306
    CNN_ConcatLastAxis_Generator("S238_Concat", 0, -1, 690, 64, 64, 0, 0, KOP_CONCAT);
    
    // generator for Conv_307_fusion
    CNN_ConvolutionNE16("S241_Conv2d_128x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 128, 30, 23,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_311_fusion
    CNN_ConvolutionNE16("S244_Conv2d_128x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 128, 30, 23,
                        KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_315_fusion
    CNN_ConvolutionNE16("S247_Conv2d_128x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 128, 15, 12,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    
    // generator for Concat_319
    CNN_ConcatLastAxis_Generator("S248_Concat", 0, -1, 180, 128, 128, 0, 0, KOP_CONCAT);
    
    // generator for Conv_320_fusion
    CNN_ConvolutionNE16("S251_Conv2d_128x256x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        256, 128, 15, 12,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_324_fusion
    CNN_ConvolutionNE16("S254_Conv2d_128x256x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        256, 128, 15, 12,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_328_fusion
    CNN_ConvolutionNE16("S257_Conv2d_128x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 128, 15, 12,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_332_fusion
    CNN_ConvolutionNE16("S260_Conv2d_128x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 128, 15, 12,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_336_fusion
    CNN_ConvolutionNE16("S263_Conv2d_128x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 128, 15, 12,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    
    // generator for Concat_340
    CNN_ConcatLastAxis_Generator("S264_Concat", 0, -1, 180, 128, 128, 0, 0, KOP_CONCAT);
    
    // generator for Conv_341_fusion
    CNN_ConvolutionNE16("S267_Conv2d_256x256x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        256, 256, 15, 12,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_345_fusion
    CNN_ConvolutionNE16("S270_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 60, 45,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_349_fusion
    CNN_ConvolutionNE16("S273_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 60, 45,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_353_fusion
    CNN_ConvolutionNE16("S276_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 60, 45,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_357_fusion
    CNN_ConvolutionNE16("S279_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 60, 45,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_361_fusion
    CNN_ConvolutionNE16("S282_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 60, 45,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_365_fusion
    CNN_ConvolutionNE16("S285_Conv2d_1x64x1x1_Sigmoid", 0,
                        -1, -1, 4, 1, 8,
                        64, 1, 60, 45,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_SIGMOID);
    
    
    // generator for Concat_386_qin2
    CNN_Convert("S286_Op_Concat_386_qin2", -1, -1, 2700, KOP_CONVERT_FP_FP_SCALE);
    
    // generator for Conv_366_fusion
    CNN_ConvolutionNE16("S289_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 60, 45,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_370_fusion
    CNN_ConvolutionNE16("S292_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 60, 45,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_374_fusion
    CNN_ConvolutionNE16("S295_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 60, 45,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_378_fusion
    CNN_ConvolutionNE16("S298_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 60, 45,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_382
    CNN_ConvolutionNE16("S301_Conv2d_4x64x1x1", 0,
                        -1, -1, 4, 1, 8,
                        64, 4, 60, 45,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    // generator for Conv_383_fusion
    CNN_ConvolutionNE16("S304_Conv2d_1x64x1x1_Sigmoid", 0,
                        -1, -1, 4, 1, 8,
                        64, 1, 60, 45,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_SIGMOID);
    
    
    // generator for Concat_386_qin1
    CNN_Convert("S305_Op_Concat_386_qin1", -1, -1, 2700, KOP_CONVERT_FP_FP_SCALE);
    
    
    // generator for Concat_386
    CNN_ConcatLastAxis_Generator("S306_Concat", 0, -1, 2700, 4, 1, 1, 0, KOP_CONCAT);
    
    
    // generator for Concat_495_qin0
    CNN_Convert("S308_Op_Concat_495_qin0", -1, -1, 16200, KOP_CONVERT_FP_FP_SCALE);
    
    // generator for Conv_387_fusion
    CNN_ConvolutionNE16("S311_Conv2d_64x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 64, 30, 23,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_391_fusion
    CNN_ConvolutionNE16("S314_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 30, 23,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_395_fusion
    CNN_ConvolutionNE16("S317_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 30, 23,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_399_fusion
    CNN_ConvolutionNE16("S320_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 30, 23,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_403_fusion
    CNN_ConvolutionNE16("S323_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 30, 23,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_407_fusion
    CNN_ConvolutionNE16("S326_Conv2d_1x64x1x1_Sigmoid", 0,
                        -1, -1, 4, 1, 8,
                        64, 1, 30, 23,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_SIGMOID);
    
    
    // generator for Concat_428_qin2
    CNN_Convert("S327_Op_Concat_428_qin2", -1, -1, 690, KOP_CONVERT_FP_FP_SCALE);
    
    // generator for Conv_408_fusion
    CNN_ConvolutionNE16("S330_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 30, 23,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_412_fusion
    CNN_ConvolutionNE16("S333_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 30, 23,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_416_fusion
    CNN_ConvolutionNE16("S336_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 30, 23,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_420_fusion
    CNN_ConvolutionNE16("S339_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 30, 23,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_424
    CNN_ConvolutionNE16("S342_Conv2d_4x64x1x1", 0,
                        -1, -1, 4, 1, 8,
                        64, 4, 30, 23,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    // generator for Conv_425_fusion
    CNN_ConvolutionNE16("S345_Conv2d_1x64x1x1_Sigmoid", 0,
                        -1, -1, 4, 1, 8,
                        64, 1, 30, 23,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_SIGMOID);
    
    
    // generator for Concat_428_qin1
    CNN_Convert("S346_Op_Concat_428_qin1", -1, -1, 690, KOP_CONVERT_FP_FP_SCALE);
    
    
    // generator for Concat_428
    CNN_ConcatLastAxis_Generator("S347_Concat", 0, -1, 690, 4, 1, 1, 0, KOP_CONCAT);
    
    
    // generator for Concat_495_qin1
    CNN_Convert("S349_Op_Concat_495_qin1", -1, -1, 4140, KOP_CONVERT_FP_FP_SCALE);
    
    // generator for Conv_429_fusion
    CNN_ConvolutionNE16("S352_Conv2d_64x256x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        256, 64, 15, 12,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_433_fusion
    CNN_ConvolutionNE16("S355_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 15, 12,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_437_fusion
    CNN_ConvolutionNE16("S358_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 15, 12,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_441_fusion
    CNN_ConvolutionNE16("S361_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 15, 12,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_445_fusion
    CNN_ConvolutionNE16("S364_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 15, 12,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_449_fusion
    CNN_ConvolutionNE16("S367_Conv2d_1x64x1x1_Sigmoid", 0,
                        -1, -1, 4, 1, 8,
                        64, 1, 15, 12,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_SIGMOID);
    
    
    // generator for Concat_470_qin2
    CNN_Convert("S368_Op_Concat_470_qin2", -1, -1, 180, KOP_CONVERT_FP_FP_SCALE);
    
    // generator for Conv_450_fusion
    CNN_ConvolutionNE16("S371_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 15, 12,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_454_fusion
    CNN_ConvolutionNE16("S374_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 15, 12,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_458_fusion
    CNN_ConvolutionNE16("S377_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 15, 12,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_462_fusion
    CNN_ConvolutionNE16("S380_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 15, 12,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_466
    CNN_ConvolutionNE16("S383_Conv2d_4x64x1x1", 0,
                        -1, -1, 4, 1, 8,
                        64, 4, 15, 12,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    // generator for Conv_467_fusion
    CNN_ConvolutionNE16("S386_Conv2d_1x64x1x1_Sigmoid", 0,
                        -1, -1, 4, 1, 8,
                        64, 1, 15, 12,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_SIGMOID);
    
    
    // generator for Concat_470_qin1
    CNN_Convert("S387_Op_Concat_470_qin1", -1, -1, 180, KOP_CONVERT_FP_FP_SCALE);
    
    
    // generator for Concat_470
    CNN_ConcatLastAxis_Generator("S388_Concat", 0, -1, 180, 4, 1, 1, 0, KOP_CONCAT);
    
    
    // generator for output_1_qin0
    CNN_Convert("S391_Op_output_1_qin0", -1, 4, 21420, KOP_CONVERT_FP_FL);
    

#define GRAPH
#ifdef GRAPH
    CreateGraph("mainCNN",
        /* Arguments either passed or globals */
            CArgs(586,
                TCArgInfo("unsigned char * __restrict__", "Input_1", ARG_SCOPE_ARG_ALLOC, ARG_DIR_IN, AT_MEM_L2, AT_MEM_L2, 0),
                TCArgInfo("unsigned char * __restrict__", "Conv_0_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_0_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1252", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1252.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S4_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S4_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S4_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S4_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S4_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S4_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_4_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_4_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1255", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1255.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S7_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S7_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S7_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S7_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S7_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S7_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_8_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_8_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1258", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1258.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S10_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S10_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S10_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S10_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S10_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S10_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_12_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_12_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1261", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1261.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S13_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S13_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S13_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S13_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S13_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S13_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_16_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_16_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1264", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1264.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S16_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S16_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S16_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S16_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03232 out: 0.03232  actscale: [1] actscalen: [0] a0: [0] b0: [186] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S16_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S16_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_20_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_20_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1267", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1267.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S19_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S19_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S19_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S19_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S19_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S19_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_24_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_24_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1270", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1270.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S22_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S22_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S22_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S22_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S22_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S22_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_28_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_28_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1273", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1273.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S25_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S25_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S25_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S25_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S25_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S25_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [1] IN1SCALEN: [0] OUTSCALE: [93] OUTSCALEN: [7] ADD_BIAS: 0
                TCArgInfo("signed char * __restrict__", "S26_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S26_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_34_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_34_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1276", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1276.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S30_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S30_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S30_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S30_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S30_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S30_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_38_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_38_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1279", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1279.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S33_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S33_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S33_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S33_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S33_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S33_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_42_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_42_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1282", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1282.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S36_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S36_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S36_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S36_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S36_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S36_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_46_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_46_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1285", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1285.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S39_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S39_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S39_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S39_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S39_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S39_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_50_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_50_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1288", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1288.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S42_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S42_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S42_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S42_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.06674 out: 0.06674  actscale: [1] actscalen: [0] a0: [0] b0: [90] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S42_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S42_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_54_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_54_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1291", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1291.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S45_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S45_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S45_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S45_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S45_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S45_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_58_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_58_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1294", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1294.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S48_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S48_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S48_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S48_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S48_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S48_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_62_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_62_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1297", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1297.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S51_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S51_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S51_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S51_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S51_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S51_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [1] IN1SCALEN: [0] OUTSCALE: [39] OUTSCALEN: [6] ADD_BIAS: 0
                TCArgInfo("signed char * __restrict__", "S52_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S52_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_67_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_67_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1300", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1300.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S55_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S55_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S55_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S55_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S55_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S55_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_71_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_71_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1303", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1303.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S58_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S58_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S58_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S58_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S58_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S58_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_75_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_75_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1306", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1306.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S61_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S61_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S61_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S61_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S61_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S61_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [209] IN1SCALEN: [7] OUTSCALE: [241] OUTSCALEN: [9] ADD_BIAS: 0
                TCArgInfo("signed char * __restrict__", "S62_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S62_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_80_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_80_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1309", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1309.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S65_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S65_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S65_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S65_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S65_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S65_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_84_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_84_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1312", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1312.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S68_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S68_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S68_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S68_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S68_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S68_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_88_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_88_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1315", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1315.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S71_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S71_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S71_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S71_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S71_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S71_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [17] IN1SCALEN: [3] OUTSCALE: [181] OUTSCALEN: [9] ADD_BIAS: 0
                TCArgInfo("signed char * __restrict__", "S72_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S72_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_94_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_94_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1318", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1318.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S76_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S76_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S76_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S76_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S76_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S76_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_98_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_98_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1321", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1321.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S79_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S79_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S79_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S79_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S79_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S79_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_102_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_102_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1324", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1324.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S82_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S82_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S82_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S82_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S82_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S82_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_106_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_106_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1327", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1327.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S85_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S85_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S85_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S85_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S85_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S85_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_110_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_110_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1330", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1330.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S88_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S88_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S88_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S88_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.06991 out: 0.06991  actscale: [1] actscalen: [0] a0: [0] b0: [86] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S88_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S88_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_114_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_114_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1333", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1333.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S91_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S91_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S91_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S91_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S91_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S91_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_118_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_118_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1336", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1336.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S94_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S94_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S94_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S94_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S94_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S94_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_122_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_122_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1339", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1339.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S97_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S97_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S97_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S97_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S97_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S97_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [1] IN1SCALEN: [0] OUTSCALE: [183] OUTSCALEN: [8] ADD_BIAS: 0
                TCArgInfo("signed char * __restrict__", "S98_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S98_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_127_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_127_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1342", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1342.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S101_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S101_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S101_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S101_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S101_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S101_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_131_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_131_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1345", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1345.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S104_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S104_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S104_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S104_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S104_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S104_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_135_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_135_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1348", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1348.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S107_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S107_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S107_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S107_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S107_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S107_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [179] IN1SCALEN: [7] OUTSCALE: [119] OUTSCALEN: [8] ADD_BIAS: 0
                TCArgInfo("signed char * __restrict__", "S108_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S108_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_140_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_140_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1351", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1351.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S111_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S111_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S111_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S111_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S111_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S111_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_144_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_144_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1354", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1354.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S114_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S114_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S114_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S114_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S114_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S114_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_148_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_148_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1357", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1357.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S117_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S117_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S117_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S117_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S117_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S117_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [69] IN1SCALEN: [5] OUTSCALE: [43] OUTSCALEN: [7] ADD_BIAS: 0
                TCArgInfo("signed char * __restrict__", "S118_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S118_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_154_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_154_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1360", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1360.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S122_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S122_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S122_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S122_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S122_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S122_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_158_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_158_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1363", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1363.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S125_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S125_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S125_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S125_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S125_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S125_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_162_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_162_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1366", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1366.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S128_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S128_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S128_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S128_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S128_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S128_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_166_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_166_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1369", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1369.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S131_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S131_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S131_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S131_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S131_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S131_Infos.tensor", 1, 1, 8, 0)),
                // no activation ACTSCALE: [1] ACTSCALEN: [0]
                TCArgInfo("signed char * __restrict__", "S132_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S132_Infos.tensor", 1, 1, 8, 0)),
                // no activation ACTSCALE: [1] ACTSCALEN: [0]
                TCArgInfo("signed char * __restrict__", "S133_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S133_Infos.tensor", 1, 1, 8, 0)),
                // no activation ACTSCALE: [1] ACTSCALEN: [0]
                TCArgInfo("signed char * __restrict__", "S134_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S134_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_174_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_174_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1372", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1372.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S138_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S138_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S138_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S138_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S138_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S138_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_178_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_178_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1375", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1375.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S141_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S141_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S141_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S141_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S141_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S141_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_182_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_182_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1378", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1378.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S144_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S144_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S144_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S144_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S144_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S144_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_186_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_186_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1381", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1381.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S147_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S147_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S147_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S147_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S147_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S147_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_190_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_190_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1384", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1384.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S150_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S150_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S150_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S150_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S150_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S150_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_194_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_194_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1387", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1387.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S153_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S153_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S153_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S153_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S153_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S153_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_199_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_199_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1390", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1390.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S157_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S157_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S157_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S157_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S157_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S157_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_203_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_203_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1393", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1393.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S160_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S160_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S160_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S160_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S160_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S160_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_215_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_215_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1396", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1396.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S170_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S170_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S170_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S170_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S170_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S170_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_219_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_219_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1399", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1399.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S173_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S173_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S173_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S173_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S173_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S173_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_223_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_223_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1402", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1402.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S176_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S176_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S176_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S176_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S176_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S176_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_227_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_227_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1405", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1405.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S179_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S179_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S179_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S179_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S179_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S179_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_231_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_231_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1408", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1408.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S182_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S182_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S182_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S182_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S182_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S182_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_236_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_236_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1411", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1411.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S186_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S186_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S186_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S186_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S186_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S186_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_240_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_240_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1414", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1414.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S189_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S189_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S189_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S189_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S189_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S189_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_252_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_252_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1417", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1417.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S199_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S199_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S199_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S199_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S199_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S199_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_256_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_256_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1420", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1420.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S202_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S202_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S202_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S202_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S202_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S202_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_260_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_260_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1423", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1423.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S205_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S205_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S205_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S205_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S205_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S205_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_264_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_264_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1426", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1426.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S208_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S208_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S208_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S208_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S208_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S208_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_268_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_268_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1429", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1429.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S211_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S211_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S211_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S211_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S211_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S211_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_273_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_273_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1432", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1432.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S215_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S215_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S215_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S215_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S215_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S215_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_277_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_277_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1435", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1435.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S218_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S218_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S218_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S218_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S218_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S218_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_281_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_281_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1438", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1438.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S221_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S221_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S221_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S221_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S221_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S221_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_286_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_286_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1441", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1441.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S225_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S225_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S225_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S225_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S225_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S225_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_290_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_290_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1444", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1444.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S228_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S228_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S228_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S228_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S228_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S228_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_294_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_294_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1447", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1447.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S231_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S231_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S231_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S231_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S231_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S231_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_298_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_298_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1450", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1450.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S234_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S234_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S234_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S234_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S234_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S234_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_302_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_302_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1453", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1453.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S237_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S237_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S237_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S237_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S237_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S237_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_307_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_307_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1456", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1456.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S241_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S241_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S241_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S241_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S241_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S241_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_311_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_311_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1459", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1459.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S244_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S244_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S244_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S244_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S244_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S244_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_315_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_315_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1462", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1462.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S247_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S247_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S247_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S247_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S247_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S247_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_320_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_320_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1465", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1465.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S251_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S251_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S251_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S251_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S251_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S251_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_324_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_324_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1468", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1468.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S254_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S254_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S254_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S254_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S254_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S254_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_328_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_328_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1471", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1471.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S257_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S257_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S257_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S257_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S257_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S257_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_332_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_332_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1474", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1474.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S260_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S260_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S260_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S260_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S260_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S260_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_336_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_336_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1477", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1477.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S263_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S263_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S263_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S263_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S263_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S263_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_341_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_341_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1480", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1480.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S267_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S267_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S267_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S267_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S267_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S267_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_345_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_345_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1483", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1483.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S270_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S270_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S270_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S270_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S270_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S270_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_349_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_349_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1486", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1486.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S273_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S273_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S273_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S273_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S273_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S273_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_353_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_353_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1489", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1489.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S276_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S276_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S276_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S276_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S276_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S276_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_357_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_357_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1492", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1492.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S279_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S279_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S279_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S279_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S279_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S279_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_361_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_361_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1495", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1495.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S282_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S282_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S282_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S282_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S282_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S282_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_365_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_365_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_head_cls_preds_0_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_head_cls_preds_0_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S285_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S285_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S285_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S285_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00024 out: 0.00392  actscale: [255] actscalen: [15] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S285_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S285_Infos.tensor", 1, 1, 8, 0)),
                // in q: 0.00<(u8-0.00)*0.00392157<1.00 out_q: -3.70<(u8-104.00)*0.03556376<5.37
                TCArgInfo("signed char * __restrict__", "S286_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S286_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_366_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_366_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1498", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1498.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S289_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S289_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S289_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S289_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S289_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S289_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_370_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_370_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1501", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1501.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S292_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S292_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S292_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S292_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S292_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S292_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_374_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_374_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1504", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1504.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S295_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S295_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S295_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S295_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S295_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S295_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_378_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_378_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1507", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1507.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S298_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S298_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S298_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S298_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S298_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S298_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_382_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_382_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_head_reg_preds_0_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_head_reg_preds_0_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S301_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S301_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S301_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S301_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S301_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S301_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_383_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_383_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_head_obj_preds_0_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_head_obj_preds_0_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S304_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S304_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S304_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S304_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00024 out: 0.00392  actscale: [255] actscalen: [15] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S304_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S304_Infos.tensor", 1, 1, 8, 0)),
                // in q: 0.00<(u8-0.00)*0.00392157<1.00 out_q: -3.70<(u8-104.00)*0.03556376<5.37
                TCArgInfo("signed char * __restrict__", "S305_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S305_Infos.tensor", 1, 1, 8, 0)),
                // in q: -3.70<(u8-104.00)*0.03556376<5.37 out_q: -4.69<(u8-102.00)*0.04595577<7.03
                TCArgInfo("signed char * __restrict__", "S308_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S308_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_387_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_387_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1510", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1510.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S311_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S311_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S311_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S311_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S311_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S311_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_391_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_391_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1513", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1513.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S314_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S314_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S314_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S314_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S314_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S314_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_395_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_395_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1516", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1516.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S317_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S317_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S317_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S317_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S317_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S317_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_399_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_399_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1519", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1519.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S320_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S320_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S320_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S320_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S320_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S320_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_403_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_403_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1522", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1522.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S323_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S323_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S323_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S323_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S323_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S323_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_407_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_407_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_head_cls_preds_1_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_head_cls_preds_1_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S326_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S326_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S326_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S326_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00024 out: 0.00392  actscale: [255] actscalen: [15] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S326_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S326_Infos.tensor", 1, 1, 8, 0)),
                // in q: 0.00<(u8-0.00)*0.00392157<1.00 out_q: -1.65<(u8-110.00)*0.01499095<2.17
                TCArgInfo("signed char * __restrict__", "S327_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S327_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_408_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_408_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1525", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1525.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S330_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S330_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S330_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S330_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S330_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S330_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_412_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_412_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1528", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1528.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S333_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S333_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S333_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S333_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S333_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S333_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_416_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_416_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1531", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1531.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S336_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S336_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S336_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S336_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S336_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S336_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_420_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_420_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1534", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1534.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S339_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S339_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S339_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S339_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S339_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S339_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_424_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_424_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_head_reg_preds_1_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_head_reg_preds_1_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S342_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S342_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S342_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S342_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S342_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S342_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_425_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_425_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_head_obj_preds_1_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_head_obj_preds_1_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S345_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S345_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S345_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S345_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00024 out: 0.00392  actscale: [255] actscalen: [15] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S345_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S345_Infos.tensor", 1, 1, 8, 0)),
                // in q: 0.00<(u8-0.00)*0.00392157<1.00 out_q: -1.65<(u8-110.00)*0.01499095<2.17
                TCArgInfo("signed char * __restrict__", "S346_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S346_Infos.tensor", 1, 1, 8, 0)),
                // in q: -1.65<(u8-110.00)*0.01499095<2.17 out_q: -4.69<(u8-102.00)*0.04595577<7.03
                TCArgInfo("signed char * __restrict__", "S349_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S349_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_429_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_429_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1537", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1537.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S352_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S352_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S352_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S352_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S352_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S352_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_433_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_433_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1540", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1540.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S355_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S355_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S355_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S355_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S355_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S355_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_437_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_437_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1543", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1543.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S358_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S358_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S358_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S358_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S358_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S358_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_441_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_441_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1546", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1546.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S361_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S361_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S361_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S361_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S361_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S361_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_445_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_445_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1549", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1549.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S364_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S364_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S364_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S364_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S364_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S364_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_449_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_449_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_head_cls_preds_2_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_head_cls_preds_2_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S367_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S367_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S367_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S367_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00024 out: 0.00392  actscale: [255] actscalen: [15] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S367_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S367_Infos.tensor", 1, 1, 8, 0)),
                // in q: 0.00<(u8-0.00)*0.00392157<1.00 out_q: -4.69<(u8-102.00)*0.04595577<7.03
                TCArgInfo("signed char * __restrict__", "S368_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S368_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_450_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_450_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1552", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1552.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S371_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S371_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S371_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S371_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S371_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S371_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_454_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_454_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1555", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1555.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S374_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S374_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S374_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S374_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S374_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S374_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_458_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_458_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1558", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1558.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S377_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S377_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S377_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S377_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S377_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S377_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_462_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_462_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1561", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1561.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S380_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S380_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S380_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S380_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S380_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S380_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_466_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_466_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_head_reg_preds_2_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_head_reg_preds_2_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S383_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S383_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S383_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S383_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S383_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S383_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_467_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_467_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_head_obj_preds_2_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_head_obj_preds_2_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S386_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S386_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S386_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S386_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00024 out: 0.00392  actscale: [255] actscalen: [15] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S386_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S386_Infos.tensor", 1, 1, 8, 0)),
                // in q: 0.00<(u8-0.00)*0.00392157<1.00 out_q: -4.69<(u8-102.00)*0.04595577<7.03
                TCArgInfo("signed char * __restrict__", "S387_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S387_Infos.tensor", 1, 1, 8, 0)),
                // in q: -4.69<(u8-102.00)*0.04595577<7.03 out_q: f32
                TCArgInfo("signed char * __restrict__", "S391_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S391_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("float * __restrict__", "Output_1", ARG_SCOPE_ARG, ARG_DIR_OUT, AT_MEM_L2, AT_MEM_L2, 0)
            ),
        /* Locals, allocated dynamically */
        CArgs(153,
            TCArgInfo("unsigned char * __restrict__", "S4_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S7_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S10_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S13_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S16_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S19_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S22_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S25_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S26_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S27_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S30_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S33_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S36_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S39_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S42_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S45_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S48_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S51_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S52_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S55_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S58_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S61_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S62_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S65_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S68_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S71_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S72_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S73_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S76_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S79_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S82_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S85_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S88_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S91_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S94_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S97_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S98_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S101_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S104_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S107_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S108_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S111_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S114_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S117_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S118_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S119_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S122_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S125_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S128_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S131_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S132_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S133_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S134_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S135_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S138_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S141_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S144_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S147_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S150_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S153_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S154_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S157_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S160_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S161_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S162_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S163_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S165_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S166_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S170_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S173_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S176_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S179_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S182_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S183_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S186_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S189_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S190_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S191_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S192_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S194_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S195_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S199_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S202_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S205_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S208_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S211_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S212_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S215_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S218_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S221_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S222_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S225_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S228_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S231_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S234_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S237_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S238_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S241_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S244_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S247_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S248_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S251_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S254_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S257_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S260_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S263_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S264_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S267_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S270_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S273_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S276_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S279_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S282_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S285_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S286_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S289_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S292_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S295_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S298_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S301_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S304_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S305_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S306_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S311_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S314_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S317_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S320_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S323_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S326_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S327_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S330_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S333_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S336_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S339_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S342_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S345_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S346_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S347_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S352_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S355_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S358_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S361_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S364_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S367_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S368_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S371_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S374_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S377_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S380_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S383_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S386_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S387_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S390_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0)
        )
    );

    // Stacked tensors for concats and splits
    AddStackedTensors("S390_Output", 3, "S308_Output", "S349_Output", "S388_Output");
    AddStackedTensors("S163_Output", 2, "S164_Output_0", AT_UnusedStackMember("Slice_213_unused", 3840));
    AddStackedTensors("S192_Output", 2, "S193_Output_0", AT_UnusedStackMember("Slice_250_unused", 3840));

    // Node S4_Conv2d_8x12x3x3_Relu6 inq 0.00<(u8-0.00)*0.97254902<248.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S4_Conv2d_8x12x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "Input_1", 0),
            GNodeArg(GNA_IN, "Conv_0_weights", 0),
            GNodeArg(GNA_IN, "Constant__1252", 0),
            GNodeArg(GNA_OUT, "S4_Output", 0),
            GNodeArg(GNA_IN, "S4_Mul_scale", 0),
            GNodeArg(GNA_IN, "S4_Mul_shift", 0),
            GNodeArg(GNA_IN, "S4_Infos", 0)
        )
    );
    // Node S7_Conv2d_8x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S7_Conv2d_8x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S4_Output", 0),
            GNodeArg(GNA_IN, "Conv_4_weights", 0),
            GNodeArg(GNA_IN, "Constant__1255", 0),
            GNodeArg(GNA_OUT, "S7_Output", 0),
            GNodeArg(GNA_IN, "S7_Mul_scale", 0),
            GNodeArg(GNA_IN, "S7_Mul_shift", 0),
            GNodeArg(GNA_IN, "S7_Infos", 0)
        )
    );
    // Node S10_Conv2d_8x8x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S10_Conv2d_8x8x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S7_Output", 0),
            GNodeArg(GNA_IN, "Conv_8_weights", 0),
            GNodeArg(GNA_IN, "Constant__1258", 0),
            GNodeArg(GNA_OUT, "S10_Output", 0),
            GNodeArg(GNA_IN, "S10_Mul_scale", 0),
            GNodeArg(GNA_IN, "S10_Mul_shift", 0),
            GNodeArg(GNA_IN, "S10_Infos", 0)
        )
    );
    // Node S13_Conv2d_8x8x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S13_Conv2d_8x8x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S10_Output", 0),
            GNodeArg(GNA_IN, "Conv_12_weights", 0),
            GNodeArg(GNA_IN, "Constant__1261", 0),
            GNodeArg(GNA_OUT, "S13_Output", 0),
            GNodeArg(GNA_IN, "S13_Mul_scale", 0),
            GNodeArg(GNA_IN, "S13_Mul_shift", 0),
            GNodeArg(GNA_IN, "S13_Infos", 0)
        )
    );
    // Node S16_Conv2d_8x8x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.03231570<8.24 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S16_Conv2d_8x8x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S10_Output", 0),
            GNodeArg(GNA_IN, "Conv_16_weights", 0),
            GNodeArg(GNA_IN, "Constant__1264", 0),
            GNodeArg(GNA_OUT, "S16_Output", 0),
            GNodeArg(GNA_IN, "S16_Mul_scale", 0),
            GNodeArg(GNA_IN, "S16_Mul_shift", 0),
            GNodeArg(GNA_IN, "S16_Infos", 0)
        )
    );
    // Node S19_Conv2d_8x8x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S19_Conv2d_8x8x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S13_Output", 0),
            GNodeArg(GNA_IN, "Conv_20_weights", 0),
            GNodeArg(GNA_IN, "Constant__1267", 0),
            GNodeArg(GNA_OUT, "S19_Output", 0),
            GNodeArg(GNA_IN, "S19_Mul_scale", 0),
            GNodeArg(GNA_IN, "S19_Mul_shift", 0),
            GNodeArg(GNA_IN, "S19_Infos", 0)
        )
    );
    // Node S22_Conv2d_8x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S22_Conv2d_8x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S19_Output", 0),
            GNodeArg(GNA_IN, "Conv_24_weights", 0),
            GNodeArg(GNA_IN, "Constant__1270", 0),
            GNodeArg(GNA_OUT, "S22_Output", 0),
            GNodeArg(GNA_IN, "S22_Mul_scale", 0),
            GNodeArg(GNA_IN, "S22_Mul_shift", 0),
            GNodeArg(GNA_IN, "S22_Infos", 0)
        )
    );
    // Node S25_Conv2d_8x8x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S25_Conv2d_8x8x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S22_Output", 0),
            GNodeArg(GNA_IN, "Conv_28_weights", 0),
            GNodeArg(GNA_IN, "Constant__1273", 0),
            GNodeArg(GNA_OUT, "S25_Output", 0),
            GNodeArg(GNA_IN, "S25_Mul_scale", 0),
            GNodeArg(GNA_IN, "S25_Mul_shift", 0),
            GNodeArg(GNA_IN, "S25_Infos", 0)
        )
    );
    // Node S26_MatAdd_90x120x8 in1q 0.00<(u8-0.00)*0.02352941<6.00 forced
    //   in2q 0.00<(u8-0.00)*0.02352941<6.00 forced
    //   outq 0.00<(u8-0.00)*0.03231570<8.24 forced scaled input 0 is node input 0
    AddNode("S26_MatAdd_90x120x8",
        Bindings(4,
            GNodeArg(GNA_IN, "S25_Output", 0),
            GNodeArg(GNA_IN, "S13_Output", 0),
            GNodeArg(GNA_OUT, "S26_Output", 0),
            GNodeArg(GNA_IN, "S26_Infos", 0)
        )
    );
    // Node Concat_33 inq ['0.00<(u8-0.00)*0.03231570<8.24 forced', '0.00<(u8-0.00)*0.03231570<8.24 forced'] outq ['0.00<(u8-0.00)*0.03231570<8.24 forced']
    AddNode("S27_Concat",
        Bindings(3,
            GNodeArg(GNA_IN, "S26_Output", 0),
            GNodeArg(GNA_IN, "S16_Output", 0),
            GNodeArg(GNA_OUT, "S27_Output", 0)
        )
    );
    // Node S30_Conv2d_16x16x1x1_Relu6 inq 0.00<(u8-0.00)*0.03231570<8.24 forced weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S30_Conv2d_16x16x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S27_Output", 0),
            GNodeArg(GNA_IN, "Conv_34_weights", 0),
            GNodeArg(GNA_IN, "Constant__1276", 0),
            GNodeArg(GNA_OUT, "S30_Output", 0),
            GNodeArg(GNA_IN, "S30_Mul_scale", 0),
            GNodeArg(GNA_IN, "S30_Mul_shift", 0),
            GNodeArg(GNA_IN, "S30_Infos", 0)
        )
    );
    // Node S33_Conv2d_16x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S33_Conv2d_16x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S30_Output", 0),
            GNodeArg(GNA_IN, "Conv_38_weights", 0),
            GNodeArg(GNA_IN, "Constant__1279", 0),
            GNodeArg(GNA_OUT, "S33_Output", 0),
            GNodeArg(GNA_IN, "S33_Mul_scale", 0),
            GNodeArg(GNA_IN, "S33_Mul_shift", 0),
            GNodeArg(GNA_IN, "S33_Infos", 0)
        )
    );
    // Node S36_Conv2d_64x16x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S36_Conv2d_64x16x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S33_Output", 0),
            GNodeArg(GNA_IN, "Conv_42_weights", 0),
            GNodeArg(GNA_IN, "Constant__1282", 0),
            GNodeArg(GNA_OUT, "S36_Output", 0),
            GNodeArg(GNA_IN, "S36_Mul_scale", 0),
            GNodeArg(GNA_IN, "S36_Mul_shift", 0),
            GNodeArg(GNA_IN, "S36_Infos", 0)
        )
    );
    // Node S39_Conv2d_32x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S39_Conv2d_32x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S36_Output", 0),
            GNodeArg(GNA_IN, "Conv_46_weights", 0),
            GNodeArg(GNA_IN, "Constant__1285", 0),
            GNodeArg(GNA_OUT, "S39_Output", 0),
            GNodeArg(GNA_IN, "S39_Mul_scale", 0),
            GNodeArg(GNA_IN, "S39_Mul_shift", 0),
            GNodeArg(GNA_IN, "S39_Infos", 0)
        )
    );
    // Node S42_Conv2d_32x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.06673826<17.02 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S42_Conv2d_32x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S36_Output", 0),
            GNodeArg(GNA_IN, "Conv_50_weights", 0),
            GNodeArg(GNA_IN, "Constant__1288", 0),
            GNodeArg(GNA_OUT, "S42_Output", 0),
            GNodeArg(GNA_IN, "S42_Mul_scale", 0),
            GNodeArg(GNA_IN, "S42_Mul_shift", 0),
            GNodeArg(GNA_IN, "S42_Infos", 0)
        )
    );
    // Node S45_Conv2d_32x32x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S45_Conv2d_32x32x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S39_Output", 0),
            GNodeArg(GNA_IN, "Conv_54_weights", 0),
            GNodeArg(GNA_IN, "Constant__1291", 0),
            GNodeArg(GNA_OUT, "S45_Output", 0),
            GNodeArg(GNA_IN, "S45_Mul_scale", 0),
            GNodeArg(GNA_IN, "S45_Mul_shift", 0),
            GNodeArg(GNA_IN, "S45_Infos", 0)
        )
    );
    // Node S48_Conv2d_32x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S48_Conv2d_32x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S45_Output", 0),
            GNodeArg(GNA_IN, "Conv_58_weights", 0),
            GNodeArg(GNA_IN, "Constant__1294", 0),
            GNodeArg(GNA_OUT, "S48_Output", 0),
            GNodeArg(GNA_IN, "S48_Mul_scale", 0),
            GNodeArg(GNA_IN, "S48_Mul_shift", 0),
            GNodeArg(GNA_IN, "S48_Infos", 0)
        )
    );
    // Node S51_Conv2d_32x32x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S51_Conv2d_32x32x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S48_Output", 0),
            GNodeArg(GNA_IN, "Conv_62_weights", 0),
            GNodeArg(GNA_IN, "Constant__1297", 0),
            GNodeArg(GNA_OUT, "S51_Output", 0),
            GNodeArg(GNA_IN, "S51_Mul_scale", 0),
            GNodeArg(GNA_IN, "S51_Mul_shift", 0),
            GNodeArg(GNA_IN, "S51_Infos", 0)
        )
    );
    // Node S52_MatAdd_45x60x32 in1q 0.00<(u8-0.00)*0.02352941<6.00 forced
    //   in2q 0.00<(u8-0.00)*0.02352941<6.00 forced
    //   outq 0.00<(u8-0.00)*0.03850398<9.82 forced scaled input 0 is node input 0
    AddNode("S52_MatAdd_45x60x32",
        Bindings(4,
            GNodeArg(GNA_IN, "S51_Output", 0),
            GNodeArg(GNA_IN, "S39_Output", 0),
            GNodeArg(GNA_OUT, "S52_Output", 0),
            GNodeArg(GNA_IN, "S52_Infos", 0)
        )
    );
    // Node S55_Conv2d_32x32x1x1_Relu6 inq 0.00<(u8-0.00)*0.03850398<9.82 forced weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S55_Conv2d_32x32x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S52_Output", 0),
            GNodeArg(GNA_IN, "Conv_67_weights", 0),
            GNodeArg(GNA_IN, "Constant__1300", 0),
            GNodeArg(GNA_OUT, "S55_Output", 0),
            GNodeArg(GNA_IN, "S55_Mul_scale", 0),
            GNodeArg(GNA_IN, "S55_Mul_shift", 0),
            GNodeArg(GNA_IN, "S55_Infos", 0)
        )
    );
    // Node S58_Conv2d_32x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S58_Conv2d_32x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S55_Output", 0),
            GNodeArg(GNA_IN, "Conv_71_weights", 0),
            GNodeArg(GNA_IN, "Constant__1303", 0),
            GNodeArg(GNA_OUT, "S58_Output", 0),
            GNodeArg(GNA_IN, "S58_Mul_scale", 0),
            GNodeArg(GNA_IN, "S58_Mul_shift", 0),
            GNodeArg(GNA_IN, "S58_Infos", 0)
        )
    );
    // Node S61_Conv2d_32x32x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S61_Conv2d_32x32x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S58_Output", 0),
            GNodeArg(GNA_IN, "Conv_75_weights", 0),
            GNodeArg(GNA_IN, "Constant__1306", 0),
            GNodeArg(GNA_OUT, "S61_Output", 0),
            GNodeArg(GNA_IN, "S61_Mul_scale", 0),
            GNodeArg(GNA_IN, "S61_Mul_shift", 0),
            GNodeArg(GNA_IN, "S61_Infos", 0)
        )
    );
    // Node S62_MatAdd_45x60x32 in1q 0.00<(u8-0.00)*0.03850398<9.82 forced
    //   in2q 0.00<(u8-0.00)*0.02352941<6.00 forced
    //   outq 0.00<(u8-0.00)*0.05007586<12.77 forced scaled input 0 is node input 1
    AddNode("S62_MatAdd_45x60x32",
        Bindings(4,
            GNodeArg(GNA_IN, "S52_Output", 0),
            GNodeArg(GNA_IN, "S61_Output", 0),
            GNodeArg(GNA_OUT, "S62_Output", 0),
            GNodeArg(GNA_IN, "S62_Infos", 0)
        )
    );
    // Node S65_Conv2d_32x32x1x1_Relu6 inq 0.00<(u8-0.00)*0.05007586<12.77 forced weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S65_Conv2d_32x32x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S62_Output", 0),
            GNodeArg(GNA_IN, "Conv_80_weights", 0),
            GNodeArg(GNA_IN, "Constant__1309", 0),
            GNodeArg(GNA_OUT, "S65_Output", 0),
            GNodeArg(GNA_IN, "S65_Mul_scale", 0),
            GNodeArg(GNA_IN, "S65_Mul_shift", 0),
            GNodeArg(GNA_IN, "S65_Infos", 0)
        )
    );
    // Node S68_Conv2d_32x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S68_Conv2d_32x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S65_Output", 0),
            GNodeArg(GNA_IN, "Conv_84_weights", 0),
            GNodeArg(GNA_IN, "Constant__1312", 0),
            GNodeArg(GNA_OUT, "S68_Output", 0),
            GNodeArg(GNA_IN, "S68_Mul_scale", 0),
            GNodeArg(GNA_IN, "S68_Mul_shift", 0),
            GNodeArg(GNA_IN, "S68_Infos", 0)
        )
    );
    // Node S71_Conv2d_32x32x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S71_Conv2d_32x32x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S68_Output", 0),
            GNodeArg(GNA_IN, "Conv_88_weights", 0),
            GNodeArg(GNA_IN, "Constant__1315", 0),
            GNodeArg(GNA_OUT, "S71_Output", 0),
            GNodeArg(GNA_IN, "S71_Mul_scale", 0),
            GNodeArg(GNA_IN, "S71_Mul_shift", 0),
            GNodeArg(GNA_IN, "S71_Infos", 0)
        )
    );
    // Node S72_MatAdd_45x60x32 in1q 0.00<(u8-0.00)*0.05007586<12.77 forced
    //   in2q 0.00<(u8-0.00)*0.02352941<6.00 forced
    //   outq 0.00<(u8-0.00)*0.06673826<17.02 forced scaled input 0 is node input 1
    AddNode("S72_MatAdd_45x60x32",
        Bindings(4,
            GNodeArg(GNA_IN, "S62_Output", 0),
            GNodeArg(GNA_IN, "S71_Output", 0),
            GNodeArg(GNA_OUT, "S72_Output", 0),
            GNodeArg(GNA_IN, "S72_Infos", 0)
        )
    );
    // Node Concat_93 inq ['0.00<(u8-0.00)*0.06673826<17.02 forced', '0.00<(u8-0.00)*0.06673826<17.02 forced'] outq ['0.00<(u8-0.00)*0.06673826<17.02 forced']
    AddNode("S73_Concat",
        Bindings(3,
            GNodeArg(GNA_IN, "S72_Output", 0),
            GNodeArg(GNA_IN, "S42_Output", 0),
            GNodeArg(GNA_OUT, "S73_Output", 0)
        )
    );
    // Node S76_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.06673826<17.02 forced weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S76_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S73_Output", 0),
            GNodeArg(GNA_IN, "Conv_94_weights", 0),
            GNodeArg(GNA_IN, "Constant__1318", 0),
            GNodeArg(GNA_OUT, "S76_Output", 0),
            GNodeArg(GNA_IN, "S76_Mul_scale", 0),
            GNodeArg(GNA_IN, "S76_Mul_shift", 0),
            GNodeArg(GNA_IN, "S76_Infos", 0)
        )
    );
    // Node S79_Conv2d_64x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S79_Conv2d_64x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S76_Output", 0),
            GNodeArg(GNA_IN, "Conv_98_weights", 0),
            GNodeArg(GNA_IN, "Constant__1321", 0),
            GNodeArg(GNA_OUT, "S79_Output", 0),
            GNodeArg(GNA_IN, "S79_Mul_scale", 0),
            GNodeArg(GNA_IN, "S79_Mul_shift", 0),
            GNodeArg(GNA_IN, "S79_Infos", 0)
        )
    );
    // Node S82_Conv2d_128x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S82_Conv2d_128x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S79_Output", 0),
            GNodeArg(GNA_IN, "Conv_102_weights", 0),
            GNodeArg(GNA_IN, "Constant__1324", 0),
            GNodeArg(GNA_OUT, "S82_Output", 0),
            GNodeArg(GNA_IN, "S82_Mul_scale", 0),
            GNodeArg(GNA_IN, "S82_Mul_shift", 0),
            GNodeArg(GNA_IN, "S82_Infos", 0)
        )
    );
    // Node S85_Conv2d_64x128x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S85_Conv2d_64x128x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S82_Output", 0),
            GNodeArg(GNA_IN, "Conv_106_weights", 0),
            GNodeArg(GNA_IN, "Constant__1327", 0),
            GNodeArg(GNA_OUT, "S85_Output", 0),
            GNodeArg(GNA_IN, "S85_Mul_scale", 0),
            GNodeArg(GNA_IN, "S85_Mul_shift", 0),
            GNodeArg(GNA_IN, "S85_Infos", 0)
        )
    );
    // Node S88_Conv2d_64x128x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.06990734<17.83 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S88_Conv2d_64x128x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S82_Output", 0),
            GNodeArg(GNA_IN, "Conv_110_weights", 0),
            GNodeArg(GNA_IN, "Constant__1330", 0),
            GNodeArg(GNA_OUT, "S88_Output", 0),
            GNodeArg(GNA_IN, "S88_Mul_scale", 0),
            GNodeArg(GNA_IN, "S88_Mul_shift", 0),
            GNodeArg(GNA_IN, "S88_Infos", 0)
        )
    );
    // Node S91_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S91_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S85_Output", 0),
            GNodeArg(GNA_IN, "Conv_114_weights", 0),
            GNodeArg(GNA_IN, "Constant__1333", 0),
            GNodeArg(GNA_OUT, "S91_Output", 0),
            GNodeArg(GNA_IN, "S91_Mul_scale", 0),
            GNodeArg(GNA_IN, "S91_Mul_shift", 0),
            GNodeArg(GNA_IN, "S91_Infos", 0)
        )
    );
    // Node S94_Conv2d_64x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S94_Conv2d_64x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S91_Output", 0),
            GNodeArg(GNA_IN, "Conv_118_weights", 0),
            GNodeArg(GNA_IN, "Constant__1336", 0),
            GNodeArg(GNA_OUT, "S94_Output", 0),
            GNodeArg(GNA_IN, "S94_Mul_scale", 0),
            GNodeArg(GNA_IN, "S94_Mul_shift", 0),
            GNodeArg(GNA_IN, "S94_Infos", 0)
        )
    );
    // Node S97_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S97_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S94_Output", 0),
            GNodeArg(GNA_IN, "Conv_122_weights", 0),
            GNodeArg(GNA_IN, "Constant__1339", 0),
            GNodeArg(GNA_OUT, "S97_Output", 0),
            GNodeArg(GNA_IN, "S97_Mul_scale", 0),
            GNodeArg(GNA_IN, "S97_Mul_shift", 0),
            GNodeArg(GNA_IN, "S97_Infos", 0)
        )
    );
    // Node S98_MatAdd_23x30x64 in1q 0.00<(u8-0.00)*0.02352941<6.00 forced
    //   in2q 0.00<(u8-0.00)*0.02352941<6.00 forced
    //   outq 0.00<(u8-0.00)*0.03286810<8.38 forced scaled input 0 is node input 0
    AddNode("S98_MatAdd_23x30x64",
        Bindings(4,
            GNodeArg(GNA_IN, "S97_Output", 0),
            GNodeArg(GNA_IN, "S85_Output", 0),
            GNodeArg(GNA_OUT, "S98_Output", 0),
            GNodeArg(GNA_IN, "S98_Infos", 0)
        )
    );
    // Node S101_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.03286810<8.38 forced weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S101_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S98_Output", 0),
            GNodeArg(GNA_IN, "Conv_127_weights", 0),
            GNodeArg(GNA_IN, "Constant__1342", 0),
            GNodeArg(GNA_OUT, "S101_Output", 0),
            GNodeArg(GNA_IN, "S101_Mul_scale", 0),
            GNodeArg(GNA_IN, "S101_Mul_shift", 0),
            GNodeArg(GNA_IN, "S101_Infos", 0)
        )
    );
    // Node S104_Conv2d_64x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S104_Conv2d_64x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S101_Output", 0),
            GNodeArg(GNA_IN, "Conv_131_weights", 0),
            GNodeArg(GNA_IN, "Constant__1345", 0),
            GNodeArg(GNA_OUT, "S104_Output", 0),
            GNodeArg(GNA_IN, "S104_Mul_scale", 0),
            GNodeArg(GNA_IN, "S104_Mul_shift", 0),
            GNodeArg(GNA_IN, "S104_Infos", 0)
        )
    );
    // Node S107_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S107_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S104_Output", 0),
            GNodeArg(GNA_IN, "Conv_135_weights", 0),
            GNodeArg(GNA_IN, "Constant__1348", 0),
            GNodeArg(GNA_OUT, "S107_Output", 0),
            GNodeArg(GNA_IN, "S107_Mul_scale", 0),
            GNodeArg(GNA_IN, "S107_Mul_shift", 0),
            GNodeArg(GNA_IN, "S107_Infos", 0)
        )
    );
    // Node S108_MatAdd_23x30x64 in1q 0.00<(u8-0.00)*0.03286810<8.38 forced
    //   in2q 0.00<(u8-0.00)*0.02352941<6.00 forced
    //   outq 0.00<(u8-0.00)*0.05058009<12.90 forced scaled input 0 is node input 1
    AddNode("S108_MatAdd_23x30x64",
        Bindings(4,
            GNodeArg(GNA_IN, "S98_Output", 0),
            GNodeArg(GNA_IN, "S107_Output", 0),
            GNodeArg(GNA_OUT, "S108_Output", 0),
            GNodeArg(GNA_IN, "S108_Infos", 0)
        )
    );
    // Node S111_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.05058009<12.90 forced weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S111_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S108_Output", 0),
            GNodeArg(GNA_IN, "Conv_140_weights", 0),
            GNodeArg(GNA_IN, "Constant__1351", 0),
            GNodeArg(GNA_OUT, "S111_Output", 0),
            GNodeArg(GNA_IN, "S111_Mul_scale", 0),
            GNodeArg(GNA_IN, "S111_Mul_shift", 0),
            GNodeArg(GNA_IN, "S111_Infos", 0)
        )
    );
    // Node S114_Conv2d_64x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S114_Conv2d_64x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S111_Output", 0),
            GNodeArg(GNA_IN, "Conv_144_weights", 0),
            GNodeArg(GNA_IN, "Constant__1354", 0),
            GNodeArg(GNA_OUT, "S114_Output", 0),
            GNodeArg(GNA_IN, "S114_Mul_scale", 0),
            GNodeArg(GNA_IN, "S114_Mul_shift", 0),
            GNodeArg(GNA_IN, "S114_Infos", 0)
        )
    );
    // Node S117_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S117_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S114_Output", 0),
            GNodeArg(GNA_IN, "Conv_148_weights", 0),
            GNodeArg(GNA_IN, "Constant__1357", 0),
            GNodeArg(GNA_OUT, "S117_Output", 0),
            GNodeArg(GNA_IN, "S117_Mul_scale", 0),
            GNodeArg(GNA_IN, "S117_Mul_shift", 0),
            GNodeArg(GNA_IN, "S117_Infos", 0)
        )
    );
    // Node S118_MatAdd_23x30x64 in1q 0.00<(u8-0.00)*0.05058009<12.90 forced
    //   in2q 0.00<(u8-0.00)*0.02352941<6.00 forced
    //   outq 0.00<(u8-0.00)*0.06990734<17.83 forced scaled input 0 is node input 1
    AddNode("S118_MatAdd_23x30x64",
        Bindings(4,
            GNodeArg(GNA_IN, "S108_Output", 0),
            GNodeArg(GNA_IN, "S117_Output", 0),
            GNodeArg(GNA_OUT, "S118_Output", 0),
            GNodeArg(GNA_IN, "S118_Infos", 0)
        )
    );
    // Node Concat_153 inq ['0.00<(u8-0.00)*0.06990734<17.83 forced', '0.00<(u8-0.00)*0.06990734<17.83 forced'] outq ['0.00<(u8-0.00)*0.06990734<17.83 forced']
    AddNode("S119_Concat",
        Bindings(3,
            GNodeArg(GNA_IN, "S118_Output", 0),
            GNodeArg(GNA_IN, "S88_Output", 0),
            GNodeArg(GNA_OUT, "S119_Output", 0)
        )
    );
    // Node S122_Conv2d_128x128x1x1_Relu6 inq 0.00<(u8-0.00)*0.06990734<17.83 forced weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S122_Conv2d_128x128x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S119_Output", 0),
            GNodeArg(GNA_IN, "Conv_154_weights", 0),
            GNodeArg(GNA_IN, "Constant__1360", 0),
            GNodeArg(GNA_OUT, "S122_Output", 0),
            GNodeArg(GNA_IN, "S122_Mul_scale", 0),
            GNodeArg(GNA_IN, "S122_Mul_shift", 0),
            GNodeArg(GNA_IN, "S122_Infos", 0)
        )
    );
    // Node S125_Conv2d_128x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S125_Conv2d_128x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S122_Output", 0),
            GNodeArg(GNA_IN, "Conv_158_weights", 0),
            GNodeArg(GNA_IN, "Constant__1363", 0),
            GNodeArg(GNA_OUT, "S125_Output", 0),
            GNodeArg(GNA_IN, "S125_Mul_scale", 0),
            GNodeArg(GNA_IN, "S125_Mul_shift", 0),
            GNodeArg(GNA_IN, "S125_Infos", 0)
        )
    );
    // Node S128_Conv2d_256x128x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S128_Conv2d_256x128x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S125_Output", 0),
            GNodeArg(GNA_IN, "Conv_162_weights", 0),
            GNodeArg(GNA_IN, "Constant__1366", 0),
            GNodeArg(GNA_OUT, "S128_Output", 0),
            GNodeArg(GNA_IN, "S128_Mul_scale", 0),
            GNodeArg(GNA_IN, "S128_Mul_shift", 0),
            GNodeArg(GNA_IN, "S128_Infos", 0)
        )
    );
    // Node S131_Conv2d_128x256x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S131_Conv2d_128x256x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S128_Output", 0),
            GNodeArg(GNA_IN, "Conv_166_weights", 0),
            GNodeArg(GNA_IN, "Constant__1369", 0),
            GNodeArg(GNA_OUT, "S131_Output", 0),
            GNodeArg(GNA_IN, "S131_Mul_scale", 0),
            GNodeArg(GNA_IN, "S131_Mul_shift", 0),
            GNodeArg(GNA_IN, "S131_Infos", 0)
        )
    );
    // Node MaxPool_170 inq 0.00<(u8-0.00)*0.02352941<6.00 outq 0.00<(u8-0.00)*0.02352941<6.00
    AddNode("S132_MaxPool_5x5",
        Bindings(3,
            GNodeArg(GNA_IN, "S131_Output", 0),
            GNodeArg(GNA_OUT, "S132_Output", 0),
            GNodeArg(GNA_IN, "S132_Infos", 0)
        )
    );
    // Node MaxPool_171 inq 0.00<(u8-0.00)*0.02352941<6.00 outq 0.00<(u8-0.00)*0.02352941<6.00
    AddNode("S133_MaxPool_9x9",
        Bindings(3,
            GNodeArg(GNA_IN, "S131_Output", 0),
            GNodeArg(GNA_OUT, "S133_Output", 0),
            GNodeArg(GNA_IN, "S133_Infos", 0)
        )
    );
    // Node MaxPool_172 inq 0.00<(u8-0.00)*0.02352941<6.00 outq 0.00<(u8-0.00)*0.02352941<6.00
    AddNode("S134_MaxPool_13x13",
        Bindings(3,
            GNodeArg(GNA_IN, "S131_Output", 0),
            GNodeArg(GNA_OUT, "S134_Output", 0),
            GNodeArg(GNA_IN, "S134_Infos", 0)
        )
    );
    // Node Concat_173 inq ['0.00<(u8-0.00)*0.02352941<6.00', '0.00<(u8-0.00)*0.02352941<6.00', '0.00<(u8-0.00)*0.02352941<6.00', '0.00<(u8-0.00)*0.02352941<6.00'] outq ['0.00<(u8-0.00)*0.02352941<6.00']
    AddNode("S135_Concat",
        Bindings(5,
            GNodeArg(GNA_IN, "S131_Output", 0),
            GNodeArg(GNA_IN, "S132_Output", 0),
            GNodeArg(GNA_IN, "S133_Output", 0),
            GNodeArg(GNA_IN, "S134_Output", 0),
            GNodeArg(GNA_OUT, "S135_Output", 0)
        )
    );
    // Node S138_Conv2d_256x512x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S138_Conv2d_256x512x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S135_Output", 0),
            GNodeArg(GNA_IN, "Conv_174_weights", 0),
            GNodeArg(GNA_IN, "Constant__1372", 0),
            GNodeArg(GNA_OUT, "S138_Output", 0),
            GNodeArg(GNA_IN, "S138_Mul_scale", 0),
            GNodeArg(GNA_IN, "S138_Mul_shift", 0),
            GNodeArg(GNA_IN, "S138_Infos", 0)
        )
    );
    // Node S141_Conv2d_128x256x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S141_Conv2d_128x256x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S138_Output", 0),
            GNodeArg(GNA_IN, "Conv_178_weights", 0),
            GNodeArg(GNA_IN, "Constant__1375", 0),
            GNodeArg(GNA_OUT, "S141_Output", 0),
            GNodeArg(GNA_IN, "S141_Mul_scale", 0),
            GNodeArg(GNA_IN, "S141_Mul_shift", 0),
            GNodeArg(GNA_IN, "S141_Infos", 0)
        )
    );
    // Node S144_Conv2d_128x256x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S144_Conv2d_128x256x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S138_Output", 0),
            GNodeArg(GNA_IN, "Conv_182_weights", 0),
            GNodeArg(GNA_IN, "Constant__1378", 0),
            GNodeArg(GNA_OUT, "S144_Output", 0),
            GNodeArg(GNA_IN, "S144_Mul_scale", 0),
            GNodeArg(GNA_IN, "S144_Mul_shift", 0),
            GNodeArg(GNA_IN, "S144_Infos", 0)
        )
    );
    // Node S147_Conv2d_128x128x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S147_Conv2d_128x128x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S141_Output", 0),
            GNodeArg(GNA_IN, "Conv_186_weights", 0),
            GNodeArg(GNA_IN, "Constant__1381", 0),
            GNodeArg(GNA_OUT, "S147_Output", 0),
            GNodeArg(GNA_IN, "S147_Mul_scale", 0),
            GNodeArg(GNA_IN, "S147_Mul_shift", 0),
            GNodeArg(GNA_IN, "S147_Infos", 0)
        )
    );
    // Node S150_Conv2d_128x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S150_Conv2d_128x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S147_Output", 0),
            GNodeArg(GNA_IN, "Conv_190_weights", 0),
            GNodeArg(GNA_IN, "Constant__1384", 0),
            GNodeArg(GNA_OUT, "S150_Output", 0),
            GNodeArg(GNA_IN, "S150_Mul_scale", 0),
            GNodeArg(GNA_IN, "S150_Mul_shift", 0),
            GNodeArg(GNA_IN, "S150_Infos", 0)
        )
    );
    // Node S153_Conv2d_128x128x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S153_Conv2d_128x128x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S150_Output", 0),
            GNodeArg(GNA_IN, "Conv_194_weights", 0),
            GNodeArg(GNA_IN, "Constant__1387", 0),
            GNodeArg(GNA_OUT, "S153_Output", 0),
            GNodeArg(GNA_IN, "S153_Mul_scale", 0),
            GNodeArg(GNA_IN, "S153_Mul_shift", 0),
            GNodeArg(GNA_IN, "S153_Infos", 0)
        )
    );
    // Node Concat_198 inq ['0.00<(u8-0.00)*0.02352941<6.00', '0.00<(u8-0.00)*0.02352941<6.00'] outq ['0.00<(u8-0.00)*0.02352941<6.00']
    AddNode("S154_Concat",
        Bindings(3,
            GNodeArg(GNA_IN, "S153_Output", 0),
            GNodeArg(GNA_IN, "S144_Output", 0),
            GNodeArg(GNA_OUT, "S154_Output", 0)
        )
    );
    // Node S157_Conv2d_256x256x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S157_Conv2d_256x256x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S154_Output", 0),
            GNodeArg(GNA_IN, "Conv_199_weights", 0),
            GNodeArg(GNA_IN, "Constant__1390", 0),
            GNodeArg(GNA_OUT, "S157_Output", 0),
            GNodeArg(GNA_IN, "S157_Mul_scale", 0),
            GNodeArg(GNA_IN, "S157_Mul_shift", 0),
            GNodeArg(GNA_IN, "S157_Infos", 0)
        )
    );
    // Node S160_Conv2d_128x256x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S160_Conv2d_128x256x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S157_Output", 0),
            GNodeArg(GNA_IN, "Conv_203_weights", 0),
            GNodeArg(GNA_IN, "Constant__1393", 0),
            GNodeArg(GNA_OUT, "S160_Output", 0),
            GNodeArg(GNA_IN, "S160_Mul_scale", 0),
            GNodeArg(GNA_IN, "S160_Mul_shift", 0),
            GNodeArg(GNA_IN, "S160_Infos", 0)
        )
    );
    // Node Resize_208_trans inq 0.00<(u8-0.00)*0.02352941<6.00 outq 0.00<(u8-0.00)*0.02352941<6.00
    AddNode("S161_Op_Resize_208_trans",
        Bindings(2,
            GNodeArg(GNA_IN, "S160_Output", 0),
            GNodeArg(GNA_OUT, "S161_Output", 0)
        )
    );
    // Node Resize_208 inq 0.00<(u8-0.00)*0.02352941<6.00 outq 0.00<(u8-0.00)*0.02352941<6.00
    AddNode("S162_Op_Resize_208",
        Bindings(2,
            GNodeArg(GNA_IN, "S161_Output", 0),
            GNodeArg(GNA_OUT, "S162_Output", 0)
        )
    );
    // Node Slice_213_trans_in0 inq 0.00<(u8-0.00)*0.02352941<6.00 outq 0.00<(u8-0.00)*0.02352941<6.00
    AddNode("S163_Op_Slice_213_trans_in0",
        Bindings(2,
            GNodeArg(GNA_IN, "S162_Output", 0),
            GNodeArg(GNA_OUT, "S163_Output", 0)
        )
    );
    // Node Slice_213_trans_out0 inq 0.00<(u8-0.00)*0.02352941<6.00 outq 0.00<(u8-0.00)*0.02352941<6.00
    AddNode("S165_Op_Slice_213_trans_out0",
        Bindings(2,
            GNodeArg(GNA_IN, "S164_Output_0", 0),
            GNodeArg(GNA_OUT, "S165_Output", 0)
        )
    );
    // Node Concat_214 inq ['0.00<(u8-0.00)*0.02352941<6.00', '0.00<(u8-0.00)*0.02352941<6.00'] outq ['0.00<(u8-0.00)*0.02352941<6.00']
    AddNode("S166_Concat",
        Bindings(3,
            GNodeArg(GNA_IN, "S165_Output", 0),
            GNodeArg(GNA_IN, "S122_Output", 0),
            GNodeArg(GNA_OUT, "S166_Output", 0)
        )
    );
    // Node S170_Conv2d_64x256x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S170_Conv2d_64x256x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S166_Output", 0),
            GNodeArg(GNA_IN, "Conv_215_weights", 0),
            GNodeArg(GNA_IN, "Constant__1396", 0),
            GNodeArg(GNA_OUT, "S170_Output", 0),
            GNodeArg(GNA_IN, "S170_Mul_scale", 0),
            GNodeArg(GNA_IN, "S170_Mul_shift", 0),
            GNodeArg(GNA_IN, "S170_Infos", 0)
        )
    );
    // Node S173_Conv2d_64x256x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S173_Conv2d_64x256x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S166_Output", 0),
            GNodeArg(GNA_IN, "Conv_219_weights", 0),
            GNodeArg(GNA_IN, "Constant__1399", 0),
            GNodeArg(GNA_OUT, "S173_Output", 0),
            GNodeArg(GNA_IN, "S173_Mul_scale", 0),
            GNodeArg(GNA_IN, "S173_Mul_shift", 0),
            GNodeArg(GNA_IN, "S173_Infos", 0)
        )
    );
    // Node S176_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S176_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S170_Output", 0),
            GNodeArg(GNA_IN, "Conv_223_weights", 0),
            GNodeArg(GNA_IN, "Constant__1402", 0),
            GNodeArg(GNA_OUT, "S176_Output", 0),
            GNodeArg(GNA_IN, "S176_Mul_scale", 0),
            GNodeArg(GNA_IN, "S176_Mul_shift", 0),
            GNodeArg(GNA_IN, "S176_Infos", 0)
        )
    );
    // Node S179_Conv2d_64x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S179_Conv2d_64x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S176_Output", 0),
            GNodeArg(GNA_IN, "Conv_227_weights", 0),
            GNodeArg(GNA_IN, "Constant__1405", 0),
            GNodeArg(GNA_OUT, "S179_Output", 0),
            GNodeArg(GNA_IN, "S179_Mul_scale", 0),
            GNodeArg(GNA_IN, "S179_Mul_shift", 0),
            GNodeArg(GNA_IN, "S179_Infos", 0)
        )
    );
    // Node S182_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S182_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S179_Output", 0),
            GNodeArg(GNA_IN, "Conv_231_weights", 0),
            GNodeArg(GNA_IN, "Constant__1408", 0),
            GNodeArg(GNA_OUT, "S182_Output", 0),
            GNodeArg(GNA_IN, "S182_Mul_scale", 0),
            GNodeArg(GNA_IN, "S182_Mul_shift", 0),
            GNodeArg(GNA_IN, "S182_Infos", 0)
        )
    );
    // Node Concat_235 inq ['0.00<(u8-0.00)*0.02352941<6.00', '0.00<(u8-0.00)*0.02352941<6.00'] outq ['0.00<(u8-0.00)*0.02352941<6.00']
    AddNode("S183_Concat",
        Bindings(3,
            GNodeArg(GNA_IN, "S182_Output", 0),
            GNodeArg(GNA_IN, "S173_Output", 0),
            GNodeArg(GNA_OUT, "S183_Output", 0)
        )
    );
    // Node S186_Conv2d_128x128x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S186_Conv2d_128x128x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S183_Output", 0),
            GNodeArg(GNA_IN, "Conv_236_weights", 0),
            GNodeArg(GNA_IN, "Constant__1411", 0),
            GNodeArg(GNA_OUT, "S186_Output", 0),
            GNodeArg(GNA_IN, "S186_Mul_scale", 0),
            GNodeArg(GNA_IN, "S186_Mul_shift", 0),
            GNodeArg(GNA_IN, "S186_Infos", 0)
        )
    );
    // Node S189_Conv2d_64x128x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S189_Conv2d_64x128x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S186_Output", 0),
            GNodeArg(GNA_IN, "Conv_240_weights", 0),
            GNodeArg(GNA_IN, "Constant__1414", 0),
            GNodeArg(GNA_OUT, "S189_Output", 0),
            GNodeArg(GNA_IN, "S189_Mul_scale", 0),
            GNodeArg(GNA_IN, "S189_Mul_shift", 0),
            GNodeArg(GNA_IN, "S189_Infos", 0)
        )
    );
    // Node Resize_245_trans inq 0.00<(u8-0.00)*0.02352941<6.00 outq 0.00<(u8-0.00)*0.02352941<6.00
    AddNode("S190_Op_Resize_245_trans",
        Bindings(2,
            GNodeArg(GNA_IN, "S189_Output", 0),
            GNodeArg(GNA_OUT, "S190_Output", 0)
        )
    );
    // Node Resize_245 inq 0.00<(u8-0.00)*0.02352941<6.00 outq 0.00<(u8-0.00)*0.02352941<6.00
    AddNode("S191_Op_Resize_245",
        Bindings(2,
            GNodeArg(GNA_IN, "S190_Output", 0),
            GNodeArg(GNA_OUT, "S191_Output", 0)
        )
    );
    // Node Slice_250_trans_in0 inq 0.00<(u8-0.00)*0.02352941<6.00 outq 0.00<(u8-0.00)*0.02352941<6.00
    AddNode("S192_Op_Slice_250_trans_in0",
        Bindings(2,
            GNodeArg(GNA_IN, "S191_Output", 0),
            GNodeArg(GNA_OUT, "S192_Output", 0)
        )
    );
    // Node Slice_250_trans_out0 inq 0.00<(u8-0.00)*0.02352941<6.00 outq 0.00<(u8-0.00)*0.02352941<6.00
    AddNode("S194_Op_Slice_250_trans_out0",
        Bindings(2,
            GNodeArg(GNA_IN, "S193_Output_0", 0),
            GNodeArg(GNA_OUT, "S194_Output", 0)
        )
    );
    // Node Concat_251 inq ['0.00<(u8-0.00)*0.02352941<6.00', '0.00<(u8-0.00)*0.02352941<6.00'] outq ['0.00<(u8-0.00)*0.02352941<6.00']
    AddNode("S195_Concat",
        Bindings(3,
            GNodeArg(GNA_IN, "S194_Output", 0),
            GNodeArg(GNA_IN, "S76_Output", 0),
            GNodeArg(GNA_OUT, "S195_Output", 0)
        )
    );
    // Node S199_Conv2d_32x128x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S199_Conv2d_32x128x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S195_Output", 0),
            GNodeArg(GNA_IN, "Conv_252_weights", 0),
            GNodeArg(GNA_IN, "Constant__1417", 0),
            GNodeArg(GNA_OUT, "S199_Output", 0),
            GNodeArg(GNA_IN, "S199_Mul_scale", 0),
            GNodeArg(GNA_IN, "S199_Mul_shift", 0),
            GNodeArg(GNA_IN, "S199_Infos", 0)
        )
    );
    // Node S202_Conv2d_32x128x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S202_Conv2d_32x128x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S195_Output", 0),
            GNodeArg(GNA_IN, "Conv_256_weights", 0),
            GNodeArg(GNA_IN, "Constant__1420", 0),
            GNodeArg(GNA_OUT, "S202_Output", 0),
            GNodeArg(GNA_IN, "S202_Mul_scale", 0),
            GNodeArg(GNA_IN, "S202_Mul_shift", 0),
            GNodeArg(GNA_IN, "S202_Infos", 0)
        )
    );
    // Node S205_Conv2d_32x32x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S205_Conv2d_32x32x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S199_Output", 0),
            GNodeArg(GNA_IN, "Conv_260_weights", 0),
            GNodeArg(GNA_IN, "Constant__1423", 0),
            GNodeArg(GNA_OUT, "S205_Output", 0),
            GNodeArg(GNA_IN, "S205_Mul_scale", 0),
            GNodeArg(GNA_IN, "S205_Mul_shift", 0),
            GNodeArg(GNA_IN, "S205_Infos", 0)
        )
    );
    // Node S208_Conv2d_32x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S208_Conv2d_32x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S205_Output", 0),
            GNodeArg(GNA_IN, "Conv_264_weights", 0),
            GNodeArg(GNA_IN, "Constant__1426", 0),
            GNodeArg(GNA_OUT, "S208_Output", 0),
            GNodeArg(GNA_IN, "S208_Mul_scale", 0),
            GNodeArg(GNA_IN, "S208_Mul_shift", 0),
            GNodeArg(GNA_IN, "S208_Infos", 0)
        )
    );
    // Node S211_Conv2d_32x32x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S211_Conv2d_32x32x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S208_Output", 0),
            GNodeArg(GNA_IN, "Conv_268_weights", 0),
            GNodeArg(GNA_IN, "Constant__1429", 0),
            GNodeArg(GNA_OUT, "S211_Output", 0),
            GNodeArg(GNA_IN, "S211_Mul_scale", 0),
            GNodeArg(GNA_IN, "S211_Mul_shift", 0),
            GNodeArg(GNA_IN, "S211_Infos", 0)
        )
    );
    // Node Concat_272 inq ['0.00<(u8-0.00)*0.02352941<6.00', '0.00<(u8-0.00)*0.02352941<6.00'] outq ['0.00<(u8-0.00)*0.02352941<6.00']
    AddNode("S212_Concat",
        Bindings(3,
            GNodeArg(GNA_IN, "S211_Output", 0),
            GNodeArg(GNA_IN, "S202_Output", 0),
            GNodeArg(GNA_OUT, "S212_Output", 0)
        )
    );
    // Node S215_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S215_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S212_Output", 0),
            GNodeArg(GNA_IN, "Conv_273_weights", 0),
            GNodeArg(GNA_IN, "Constant__1432", 0),
            GNodeArg(GNA_OUT, "S215_Output", 0),
            GNodeArg(GNA_IN, "S215_Mul_scale", 0),
            GNodeArg(GNA_IN, "S215_Mul_shift", 0),
            GNodeArg(GNA_IN, "S215_Infos", 0)
        )
    );
    // Node S218_Conv2d_64x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S218_Conv2d_64x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S215_Output", 0),
            GNodeArg(GNA_IN, "Conv_277_weights", 0),
            GNodeArg(GNA_IN, "Constant__1435", 0),
            GNodeArg(GNA_OUT, "S218_Output", 0),
            GNodeArg(GNA_IN, "S218_Mul_scale", 0),
            GNodeArg(GNA_IN, "S218_Mul_shift", 0),
            GNodeArg(GNA_IN, "S218_Infos", 0)
        )
    );
    // Node S221_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S221_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S218_Output", 0),
            GNodeArg(GNA_IN, "Conv_281_weights", 0),
            GNodeArg(GNA_IN, "Constant__1438", 0),
            GNodeArg(GNA_OUT, "S221_Output", 0),
            GNodeArg(GNA_IN, "S221_Mul_scale", 0),
            GNodeArg(GNA_IN, "S221_Mul_shift", 0),
            GNodeArg(GNA_IN, "S221_Infos", 0)
        )
    );
    // Node Concat_285 inq ['0.00<(u8-0.00)*0.02352941<6.00', '0.00<(u8-0.00)*0.02352941<6.00'] outq ['0.00<(u8-0.00)*0.02352941<6.00']
    AddNode("S222_Concat",
        Bindings(3,
            GNodeArg(GNA_IN, "S221_Output", 0),
            GNodeArg(GNA_IN, "S189_Output", 0),
            GNodeArg(GNA_OUT, "S222_Output", 0)
        )
    );
    // Node S225_Conv2d_64x128x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S225_Conv2d_64x128x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S222_Output", 0),
            GNodeArg(GNA_IN, "Conv_286_weights", 0),
            GNodeArg(GNA_IN, "Constant__1441", 0),
            GNodeArg(GNA_OUT, "S225_Output", 0),
            GNodeArg(GNA_IN, "S225_Mul_scale", 0),
            GNodeArg(GNA_IN, "S225_Mul_shift", 0),
            GNodeArg(GNA_IN, "S225_Infos", 0)
        )
    );
    // Node S228_Conv2d_64x128x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S228_Conv2d_64x128x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S222_Output", 0),
            GNodeArg(GNA_IN, "Conv_290_weights", 0),
            GNodeArg(GNA_IN, "Constant__1444", 0),
            GNodeArg(GNA_OUT, "S228_Output", 0),
            GNodeArg(GNA_IN, "S228_Mul_scale", 0),
            GNodeArg(GNA_IN, "S228_Mul_shift", 0),
            GNodeArg(GNA_IN, "S228_Infos", 0)
        )
    );
    // Node S231_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S231_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S225_Output", 0),
            GNodeArg(GNA_IN, "Conv_294_weights", 0),
            GNodeArg(GNA_IN, "Constant__1447", 0),
            GNodeArg(GNA_OUT, "S231_Output", 0),
            GNodeArg(GNA_IN, "S231_Mul_scale", 0),
            GNodeArg(GNA_IN, "S231_Mul_shift", 0),
            GNodeArg(GNA_IN, "S231_Infos", 0)
        )
    );
    // Node S234_Conv2d_64x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S234_Conv2d_64x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S231_Output", 0),
            GNodeArg(GNA_IN, "Conv_298_weights", 0),
            GNodeArg(GNA_IN, "Constant__1450", 0),
            GNodeArg(GNA_OUT, "S234_Output", 0),
            GNodeArg(GNA_IN, "S234_Mul_scale", 0),
            GNodeArg(GNA_IN, "S234_Mul_shift", 0),
            GNodeArg(GNA_IN, "S234_Infos", 0)
        )
    );
    // Node S237_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S237_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S234_Output", 0),
            GNodeArg(GNA_IN, "Conv_302_weights", 0),
            GNodeArg(GNA_IN, "Constant__1453", 0),
            GNodeArg(GNA_OUT, "S237_Output", 0),
            GNodeArg(GNA_IN, "S237_Mul_scale", 0),
            GNodeArg(GNA_IN, "S237_Mul_shift", 0),
            GNodeArg(GNA_IN, "S237_Infos", 0)
        )
    );
    // Node Concat_306 inq ['0.00<(u8-0.00)*0.02352941<6.00', '0.00<(u8-0.00)*0.02352941<6.00'] outq ['0.00<(u8-0.00)*0.02352941<6.00']
    AddNode("S238_Concat",
        Bindings(3,
            GNodeArg(GNA_IN, "S237_Output", 0),
            GNodeArg(GNA_IN, "S228_Output", 0),
            GNodeArg(GNA_OUT, "S238_Output", 0)
        )
    );
    // Node S241_Conv2d_128x128x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S241_Conv2d_128x128x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S238_Output", 0),
            GNodeArg(GNA_IN, "Conv_307_weights", 0),
            GNodeArg(GNA_IN, "Constant__1456", 0),
            GNodeArg(GNA_OUT, "S241_Output", 0),
            GNodeArg(GNA_IN, "S241_Mul_scale", 0),
            GNodeArg(GNA_IN, "S241_Mul_shift", 0),
            GNodeArg(GNA_IN, "S241_Infos", 0)
        )
    );
    // Node S244_Conv2d_128x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S244_Conv2d_128x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S241_Output", 0),
            GNodeArg(GNA_IN, "Conv_311_weights", 0),
            GNodeArg(GNA_IN, "Constant__1459", 0),
            GNodeArg(GNA_OUT, "S244_Output", 0),
            GNodeArg(GNA_IN, "S244_Mul_scale", 0),
            GNodeArg(GNA_IN, "S244_Mul_shift", 0),
            GNodeArg(GNA_IN, "S244_Infos", 0)
        )
    );
    // Node S247_Conv2d_128x128x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S247_Conv2d_128x128x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S244_Output", 0),
            GNodeArg(GNA_IN, "Conv_315_weights", 0),
            GNodeArg(GNA_IN, "Constant__1462", 0),
            GNodeArg(GNA_OUT, "S247_Output", 0),
            GNodeArg(GNA_IN, "S247_Mul_scale", 0),
            GNodeArg(GNA_IN, "S247_Mul_shift", 0),
            GNodeArg(GNA_IN, "S247_Infos", 0)
        )
    );
    // Node Concat_319 inq ['0.00<(u8-0.00)*0.02352941<6.00', '0.00<(u8-0.00)*0.02352941<6.00'] outq ['0.00<(u8-0.00)*0.02352941<6.00']
    AddNode("S248_Concat",
        Bindings(3,
            GNodeArg(GNA_IN, "S247_Output", 0),
            GNodeArg(GNA_IN, "S160_Output", 0),
            GNodeArg(GNA_OUT, "S248_Output", 0)
        )
    );
    // Node S251_Conv2d_128x256x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S251_Conv2d_128x256x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S248_Output", 0),
            GNodeArg(GNA_IN, "Conv_320_weights", 0),
            GNodeArg(GNA_IN, "Constant__1465", 0),
            GNodeArg(GNA_OUT, "S251_Output", 0),
            GNodeArg(GNA_IN, "S251_Mul_scale", 0),
            GNodeArg(GNA_IN, "S251_Mul_shift", 0),
            GNodeArg(GNA_IN, "S251_Infos", 0)
        )
    );
    // Node S254_Conv2d_128x256x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S254_Conv2d_128x256x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S248_Output", 0),
            GNodeArg(GNA_IN, "Conv_324_weights", 0),
            GNodeArg(GNA_IN, "Constant__1468", 0),
            GNodeArg(GNA_OUT, "S254_Output", 0),
            GNodeArg(GNA_IN, "S254_Mul_scale", 0),
            GNodeArg(GNA_IN, "S254_Mul_shift", 0),
            GNodeArg(GNA_IN, "S254_Infos", 0)
        )
    );
    // Node S257_Conv2d_128x128x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S257_Conv2d_128x128x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S251_Output", 0),
            GNodeArg(GNA_IN, "Conv_328_weights", 0),
            GNodeArg(GNA_IN, "Constant__1471", 0),
            GNodeArg(GNA_OUT, "S257_Output", 0),
            GNodeArg(GNA_IN, "S257_Mul_scale", 0),
            GNodeArg(GNA_IN, "S257_Mul_shift", 0),
            GNodeArg(GNA_IN, "S257_Infos", 0)
        )
    );
    // Node S260_Conv2d_128x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S260_Conv2d_128x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S257_Output", 0),
            GNodeArg(GNA_IN, "Conv_332_weights", 0),
            GNodeArg(GNA_IN, "Constant__1474", 0),
            GNodeArg(GNA_OUT, "S260_Output", 0),
            GNodeArg(GNA_IN, "S260_Mul_scale", 0),
            GNodeArg(GNA_IN, "S260_Mul_shift", 0),
            GNodeArg(GNA_IN, "S260_Infos", 0)
        )
    );
    // Node S263_Conv2d_128x128x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S263_Conv2d_128x128x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S260_Output", 0),
            GNodeArg(GNA_IN, "Conv_336_weights", 0),
            GNodeArg(GNA_IN, "Constant__1477", 0),
            GNodeArg(GNA_OUT, "S263_Output", 0),
            GNodeArg(GNA_IN, "S263_Mul_scale", 0),
            GNodeArg(GNA_IN, "S263_Mul_shift", 0),
            GNodeArg(GNA_IN, "S263_Infos", 0)
        )
    );
    // Node Concat_340 inq ['0.00<(u8-0.00)*0.02352941<6.00', '0.00<(u8-0.00)*0.02352941<6.00'] outq ['0.00<(u8-0.00)*0.02352941<6.00']
    AddNode("S264_Concat",
        Bindings(3,
            GNodeArg(GNA_IN, "S263_Output", 0),
            GNodeArg(GNA_IN, "S254_Output", 0),
            GNodeArg(GNA_OUT, "S264_Output", 0)
        )
    );
    // Node S267_Conv2d_256x256x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S267_Conv2d_256x256x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S264_Output", 0),
            GNodeArg(GNA_IN, "Conv_341_weights", 0),
            GNodeArg(GNA_IN, "Constant__1480", 0),
            GNodeArg(GNA_OUT, "S267_Output", 0),
            GNodeArg(GNA_IN, "S267_Mul_scale", 0),
            GNodeArg(GNA_IN, "S267_Mul_shift", 0),
            GNodeArg(GNA_IN, "S267_Infos", 0)
        )
    );
    // Node S270_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S270_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S215_Output", 0),
            GNodeArg(GNA_IN, "Conv_345_weights", 0),
            GNodeArg(GNA_IN, "Constant__1483", 0),
            GNodeArg(GNA_OUT, "S270_Output", 0),
            GNodeArg(GNA_IN, "S270_Mul_scale", 0),
            GNodeArg(GNA_IN, "S270_Mul_shift", 0),
            GNodeArg(GNA_IN, "S270_Infos", 0)
        )
    );
    // Node S273_Conv2d_64x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S273_Conv2d_64x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S270_Output", 0),
            GNodeArg(GNA_IN, "Conv_349_weights", 0),
            GNodeArg(GNA_IN, "Constant__1486", 0),
            GNodeArg(GNA_OUT, "S273_Output", 0),
            GNodeArg(GNA_IN, "S273_Mul_scale", 0),
            GNodeArg(GNA_IN, "S273_Mul_shift", 0),
            GNodeArg(GNA_IN, "S273_Infos", 0)
        )
    );
    // Node S276_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S276_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S273_Output", 0),
            GNodeArg(GNA_IN, "Conv_353_weights", 0),
            GNodeArg(GNA_IN, "Constant__1489", 0),
            GNodeArg(GNA_OUT, "S276_Output", 0),
            GNodeArg(GNA_IN, "S276_Mul_scale", 0),
            GNodeArg(GNA_IN, "S276_Mul_shift", 0),
            GNodeArg(GNA_IN, "S276_Infos", 0)
        )
    );
    // Node S279_Conv2d_64x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S279_Conv2d_64x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S276_Output", 0),
            GNodeArg(GNA_IN, "Conv_357_weights", 0),
            GNodeArg(GNA_IN, "Constant__1492", 0),
            GNodeArg(GNA_OUT, "S279_Output", 0),
            GNodeArg(GNA_IN, "S279_Mul_scale", 0),
            GNodeArg(GNA_IN, "S279_Mul_shift", 0),
            GNodeArg(GNA_IN, "S279_Infos", 0)
        )
    );
    // Node S282_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S282_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S279_Output", 0),
            GNodeArg(GNA_IN, "Conv_361_weights", 0),
            GNodeArg(GNA_IN, "Constant__1495", 0),
            GNodeArg(GNA_OUT, "S282_Output", 0),
            GNodeArg(GNA_IN, "S282_Mul_scale", 0),
            GNodeArg(GNA_IN, "S282_Mul_shift", 0),
            GNodeArg(GNA_IN, "S282_Infos", 0)
        )
    );
    // Node S285_Conv2d_1x64x1x1_Sigmoid inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq -0.10<(u8-128.00)*0.00079523<0.10 outq 0.00<(u8-0.00)*0.00392157<1.00 biasesq -40182.20<(i32-0.00)*0.00001871<40182.20
    AddNode("S285_Conv2d_1x64x1x1_Sigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S282_Output", 0),
            GNodeArg(GNA_IN, "Conv_365_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_cls_preds_0_bias", 0),
            GNodeArg(GNA_OUT, "S285_Output", 0),
            GNodeArg(GNA_IN, "S285_Mul_scale", 0),
            GNodeArg(GNA_IN, "S285_Mul_shift", 0),
            GNodeArg(GNA_IN, "S285_Infos", 0)
        )
    );
    // Node Concat_386_qin2 inq 0.00<(u8-0.00)*0.00392157<1.00 outq -3.70<(u8-104.00)*0.03556376<5.37
    AddNode("S286_Op_Concat_386_qin2",
        Bindings(3,
            GNodeArg(GNA_IN, "S285_Output", 0),
            GNodeArg(GNA_OUT, "S286_Output", 0),
            GNodeArg(GNA_IN, "S286_Infos", 0)
        )
    );
    // Node S289_Conv2d_64x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S289_Conv2d_64x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S270_Output", 0),
            GNodeArg(GNA_IN, "Conv_366_weights", 0),
            GNodeArg(GNA_IN, "Constant__1498", 0),
            GNodeArg(GNA_OUT, "S289_Output", 0),
            GNodeArg(GNA_IN, "S289_Mul_scale", 0),
            GNodeArg(GNA_IN, "S289_Mul_shift", 0),
            GNodeArg(GNA_IN, "S289_Infos", 0)
        )
    );
    // Node S292_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S292_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S289_Output", 0),
            GNodeArg(GNA_IN, "Conv_370_weights", 0),
            GNodeArg(GNA_IN, "Constant__1501", 0),
            GNodeArg(GNA_OUT, "S292_Output", 0),
            GNodeArg(GNA_IN, "S292_Mul_scale", 0),
            GNodeArg(GNA_IN, "S292_Mul_shift", 0),
            GNodeArg(GNA_IN, "S292_Infos", 0)
        )
    );
    // Node S295_Conv2d_64x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S295_Conv2d_64x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S292_Output", 0),
            GNodeArg(GNA_IN, "Conv_374_weights", 0),
            GNodeArg(GNA_IN, "Constant__1504", 0),
            GNodeArg(GNA_OUT, "S295_Output", 0),
            GNodeArg(GNA_IN, "S295_Mul_scale", 0),
            GNodeArg(GNA_IN, "S295_Mul_shift", 0),
            GNodeArg(GNA_IN, "S295_Infos", 0)
        )
    );
    // Node S298_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S298_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S295_Output", 0),
            GNodeArg(GNA_IN, "Conv_378_weights", 0),
            GNodeArg(GNA_IN, "Constant__1507", 0),
            GNodeArg(GNA_OUT, "S298_Output", 0),
            GNodeArg(GNA_IN, "S298_Mul_scale", 0),
            GNodeArg(GNA_IN, "S298_Mul_shift", 0),
            GNodeArg(GNA_IN, "S298_Infos", 0)
        )
    );
    // Node S301_Conv2d_4x64x1x1 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq -3.70<(u8-104.00)*0.03556376<5.37 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S301_Conv2d_4x64x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S298_Output", 0),
            GNodeArg(GNA_IN, "Conv_382_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_reg_preds_0_bias", 0),
            GNodeArg(GNA_OUT, "S301_Output", 0),
            GNodeArg(GNA_IN, "S301_Mul_scale", 0),
            GNodeArg(GNA_IN, "S301_Mul_shift", 0),
            GNodeArg(GNA_IN, "S301_Infos", 0)
        )
    );
    // Node S304_Conv2d_1x64x1x1_Sigmoid inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq -0.90<(u8-128.00)*0.00711760<0.90 outq 0.00<(u8-0.00)*0.00392157<1.00 biasesq -359645.53<(i32-0.00)*0.00016747<359645.53
    AddNode("S304_Conv2d_1x64x1x1_Sigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S298_Output", 0),
            GNodeArg(GNA_IN, "Conv_383_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_obj_preds_0_bias", 0),
            GNodeArg(GNA_OUT, "S304_Output", 0),
            GNodeArg(GNA_IN, "S304_Mul_scale", 0),
            GNodeArg(GNA_IN, "S304_Mul_shift", 0),
            GNodeArg(GNA_IN, "S304_Infos", 0)
        )
    );
    // Node Concat_386_qin1 inq 0.00<(u8-0.00)*0.00392157<1.00 outq -3.70<(u8-104.00)*0.03556376<5.37
    AddNode("S305_Op_Concat_386_qin1",
        Bindings(3,
            GNodeArg(GNA_IN, "S304_Output", 0),
            GNodeArg(GNA_OUT, "S305_Output", 0),
            GNodeArg(GNA_IN, "S305_Infos", 0)
        )
    );
    // Node Concat_386 inq ['-3.70<(u8-104.00)*0.03556376<5.37', '-3.70<(u8-104.00)*0.03556376<5.37', '-3.70<(u8-104.00)*0.03556376<5.37'] outq ['-3.70<(u8-104.00)*0.03556376<5.37']
    AddNode("S306_Concat",
        Bindings(4,
            GNodeArg(GNA_IN, "S301_Output", 0),
            GNodeArg(GNA_IN, "S305_Output", 0),
            GNodeArg(GNA_IN, "S286_Output", 0),
            GNodeArg(GNA_OUT, "S306_Output", 0)
        )
    );
    // Node Concat_495_qin0 inq -3.70<(u8-104.00)*0.03556376<5.37 outq -4.69<(u8-102.00)*0.04595577<7.03
    AddNode("S308_Op_Concat_495_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S306_Output", 0),
            GNodeArg(GNA_OUT, "S308_Output", 0),
            GNodeArg(GNA_IN, "S308_Infos", 0)
        )
    );
    // Node S311_Conv2d_64x128x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S311_Conv2d_64x128x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S241_Output", 0),
            GNodeArg(GNA_IN, "Conv_387_weights", 0),
            GNodeArg(GNA_IN, "Constant__1510", 0),
            GNodeArg(GNA_OUT, "S311_Output", 0),
            GNodeArg(GNA_IN, "S311_Mul_scale", 0),
            GNodeArg(GNA_IN, "S311_Mul_shift", 0),
            GNodeArg(GNA_IN, "S311_Infos", 0)
        )
    );
    // Node S314_Conv2d_64x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S314_Conv2d_64x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S311_Output", 0),
            GNodeArg(GNA_IN, "Conv_391_weights", 0),
            GNodeArg(GNA_IN, "Constant__1513", 0),
            GNodeArg(GNA_OUT, "S314_Output", 0),
            GNodeArg(GNA_IN, "S314_Mul_scale", 0),
            GNodeArg(GNA_IN, "S314_Mul_shift", 0),
            GNodeArg(GNA_IN, "S314_Infos", 0)
        )
    );
    // Node S317_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S317_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S314_Output", 0),
            GNodeArg(GNA_IN, "Conv_395_weights", 0),
            GNodeArg(GNA_IN, "Constant__1516", 0),
            GNodeArg(GNA_OUT, "S317_Output", 0),
            GNodeArg(GNA_IN, "S317_Mul_scale", 0),
            GNodeArg(GNA_IN, "S317_Mul_shift", 0),
            GNodeArg(GNA_IN, "S317_Infos", 0)
        )
    );
    // Node S320_Conv2d_64x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S320_Conv2d_64x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S317_Output", 0),
            GNodeArg(GNA_IN, "Conv_399_weights", 0),
            GNodeArg(GNA_IN, "Constant__1519", 0),
            GNodeArg(GNA_OUT, "S320_Output", 0),
            GNodeArg(GNA_IN, "S320_Mul_scale", 0),
            GNodeArg(GNA_IN, "S320_Mul_shift", 0),
            GNodeArg(GNA_IN, "S320_Infos", 0)
        )
    );
    // Node S323_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S323_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S320_Output", 0),
            GNodeArg(GNA_IN, "Conv_403_weights", 0),
            GNodeArg(GNA_IN, "Constant__1522", 0),
            GNodeArg(GNA_OUT, "S323_Output", 0),
            GNodeArg(GNA_IN, "S323_Mul_scale", 0),
            GNodeArg(GNA_IN, "S323_Mul_shift", 0),
            GNodeArg(GNA_IN, "S323_Infos", 0)
        )
    );
    // Node S326_Conv2d_1x64x1x1_Sigmoid inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq -0.06<(u8-128.00)*0.00048565<0.06 outq 0.00<(u8-0.00)*0.00392157<1.00 biasesq -24539.18<(i32-0.00)*0.00001143<24539.18
    AddNode("S326_Conv2d_1x64x1x1_Sigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S323_Output", 0),
            GNodeArg(GNA_IN, "Conv_407_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_cls_preds_1_bias", 0),
            GNodeArg(GNA_OUT, "S326_Output", 0),
            GNodeArg(GNA_IN, "S326_Mul_scale", 0),
            GNodeArg(GNA_IN, "S326_Mul_shift", 0),
            GNodeArg(GNA_IN, "S326_Infos", 0)
        )
    );
    // Node Concat_428_qin2 inq 0.00<(u8-0.00)*0.00392157<1.00 outq -1.65<(u8-110.00)*0.01499095<2.17
    AddNode("S327_Op_Concat_428_qin2",
        Bindings(3,
            GNodeArg(GNA_IN, "S326_Output", 0),
            GNodeArg(GNA_OUT, "S327_Output", 0),
            GNodeArg(GNA_IN, "S327_Infos", 0)
        )
    );
    // Node S330_Conv2d_64x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S330_Conv2d_64x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S311_Output", 0),
            GNodeArg(GNA_IN, "Conv_408_weights", 0),
            GNodeArg(GNA_IN, "Constant__1525", 0),
            GNodeArg(GNA_OUT, "S330_Output", 0),
            GNodeArg(GNA_IN, "S330_Mul_scale", 0),
            GNodeArg(GNA_IN, "S330_Mul_shift", 0),
            GNodeArg(GNA_IN, "S330_Infos", 0)
        )
    );
    // Node S333_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S333_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S330_Output", 0),
            GNodeArg(GNA_IN, "Conv_412_weights", 0),
            GNodeArg(GNA_IN, "Constant__1528", 0),
            GNodeArg(GNA_OUT, "S333_Output", 0),
            GNodeArg(GNA_IN, "S333_Mul_scale", 0),
            GNodeArg(GNA_IN, "S333_Mul_shift", 0),
            GNodeArg(GNA_IN, "S333_Infos", 0)
        )
    );
    // Node S336_Conv2d_64x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S336_Conv2d_64x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S333_Output", 0),
            GNodeArg(GNA_IN, "Conv_416_weights", 0),
            GNodeArg(GNA_IN, "Constant__1531", 0),
            GNodeArg(GNA_OUT, "S336_Output", 0),
            GNodeArg(GNA_IN, "S336_Mul_scale", 0),
            GNodeArg(GNA_IN, "S336_Mul_shift", 0),
            GNodeArg(GNA_IN, "S336_Infos", 0)
        )
    );
    // Node S339_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S339_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S336_Output", 0),
            GNodeArg(GNA_IN, "Conv_420_weights", 0),
            GNodeArg(GNA_IN, "Constant__1534", 0),
            GNodeArg(GNA_OUT, "S339_Output", 0),
            GNodeArg(GNA_IN, "S339_Mul_scale", 0),
            GNodeArg(GNA_IN, "S339_Mul_shift", 0),
            GNodeArg(GNA_IN, "S339_Infos", 0)
        )
    );
    // Node S342_Conv2d_4x64x1x1 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq -1.65<(u8-110.00)*0.01499095<2.17 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S342_Conv2d_4x64x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S339_Output", 0),
            GNodeArg(GNA_IN, "Conv_424_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_reg_preds_1_bias", 0),
            GNodeArg(GNA_OUT, "S342_Output", 0),
            GNodeArg(GNA_IN, "S342_Mul_scale", 0),
            GNodeArg(GNA_IN, "S342_Mul_shift", 0),
            GNodeArg(GNA_IN, "S342_Infos", 0)
        )
    );
    // Node S345_Conv2d_1x64x1x1_Sigmoid inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq -0.99<(u8-128.00)*0.00780021<0.99 outq 0.00<(u8-0.00)*0.00392157<1.00 biasesq -394137.25<(i32-0.00)*0.00018353<394137.25
    AddNode("S345_Conv2d_1x64x1x1_Sigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S339_Output", 0),
            GNodeArg(GNA_IN, "Conv_425_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_obj_preds_1_bias", 0),
            GNodeArg(GNA_OUT, "S345_Output", 0),
            GNodeArg(GNA_IN, "S345_Mul_scale", 0),
            GNodeArg(GNA_IN, "S345_Mul_shift", 0),
            GNodeArg(GNA_IN, "S345_Infos", 0)
        )
    );
    // Node Concat_428_qin1 inq 0.00<(u8-0.00)*0.00392157<1.00 outq -1.65<(u8-110.00)*0.01499095<2.17
    AddNode("S346_Op_Concat_428_qin1",
        Bindings(3,
            GNodeArg(GNA_IN, "S345_Output", 0),
            GNodeArg(GNA_OUT, "S346_Output", 0),
            GNodeArg(GNA_IN, "S346_Infos", 0)
        )
    );
    // Node Concat_428 inq ['-1.65<(u8-110.00)*0.01499095<2.17', '-1.65<(u8-110.00)*0.01499095<2.17', '-1.65<(u8-110.00)*0.01499095<2.17'] outq ['-1.65<(u8-110.00)*0.01499095<2.17']
    AddNode("S347_Concat",
        Bindings(4,
            GNodeArg(GNA_IN, "S342_Output", 0),
            GNodeArg(GNA_IN, "S346_Output", 0),
            GNodeArg(GNA_IN, "S327_Output", 0),
            GNodeArg(GNA_OUT, "S347_Output", 0)
        )
    );
    // Node Concat_495_qin1 inq -1.65<(u8-110.00)*0.01499095<2.17 outq -4.69<(u8-102.00)*0.04595577<7.03
    AddNode("S349_Op_Concat_495_qin1",
        Bindings(3,
            GNodeArg(GNA_IN, "S347_Output", 0),
            GNodeArg(GNA_OUT, "S349_Output", 0),
            GNodeArg(GNA_IN, "S349_Infos", 0)
        )
    );
    // Node S352_Conv2d_64x256x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S352_Conv2d_64x256x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S267_Output", 0),
            GNodeArg(GNA_IN, "Conv_429_weights", 0),
            GNodeArg(GNA_IN, "Constant__1537", 0),
            GNodeArg(GNA_OUT, "S352_Output", 0),
            GNodeArg(GNA_IN, "S352_Mul_scale", 0),
            GNodeArg(GNA_IN, "S352_Mul_shift", 0),
            GNodeArg(GNA_IN, "S352_Infos", 0)
        )
    );
    // Node S355_Conv2d_64x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S355_Conv2d_64x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S352_Output", 0),
            GNodeArg(GNA_IN, "Conv_433_weights", 0),
            GNodeArg(GNA_IN, "Constant__1540", 0),
            GNodeArg(GNA_OUT, "S355_Output", 0),
            GNodeArg(GNA_IN, "S355_Mul_scale", 0),
            GNodeArg(GNA_IN, "S355_Mul_shift", 0),
            GNodeArg(GNA_IN, "S355_Infos", 0)
        )
    );
    // Node S358_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S358_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S355_Output", 0),
            GNodeArg(GNA_IN, "Conv_437_weights", 0),
            GNodeArg(GNA_IN, "Constant__1543", 0),
            GNodeArg(GNA_OUT, "S358_Output", 0),
            GNodeArg(GNA_IN, "S358_Mul_scale", 0),
            GNodeArg(GNA_IN, "S358_Mul_shift", 0),
            GNodeArg(GNA_IN, "S358_Infos", 0)
        )
    );
    // Node S361_Conv2d_64x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S361_Conv2d_64x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S358_Output", 0),
            GNodeArg(GNA_IN, "Conv_441_weights", 0),
            GNodeArg(GNA_IN, "Constant__1546", 0),
            GNodeArg(GNA_OUT, "S361_Output", 0),
            GNodeArg(GNA_IN, "S361_Mul_scale", 0),
            GNodeArg(GNA_IN, "S361_Mul_shift", 0),
            GNodeArg(GNA_IN, "S361_Infos", 0)
        )
    );
    // Node S364_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S364_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S361_Output", 0),
            GNodeArg(GNA_IN, "Conv_445_weights", 0),
            GNodeArg(GNA_IN, "Constant__1549", 0),
            GNodeArg(GNA_OUT, "S364_Output", 0),
            GNodeArg(GNA_IN, "S364_Mul_scale", 0),
            GNodeArg(GNA_IN, "S364_Mul_shift", 0),
            GNodeArg(GNA_IN, "S364_Infos", 0)
        )
    );
    // Node S367_Conv2d_1x64x1x1_Sigmoid inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq -0.11<(u8-128.00)*0.00084849<0.11 outq 0.00<(u8-0.00)*0.00392157<1.00 biasesq -42873.30<(i32-0.00)*0.00001996<42873.30
    AddNode("S367_Conv2d_1x64x1x1_Sigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S364_Output", 0),
            GNodeArg(GNA_IN, "Conv_449_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_cls_preds_2_bias", 0),
            GNodeArg(GNA_OUT, "S367_Output", 0),
            GNodeArg(GNA_IN, "S367_Mul_scale", 0),
            GNodeArg(GNA_IN, "S367_Mul_shift", 0),
            GNodeArg(GNA_IN, "S367_Infos", 0)
        )
    );
    // Node Concat_470_qin2 inq 0.00<(u8-0.00)*0.00392157<1.00 outq -4.69<(u8-102.00)*0.04595577<7.03
    AddNode("S368_Op_Concat_470_qin2",
        Bindings(3,
            GNodeArg(GNA_IN, "S367_Output", 0),
            GNodeArg(GNA_OUT, "S368_Output", 0),
            GNodeArg(GNA_IN, "S368_Infos", 0)
        )
    );
    // Node S371_Conv2d_64x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S371_Conv2d_64x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S352_Output", 0),
            GNodeArg(GNA_IN, "Conv_450_weights", 0),
            GNodeArg(GNA_IN, "Constant__1552", 0),
            GNodeArg(GNA_OUT, "S371_Output", 0),
            GNodeArg(GNA_IN, "S371_Mul_scale", 0),
            GNodeArg(GNA_IN, "S371_Mul_shift", 0),
            GNodeArg(GNA_IN, "S371_Infos", 0)
        )
    );
    // Node S374_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S374_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S371_Output", 0),
            GNodeArg(GNA_IN, "Conv_454_weights", 0),
            GNodeArg(GNA_IN, "Constant__1555", 0),
            GNodeArg(GNA_OUT, "S374_Output", 0),
            GNodeArg(GNA_IN, "S374_Mul_scale", 0),
            GNodeArg(GNA_IN, "S374_Mul_shift", 0),
            GNodeArg(GNA_IN, "S374_Infos", 0)
        )
    );
    // Node S377_Conv2d_64x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S377_Conv2d_64x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S374_Output", 0),
            GNodeArg(GNA_IN, "Conv_458_weights", 0),
            GNodeArg(GNA_IN, "Constant__1558", 0),
            GNodeArg(GNA_OUT, "S377_Output", 0),
            GNodeArg(GNA_IN, "S377_Mul_scale", 0),
            GNodeArg(GNA_IN, "S377_Mul_shift", 0),
            GNodeArg(GNA_IN, "S377_Infos", 0)
        )
    );
    // Node S380_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S380_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S377_Output", 0),
            GNodeArg(GNA_IN, "Conv_462_weights", 0),
            GNodeArg(GNA_IN, "Constant__1561", 0),
            GNodeArg(GNA_OUT, "S380_Output", 0),
            GNodeArg(GNA_IN, "S380_Mul_scale", 0),
            GNodeArg(GNA_IN, "S380_Mul_shift", 0),
            GNodeArg(GNA_IN, "S380_Infos", 0)
        )
    );
    // Node S383_Conv2d_4x64x1x1 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq -4.69<(u8-102.00)*0.04595577<7.03 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S383_Conv2d_4x64x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S380_Output", 0),
            GNodeArg(GNA_IN, "Conv_466_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_reg_preds_2_bias", 0),
            GNodeArg(GNA_OUT, "S383_Output", 0),
            GNodeArg(GNA_IN, "S383_Mul_scale", 0),
            GNodeArg(GNA_IN, "S383_Mul_shift", 0),
            GNodeArg(GNA_IN, "S383_Infos", 0)
        )
    );
    // Node S386_Conv2d_1x64x1x1_Sigmoid inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq -0.92<(u8-128.00)*0.00727906<0.92 outq 0.00<(u8-0.00)*0.00392157<1.00 biasesq -367804.00<(i32-0.00)*0.00017127<367804.00
    AddNode("S386_Conv2d_1x64x1x1_Sigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S380_Output", 0),
            GNodeArg(GNA_IN, "Conv_467_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_obj_preds_2_bias", 0),
            GNodeArg(GNA_OUT, "S386_Output", 0),
            GNodeArg(GNA_IN, "S386_Mul_scale", 0),
            GNodeArg(GNA_IN, "S386_Mul_shift", 0),
            GNodeArg(GNA_IN, "S386_Infos", 0)
        )
    );
    // Node Concat_470_qin1 inq 0.00<(u8-0.00)*0.00392157<1.00 outq -4.69<(u8-102.00)*0.04595577<7.03
    AddNode("S387_Op_Concat_470_qin1",
        Bindings(3,
            GNodeArg(GNA_IN, "S386_Output", 0),
            GNodeArg(GNA_OUT, "S387_Output", 0),
            GNodeArg(GNA_IN, "S387_Infos", 0)
        )
    );
    // Node Concat_470 inq ['-4.69<(u8-102.00)*0.04595577<7.03', '-4.69<(u8-102.00)*0.04595577<7.03', '-4.69<(u8-102.00)*0.04595577<7.03'] outq ['-4.69<(u8-102.00)*0.04595577<7.03']
    AddNode("S388_Concat",
        Bindings(4,
            GNodeArg(GNA_IN, "S383_Output", 0),
            GNodeArg(GNA_IN, "S387_Output", 0),
            GNodeArg(GNA_IN, "S368_Output", 0),
            GNodeArg(GNA_OUT, "S388_Output", 0)
        )
    );
    // Node output_1_qin0 inq -4.69<(u8-102.00)*0.04595577<7.03 outq f32
    AddNode("S391_Op_output_1_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S390_Output", 0),
            GNodeArg(GNA_OUT, "Output_1", 0),
            GNodeArg(GNA_IN, "S391_Infos", 0)
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
    mainModel(128000, 1000000, 8000000, 64*1024*1024);
    GenerateTilingCode();
    return 0;
}
