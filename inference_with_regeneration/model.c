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
    CNN_ConvolutionNE16("S4_Conv2d_16x12x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        12, 16, 160, 120,
                        KOP_CONV, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_4_fusion
    CNN_ConvolutionNE16("S7_Conv2d_16x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        16, 16, 160, 120,
                        KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_8_fusion
    CNN_ConvolutionNE16("S10_Conv2d_32x16x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        16, 32, 80, 60,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_12_fusion
    CNN_ConvolutionNE16("S13_Conv2d_16x32x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        32, 16, 80, 60,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_16_fusion
    CNN_ConvolutionNE16("S16_Conv2d_16x32x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        32, 16, 80, 60,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_20_fusion
    CNN_ConvolutionNE16("S19_Conv2d_16x16x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        16, 16, 80, 60,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_24_fusion
    CNN_ConvolutionNE16("S22_Conv2d_16x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        16, 16, 80, 60,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_28_fusion
    CNN_ConvolutionNE16("S25_Conv2d_16x16x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        16, 16, 80, 60,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S26_MatAdd_60x80x16;
    CNN_InitGenCtrl(&gen_ctrl_S26_MatAdd_60x80x16);
    CNN_SetGenCtrl(&gen_ctrl_S26_MatAdd_60x80x16, "OUTPUT_DATASIZE", AT_OPT_VAL(-1));
    CNN_SetGenCtrl(&gen_ctrl_S26_MatAdd_60x80x16, "INPUT_DATASIZE", AT_OPT_VAL(-1));
    
    // generator for Add_32
    CNN_MatAddAct_SQ8("S26_MatAdd_60x80x16", &gen_ctrl_S26_MatAdd_60x80x16, 60, 80, 16, KOP_MATADD, KOP_NONE);
    
    
    // generator for Concat_33
    CNN_ConcatLastAxis_Generator("S27_Concat", 0, -1, 4800, 16, 16, 0, 0, KOP_CONCAT);
    
    // generator for Conv_34_fusion
    CNN_ConvolutionNE16("S30_Conv2d_32x32x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        32, 32, 80, 60,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_38_fusion
    CNN_ConvolutionNE16("S33_Conv2d_32x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        32, 32, 80, 60,
                        KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_42_fusion
    CNN_ConvolutionNE16("S36_Conv2d_64x32x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        32, 64, 40, 30,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_46_fusion
    CNN_ConvolutionNE16("S39_Conv2d_32x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 32, 40, 30,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_50_fusion
    CNN_ConvolutionNE16("S42_Conv2d_32x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 32, 40, 30,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_54_fusion
    CNN_ConvolutionNE16("S45_Conv2d_32x32x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        32, 32, 40, 30,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_58_fusion
    CNN_ConvolutionNE16("S48_Conv2d_32x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        32, 32, 40, 30,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_62_fusion
    CNN_ConvolutionNE16("S51_Conv2d_32x32x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        32, 32, 40, 30,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S52_MatAdd_30x40x32;
    CNN_InitGenCtrl(&gen_ctrl_S52_MatAdd_30x40x32);
    CNN_SetGenCtrl(&gen_ctrl_S52_MatAdd_30x40x32, "OUTPUT_DATASIZE", AT_OPT_VAL(-1));
    CNN_SetGenCtrl(&gen_ctrl_S52_MatAdd_30x40x32, "INPUT_DATASIZE", AT_OPT_VAL(-1));
    
    // generator for Add_66
    CNN_MatAddAct_SQ8("S52_MatAdd_30x40x32", &gen_ctrl_S52_MatAdd_30x40x32, 30, 40, 32, KOP_MATADD, KOP_NONE);
    
    // generator for Conv_67_fusion
    CNN_ConvolutionNE16("S55_Conv2d_32x32x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        32, 32, 40, 30,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_71_fusion
    CNN_ConvolutionNE16("S58_Conv2d_32x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        32, 32, 40, 30,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_75_fusion
    CNN_ConvolutionNE16("S61_Conv2d_32x32x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        32, 32, 40, 30,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S62_MatAdd_30x40x32;
    CNN_InitGenCtrl(&gen_ctrl_S62_MatAdd_30x40x32);
    CNN_SetGenCtrl(&gen_ctrl_S62_MatAdd_30x40x32, "OUTPUT_DATASIZE", AT_OPT_VAL(-1));
    CNN_SetGenCtrl(&gen_ctrl_S62_MatAdd_30x40x32, "INPUT_DATASIZE", AT_OPT_VAL(-1));
    
    // generator for Add_79
    CNN_MatAddAct_SQ8("S62_MatAdd_30x40x32", &gen_ctrl_S62_MatAdd_30x40x32, 30, 40, 32, KOP_MATADD, KOP_NONE);
    
    // generator for Conv_80_fusion
    CNN_ConvolutionNE16("S65_Conv2d_32x32x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        32, 32, 40, 30,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_84_fusion
    CNN_ConvolutionNE16("S68_Conv2d_32x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        32, 32, 40, 30,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_88_fusion
    CNN_ConvolutionNE16("S71_Conv2d_32x32x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        32, 32, 40, 30,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S72_MatAdd_30x40x32;
    CNN_InitGenCtrl(&gen_ctrl_S72_MatAdd_30x40x32);
    CNN_SetGenCtrl(&gen_ctrl_S72_MatAdd_30x40x32, "OUTPUT_DATASIZE", AT_OPT_VAL(-1));
    CNN_SetGenCtrl(&gen_ctrl_S72_MatAdd_30x40x32, "INPUT_DATASIZE", AT_OPT_VAL(-1));
    
    // generator for Add_92
    CNN_MatAddAct_SQ8("S72_MatAdd_30x40x32", &gen_ctrl_S72_MatAdd_30x40x32, 30, 40, 32, KOP_MATADD, KOP_NONE);
    
    
    // generator for Concat_93
    CNN_ConcatLastAxis_Generator("S73_Concat", 0, -1, 1200, 32, 32, 0, 0, KOP_CONCAT);
    
    // generator for Conv_94_fusion
    CNN_ConvolutionNE16("S76_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 40, 30,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_98_fusion
    CNN_ConvolutionNE16("S79_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 40, 30,
                        KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_102_fusion
    CNN_ConvolutionNE16("S82_Conv2d_128x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 128, 20, 15,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_106_fusion
    CNN_ConvolutionNE16("S85_Conv2d_64x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 64, 20, 15,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_110_fusion
    CNN_ConvolutionNE16("S88_Conv2d_64x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 64, 20, 15,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_114_fusion
    CNN_ConvolutionNE16("S91_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 20, 15,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_118_fusion
    CNN_ConvolutionNE16("S94_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 20, 15,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_122_fusion
    CNN_ConvolutionNE16("S97_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 20, 15,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S98_MatAdd_15x20x64;
    CNN_InitGenCtrl(&gen_ctrl_S98_MatAdd_15x20x64);
    CNN_SetGenCtrl(&gen_ctrl_S98_MatAdd_15x20x64, "OUTPUT_DATASIZE", AT_OPT_VAL(-1));
    CNN_SetGenCtrl(&gen_ctrl_S98_MatAdd_15x20x64, "INPUT_DATASIZE", AT_OPT_VAL(-1));
    
    // generator for Add_126
    CNN_MatAddAct_SQ8("S98_MatAdd_15x20x64", &gen_ctrl_S98_MatAdd_15x20x64, 15, 20, 64, KOP_MATADD, KOP_NONE);
    
    // generator for Conv_127_fusion
    CNN_ConvolutionNE16("S101_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 20, 15,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_131_fusion
    CNN_ConvolutionNE16("S104_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 20, 15,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_135_fusion
    CNN_ConvolutionNE16("S107_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 20, 15,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S108_MatAdd_15x20x64;
    CNN_InitGenCtrl(&gen_ctrl_S108_MatAdd_15x20x64);
    CNN_SetGenCtrl(&gen_ctrl_S108_MatAdd_15x20x64, "OUTPUT_DATASIZE", AT_OPT_VAL(-1));
    CNN_SetGenCtrl(&gen_ctrl_S108_MatAdd_15x20x64, "INPUT_DATASIZE", AT_OPT_VAL(-1));
    
    // generator for Add_139
    CNN_MatAddAct_SQ8("S108_MatAdd_15x20x64", &gen_ctrl_S108_MatAdd_15x20x64, 15, 20, 64, KOP_MATADD, KOP_NONE);
    
    // generator for Conv_140_fusion
    CNN_ConvolutionNE16("S111_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 20, 15,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_144_fusion
    CNN_ConvolutionNE16("S114_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 20, 15,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_148_fusion
    CNN_ConvolutionNE16("S117_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 20, 15,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S118_MatAdd_15x20x64;
    CNN_InitGenCtrl(&gen_ctrl_S118_MatAdd_15x20x64);
    CNN_SetGenCtrl(&gen_ctrl_S118_MatAdd_15x20x64, "OUTPUT_DATASIZE", AT_OPT_VAL(-1));
    CNN_SetGenCtrl(&gen_ctrl_S118_MatAdd_15x20x64, "INPUT_DATASIZE", AT_OPT_VAL(-1));
    
    // generator for Add_152
    CNN_MatAddAct_SQ8("S118_MatAdd_15x20x64", &gen_ctrl_S118_MatAdd_15x20x64, 15, 20, 64, KOP_MATADD, KOP_NONE);
    
    
    // generator for Concat_153
    CNN_ConcatLastAxis_Generator("S119_Concat", 0, -1, 300, 64, 64, 0, 0, KOP_CONCAT);
    
    // generator for Conv_154_fusion
    CNN_ConvolutionNE16("S122_Conv2d_128x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 128, 20, 15,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_158_fusion
    CNN_ConvolutionNE16("S125_Conv2d_128x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 128, 20, 15,
                        KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_162_fusion
    CNN_ConvolutionNE16("S128_Conv2d_256x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 256, 10, 8,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_166_fusion
    CNN_ConvolutionNE16("S131_Conv2d_128x256x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        256, 128, 10, 8,
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
                    128, 10, 8,
                    KOP_MAXPOOL, 5, 5, 1, 1, 1, 1, 1,
                    KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S133_MaxPool_9x9;
    CNN_InitGenCtrl(&gen_ctrl_S133_MaxPool_9x9);
    CNN_SetGenCtrl(&gen_ctrl_S133_MaxPool_9x9, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S133_MaxPool_9x9, "OUTPUT_DATASIZE", AT_OPT_VAL(-1));
    CNN_SetGenCtrl(&gen_ctrl_S133_MaxPool_9x9, "INPUT_DATASIZE", AT_OPT_VAL(-1));
    // generator for MaxPool_171
    CNN_PoolAct_SQ8("S133_MaxPool_9x9", &gen_ctrl_S133_MaxPool_9x9,
                    128, 10, 8,
                    KOP_MAXPOOL, 9, 9, 1, 1, 1, 1, 1,
                    KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S134_MaxPool_13x13;
    CNN_InitGenCtrl(&gen_ctrl_S134_MaxPool_13x13);
    CNN_SetGenCtrl(&gen_ctrl_S134_MaxPool_13x13, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S134_MaxPool_13x13, "OUTPUT_DATASIZE", AT_OPT_VAL(-1));
    CNN_SetGenCtrl(&gen_ctrl_S134_MaxPool_13x13, "INPUT_DATASIZE", AT_OPT_VAL(-1));
    // generator for MaxPool_172
    CNN_PoolAct_SQ8("S134_MaxPool_13x13", &gen_ctrl_S134_MaxPool_13x13,
                    128, 10, 8,
                    KOP_MAXPOOL, 13, 13, 1, 1, 1, 1, 1,
                    KOP_NONE);
    
    
    // generator for Concat_173
    CNN_ConcatLastAxis_Generator("S135_Concat", 0, -1, 80, 128, 128, 128, 128, KOP_CONCAT);
    
    // generator for Conv_174_fusion
    CNN_ConvolutionNE16("S138_Conv2d_256x512x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        512, 256, 10, 8,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_178_fusion
    CNN_ConvolutionNE16("S141_Conv2d_128x256x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        256, 128, 10, 8,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_182_fusion
    CNN_ConvolutionNE16("S144_Conv2d_128x256x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        256, 128, 10, 8,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_186_fusion
    CNN_ConvolutionNE16("S147_Conv2d_128x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 128, 10, 8,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_190_fusion
    CNN_ConvolutionNE16("S150_Conv2d_128x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 128, 10, 8,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_194_fusion
    CNN_ConvolutionNE16("S153_Conv2d_128x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 128, 10, 8,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    
    // generator for Concat_198
    CNN_ConcatLastAxis_Generator("S154_Concat", 0, -1, 80, 128, 128, 0, 0, KOP_CONCAT);
    
    // generator for Conv_199_fusion
    CNN_ConvolutionNE16("S157_Conv2d_256x256x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        256, 256, 10, 8,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_203_fusion
    CNN_ConvolutionNE16("S160_Conv2d_128x256x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        256, 128, 10, 8,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    
    // generator for Resize_208_trans Transpose 8x10x128 -> 128x8x10 ((1, 0))
    CNN_MatTranspose("S161_Op_Resize_208_trans", 0, -1,
                      1, 128, 80, KOP_MATTRANSP);
    
    
    // generator for Resize_208
    GenerateResizeMultiChannel("S162_Op_Resize_208", 10, 8, 20, 16, 128, UNSIGNED_INOUT, KOP_NEAREST_NEIGHBOR_RESIZE);
    
    
    // generator for Slice_213_trans_in0 Transpose 128x16x20 -> 16x128x20 ((1, 0, 2))
    CNN_3DTensorPermute("S163_Op_Slice_213_trans_in0", 0, -1,
                         128, 20, 16, KOP_MATPERM_CHW2HCW);
    
    
    // generator for Slice_213_trans_out0 Transpose 15x128x20 -> 15x20x128 ((0, 2, 1))
    CNN_3DTensorPermute("S165_Op_Slice_213_trans_out0", 0, -1,
                         15, 20, 128, KOP_MATPERM_CHW2CWH);
    
    
    // generator for Concat_214
    CNN_ConcatLastAxis_Generator("S166_Concat", 0, -1, 300, 128, 128, 0, 0, KOP_CONCAT);
    
    // generator for Conv_215_fusion
    CNN_ConvolutionNE16("S170_Conv2d_64x256x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        256, 64, 20, 15,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_219_fusion
    CNN_ConvolutionNE16("S173_Conv2d_64x256x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        256, 64, 20, 15,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_223_fusion
    CNN_ConvolutionNE16("S176_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 20, 15,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_227_fusion
    CNN_ConvolutionNE16("S179_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 20, 15,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_231_fusion
    CNN_ConvolutionNE16("S182_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 20, 15,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    
    // generator for Concat_235
    CNN_ConcatLastAxis_Generator("S183_Concat", 0, -1, 300, 64, 64, 0, 0, KOP_CONCAT);
    
    // generator for Conv_236_fusion
    CNN_ConvolutionNE16("S186_Conv2d_128x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 128, 20, 15,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_240_fusion
    CNN_ConvolutionNE16("S189_Conv2d_64x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 64, 20, 15,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    
    // generator for Resize_245_trans Transpose 15x20x64 -> 64x15x20 ((1, 0))
    CNN_MatTranspose("S190_Op_Resize_245_trans", 0, -1,
                      1, 64, 300, KOP_MATTRANSP);
    
    
    // generator for Resize_245
    GenerateResizeMultiChannel("S191_Op_Resize_245", 20, 15, 40, 30, 64, UNSIGNED_INOUT, KOP_NEAREST_NEIGHBOR_RESIZE);
    
    
    // generator for Resize_245_trans_0 Transpose 64x30x40 -> 30x40x64 ((1, 0))
    CNN_MatTranspose("S192_Op_Resize_245_trans_0", 0, -1,
                      1, 1200, 64, KOP_MATTRANSP);
    
    
    // generator for Concat_246
    CNN_ConcatLastAxis_Generator("S193_Concat", 0, -1, 1200, 64, 64, 0, 0, KOP_CONCAT);
    
    // generator for Conv_247_fusion
    CNN_ConvolutionNE16("S196_Conv2d_32x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 32, 40, 30,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_251_fusion
    CNN_ConvolutionNE16("S199_Conv2d_32x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 32, 40, 30,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_255_fusion
    CNN_ConvolutionNE16("S202_Conv2d_32x32x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        32, 32, 40, 30,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_259_fusion
    CNN_ConvolutionNE16("S205_Conv2d_32x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        32, 32, 40, 30,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_263_fusion
    CNN_ConvolutionNE16("S208_Conv2d_32x32x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        32, 32, 40, 30,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    
    // generator for Concat_267
    CNN_ConcatLastAxis_Generator("S209_Concat", 0, -1, 1200, 32, 32, 0, 0, KOP_CONCAT);
    
    // generator for Conv_268_fusion
    CNN_ConvolutionNE16("S212_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 40, 30,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_272_fusion
    CNN_ConvolutionNE16("S215_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 40, 30,
                        KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_276_fusion
    CNN_ConvolutionNE16("S218_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 20, 15,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    
    // generator for Concat_280
    CNN_ConcatLastAxis_Generator("S219_Concat", 0, -1, 300, 64, 64, 0, 0, KOP_CONCAT);
    
    // generator for Conv_281_fusion
    CNN_ConvolutionNE16("S222_Conv2d_64x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 64, 20, 15,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_285_fusion
    CNN_ConvolutionNE16("S225_Conv2d_64x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 64, 20, 15,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_289_fusion
    CNN_ConvolutionNE16("S228_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 20, 15,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_293_fusion
    CNN_ConvolutionNE16("S231_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 20, 15,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_297_fusion
    CNN_ConvolutionNE16("S234_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 20, 15,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    
    // generator for Concat_301
    CNN_ConcatLastAxis_Generator("S235_Concat", 0, -1, 300, 64, 64, 0, 0, KOP_CONCAT);
    
    // generator for Conv_302_fusion
    CNN_ConvolutionNE16("S238_Conv2d_128x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 128, 20, 15,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_306_fusion
    CNN_ConvolutionNE16("S241_Conv2d_128x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 128, 20, 15,
                        KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_310_fusion
    CNN_ConvolutionNE16("S244_Conv2d_128x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 128, 10, 8,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    
    // generator for Concat_314
    CNN_ConcatLastAxis_Generator("S245_Concat", 0, -1, 80, 128, 128, 0, 0, KOP_CONCAT);
    
    // generator for Conv_315_fusion
    CNN_ConvolutionNE16("S248_Conv2d_128x256x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        256, 128, 10, 8,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_319_fusion
    CNN_ConvolutionNE16("S251_Conv2d_128x256x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        256, 128, 10, 8,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_323_fusion
    CNN_ConvolutionNE16("S254_Conv2d_128x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 128, 10, 8,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_327_fusion
    CNN_ConvolutionNE16("S257_Conv2d_128x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 128, 10, 8,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_331_fusion
    CNN_ConvolutionNE16("S260_Conv2d_128x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 128, 10, 8,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    
    // generator for Concat_335
    CNN_ConcatLastAxis_Generator("S261_Concat", 0, -1, 80, 128, 128, 0, 0, KOP_CONCAT);
    
    // generator for Conv_336_fusion
    CNN_ConvolutionNE16("S264_Conv2d_256x256x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        256, 256, 10, 8,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_340_fusion
    CNN_ConvolutionNE16("S267_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 40, 30,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_344_fusion
    CNN_ConvolutionNE16("S270_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 40, 30,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_348_fusion
    CNN_ConvolutionNE16("S273_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 40, 30,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_352_fusion
    CNN_ConvolutionNE16("S276_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 40, 30,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_356_fusion
    CNN_ConvolutionNE16("S279_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 40, 30,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_360_fusion
    CNN_ConvolutionNE16("S282_Conv2d_1x64x1x1_Sigmoid", 0,
                        -1, -1, 4, 1, 8,
                        64, 1, 40, 30,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_SIGMOID);
    
    
    // generator for Concat_381_qin2
    CNN_Convert("S283_Op_Concat_381_qin2", -1, -1, 1200, KOP_CONVERT_FP_FP_SCALE);
    
    // generator for Conv_361_fusion
    CNN_ConvolutionNE16("S286_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 40, 30,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_365_fusion
    CNN_ConvolutionNE16("S289_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 40, 30,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_369_fusion
    CNN_ConvolutionNE16("S292_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 40, 30,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_373_fusion
    CNN_ConvolutionNE16("S295_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 40, 30,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_377
    CNN_ConvolutionNE16("S298_Conv2d_4x64x1x1", 0,
                        -1, -1, 4, 1, 8,
                        64, 4, 40, 30,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    // generator for Conv_378_fusion
    CNN_ConvolutionNE16("S301_Conv2d_1x64x1x1_Sigmoid", 0,
                        -1, -1, 4, 1, 8,
                        64, 1, 40, 30,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_SIGMOID);
    
    
    // generator for Concat_381_qin1
    CNN_Convert("S302_Op_Concat_381_qin1", -1, -1, 1200, KOP_CONVERT_FP_FP_SCALE);
    
    
    // generator for Concat_381
    CNN_ConcatLastAxis_Generator("S303_Concat", 0, -1, 1200, 4, 1, 1, 0, KOP_CONCAT);
    
    
    // generator for Concat_490_qin0
    CNN_Convert("S305_Op_Concat_490_qin0", -1, -1, 7200, KOP_CONVERT_FP_FP_SCALE);
    
    // generator for Conv_382_fusion
    CNN_ConvolutionNE16("S308_Conv2d_64x128x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        128, 64, 20, 15,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_386_fusion
    CNN_ConvolutionNE16("S311_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 20, 15,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_390_fusion
    CNN_ConvolutionNE16("S314_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 20, 15,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_394_fusion
    CNN_ConvolutionNE16("S317_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 20, 15,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_398_fusion
    CNN_ConvolutionNE16("S320_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 20, 15,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_402_fusion
    CNN_ConvolutionNE16("S323_Conv2d_1x64x1x1_Sigmoid", 0,
                        -1, -1, 4, 1, 8,
                        64, 1, 20, 15,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_SIGMOID);
    
    
    // generator for Concat_423_qin2
    CNN_Convert("S324_Op_Concat_423_qin2", -1, -1, 300, KOP_CONVERT_FP_FP_SCALE);
    
    // generator for Conv_403_fusion
    CNN_ConvolutionNE16("S327_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 20, 15,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_407_fusion
    CNN_ConvolutionNE16("S330_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 20, 15,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_411_fusion
    CNN_ConvolutionNE16("S333_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 20, 15,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_415_fusion
    CNN_ConvolutionNE16("S336_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 20, 15,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_419
    CNN_ConvolutionNE16("S339_Conv2d_4x64x1x1", 0,
                        -1, -1, 4, 1, 8,
                        64, 4, 20, 15,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    // generator for Conv_420_fusion
    CNN_ConvolutionNE16("S342_Conv2d_1x64x1x1_Sigmoid", 0,
                        -1, -1, 4, 1, 8,
                        64, 1, 20, 15,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_SIGMOID);
    
    
    // generator for Concat_423_qin1
    CNN_Convert("S343_Op_Concat_423_qin1", -1, -1, 300, KOP_CONVERT_FP_FP_SCALE);
    
    
    // generator for Concat_423
    CNN_ConcatLastAxis_Generator("S344_Concat", 0, -1, 300, 4, 1, 1, 0, KOP_CONCAT);
    
    // generator for Conv_424_fusion
    CNN_ConvolutionNE16("S348_Conv2d_64x256x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        256, 64, 10, 8,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_428_fusion
    CNN_ConvolutionNE16("S351_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 10, 8,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_432_fusion
    CNN_ConvolutionNE16("S354_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 10, 8,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_436_fusion
    CNN_ConvolutionNE16("S357_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 10, 8,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_440_fusion
    CNN_ConvolutionNE16("S360_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 10, 8,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_444_fusion
    CNN_ConvolutionNE16("S363_Conv2d_1x64x1x1_Sigmoid", 0,
                        -1, -1, 4, 1, 8,
                        64, 1, 10, 8,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_SIGMOID);
    
    
    // generator for Concat_465_qin2
    CNN_Convert("S364_Op_Concat_465_qin2", -1, -1, 80, KOP_CONVERT_FP_FP_SCALE);
    
    // generator for Conv_445_fusion
    CNN_ConvolutionNE16("S367_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 10, 8,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_449_fusion
    CNN_ConvolutionNE16("S370_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 10, 8,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_453_fusion
    CNN_ConvolutionNE16("S373_Conv2d_64x1x3x3_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 10, 8,
                        KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_457_fusion
    CNN_ConvolutionNE16("S376_Conv2d_64x64x1x1_Relu6", 0,
                        -1, -1, 4, 1, 8,
                        64, 64, 10, 8,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_RELU);
    
    // generator for Conv_461
    CNN_ConvolutionNE16("S379_Conv2d_4x64x1x1", 0,
                        -1, -1, 4, 1, 8,
                        64, 4, 10, 8,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_NONE);
    
    // generator for Conv_462_fusion
    CNN_ConvolutionNE16("S382_Conv2d_1x64x1x1_Sigmoid", 0,
                        -1, -1, 4, 1, 8,
                        64, 1, 10, 8,
                        KOP_CONV, 1, 1, 1, 1, 1, 1, 0, 0,
                        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                        KOP_SIGMOID);
    
    
    // generator for Concat_465_qin1
    CNN_Convert("S383_Op_Concat_465_qin1", -1, -1, 80, KOP_CONVERT_FP_FP_SCALE);
    
    
    // generator for Concat_465
    CNN_ConcatLastAxis_Generator("S384_Concat", 0, -1, 80, 4, 1, 1, 0, KOP_CONCAT);
    
    
    // generator for Concat_490_qin2
    CNN_Convert("S386_Op_Concat_490_qin2", -1, -1, 480, KOP_CONVERT_FP_FP_SCALE);
    
    
    // generator for output_1_qin0
    CNN_Convert("S388_Op_output_1_qin0", -1, 4, 9480, KOP_CONVERT_FP_FL);
    

#define GRAPH
#ifdef GRAPH
    CreateGraph("mainCNN",
        /* Arguments either passed or globals */
            CArgs(586,
                TCArgInfo("unsigned char * __restrict__", "Input_1", ARG_SCOPE_ARG_ALLOC, ARG_DIR_IN, AT_MEM_L2, AT_MEM_L2, 0),
                TCArgInfo("unsigned char * __restrict__", "Conv_0_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_0_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1247", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1247.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S4_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S4_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S4_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S4_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S4_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S4_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_4_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_4_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1250", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1250.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S7_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S7_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S7_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S7_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S7_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S7_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_8_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_8_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1253", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1253.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S10_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S10_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S10_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S10_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S10_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S10_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_12_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_12_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1256", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1256.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S13_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S13_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S13_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S13_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S13_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S13_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_16_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_16_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1259", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1259.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S16_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S16_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S16_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S16_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04065 out: 0.04065  actscale: [1] actscalen: [0] a0: [0] b0: [148] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S16_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S16_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_20_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_20_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1262", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1262.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S19_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S19_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S19_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S19_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S19_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S19_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_24_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_24_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1265", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1265.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S22_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S22_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S22_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S22_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S22_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S22_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_28_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_28_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1268", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1268.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S25_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S25_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S25_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S25_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S25_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S25_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [1] IN1SCALEN: [0] OUTSCALE: [37] OUTSCALEN: [6] ADD_BIAS: 0
                TCArgInfo("signed char * __restrict__", "S26_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S26_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_34_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_34_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1271", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1271.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S30_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S30_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S30_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S30_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S30_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S30_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_38_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_38_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1274", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1274.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S33_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S33_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S33_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S33_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S33_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S33_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_42_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_42_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1277", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1277.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S36_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S36_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S36_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S36_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S36_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S36_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_46_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_46_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1280", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1280.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S39_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S39_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S39_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S39_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S39_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S39_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_50_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_50_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1283", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1283.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S42_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S42_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S42_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S42_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.06783 out: 0.06783  actscale: [1] actscalen: [0] a0: [0] b0: [88] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S42_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S42_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_54_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_54_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1286", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1286.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S45_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S45_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S45_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S45_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S45_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S45_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_58_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_58_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1289", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1289.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S48_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S48_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S48_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S48_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S48_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S48_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_62_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_62_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1292", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1292.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S51_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S51_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S51_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S51_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S51_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S51_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [1] IN1SCALEN: [0] OUTSCALE: [43] OUTSCALEN: [6] ADD_BIAS: 0
                TCArgInfo("signed char * __restrict__", "S52_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S52_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_67_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_67_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1295", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1295.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S55_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S55_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S55_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S55_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S55_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S55_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_71_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_71_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1298", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1298.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S58_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S58_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S58_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S58_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S58_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S58_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_75_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_75_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1301", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1301.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S61_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S61_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S61_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S61_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S61_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S61_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [191] IN1SCALEN: [7] OUTSCALE: [227] OUTSCALEN: [9] ADD_BIAS: 0
                TCArgInfo("signed char * __restrict__", "S62_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S62_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_80_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_80_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1304", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1304.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S65_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S65_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S65_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S65_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S65_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S65_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_84_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_84_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1307", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1307.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S68_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S68_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S68_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S68_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S68_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S68_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_88_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_88_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1310", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1310.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S71_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S71_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S71_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S71_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S71_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S71_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [9] IN1SCALEN: [2] OUTSCALE: [89] OUTSCALEN: [8] ADD_BIAS: 0
                TCArgInfo("signed char * __restrict__", "S72_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S72_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_94_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_94_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1313", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1313.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S76_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S76_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S76_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S76_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S76_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S76_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_98_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_98_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1316", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1316.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S79_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S79_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S79_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S79_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S79_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S79_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_102_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_102_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1319", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1319.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S82_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S82_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S82_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S82_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S82_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S82_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_106_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_106_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1322", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1322.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S85_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S85_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S85_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S85_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S85_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S85_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_110_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_110_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1325", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1325.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S88_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S88_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S88_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S88_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.07059 out: 0.07059  actscale: [1] actscalen: [0] a0: [0] b0: [85] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S88_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S88_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_114_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_114_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1328", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1328.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S91_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S91_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S91_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S91_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S91_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S91_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_118_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_118_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1331", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1331.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S94_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S94_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S94_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S94_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S94_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S94_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_122_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_122_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1334", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1334.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S97_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S97_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S97_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S97_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S97_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S97_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [1] IN1SCALEN: [0] OUTSCALE: [83] OUTSCALEN: [7] ADD_BIAS: 0
                TCArgInfo("signed char * __restrict__", "S98_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S98_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_127_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_127_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1337", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1337.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S101_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S101_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S101_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S101_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S101_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S101_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_131_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_131_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1340", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1340.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S104_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S104_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S104_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S104_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S104_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S104_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_135_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_135_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1343", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1343.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S107_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S107_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S107_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S107_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S107_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S107_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [99] IN1SCALEN: [6] OUTSCALE: [233] OUTSCALEN: [9] ADD_BIAS: 0
                TCArgInfo("signed char * __restrict__", "S108_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S108_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_140_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_140_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1346", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1346.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S111_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S111_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S111_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S111_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S111_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S111_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_144_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_144_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1349", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1349.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S114_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S114_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S114_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S114_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S114_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S114_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_148_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_148_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1352", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1352.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S117_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S117_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S117_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S117_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S117_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S117_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [35] IN1SCALEN: [4] OUTSCALE: [171] OUTSCALEN: [9] ADD_BIAS: 0
                TCArgInfo("signed char * __restrict__", "S118_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S118_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_154_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_154_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1355", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1355.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S122_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S122_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S122_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S122_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S122_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S122_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_158_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_158_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1358", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1358.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S125_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S125_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S125_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S125_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S125_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S125_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_162_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_162_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1361", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1361.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S128_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S128_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S128_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S128_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S128_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S128_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_166_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_166_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1364", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1364.tensor", 1, 1, 32, 0)),
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
                TCArgInfo("signed int * __restrict__", "Constant__1367", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1367.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S138_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S138_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S138_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S138_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S138_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S138_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_178_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_178_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1370", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1370.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S141_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S141_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S141_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S141_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S141_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S141_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_182_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_182_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1373", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1373.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S144_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S144_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S144_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S144_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S144_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S144_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_186_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_186_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1376", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1376.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S147_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S147_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S147_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S147_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S147_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S147_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_190_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_190_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1379", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1379.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S150_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S150_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S150_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S150_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S150_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S150_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_194_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_194_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1382", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1382.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S153_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S153_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S153_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S153_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S153_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S153_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_199_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_199_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1385", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1385.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S157_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S157_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S157_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S157_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S157_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S157_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_203_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_203_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1388", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1388.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S160_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S160_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S160_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S160_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S160_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S160_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_215_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_215_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1391", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1391.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S170_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S170_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S170_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S170_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S170_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S170_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_219_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_219_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1394", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1394.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S173_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S173_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S173_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S173_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S173_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S173_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_223_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_223_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1397", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1397.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S176_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S176_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S176_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S176_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S176_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S176_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_227_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_227_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1400", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1400.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S179_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S179_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S179_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S179_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S179_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S179_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_231_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_231_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1403", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1403.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S182_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S182_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S182_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S182_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S182_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S182_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_236_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_236_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1406", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1406.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S186_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S186_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S186_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S186_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S186_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S186_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_240_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_240_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1409", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1409.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S189_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S189_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S189_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S189_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S189_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S189_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_247_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_247_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1412", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1412.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S196_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S196_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S196_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S196_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S196_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S196_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_251_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_251_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1415", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1415.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S199_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S199_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S199_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S199_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S199_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S199_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_255_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_255_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1418", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1418.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S202_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S202_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S202_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S202_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S202_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S202_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_259_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_259_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1421", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1421.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S205_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S205_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S205_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S205_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S205_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S205_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_263_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_263_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1424", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1424.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S208_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S208_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S208_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S208_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S208_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S208_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_268_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_268_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1427", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1427.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S212_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S212_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S212_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S212_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S212_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S212_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_272_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_272_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1430", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1430.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S215_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S215_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S215_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S215_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S215_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S215_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_276_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_276_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1433", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1433.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S218_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S218_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S218_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S218_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S218_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S218_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_281_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_281_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1436", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1436.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S222_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S222_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S222_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S222_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S222_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S222_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_285_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_285_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1439", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1439.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S225_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S225_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S225_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S225_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S225_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S225_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_289_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_289_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1442", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1442.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S228_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S228_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S228_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S228_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S228_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S228_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_293_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_293_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1445", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1445.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S231_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S231_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S231_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S231_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S231_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S231_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_297_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_297_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1448", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1448.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S234_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S234_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S234_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S234_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S234_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S234_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_302_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_302_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1451", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1451.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S238_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S238_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S238_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S238_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S238_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S238_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_306_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_306_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1454", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1454.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S241_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S241_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S241_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S241_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S241_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S241_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_310_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_310_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1457", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1457.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S244_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S244_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S244_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S244_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S244_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S244_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_315_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_315_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1460", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1460.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S248_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S248_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S248_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S248_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S248_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S248_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_319_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_319_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1463", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1463.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S251_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S251_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S251_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S251_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S251_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S251_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_323_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_323_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1466", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1466.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S254_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S254_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S254_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S254_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S254_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S254_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_327_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_327_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1469", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1469.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S257_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S257_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S257_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S257_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S257_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S257_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_331_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_331_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1472", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1472.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S260_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S260_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S260_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S260_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S260_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S260_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_336_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_336_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1475", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1475.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S264_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S264_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S264_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S264_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S264_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S264_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_340_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_340_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1478", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1478.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S267_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S267_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S267_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S267_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S267_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S267_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_344_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_344_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1481", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1481.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S270_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S270_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S270_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S270_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S270_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S270_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_348_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_348_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1484", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1484.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S273_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S273_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S273_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S273_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S273_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S273_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_352_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_352_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1487", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1487.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S276_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S276_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S276_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S276_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S276_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S276_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_356_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_356_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1490", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1490.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S279_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S279_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S279_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S279_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S279_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S279_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_360_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_360_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_head_cls_preds_0_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_head_cls_preds_0_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S282_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S282_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S282_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S282_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00024 out: 0.00392  actscale: [255] actscalen: [15] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S282_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S282_Infos.tensor", 1, 1, 8, 0)),
                // in q: 0.00<(u8-0.00)*0.00392157<1.00 out_q: -2.03<(u8-86.00)*0.02357237<3.98
                TCArgInfo("signed char * __restrict__", "S283_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S283_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_361_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_361_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1493", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1493.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S286_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S286_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S286_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S286_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S286_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S286_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_365_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_365_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1496", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1496.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S289_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S289_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S289_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S289_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S289_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S289_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_369_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_369_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1499", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1499.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S292_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S292_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S292_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S292_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S292_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S292_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_373_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_373_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1502", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1502.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S295_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S295_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S295_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S295_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S295_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S295_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_377_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_377_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_head_reg_preds_0_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_head_reg_preds_0_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S298_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S298_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S298_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S298_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S298_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S298_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_378_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_378_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_head_obj_preds_0_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_head_obj_preds_0_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S301_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S301_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S301_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S301_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00024 out: 0.00392  actscale: [255] actscalen: [15] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S301_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S301_Infos.tensor", 1, 1, 8, 0)),
                // in q: 0.00<(u8-0.00)*0.00392157<1.00 out_q: -2.03<(u8-86.00)*0.02357237<3.98
                TCArgInfo("signed char * __restrict__", "S302_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S302_Infos.tensor", 1, 1, 8, 0)),
                // in q: -2.03<(u8-86.00)*0.02357237<3.98 out_q: -6.49<(u8-117.00)*0.05544940<7.65
                TCArgInfo("signed char * __restrict__", "S305_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S305_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_382_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_382_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1505", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1505.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S308_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S308_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S308_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S308_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S308_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S308_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_386_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_386_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1508", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1508.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S311_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S311_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S311_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S311_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S311_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S311_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_390_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_390_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1511", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1511.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S314_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S314_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S314_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S314_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S314_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S314_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_394_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_394_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1514", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1514.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S317_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S317_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S317_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S317_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S317_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S317_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_398_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_398_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1517", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1517.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S320_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S320_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S320_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S320_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S320_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S320_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_402_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_402_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_head_cls_preds_1_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_head_cls_preds_1_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S323_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S323_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S323_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S323_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00024 out: 0.00392  actscale: [255] actscalen: [15] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S323_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S323_Infos.tensor", 1, 1, 8, 0)),
                // in q: 0.00<(u8-0.00)*0.00392157<1.00 out_q: -6.49<(u8-117.00)*0.05544940<7.65
                TCArgInfo("signed char * __restrict__", "S324_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S324_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_403_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_403_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1520", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1520.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S327_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S327_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S327_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S327_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S327_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S327_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_407_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_407_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1523", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1523.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S330_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S330_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S330_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S330_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S330_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S330_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_411_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_411_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1526", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1526.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S333_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S333_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S333_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S333_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S333_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S333_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_415_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_415_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1529", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1529.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S336_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S336_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S336_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S336_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S336_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S336_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_419_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_419_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_head_reg_preds_1_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_head_reg_preds_1_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S339_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S339_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S339_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S339_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S339_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S339_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_420_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_420_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_head_obj_preds_1_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_head_obj_preds_1_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S342_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S342_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S342_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S342_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00024 out: 0.00392  actscale: [255] actscalen: [15] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S342_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S342_Infos.tensor", 1, 1, 8, 0)),
                // in q: 0.00<(u8-0.00)*0.00392157<1.00 out_q: -6.49<(u8-117.00)*0.05544940<7.65
                TCArgInfo("signed char * __restrict__", "S343_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S343_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_424_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_424_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1532", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1532.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S348_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S348_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S348_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S348_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S348_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S348_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_428_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_428_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1535", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1535.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S351_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S351_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S351_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S351_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S351_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S351_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_432_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_432_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1538", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1538.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S354_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S354_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S354_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S354_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S354_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S354_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_436_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_436_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1541", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1541.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S357_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S357_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S357_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S357_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S357_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S357_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_440_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_440_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1544", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1544.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S360_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S360_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S360_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S360_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S360_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S360_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_444_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_444_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_head_cls_preds_2_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_head_cls_preds_2_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S363_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S363_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S363_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S363_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00024 out: 0.00392  actscale: [255] actscalen: [15] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S363_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S363_Infos.tensor", 1, 1, 8, 0)),
                // in q: 0.00<(u8-0.00)*0.00392157<1.00 out_q: -2.12<(u8-173.00)*0.01224599<1.00
                TCArgInfo("signed char * __restrict__", "S364_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S364_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_445_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_445_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1547", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1547.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S367_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S367_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S367_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S367_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S367_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S367_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_449_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_449_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1550", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1550.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S370_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S370_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S370_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S370_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S370_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S370_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_453_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_453_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1553", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1553.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S373_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S373_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S373_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S373_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S373_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S373_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_457_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_457_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1556", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant__1556.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S376_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S376_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S376_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S376_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [0] b0: [255] c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S376_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S376_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_461_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_461_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_head_reg_preds_2_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_head_reg_preds_2_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S379_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S379_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S379_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S379_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S379_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S379_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("unsigned char * __restrict__", "Conv_462_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Conv_462_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_head_obj_preds_2_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_head_obj_preds_2_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S382_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S382_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S382_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S382_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00024 out: 0.00392  actscale: [255] actscalen: [15] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0 NE16_PADVAL: [0] NE16_WOFFSET: [-128]
                TCArgInfo("signed char * __restrict__", "S382_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S382_Infos.tensor", 1, 1, 8, 0)),
                // in q: 0.00<(u8-0.00)*0.00392157<1.00 out_q: -2.12<(u8-173.00)*0.01224599<1.00
                TCArgInfo("signed char * __restrict__", "S383_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S383_Infos.tensor", 1, 1, 8, 0)),
                // in q: -2.12<(u8-173.00)*0.01224599<1.00 out_q: -6.49<(u8-117.00)*0.05544940<7.65
                TCArgInfo("signed char * __restrict__", "S386_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S386_Infos.tensor", 1, 1, 8, 0)),
                // in q: -6.49<(u8-117.00)*0.05544940<7.65 out_q: f32
                TCArgInfo("signed char * __restrict__", "S388_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("S388_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("float * __restrict__", "Output_1", ARG_SCOPE_ARG, ARG_DIR_OUT, AT_MEM_L2, AT_MEM_L2, 0)
            ),
        /* Locals, allocated dynamically */
        CArgs(152,
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
            TCArgInfo("unsigned char * __restrict__", "S193_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S196_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S199_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S202_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S205_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S208_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S209_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S212_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S215_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S218_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S219_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S222_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S225_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S228_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S231_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S234_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S235_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S238_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S241_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S244_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S245_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S248_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S251_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S254_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S257_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S260_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S261_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S264_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S267_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S270_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S273_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S276_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S279_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S282_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S283_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S286_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S289_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S292_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S295_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S298_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S301_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S302_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S303_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S308_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S311_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S314_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S317_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S320_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S323_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S324_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S327_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S330_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S333_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S336_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S339_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S342_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S343_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S348_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S351_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S354_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S357_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S360_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S363_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S364_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S367_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S370_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S373_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S376_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S379_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S382_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S383_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S384_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("unsigned char * __restrict__", "S387_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0)
        )
    );

    // Stacked tensors for concats and splits
    AddStackedTensors("S387_Output", 3, "S305_Output", "S344_Output", "S386_Output");
    AddStackedTensors("S163_Output", 2, "S164_Output_0", AT_UnusedStackMember("Slice_213_unused", 2560));

    // Node S4_Conv2d_16x12x3x3_Relu6 inq 0.00<(u8-0.00)*1.00000000<255.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S4_Conv2d_16x12x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "Input_1", 0),
            GNodeArg(GNA_IN, "Conv_0_weights", 0),
            GNodeArg(GNA_IN, "Constant__1247", 0),
            GNodeArg(GNA_OUT, "S4_Output", 0),
            GNodeArg(GNA_IN, "S4_Mul_scale", 0),
            GNodeArg(GNA_IN, "S4_Mul_shift", 0),
            GNodeArg(GNA_IN, "S4_Infos", 0)
        )
    );
    // Node S7_Conv2d_16x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S7_Conv2d_16x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S4_Output", 0),
            GNodeArg(GNA_IN, "Conv_4_weights", 0),
            GNodeArg(GNA_IN, "Constant__1250", 0),
            GNodeArg(GNA_OUT, "S7_Output", 0),
            GNodeArg(GNA_IN, "S7_Mul_scale", 0),
            GNodeArg(GNA_IN, "S7_Mul_shift", 0),
            GNodeArg(GNA_IN, "S7_Infos", 0)
        )
    );
    // Node S10_Conv2d_32x16x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S10_Conv2d_32x16x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S7_Output", 0),
            GNodeArg(GNA_IN, "Conv_8_weights", 0),
            GNodeArg(GNA_IN, "Constant__1253", 0),
            GNodeArg(GNA_OUT, "S10_Output", 0),
            GNodeArg(GNA_IN, "S10_Mul_scale", 0),
            GNodeArg(GNA_IN, "S10_Mul_shift", 0),
            GNodeArg(GNA_IN, "S10_Infos", 0)
        )
    );
    // Node S13_Conv2d_16x32x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S13_Conv2d_16x32x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S10_Output", 0),
            GNodeArg(GNA_IN, "Conv_12_weights", 0),
            GNodeArg(GNA_IN, "Constant__1256", 0),
            GNodeArg(GNA_OUT, "S13_Output", 0),
            GNodeArg(GNA_IN, "S13_Mul_scale", 0),
            GNodeArg(GNA_IN, "S13_Mul_shift", 0),
            GNodeArg(GNA_IN, "S13_Infos", 0)
        )
    );
    // Node S16_Conv2d_16x32x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.04064656<10.36 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S16_Conv2d_16x32x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S10_Output", 0),
            GNodeArg(GNA_IN, "Conv_16_weights", 0),
            GNodeArg(GNA_IN, "Constant__1259", 0),
            GNodeArg(GNA_OUT, "S16_Output", 0),
            GNodeArg(GNA_IN, "S16_Mul_scale", 0),
            GNodeArg(GNA_IN, "S16_Mul_shift", 0),
            GNodeArg(GNA_IN, "S16_Infos", 0)
        )
    );
    // Node S19_Conv2d_16x16x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S19_Conv2d_16x16x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S13_Output", 0),
            GNodeArg(GNA_IN, "Conv_20_weights", 0),
            GNodeArg(GNA_IN, "Constant__1262", 0),
            GNodeArg(GNA_OUT, "S19_Output", 0),
            GNodeArg(GNA_IN, "S19_Mul_scale", 0),
            GNodeArg(GNA_IN, "S19_Mul_shift", 0),
            GNodeArg(GNA_IN, "S19_Infos", 0)
        )
    );
    // Node S22_Conv2d_16x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S22_Conv2d_16x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S19_Output", 0),
            GNodeArg(GNA_IN, "Conv_24_weights", 0),
            GNodeArg(GNA_IN, "Constant__1265", 0),
            GNodeArg(GNA_OUT, "S22_Output", 0),
            GNodeArg(GNA_IN, "S22_Mul_scale", 0),
            GNodeArg(GNA_IN, "S22_Mul_shift", 0),
            GNodeArg(GNA_IN, "S22_Infos", 0)
        )
    );
    // Node S25_Conv2d_16x16x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S25_Conv2d_16x16x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S22_Output", 0),
            GNodeArg(GNA_IN, "Conv_28_weights", 0),
            GNodeArg(GNA_IN, "Constant__1268", 0),
            GNodeArg(GNA_OUT, "S25_Output", 0),
            GNodeArg(GNA_IN, "S25_Mul_scale", 0),
            GNodeArg(GNA_IN, "S25_Mul_shift", 0),
            GNodeArg(GNA_IN, "S25_Infos", 0)
        )
    );
    // Node S26_MatAdd_60x80x16 in1q 0.00<(u8-0.00)*0.02352941<6.00 forced
    //   in2q 0.00<(u8-0.00)*0.02352941<6.00 forced
    //   outq 0.00<(u8-0.00)*0.04064656<10.36 forced scaled input 0 is node input 0
    AddNode("S26_MatAdd_60x80x16",
        Bindings(4,
            GNodeArg(GNA_IN, "S25_Output", 0),
            GNodeArg(GNA_IN, "S13_Output", 0),
            GNodeArg(GNA_OUT, "S26_Output", 0),
            GNodeArg(GNA_IN, "S26_Infos", 0)
        )
    );
    // Node Concat_33 inq ['0.00<(u8-0.00)*0.04064656<10.36 forced', '0.00<(u8-0.00)*0.04064656<10.36 forced'] outq ['0.00<(u8-0.00)*0.04064656<10.36 forced']
    AddNode("S27_Concat",
        Bindings(3,
            GNodeArg(GNA_IN, "S26_Output", 0),
            GNodeArg(GNA_IN, "S16_Output", 0),
            GNodeArg(GNA_OUT, "S27_Output", 0)
        )
    );
    // Node S30_Conv2d_32x32x1x1_Relu6 inq 0.00<(u8-0.00)*0.04064656<10.36 forced weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S30_Conv2d_32x32x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S27_Output", 0),
            GNodeArg(GNA_IN, "Conv_34_weights", 0),
            GNodeArg(GNA_IN, "Constant__1271", 0),
            GNodeArg(GNA_OUT, "S30_Output", 0),
            GNodeArg(GNA_IN, "S30_Mul_scale", 0),
            GNodeArg(GNA_IN, "S30_Mul_shift", 0),
            GNodeArg(GNA_IN, "S30_Infos", 0)
        )
    );
    // Node S33_Conv2d_32x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S33_Conv2d_32x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S30_Output", 0),
            GNodeArg(GNA_IN, "Conv_38_weights", 0),
            GNodeArg(GNA_IN, "Constant__1274", 0),
            GNodeArg(GNA_OUT, "S33_Output", 0),
            GNodeArg(GNA_IN, "S33_Mul_scale", 0),
            GNodeArg(GNA_IN, "S33_Mul_shift", 0),
            GNodeArg(GNA_IN, "S33_Infos", 0)
        )
    );
    // Node S36_Conv2d_64x32x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S36_Conv2d_64x32x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S33_Output", 0),
            GNodeArg(GNA_IN, "Conv_42_weights", 0),
            GNodeArg(GNA_IN, "Constant__1277", 0),
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
            GNodeArg(GNA_IN, "Constant__1280", 0),
            GNodeArg(GNA_OUT, "S39_Output", 0),
            GNodeArg(GNA_IN, "S39_Mul_scale", 0),
            GNodeArg(GNA_IN, "S39_Mul_shift", 0),
            GNodeArg(GNA_IN, "S39_Infos", 0)
        )
    );
    // Node S42_Conv2d_32x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.06783044<17.30 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S42_Conv2d_32x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S36_Output", 0),
            GNodeArg(GNA_IN, "Conv_50_weights", 0),
            GNodeArg(GNA_IN, "Constant__1283", 0),
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
            GNodeArg(GNA_IN, "Constant__1286", 0),
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
            GNodeArg(GNA_IN, "Constant__1289", 0),
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
            GNodeArg(GNA_IN, "Constant__1292", 0),
            GNodeArg(GNA_OUT, "S51_Output", 0),
            GNodeArg(GNA_IN, "S51_Mul_scale", 0),
            GNodeArg(GNA_IN, "S51_Mul_shift", 0),
            GNodeArg(GNA_IN, "S51_Infos", 0)
        )
    );
    // Node S52_MatAdd_30x40x32 in1q 0.00<(u8-0.00)*0.02352941<6.00 forced
    //   in2q 0.00<(u8-0.00)*0.02352941<6.00 forced
    //   outq 0.00<(u8-0.00)*0.03505225<8.94 forced scaled input 0 is node input 0
    AddNode("S52_MatAdd_30x40x32",
        Bindings(4,
            GNodeArg(GNA_IN, "S51_Output", 0),
            GNodeArg(GNA_IN, "S39_Output", 0),
            GNodeArg(GNA_OUT, "S52_Output", 0),
            GNodeArg(GNA_IN, "S52_Infos", 0)
        )
    );
    // Node S55_Conv2d_32x32x1x1_Relu6 inq 0.00<(u8-0.00)*0.03505225<8.94 forced weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S55_Conv2d_32x32x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S52_Output", 0),
            GNodeArg(GNA_IN, "Conv_67_weights", 0),
            GNodeArg(GNA_IN, "Constant__1295", 0),
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
            GNodeArg(GNA_IN, "Constant__1298", 0),
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
            GNodeArg(GNA_IN, "Constant__1301", 0),
            GNodeArg(GNA_OUT, "S61_Output", 0),
            GNodeArg(GNA_IN, "S61_Mul_scale", 0),
            GNodeArg(GNA_IN, "S61_Mul_shift", 0),
            GNodeArg(GNA_IN, "S61_Infos", 0)
        )
    );
    // Node S62_MatAdd_30x40x32 in1q 0.00<(u8-0.00)*0.03505225<8.94 forced
    //   in2q 0.00<(u8-0.00)*0.02352941<6.00 forced
    //   outq 0.00<(u8-0.00)*0.05308677<13.54 forced scaled input 0 is node input 1
    AddNode("S62_MatAdd_30x40x32",
        Bindings(4,
            GNodeArg(GNA_IN, "S52_Output", 0),
            GNodeArg(GNA_IN, "S61_Output", 0),
            GNodeArg(GNA_OUT, "S62_Output", 0),
            GNodeArg(GNA_IN, "S62_Infos", 0)
        )
    );
    // Node S65_Conv2d_32x32x1x1_Relu6 inq 0.00<(u8-0.00)*0.05308677<13.54 forced weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S65_Conv2d_32x32x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S62_Output", 0),
            GNodeArg(GNA_IN, "Conv_80_weights", 0),
            GNodeArg(GNA_IN, "Constant__1304", 0),
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
            GNodeArg(GNA_IN, "Constant__1307", 0),
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
            GNodeArg(GNA_IN, "Constant__1310", 0),
            GNodeArg(GNA_OUT, "S71_Output", 0),
            GNodeArg(GNA_IN, "S71_Mul_scale", 0),
            GNodeArg(GNA_IN, "S71_Mul_shift", 0),
            GNodeArg(GNA_IN, "S71_Infos", 0)
        )
    );
    // Node S72_MatAdd_30x40x32 in1q 0.00<(u8-0.00)*0.05308677<13.54 forced
    //   in2q 0.00<(u8-0.00)*0.02352941<6.00 forced
    //   outq 0.00<(u8-0.00)*0.06783044<17.30 forced scaled input 0 is node input 1
    AddNode("S72_MatAdd_30x40x32",
        Bindings(4,
            GNodeArg(GNA_IN, "S62_Output", 0),
            GNodeArg(GNA_IN, "S71_Output", 0),
            GNodeArg(GNA_OUT, "S72_Output", 0),
            GNodeArg(GNA_IN, "S72_Infos", 0)
        )
    );
    // Node Concat_93 inq ['0.00<(u8-0.00)*0.06783044<17.30 forced', '0.00<(u8-0.00)*0.06783044<17.30 forced'] outq ['0.00<(u8-0.00)*0.06783044<17.30 forced']
    AddNode("S73_Concat",
        Bindings(3,
            GNodeArg(GNA_IN, "S72_Output", 0),
            GNodeArg(GNA_IN, "S42_Output", 0),
            GNodeArg(GNA_OUT, "S73_Output", 0)
        )
    );
    // Node S76_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.06783044<17.30 forced weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S76_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S73_Output", 0),
            GNodeArg(GNA_IN, "Conv_94_weights", 0),
            GNodeArg(GNA_IN, "Constant__1313", 0),
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
            GNodeArg(GNA_IN, "Constant__1316", 0),
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
            GNodeArg(GNA_IN, "Constant__1319", 0),
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
            GNodeArg(GNA_IN, "Constant__1322", 0),
            GNodeArg(GNA_OUT, "S85_Output", 0),
            GNodeArg(GNA_IN, "S85_Mul_scale", 0),
            GNodeArg(GNA_IN, "S85_Mul_shift", 0),
            GNodeArg(GNA_IN, "S85_Infos", 0)
        )
    );
    // Node S88_Conv2d_64x128x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.07058824<18.00 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S88_Conv2d_64x128x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S82_Output", 0),
            GNodeArg(GNA_IN, "Conv_110_weights", 0),
            GNodeArg(GNA_IN, "Constant__1325", 0),
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
            GNodeArg(GNA_IN, "Constant__1328", 0),
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
            GNodeArg(GNA_IN, "Constant__1331", 0),
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
            GNodeArg(GNA_IN, "Constant__1334", 0),
            GNodeArg(GNA_OUT, "S97_Output", 0),
            GNodeArg(GNA_IN, "S97_Mul_scale", 0),
            GNodeArg(GNA_IN, "S97_Mul_shift", 0),
            GNodeArg(GNA_IN, "S97_Infos", 0)
        )
    );
    // Node S98_MatAdd_15x20x64 in1q 0.00<(u8-0.00)*0.02352941<6.00 forced
    //   in2q 0.00<(u8-0.00)*0.02352941<6.00 forced
    //   outq 0.00<(u8-0.00)*0.03637297<9.28 forced scaled input 0 is node input 0
    AddNode("S98_MatAdd_15x20x64",
        Bindings(4,
            GNodeArg(GNA_IN, "S97_Output", 0),
            GNodeArg(GNA_IN, "S85_Output", 0),
            GNodeArg(GNA_OUT, "S98_Output", 0),
            GNodeArg(GNA_IN, "S98_Infos", 0)
        )
    );
    // Node S101_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.03637297<9.28 forced weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S101_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S98_Output", 0),
            GNodeArg(GNA_IN, "Conv_127_weights", 0),
            GNodeArg(GNA_IN, "Constant__1337", 0),
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
            GNodeArg(GNA_IN, "Constant__1340", 0),
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
            GNodeArg(GNA_IN, "Constant__1343", 0),
            GNodeArg(GNA_OUT, "S107_Output", 0),
            GNodeArg(GNA_IN, "S107_Mul_scale", 0),
            GNodeArg(GNA_IN, "S107_Mul_shift", 0),
            GNodeArg(GNA_IN, "S107_Infos", 0)
        )
    );
    // Node S108_MatAdd_15x20x64 in1q 0.00<(u8-0.00)*0.03637297<9.28 forced
    //   in2q 0.00<(u8-0.00)*0.02352941<6.00 forced
    //   outq 0.00<(u8-0.00)*0.05164580<13.17 forced scaled input 0 is node input 1
    AddNode("S108_MatAdd_15x20x64",
        Bindings(4,
            GNodeArg(GNA_IN, "S98_Output", 0),
            GNodeArg(GNA_IN, "S107_Output", 0),
            GNodeArg(GNA_OUT, "S108_Output", 0),
            GNodeArg(GNA_IN, "S108_Infos", 0)
        )
    );
    // Node S111_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.05164580<13.17 forced weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S111_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S108_Output", 0),
            GNodeArg(GNA_IN, "Conv_140_weights", 0),
            GNodeArg(GNA_IN, "Constant__1346", 0),
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
            GNodeArg(GNA_IN, "Constant__1349", 0),
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
            GNodeArg(GNA_IN, "Constant__1352", 0),
            GNodeArg(GNA_OUT, "S117_Output", 0),
            GNodeArg(GNA_IN, "S117_Mul_scale", 0),
            GNodeArg(GNA_IN, "S117_Mul_shift", 0),
            GNodeArg(GNA_IN, "S117_Infos", 0)
        )
    );
    // Node S118_MatAdd_15x20x64 in1q 0.00<(u8-0.00)*0.05164580<13.17 forced
    //   in2q 0.00<(u8-0.00)*0.02352941<6.00 forced
    //   outq 0.00<(u8-0.00)*0.07058824<18.00 forced scaled input 0 is node input 1
    AddNode("S118_MatAdd_15x20x64",
        Bindings(4,
            GNodeArg(GNA_IN, "S108_Output", 0),
            GNodeArg(GNA_IN, "S117_Output", 0),
            GNodeArg(GNA_OUT, "S118_Output", 0),
            GNodeArg(GNA_IN, "S118_Infos", 0)
        )
    );
    // Node Concat_153 inq ['0.00<(u8-0.00)*0.07058824<18.00 forced', '0.00<(u8-0.00)*0.07058824<18.00 forced'] outq ['0.00<(u8-0.00)*0.07058824<18.00 forced']
    AddNode("S119_Concat",
        Bindings(3,
            GNodeArg(GNA_IN, "S118_Output", 0),
            GNodeArg(GNA_IN, "S88_Output", 0),
            GNodeArg(GNA_OUT, "S119_Output", 0)
        )
    );
    // Node S122_Conv2d_128x128x1x1_Relu6 inq 0.00<(u8-0.00)*0.07058824<18.00 forced weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S122_Conv2d_128x128x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S119_Output", 0),
            GNodeArg(GNA_IN, "Conv_154_weights", 0),
            GNodeArg(GNA_IN, "Constant__1355", 0),
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
            GNodeArg(GNA_IN, "Constant__1358", 0),
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
            GNodeArg(GNA_IN, "Constant__1361", 0),
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
            GNodeArg(GNA_IN, "Constant__1364", 0),
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
            GNodeArg(GNA_IN, "Constant__1367", 0),
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
            GNodeArg(GNA_IN, "Constant__1370", 0),
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
            GNodeArg(GNA_IN, "Constant__1373", 0),
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
            GNodeArg(GNA_IN, "Constant__1376", 0),
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
            GNodeArg(GNA_IN, "Constant__1379", 0),
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
            GNodeArg(GNA_IN, "Constant__1382", 0),
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
            GNodeArg(GNA_IN, "Constant__1385", 0),
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
            GNodeArg(GNA_IN, "Constant__1388", 0),
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
            GNodeArg(GNA_IN, "Constant__1391", 0),
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
            GNodeArg(GNA_IN, "Constant__1394", 0),
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
            GNodeArg(GNA_IN, "Constant__1397", 0),
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
            GNodeArg(GNA_IN, "Constant__1400", 0),
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
            GNodeArg(GNA_IN, "Constant__1403", 0),
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
            GNodeArg(GNA_IN, "Constant__1406", 0),
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
            GNodeArg(GNA_IN, "Constant__1409", 0),
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
    // Node Resize_245_trans_0 inq 0.00<(u8-0.00)*0.02352941<6.00 outq 0.00<(u8-0.00)*0.02352941<6.00
    AddNode("S192_Op_Resize_245_trans_0",
        Bindings(2,
            GNodeArg(GNA_IN, "S191_Output", 0),
            GNodeArg(GNA_OUT, "S192_Output", 0)
        )
    );
    // Node Concat_246 inq ['0.00<(u8-0.00)*0.02352941<6.00', '0.00<(u8-0.00)*0.02352941<6.00'] outq ['0.00<(u8-0.00)*0.02352941<6.00']
    AddNode("S193_Concat",
        Bindings(3,
            GNodeArg(GNA_IN, "S192_Output", 0),
            GNodeArg(GNA_IN, "S76_Output", 0),
            GNodeArg(GNA_OUT, "S193_Output", 0)
        )
    );
    // Node S196_Conv2d_32x128x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S196_Conv2d_32x128x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S193_Output", 0),
            GNodeArg(GNA_IN, "Conv_247_weights", 0),
            GNodeArg(GNA_IN, "Constant__1412", 0),
            GNodeArg(GNA_OUT, "S196_Output", 0),
            GNodeArg(GNA_IN, "S196_Mul_scale", 0),
            GNodeArg(GNA_IN, "S196_Mul_shift", 0),
            GNodeArg(GNA_IN, "S196_Infos", 0)
        )
    );
    // Node S199_Conv2d_32x128x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S199_Conv2d_32x128x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S193_Output", 0),
            GNodeArg(GNA_IN, "Conv_251_weights", 0),
            GNodeArg(GNA_IN, "Constant__1415", 0),
            GNodeArg(GNA_OUT, "S199_Output", 0),
            GNodeArg(GNA_IN, "S199_Mul_scale", 0),
            GNodeArg(GNA_IN, "S199_Mul_shift", 0),
            GNodeArg(GNA_IN, "S199_Infos", 0)
        )
    );
    // Node S202_Conv2d_32x32x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S202_Conv2d_32x32x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S196_Output", 0),
            GNodeArg(GNA_IN, "Conv_255_weights", 0),
            GNodeArg(GNA_IN, "Constant__1418", 0),
            GNodeArg(GNA_OUT, "S202_Output", 0),
            GNodeArg(GNA_IN, "S202_Mul_scale", 0),
            GNodeArg(GNA_IN, "S202_Mul_shift", 0),
            GNodeArg(GNA_IN, "S202_Infos", 0)
        )
    );
    // Node S205_Conv2d_32x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S205_Conv2d_32x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S202_Output", 0),
            GNodeArg(GNA_IN, "Conv_259_weights", 0),
            GNodeArg(GNA_IN, "Constant__1421", 0),
            GNodeArg(GNA_OUT, "S205_Output", 0),
            GNodeArg(GNA_IN, "S205_Mul_scale", 0),
            GNodeArg(GNA_IN, "S205_Mul_shift", 0),
            GNodeArg(GNA_IN, "S205_Infos", 0)
        )
    );
    // Node S208_Conv2d_32x32x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S208_Conv2d_32x32x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S205_Output", 0),
            GNodeArg(GNA_IN, "Conv_263_weights", 0),
            GNodeArg(GNA_IN, "Constant__1424", 0),
            GNodeArg(GNA_OUT, "S208_Output", 0),
            GNodeArg(GNA_IN, "S208_Mul_scale", 0),
            GNodeArg(GNA_IN, "S208_Mul_shift", 0),
            GNodeArg(GNA_IN, "S208_Infos", 0)
        )
    );
    // Node Concat_267 inq ['0.00<(u8-0.00)*0.02352941<6.00', '0.00<(u8-0.00)*0.02352941<6.00'] outq ['0.00<(u8-0.00)*0.02352941<6.00']
    AddNode("S209_Concat",
        Bindings(3,
            GNodeArg(GNA_IN, "S208_Output", 0),
            GNodeArg(GNA_IN, "S199_Output", 0),
            GNodeArg(GNA_OUT, "S209_Output", 0)
        )
    );
    // Node S212_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S212_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S209_Output", 0),
            GNodeArg(GNA_IN, "Conv_268_weights", 0),
            GNodeArg(GNA_IN, "Constant__1427", 0),
            GNodeArg(GNA_OUT, "S212_Output", 0),
            GNodeArg(GNA_IN, "S212_Mul_scale", 0),
            GNodeArg(GNA_IN, "S212_Mul_shift", 0),
            GNodeArg(GNA_IN, "S212_Infos", 0)
        )
    );
    // Node S215_Conv2d_64x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S215_Conv2d_64x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S212_Output", 0),
            GNodeArg(GNA_IN, "Conv_272_weights", 0),
            GNodeArg(GNA_IN, "Constant__1430", 0),
            GNodeArg(GNA_OUT, "S215_Output", 0),
            GNodeArg(GNA_IN, "S215_Mul_scale", 0),
            GNodeArg(GNA_IN, "S215_Mul_shift", 0),
            GNodeArg(GNA_IN, "S215_Infos", 0)
        )
    );
    // Node S218_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S218_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S215_Output", 0),
            GNodeArg(GNA_IN, "Conv_276_weights", 0),
            GNodeArg(GNA_IN, "Constant__1433", 0),
            GNodeArg(GNA_OUT, "S218_Output", 0),
            GNodeArg(GNA_IN, "S218_Mul_scale", 0),
            GNodeArg(GNA_IN, "S218_Mul_shift", 0),
            GNodeArg(GNA_IN, "S218_Infos", 0)
        )
    );
    // Node Concat_280 inq ['0.00<(u8-0.00)*0.02352941<6.00', '0.00<(u8-0.00)*0.02352941<6.00'] outq ['0.00<(u8-0.00)*0.02352941<6.00']
    AddNode("S219_Concat",
        Bindings(3,
            GNodeArg(GNA_IN, "S218_Output", 0),
            GNodeArg(GNA_IN, "S189_Output", 0),
            GNodeArg(GNA_OUT, "S219_Output", 0)
        )
    );
    // Node S222_Conv2d_64x128x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S222_Conv2d_64x128x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S219_Output", 0),
            GNodeArg(GNA_IN, "Conv_281_weights", 0),
            GNodeArg(GNA_IN, "Constant__1436", 0),
            GNodeArg(GNA_OUT, "S222_Output", 0),
            GNodeArg(GNA_IN, "S222_Mul_scale", 0),
            GNodeArg(GNA_IN, "S222_Mul_shift", 0),
            GNodeArg(GNA_IN, "S222_Infos", 0)
        )
    );
    // Node S225_Conv2d_64x128x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S225_Conv2d_64x128x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S219_Output", 0),
            GNodeArg(GNA_IN, "Conv_285_weights", 0),
            GNodeArg(GNA_IN, "Constant__1439", 0),
            GNodeArg(GNA_OUT, "S225_Output", 0),
            GNodeArg(GNA_IN, "S225_Mul_scale", 0),
            GNodeArg(GNA_IN, "S225_Mul_shift", 0),
            GNodeArg(GNA_IN, "S225_Infos", 0)
        )
    );
    // Node S228_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S228_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S222_Output", 0),
            GNodeArg(GNA_IN, "Conv_289_weights", 0),
            GNodeArg(GNA_IN, "Constant__1442", 0),
            GNodeArg(GNA_OUT, "S228_Output", 0),
            GNodeArg(GNA_IN, "S228_Mul_scale", 0),
            GNodeArg(GNA_IN, "S228_Mul_shift", 0),
            GNodeArg(GNA_IN, "S228_Infos", 0)
        )
    );
    // Node S231_Conv2d_64x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S231_Conv2d_64x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S228_Output", 0),
            GNodeArg(GNA_IN, "Conv_293_weights", 0),
            GNodeArg(GNA_IN, "Constant__1445", 0),
            GNodeArg(GNA_OUT, "S231_Output", 0),
            GNodeArg(GNA_IN, "S231_Mul_scale", 0),
            GNodeArg(GNA_IN, "S231_Mul_shift", 0),
            GNodeArg(GNA_IN, "S231_Infos", 0)
        )
    );
    // Node S234_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S234_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S231_Output", 0),
            GNodeArg(GNA_IN, "Conv_297_weights", 0),
            GNodeArg(GNA_IN, "Constant__1448", 0),
            GNodeArg(GNA_OUT, "S234_Output", 0),
            GNodeArg(GNA_IN, "S234_Mul_scale", 0),
            GNodeArg(GNA_IN, "S234_Mul_shift", 0),
            GNodeArg(GNA_IN, "S234_Infos", 0)
        )
    );
    // Node Concat_301 inq ['0.00<(u8-0.00)*0.02352941<6.00', '0.00<(u8-0.00)*0.02352941<6.00'] outq ['0.00<(u8-0.00)*0.02352941<6.00']
    AddNode("S235_Concat",
        Bindings(3,
            GNodeArg(GNA_IN, "S234_Output", 0),
            GNodeArg(GNA_IN, "S225_Output", 0),
            GNodeArg(GNA_OUT, "S235_Output", 0)
        )
    );
    // Node S238_Conv2d_128x128x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S238_Conv2d_128x128x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S235_Output", 0),
            GNodeArg(GNA_IN, "Conv_302_weights", 0),
            GNodeArg(GNA_IN, "Constant__1451", 0),
            GNodeArg(GNA_OUT, "S238_Output", 0),
            GNodeArg(GNA_IN, "S238_Mul_scale", 0),
            GNodeArg(GNA_IN, "S238_Mul_shift", 0),
            GNodeArg(GNA_IN, "S238_Infos", 0)
        )
    );
    // Node S241_Conv2d_128x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S241_Conv2d_128x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S238_Output", 0),
            GNodeArg(GNA_IN, "Conv_306_weights", 0),
            GNodeArg(GNA_IN, "Constant__1454", 0),
            GNodeArg(GNA_OUT, "S241_Output", 0),
            GNodeArg(GNA_IN, "S241_Mul_scale", 0),
            GNodeArg(GNA_IN, "S241_Mul_shift", 0),
            GNodeArg(GNA_IN, "S241_Infos", 0)
        )
    );
    // Node S244_Conv2d_128x128x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S244_Conv2d_128x128x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S241_Output", 0),
            GNodeArg(GNA_IN, "Conv_310_weights", 0),
            GNodeArg(GNA_IN, "Constant__1457", 0),
            GNodeArg(GNA_OUT, "S244_Output", 0),
            GNodeArg(GNA_IN, "S244_Mul_scale", 0),
            GNodeArg(GNA_IN, "S244_Mul_shift", 0),
            GNodeArg(GNA_IN, "S244_Infos", 0)
        )
    );
    // Node Concat_314 inq ['0.00<(u8-0.00)*0.02352941<6.00', '0.00<(u8-0.00)*0.02352941<6.00'] outq ['0.00<(u8-0.00)*0.02352941<6.00']
    AddNode("S245_Concat",
        Bindings(3,
            GNodeArg(GNA_IN, "S244_Output", 0),
            GNodeArg(GNA_IN, "S160_Output", 0),
            GNodeArg(GNA_OUT, "S245_Output", 0)
        )
    );
    // Node S248_Conv2d_128x256x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S248_Conv2d_128x256x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S245_Output", 0),
            GNodeArg(GNA_IN, "Conv_315_weights", 0),
            GNodeArg(GNA_IN, "Constant__1460", 0),
            GNodeArg(GNA_OUT, "S248_Output", 0),
            GNodeArg(GNA_IN, "S248_Mul_scale", 0),
            GNodeArg(GNA_IN, "S248_Mul_shift", 0),
            GNodeArg(GNA_IN, "S248_Infos", 0)
        )
    );
    // Node S251_Conv2d_128x256x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S251_Conv2d_128x256x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S245_Output", 0),
            GNodeArg(GNA_IN, "Conv_319_weights", 0),
            GNodeArg(GNA_IN, "Constant__1463", 0),
            GNodeArg(GNA_OUT, "S251_Output", 0),
            GNodeArg(GNA_IN, "S251_Mul_scale", 0),
            GNodeArg(GNA_IN, "S251_Mul_shift", 0),
            GNodeArg(GNA_IN, "S251_Infos", 0)
        )
    );
    // Node S254_Conv2d_128x128x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S254_Conv2d_128x128x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S248_Output", 0),
            GNodeArg(GNA_IN, "Conv_323_weights", 0),
            GNodeArg(GNA_IN, "Constant__1466", 0),
            GNodeArg(GNA_OUT, "S254_Output", 0),
            GNodeArg(GNA_IN, "S254_Mul_scale", 0),
            GNodeArg(GNA_IN, "S254_Mul_shift", 0),
            GNodeArg(GNA_IN, "S254_Infos", 0)
        )
    );
    // Node S257_Conv2d_128x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S257_Conv2d_128x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S254_Output", 0),
            GNodeArg(GNA_IN, "Conv_327_weights", 0),
            GNodeArg(GNA_IN, "Constant__1469", 0),
            GNodeArg(GNA_OUT, "S257_Output", 0),
            GNodeArg(GNA_IN, "S257_Mul_scale", 0),
            GNodeArg(GNA_IN, "S257_Mul_shift", 0),
            GNodeArg(GNA_IN, "S257_Infos", 0)
        )
    );
    // Node S260_Conv2d_128x128x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S260_Conv2d_128x128x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S257_Output", 0),
            GNodeArg(GNA_IN, "Conv_331_weights", 0),
            GNodeArg(GNA_IN, "Constant__1472", 0),
            GNodeArg(GNA_OUT, "S260_Output", 0),
            GNodeArg(GNA_IN, "S260_Mul_scale", 0),
            GNodeArg(GNA_IN, "S260_Mul_shift", 0),
            GNodeArg(GNA_IN, "S260_Infos", 0)
        )
    );
    // Node Concat_335 inq ['0.00<(u8-0.00)*0.02352941<6.00', '0.00<(u8-0.00)*0.02352941<6.00'] outq ['0.00<(u8-0.00)*0.02352941<6.00']
    AddNode("S261_Concat",
        Bindings(3,
            GNodeArg(GNA_IN, "S260_Output", 0),
            GNodeArg(GNA_IN, "S251_Output", 0),
            GNodeArg(GNA_OUT, "S261_Output", 0)
        )
    );
    // Node S264_Conv2d_256x256x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S264_Conv2d_256x256x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S261_Output", 0),
            GNodeArg(GNA_IN, "Conv_336_weights", 0),
            GNodeArg(GNA_IN, "Constant__1475", 0),
            GNodeArg(GNA_OUT, "S264_Output", 0),
            GNodeArg(GNA_IN, "S264_Mul_scale", 0),
            GNodeArg(GNA_IN, "S264_Mul_shift", 0),
            GNodeArg(GNA_IN, "S264_Infos", 0)
        )
    );
    // Node S267_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S267_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S212_Output", 0),
            GNodeArg(GNA_IN, "Conv_340_weights", 0),
            GNodeArg(GNA_IN, "Constant__1478", 0),
            GNodeArg(GNA_OUT, "S267_Output", 0),
            GNodeArg(GNA_IN, "S267_Mul_scale", 0),
            GNodeArg(GNA_IN, "S267_Mul_shift", 0),
            GNodeArg(GNA_IN, "S267_Infos", 0)
        )
    );
    // Node S270_Conv2d_64x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S270_Conv2d_64x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S267_Output", 0),
            GNodeArg(GNA_IN, "Conv_344_weights", 0),
            GNodeArg(GNA_IN, "Constant__1481", 0),
            GNodeArg(GNA_OUT, "S270_Output", 0),
            GNodeArg(GNA_IN, "S270_Mul_scale", 0),
            GNodeArg(GNA_IN, "S270_Mul_shift", 0),
            GNodeArg(GNA_IN, "S270_Infos", 0)
        )
    );
    // Node S273_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S273_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S270_Output", 0),
            GNodeArg(GNA_IN, "Conv_348_weights", 0),
            GNodeArg(GNA_IN, "Constant__1484", 0),
            GNodeArg(GNA_OUT, "S273_Output", 0),
            GNodeArg(GNA_IN, "S273_Mul_scale", 0),
            GNodeArg(GNA_IN, "S273_Mul_shift", 0),
            GNodeArg(GNA_IN, "S273_Infos", 0)
        )
    );
    // Node S276_Conv2d_64x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S276_Conv2d_64x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S273_Output", 0),
            GNodeArg(GNA_IN, "Conv_352_weights", 0),
            GNodeArg(GNA_IN, "Constant__1487", 0),
            GNodeArg(GNA_OUT, "S276_Output", 0),
            GNodeArg(GNA_IN, "S276_Mul_scale", 0),
            GNodeArg(GNA_IN, "S276_Mul_shift", 0),
            GNodeArg(GNA_IN, "S276_Infos", 0)
        )
    );
    // Node S279_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S279_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S276_Output", 0),
            GNodeArg(GNA_IN, "Conv_356_weights", 0),
            GNodeArg(GNA_IN, "Constant__1490", 0),
            GNodeArg(GNA_OUT, "S279_Output", 0),
            GNodeArg(GNA_IN, "S279_Mul_scale", 0),
            GNodeArg(GNA_IN, "S279_Mul_shift", 0),
            GNodeArg(GNA_IN, "S279_Infos", 0)
        )
    );
    // Node S282_Conv2d_1x64x1x1_Sigmoid inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq -0.09<(u8-128.00)*0.00070337<0.09 outq 0.00<(u8-0.00)*0.00392157<1.00 biasesq -35540.37<(i32-0.00)*0.00001655<35540.37
    AddNode("S282_Conv2d_1x64x1x1_Sigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S279_Output", 0),
            GNodeArg(GNA_IN, "Conv_360_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_cls_preds_0_bias", 0),
            GNodeArg(GNA_OUT, "S282_Output", 0),
            GNodeArg(GNA_IN, "S282_Mul_scale", 0),
            GNodeArg(GNA_IN, "S282_Mul_shift", 0),
            GNodeArg(GNA_IN, "S282_Infos", 0)
        )
    );
    // Node Concat_381_qin2 inq 0.00<(u8-0.00)*0.00392157<1.00 outq -2.03<(u8-86.00)*0.02357237<3.98
    AddNode("S283_Op_Concat_381_qin2",
        Bindings(3,
            GNodeArg(GNA_IN, "S282_Output", 0),
            GNodeArg(GNA_OUT, "S283_Output", 0),
            GNodeArg(GNA_IN, "S283_Infos", 0)
        )
    );
    // Node S286_Conv2d_64x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S286_Conv2d_64x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S267_Output", 0),
            GNodeArg(GNA_IN, "Conv_361_weights", 0),
            GNodeArg(GNA_IN, "Constant__1493", 0),
            GNodeArg(GNA_OUT, "S286_Output", 0),
            GNodeArg(GNA_IN, "S286_Mul_scale", 0),
            GNodeArg(GNA_IN, "S286_Mul_shift", 0),
            GNodeArg(GNA_IN, "S286_Infos", 0)
        )
    );
    // Node S289_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S289_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S286_Output", 0),
            GNodeArg(GNA_IN, "Conv_365_weights", 0),
            GNodeArg(GNA_IN, "Constant__1496", 0),
            GNodeArg(GNA_OUT, "S289_Output", 0),
            GNodeArg(GNA_IN, "S289_Mul_scale", 0),
            GNodeArg(GNA_IN, "S289_Mul_shift", 0),
            GNodeArg(GNA_IN, "S289_Infos", 0)
        )
    );
    // Node S292_Conv2d_64x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S292_Conv2d_64x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S289_Output", 0),
            GNodeArg(GNA_IN, "Conv_369_weights", 0),
            GNodeArg(GNA_IN, "Constant__1499", 0),
            GNodeArg(GNA_OUT, "S292_Output", 0),
            GNodeArg(GNA_IN, "S292_Mul_scale", 0),
            GNodeArg(GNA_IN, "S292_Mul_shift", 0),
            GNodeArg(GNA_IN, "S292_Infos", 0)
        )
    );
    // Node S295_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S295_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S292_Output", 0),
            GNodeArg(GNA_IN, "Conv_373_weights", 0),
            GNodeArg(GNA_IN, "Constant__1502", 0),
            GNodeArg(GNA_OUT, "S295_Output", 0),
            GNodeArg(GNA_IN, "S295_Mul_scale", 0),
            GNodeArg(GNA_IN, "S295_Mul_shift", 0),
            GNodeArg(GNA_IN, "S295_Infos", 0)
        )
    );
    // Node S298_Conv2d_4x64x1x1 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq -2.03<(u8-86.00)*0.02357237<3.98 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S298_Conv2d_4x64x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S295_Output", 0),
            GNodeArg(GNA_IN, "Conv_377_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_reg_preds_0_bias", 0),
            GNodeArg(GNA_OUT, "S298_Output", 0),
            GNodeArg(GNA_IN, "S298_Mul_scale", 0),
            GNodeArg(GNA_IN, "S298_Mul_shift", 0),
            GNodeArg(GNA_IN, "S298_Infos", 0)
        )
    );
    // Node S301_Conv2d_1x64x1x1_Sigmoid inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq -0.99<(u8-128.00)*0.00782754<0.99 outq 0.00<(u8-0.00)*0.00392157<1.00 biasesq -395517.79<(i32-0.00)*0.00018418<395517.79
    AddNode("S301_Conv2d_1x64x1x1_Sigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S295_Output", 0),
            GNodeArg(GNA_IN, "Conv_378_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_obj_preds_0_bias", 0),
            GNodeArg(GNA_OUT, "S301_Output", 0),
            GNodeArg(GNA_IN, "S301_Mul_scale", 0),
            GNodeArg(GNA_IN, "S301_Mul_shift", 0),
            GNodeArg(GNA_IN, "S301_Infos", 0)
        )
    );
    // Node Concat_381_qin1 inq 0.00<(u8-0.00)*0.00392157<1.00 outq -2.03<(u8-86.00)*0.02357237<3.98
    AddNode("S302_Op_Concat_381_qin1",
        Bindings(3,
            GNodeArg(GNA_IN, "S301_Output", 0),
            GNodeArg(GNA_OUT, "S302_Output", 0),
            GNodeArg(GNA_IN, "S302_Infos", 0)
        )
    );
    // Node Concat_381 inq ['-2.03<(u8-86.00)*0.02357237<3.98', '-2.03<(u8-86.00)*0.02357237<3.98', '-2.03<(u8-86.00)*0.02357237<3.98'] outq ['-2.03<(u8-86.00)*0.02357237<3.98']
    AddNode("S303_Concat",
        Bindings(4,
            GNodeArg(GNA_IN, "S298_Output", 0),
            GNodeArg(GNA_IN, "S302_Output", 0),
            GNodeArg(GNA_IN, "S283_Output", 0),
            GNodeArg(GNA_OUT, "S303_Output", 0)
        )
    );
    // Node Concat_490_qin0 inq -2.03<(u8-86.00)*0.02357237<3.98 outq -6.49<(u8-117.00)*0.05544940<7.65
    AddNode("S305_Op_Concat_490_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S303_Output", 0),
            GNodeArg(GNA_OUT, "S305_Output", 0),
            GNodeArg(GNA_IN, "S305_Infos", 0)
        )
    );
    // Node S308_Conv2d_64x128x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S308_Conv2d_64x128x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S238_Output", 0),
            GNodeArg(GNA_IN, "Conv_382_weights", 0),
            GNodeArg(GNA_IN, "Constant__1505", 0),
            GNodeArg(GNA_OUT, "S308_Output", 0),
            GNodeArg(GNA_IN, "S308_Mul_scale", 0),
            GNodeArg(GNA_IN, "S308_Mul_shift", 0),
            GNodeArg(GNA_IN, "S308_Infos", 0)
        )
    );
    // Node S311_Conv2d_64x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S311_Conv2d_64x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S308_Output", 0),
            GNodeArg(GNA_IN, "Conv_386_weights", 0),
            GNodeArg(GNA_IN, "Constant__1508", 0),
            GNodeArg(GNA_OUT, "S311_Output", 0),
            GNodeArg(GNA_IN, "S311_Mul_scale", 0),
            GNodeArg(GNA_IN, "S311_Mul_shift", 0),
            GNodeArg(GNA_IN, "S311_Infos", 0)
        )
    );
    // Node S314_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S314_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S311_Output", 0),
            GNodeArg(GNA_IN, "Conv_390_weights", 0),
            GNodeArg(GNA_IN, "Constant__1511", 0),
            GNodeArg(GNA_OUT, "S314_Output", 0),
            GNodeArg(GNA_IN, "S314_Mul_scale", 0),
            GNodeArg(GNA_IN, "S314_Mul_shift", 0),
            GNodeArg(GNA_IN, "S314_Infos", 0)
        )
    );
    // Node S317_Conv2d_64x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S317_Conv2d_64x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S314_Output", 0),
            GNodeArg(GNA_IN, "Conv_394_weights", 0),
            GNodeArg(GNA_IN, "Constant__1514", 0),
            GNodeArg(GNA_OUT, "S317_Output", 0),
            GNodeArg(GNA_IN, "S317_Mul_scale", 0),
            GNodeArg(GNA_IN, "S317_Mul_shift", 0),
            GNodeArg(GNA_IN, "S317_Infos", 0)
        )
    );
    // Node S320_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S320_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S317_Output", 0),
            GNodeArg(GNA_IN, "Conv_398_weights", 0),
            GNodeArg(GNA_IN, "Constant__1517", 0),
            GNodeArg(GNA_OUT, "S320_Output", 0),
            GNodeArg(GNA_IN, "S320_Mul_scale", 0),
            GNodeArg(GNA_IN, "S320_Mul_shift", 0),
            GNodeArg(GNA_IN, "S320_Infos", 0)
        )
    );
    // Node S323_Conv2d_1x64x1x1_Sigmoid inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq -0.12<(u8-128.00)*0.00091073<0.12 outq 0.00<(u8-0.00)*0.00392157<1.00 biasesq -46018.31<(i32-0.00)*0.00002143<46018.31
    AddNode("S323_Conv2d_1x64x1x1_Sigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S320_Output", 0),
            GNodeArg(GNA_IN, "Conv_402_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_cls_preds_1_bias", 0),
            GNodeArg(GNA_OUT, "S323_Output", 0),
            GNodeArg(GNA_IN, "S323_Mul_scale", 0),
            GNodeArg(GNA_IN, "S323_Mul_shift", 0),
            GNodeArg(GNA_IN, "S323_Infos", 0)
        )
    );
    // Node Concat_423_qin2 inq 0.00<(u8-0.00)*0.00392157<1.00 outq -6.49<(u8-117.00)*0.05544940<7.65
    AddNode("S324_Op_Concat_423_qin2",
        Bindings(3,
            GNodeArg(GNA_IN, "S323_Output", 0),
            GNodeArg(GNA_OUT, "S324_Output", 0),
            GNodeArg(GNA_IN, "S324_Infos", 0)
        )
    );
    // Node S327_Conv2d_64x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S327_Conv2d_64x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S308_Output", 0),
            GNodeArg(GNA_IN, "Conv_403_weights", 0),
            GNodeArg(GNA_IN, "Constant__1520", 0),
            GNodeArg(GNA_OUT, "S327_Output", 0),
            GNodeArg(GNA_IN, "S327_Mul_scale", 0),
            GNodeArg(GNA_IN, "S327_Mul_shift", 0),
            GNodeArg(GNA_IN, "S327_Infos", 0)
        )
    );
    // Node S330_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S330_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S327_Output", 0),
            GNodeArg(GNA_IN, "Conv_407_weights", 0),
            GNodeArg(GNA_IN, "Constant__1523", 0),
            GNodeArg(GNA_OUT, "S330_Output", 0),
            GNodeArg(GNA_IN, "S330_Mul_scale", 0),
            GNodeArg(GNA_IN, "S330_Mul_shift", 0),
            GNodeArg(GNA_IN, "S330_Infos", 0)
        )
    );
    // Node S333_Conv2d_64x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S333_Conv2d_64x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S330_Output", 0),
            GNodeArg(GNA_IN, "Conv_411_weights", 0),
            GNodeArg(GNA_IN, "Constant__1526", 0),
            GNodeArg(GNA_OUT, "S333_Output", 0),
            GNodeArg(GNA_IN, "S333_Mul_scale", 0),
            GNodeArg(GNA_IN, "S333_Mul_shift", 0),
            GNodeArg(GNA_IN, "S333_Infos", 0)
        )
    );
    // Node S336_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S336_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S333_Output", 0),
            GNodeArg(GNA_IN, "Conv_415_weights", 0),
            GNodeArg(GNA_IN, "Constant__1529", 0),
            GNodeArg(GNA_OUT, "S336_Output", 0),
            GNodeArg(GNA_IN, "S336_Mul_scale", 0),
            GNodeArg(GNA_IN, "S336_Mul_shift", 0),
            GNodeArg(GNA_IN, "S336_Infos", 0)
        )
    );
    // Node S339_Conv2d_4x64x1x1 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq -6.49<(u8-117.00)*0.05544940<7.65 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S339_Conv2d_4x64x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S336_Output", 0),
            GNodeArg(GNA_IN, "Conv_419_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_reg_preds_1_bias", 0),
            GNodeArg(GNA_OUT, "S339_Output", 0),
            GNodeArg(GNA_IN, "S339_Mul_scale", 0),
            GNodeArg(GNA_IN, "S339_Mul_shift", 0),
            GNodeArg(GNA_IN, "S339_Infos", 0)
        )
    );
    // Node S342_Conv2d_1x64x1x1_Sigmoid inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq -1.27<(u8-128.00)*0.00997218<1.27 outq 0.00<(u8-0.00)*0.00392157<1.00 biasesq -503884.56<(i32-0.00)*0.00023464<503884.56
    AddNode("S342_Conv2d_1x64x1x1_Sigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S336_Output", 0),
            GNodeArg(GNA_IN, "Conv_420_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_obj_preds_1_bias", 0),
            GNodeArg(GNA_OUT, "S342_Output", 0),
            GNodeArg(GNA_IN, "S342_Mul_scale", 0),
            GNodeArg(GNA_IN, "S342_Mul_shift", 0),
            GNodeArg(GNA_IN, "S342_Infos", 0)
        )
    );
    // Node Concat_423_qin1 inq 0.00<(u8-0.00)*0.00392157<1.00 outq -6.49<(u8-117.00)*0.05544940<7.65
    AddNode("S343_Op_Concat_423_qin1",
        Bindings(3,
            GNodeArg(GNA_IN, "S342_Output", 0),
            GNodeArg(GNA_OUT, "S343_Output", 0),
            GNodeArg(GNA_IN, "S343_Infos", 0)
        )
    );
    // Node Concat_423 inq ['-6.49<(u8-117.00)*0.05544940<7.65', '-6.49<(u8-117.00)*0.05544940<7.65', '-6.49<(u8-117.00)*0.05544940<7.65'] outq ['-6.49<(u8-117.00)*0.05544940<7.65']
    AddNode("S344_Concat",
        Bindings(4,
            GNodeArg(GNA_IN, "S339_Output", 0),
            GNodeArg(GNA_IN, "S343_Output", 0),
            GNodeArg(GNA_IN, "S324_Output", 0),
            GNodeArg(GNA_OUT, "S344_Output", 0)
        )
    );
    // Node S348_Conv2d_64x256x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S348_Conv2d_64x256x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S264_Output", 0),
            GNodeArg(GNA_IN, "Conv_424_weights", 0),
            GNodeArg(GNA_IN, "Constant__1532", 0),
            GNodeArg(GNA_OUT, "S348_Output", 0),
            GNodeArg(GNA_IN, "S348_Mul_scale", 0),
            GNodeArg(GNA_IN, "S348_Mul_shift", 0),
            GNodeArg(GNA_IN, "S348_Infos", 0)
        )
    );
    // Node S351_Conv2d_64x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S351_Conv2d_64x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S348_Output", 0),
            GNodeArg(GNA_IN, "Conv_428_weights", 0),
            GNodeArg(GNA_IN, "Constant__1535", 0),
            GNodeArg(GNA_OUT, "S351_Output", 0),
            GNodeArg(GNA_IN, "S351_Mul_scale", 0),
            GNodeArg(GNA_IN, "S351_Mul_shift", 0),
            GNodeArg(GNA_IN, "S351_Infos", 0)
        )
    );
    // Node S354_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S354_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S351_Output", 0),
            GNodeArg(GNA_IN, "Conv_432_weights", 0),
            GNodeArg(GNA_IN, "Constant__1538", 0),
            GNodeArg(GNA_OUT, "S354_Output", 0),
            GNodeArg(GNA_IN, "S354_Mul_scale", 0),
            GNodeArg(GNA_IN, "S354_Mul_shift", 0),
            GNodeArg(GNA_IN, "S354_Infos", 0)
        )
    );
    // Node S357_Conv2d_64x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S357_Conv2d_64x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S354_Output", 0),
            GNodeArg(GNA_IN, "Conv_436_weights", 0),
            GNodeArg(GNA_IN, "Constant__1541", 0),
            GNodeArg(GNA_OUT, "S357_Output", 0),
            GNodeArg(GNA_IN, "S357_Mul_scale", 0),
            GNodeArg(GNA_IN, "S357_Mul_shift", 0),
            GNodeArg(GNA_IN, "S357_Infos", 0)
        )
    );
    // Node S360_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S360_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S357_Output", 0),
            GNodeArg(GNA_IN, "Conv_440_weights", 0),
            GNodeArg(GNA_IN, "Constant__1544", 0),
            GNodeArg(GNA_OUT, "S360_Output", 0),
            GNodeArg(GNA_IN, "S360_Mul_scale", 0),
            GNodeArg(GNA_IN, "S360_Mul_shift", 0),
            GNodeArg(GNA_IN, "S360_Infos", 0)
        )
    );
    // Node S363_Conv2d_1x64x1x1_Sigmoid inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq -0.05<(u8-128.00)*0.00038083<0.05 outq 0.00<(u8-0.00)*0.00392157<1.00 biasesq -19242.72<(i32-0.00)*0.00000896<19242.72
    AddNode("S363_Conv2d_1x64x1x1_Sigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S360_Output", 0),
            GNodeArg(GNA_IN, "Conv_444_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_cls_preds_2_bias", 0),
            GNodeArg(GNA_OUT, "S363_Output", 0),
            GNodeArg(GNA_IN, "S363_Mul_scale", 0),
            GNodeArg(GNA_IN, "S363_Mul_shift", 0),
            GNodeArg(GNA_IN, "S363_Infos", 0)
        )
    );
    // Node Concat_465_qin2 inq 0.00<(u8-0.00)*0.00392157<1.00 outq -2.12<(u8-173.00)*0.01224599<1.00
    AddNode("S364_Op_Concat_465_qin2",
        Bindings(3,
            GNodeArg(GNA_IN, "S363_Output", 0),
            GNodeArg(GNA_OUT, "S364_Output", 0),
            GNodeArg(GNA_IN, "S364_Infos", 0)
        )
    );
    // Node S367_Conv2d_64x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S367_Conv2d_64x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S348_Output", 0),
            GNodeArg(GNA_IN, "Conv_445_weights", 0),
            GNodeArg(GNA_IN, "Constant__1547", 0),
            GNodeArg(GNA_OUT, "S367_Output", 0),
            GNodeArg(GNA_IN, "S367_Mul_scale", 0),
            GNodeArg(GNA_IN, "S367_Mul_shift", 0),
            GNodeArg(GNA_IN, "S367_Infos", 0)
        )
    );
    // Node S370_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S370_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S367_Output", 0),
            GNodeArg(GNA_IN, "Conv_449_weights", 0),
            GNodeArg(GNA_IN, "Constant__1550", 0),
            GNodeArg(GNA_OUT, "S370_Output", 0),
            GNodeArg(GNA_IN, "S370_Mul_scale", 0),
            GNodeArg(GNA_IN, "S370_Mul_shift", 0),
            GNodeArg(GNA_IN, "S370_Infos", 0)
        )
    );
    // Node S373_Conv2d_64x1x3x3_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S373_Conv2d_64x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S370_Output", 0),
            GNodeArg(GNA_IN, "Conv_453_weights", 0),
            GNodeArg(GNA_IN, "Constant__1553", 0),
            GNodeArg(GNA_OUT, "S373_Output", 0),
            GNodeArg(GNA_IN, "S373_Mul_scale", 0),
            GNodeArg(GNA_IN, "S373_Mul_shift", 0),
            GNodeArg(GNA_IN, "S373_Infos", 0)
        )
    );
    // Node S376_Conv2d_64x64x1x1_Relu6 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq 0.00<(u8-0.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S376_Conv2d_64x64x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S373_Output", 0),
            GNodeArg(GNA_IN, "Conv_457_weights", 0),
            GNodeArg(GNA_IN, "Constant__1556", 0),
            GNodeArg(GNA_OUT, "S376_Output", 0),
            GNodeArg(GNA_IN, "S376_Mul_scale", 0),
            GNodeArg(GNA_IN, "S376_Mul_shift", 0),
            GNodeArg(GNA_IN, "S376_Infos", 0)
        )
    );
    // Node S379_Conv2d_4x64x1x1 inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq chan<(u8-128.00)*chan<chan outq -2.12<(u8-173.00)*0.01224599<1.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S379_Conv2d_4x64x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S376_Output", 0),
            GNodeArg(GNA_IN, "Conv_461_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_reg_preds_2_bias", 0),
            GNodeArg(GNA_OUT, "S379_Output", 0),
            GNodeArg(GNA_IN, "S379_Mul_scale", 0),
            GNodeArg(GNA_IN, "S379_Mul_shift", 0),
            GNodeArg(GNA_IN, "S379_Infos", 0)
        )
    );
    // Node S382_Conv2d_1x64x1x1_Sigmoid inq 0.00<(u8-0.00)*0.02352941<6.00 weightsq -0.65<(u8-128.00)*0.00508282<0.65 outq 0.00<(u8-0.00)*0.00392157<1.00 biasesq -256829.88<(i32-0.00)*0.00011960<256829.88
    AddNode("S382_Conv2d_1x64x1x1_Sigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S376_Output", 0),
            GNodeArg(GNA_IN, "Conv_462_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_obj_preds_2_bias", 0),
            GNodeArg(GNA_OUT, "S382_Output", 0),
            GNodeArg(GNA_IN, "S382_Mul_scale", 0),
            GNodeArg(GNA_IN, "S382_Mul_shift", 0),
            GNodeArg(GNA_IN, "S382_Infos", 0)
        )
    );
    // Node Concat_465_qin1 inq 0.00<(u8-0.00)*0.00392157<1.00 outq -2.12<(u8-173.00)*0.01224599<1.00
    AddNode("S383_Op_Concat_465_qin1",
        Bindings(3,
            GNodeArg(GNA_IN, "S382_Output", 0),
            GNodeArg(GNA_OUT, "S383_Output", 0),
            GNodeArg(GNA_IN, "S383_Infos", 0)
        )
    );
    // Node Concat_465 inq ['-2.12<(u8-173.00)*0.01224599<1.00', '-2.12<(u8-173.00)*0.01224599<1.00', '-2.12<(u8-173.00)*0.01224599<1.00'] outq ['-2.12<(u8-173.00)*0.01224599<1.00']
    AddNode("S384_Concat",
        Bindings(4,
            GNodeArg(GNA_IN, "S379_Output", 0),
            GNodeArg(GNA_IN, "S383_Output", 0),
            GNodeArg(GNA_IN, "S364_Output", 0),
            GNodeArg(GNA_OUT, "S384_Output", 0)
        )
    );
    // Node Concat_490_qin2 inq -2.12<(u8-173.00)*0.01224599<1.00 outq -6.49<(u8-117.00)*0.05544940<7.65
    AddNode("S386_Op_Concat_490_qin2",
        Bindings(3,
            GNodeArg(GNA_IN, "S384_Output", 0),
            GNodeArg(GNA_OUT, "S386_Output", 0),
            GNodeArg(GNA_IN, "S386_Infos", 0)
        )
    );
    // Node output_1_qin0 inq -6.49<(u8-117.00)*0.05544940<7.65 outq f32
    AddNode("S388_Op_output_1_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S387_Output", 0),
            GNodeArg(GNA_OUT, "Output_1", 0),
            GNodeArg(GNA_IN, "S388_Infos", 0)
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
