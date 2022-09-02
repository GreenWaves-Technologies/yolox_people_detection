#include <stdint.h>
#include <stdio.h>
#include "AutoTilerLib.h"
#include "CNN_Generators_SQ8.h"
#include "ResizeGenerator.h"

#include "CNN_Copy_Generators.h"

void load_expressions_kernels() {
    LibKernelTemplate(
        "s24_kernel_args_t",
        CArgs(4,
            TCArg("unsigned int", "I0"),
            TCArg("signed char *__restrict__ ", "expr_0_in_0"),
            TCArg("signed char *__restrict__ ", "expr_0_in_1"),
            TCArg("signed char *__restrict__ ", "expr_0_out_0")
        )
    );
    
    LibKernel(
        "s24_kernel",
        CALL_PARALLEL,
        0,
        "s24_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s50_kernel_args_t",
        CArgs(4,
            TCArg("unsigned int", "I0"),
            TCArg("signed char *__restrict__ ", "expr_1_in_0"),
            TCArg("signed char *__restrict__ ", "expr_1_in_1"),
            TCArg("signed char *__restrict__ ", "expr_1_out_0")
        )
    );
    
    LibKernel(
        "s50_kernel",
        CALL_PARALLEL,
        0,
        "s50_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s96_kernel_args_t",
        CArgs(4,
            TCArg("unsigned int", "I0"),
            TCArg("signed char *__restrict__ ", "expr_2_in_0"),
            TCArg("signed char *__restrict__ ", "expr_2_in_1"),
            TCArg("signed char *__restrict__ ", "expr_2_out_0")
        )
    );
    
    LibKernel(
        "s96_kernel",
        CALL_PARALLEL,
        0,
        "s96_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s271_kernel_args_t",
        CArgs(3,
            TCArg("unsigned int", "I0"),
            TCArg("signed char *__restrict__ ", "expr_57_in_0"),
            TCArg("signed char *__restrict__ ", "expr_57_out_0")
        )
    );
    
    LibKernel(
        "s271_kernel",
        CALL_PARALLEL,
        0,
        "s271_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s310_kernel_args_t",
        CArgs(3,
            TCArg("unsigned int", "I0"),
            TCArg("signed char *__restrict__ ", "expr_68_in_0"),
            TCArg("signed char *__restrict__ ", "expr_68_out_0")
        )
    );
    
    LibKernel(
        "s310_kernel",
        CALL_PARALLEL,
        0,
        "s310_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s349_kernel_args_t",
        CArgs(3,
            TCArg("unsigned int", "I0"),
            TCArg("signed char *__restrict__ ", "expr_78_in_0"),
            TCArg("signed char *__restrict__ ", "expr_78_out_0")
        )
    );
    
    LibKernel(
        "s349_kernel",
        CALL_PARALLEL,
        0,
        "s349_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "custom_0_args_t",
        CArgs(6,
            TCArg("signed char *__restrict__ ", "expr_66_in_0"),
            TCArg("signed char *__restrict__ ", "expr_66_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat"),
            TCArg("signed char * __restrict__", "Infos")
        )
    );
    
    LibKernel(
        "custom_0",
        CALL_PARALLEL,
        0,
        "custom_0_args_t",
        0
    );
    LibKernelTemplate(
        "custom_1_args_t",
        CArgs(6,
            TCArg("signed char *__restrict__ ", "expr_91_in_0"),
            TCArg("signed char *__restrict__ ", "expr_91_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat"),
            TCArg("signed char * __restrict__", "Infos")
        )
    );
    
    LibKernel(
        "custom_1",
        CALL_PARALLEL,
        0,
        "custom_1_args_t",
        0
    );
    LibKernelTemplate(
        "custom_2_args_t",
        CArgs(6,
            TCArg("signed char *__restrict__ ", "expr_27_in_0"),
            TCArg("signed char *__restrict__ ", "expr_27_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat"),
            TCArg("signed char * __restrict__", "Infos")
        )
    );
    
    LibKernel(
        "custom_2",
        CALL_PARALLEL,
        0,
        "custom_2_args_t",
        0
    );
    LibKernelTemplate(
        "custom_3_args_t",
        CArgs(6,
            TCArg("signed char *__restrict__ ", "expr_86_in_0"),
            TCArg("signed char *__restrict__ ", "expr_86_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat"),
            TCArg("signed char * __restrict__", "Infos")
        )
    );
    
    LibKernel(
        "custom_3",
        CALL_PARALLEL,
        0,
        "custom_3_args_t",
        0
    );
    LibKernelTemplate(
        "expr_39_args_t",
        CArgs(5,
            TCArg("signed char *__restrict__ ", "expr_39_in_0"),
            TCArg("signed char *__restrict__ ", "expr_39_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_39",
        CALL_PARALLEL,
        0,
        "expr_39_args_t",
        0
    );
}



int s24_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (16, 64, 80) spaces: ((0, 1, 2),) 
        // parametric_spaces: ((0, 1, 2),) 
        // exterior_shape: (81920.0,) 
        KernelIterSpace(2, IterParSpace(KER_ITER_D0, 81920, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(3,
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_0_in_0"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_0_in_1"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_0_out_0")
        ),
        Calls(1,
            Call("s24_kernel", LOC_D0,
                Bindings(4,
                    K_ArgPar("expr_0_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_Arg("expr_0_in_0", KER_ARG_TILE),
                    K_Arg("expr_0_in_1", KER_ARG_TILE),
                    K_Arg("expr_0_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_0_out_0 axes: (0,)
        // var: expr_0_in_0 axes: (0,)
        // var: expr_0_in_1 axes: (0,)
        KerArgs(3,
            KerArg("expr_0_out_0", KerArgSpace(1, KER_ITER_D0), O_OUT|O_DB, 1, 1, 1, 0, 0, 0, "expr_0_out_0"),
            KerArg("expr_0_in_0",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_0_in_0"),
            KerArg("expr_0_in_1",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_0_in_1")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 81920, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 245760, 0);
        AddKernelArgDim(Name, "expr_0_in_0",  4, 16, 64, 80, 1);
        AddKernelArgDim(Name, "expr_0_in_1",  4, 16, 64, 80, 1);
        AddKernelArgDim(Name, "expr_0_out_0", 4, 16, 64, 80, 1);
    }
    return (Kernel!=0);
}
int s50_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (32, 32, 40) spaces: ((0, 1, 2),) 
        // parametric_spaces: ((0, 1, 2),) 
        // exterior_shape: (40960.0,) 
        KernelIterSpace(2, IterParSpace(KER_ITER_D0, 40960, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(3,
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_1_in_0"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_1_in_1"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_1_out_0")
        ),
        Calls(1,
            Call("s50_kernel", LOC_D0,
                Bindings(4,
                    K_ArgPar("expr_1_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_Arg("expr_1_in_0", KER_ARG_TILE),
                    K_Arg("expr_1_in_1", KER_ARG_TILE),
                    K_Arg("expr_1_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_1_out_0 axes: (0,)
        // var: expr_1_in_0 axes: (0,)
        // var: expr_1_in_1 axes: (0,)
        KerArgs(3,
            KerArg("expr_1_out_0", KerArgSpace(1, KER_ITER_D0), O_OUT|O_DB, 1, 1, 1, 0, 0, 0, "expr_1_out_0"),
            KerArg("expr_1_in_0",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_1_in_0"),
            KerArg("expr_1_in_1",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_1_in_1")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 40960, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 122880, 0);
        AddKernelArgDim(Name, "expr_1_in_0",  4, 32, 32, 40, 1);
        AddKernelArgDim(Name, "expr_1_in_1",  4, 32, 32, 40, 1);
        AddKernelArgDim(Name, "expr_1_out_0", 4, 32, 32, 40, 1);
    }
    return (Kernel!=0);
}
int s96_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (64, 16, 20) spaces: ((0, 1, 2),) 
        // parametric_spaces: ((0, 1, 2),) 
        // exterior_shape: (20480.0,) 
        KernelIterSpace(2, IterParSpace(KER_ITER_D0, 20480, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(3,
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_2_in_0"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_2_in_1"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_2_out_0")
        ),
        Calls(1,
            Call("s96_kernel", LOC_D0,
                Bindings(4,
                    K_ArgPar("expr_2_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_Arg("expr_2_in_0", KER_ARG_TILE),
                    K_Arg("expr_2_in_1", KER_ARG_TILE),
                    K_Arg("expr_2_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_2_out_0 axes: (0,)
        // var: expr_2_in_0 axes: (0,)
        // var: expr_2_in_1 axes: (0,)
        KerArgs(3,
            KerArg("expr_2_out_0", KerArgSpace(1, KER_ITER_D0), O_OUT|O_DB, 1, 1, 1, 0, 0, 0, "expr_2_out_0"),
            KerArg("expr_2_in_0",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_2_in_0"),
            KerArg("expr_2_in_1",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_2_in_1")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 20480, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 61440, 0);
        AddKernelArgDim(Name, "expr_2_in_0",  4, 64, 16, 20, 1);
        AddKernelArgDim(Name, "expr_2_in_1",  4, 64, 16, 20, 1);
        AddKernelArgDim(Name, "expr_2_out_0", 4, 64, 16, 20, 1);
    }
    return (Kernel!=0);
}
int s271_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (64, 32, 40) spaces: ((0, 1, 2),) 
        // parametric_spaces: ((0, 1, 2),) 
        // exterior_shape: (81920.0,) 
        KernelIterSpace(2, IterParSpace(KER_ITER_D0, 81920, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(2,
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_57_in_0"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_57_out_0")
        ),
        Calls(1,
            Call("s271_kernel", LOC_D0,
                Bindings(3,
                    K_ArgPar("expr_57_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_Arg("expr_57_in_0", KER_ARG_TILE),
                    K_Arg("expr_57_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_57_out_0 axes: (0,)
        // var: expr_57_in_0 axes: (0,)
        KerArgs(2,
            KerArg("expr_57_out_0", KerArgSpace(1, KER_ITER_D0), O_OUT|O_DB, 1, 1, 1, 0, 0, 0, "expr_57_out_0"),
            KerArg("expr_57_in_0",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_57_in_0")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 81920, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 163840, 0);
        AddKernelArgDim(Name, "expr_57_in_0",  4, 64, 32, 40, 1);
        AddKernelArgDim(Name, "expr_57_out_0", 4, 64, 32, 40, 1);
    }
    return (Kernel!=0);
}
int s310_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (64, 16, 20) spaces: ((0, 1, 2),) 
        // parametric_spaces: ((0, 1, 2),) 
        // exterior_shape: (20480.0,) 
        KernelIterSpace(2, IterParSpace(KER_ITER_D0, 20480, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(2,
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_68_in_0"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_68_out_0")
        ),
        Calls(1,
            Call("s310_kernel", LOC_D0,
                Bindings(3,
                    K_ArgPar("expr_68_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_Arg("expr_68_in_0", KER_ARG_TILE),
                    K_Arg("expr_68_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_68_out_0 axes: (0,)
        // var: expr_68_in_0 axes: (0,)
        KerArgs(2,
            KerArg("expr_68_out_0", KerArgSpace(1, KER_ITER_D0), O_OUT|O_DB, 1, 1, 1, 0, 0, 0, "expr_68_out_0"),
            KerArg("expr_68_in_0",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_68_in_0")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 20480, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 40960, 0);
        AddKernelArgDim(Name, "expr_68_in_0",  4, 64, 16, 20, 1);
        AddKernelArgDim(Name, "expr_68_out_0", 4, 64, 16, 20, 1);
    }
    return (Kernel!=0);
}
int s349_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (64, 8, 10) spaces: ((0, 1, 2),) 
        // parametric_spaces: ((0, 1, 2),) 
        // exterior_shape: (5120.0,) 
        KernelIterSpace(2, IterParSpace(KER_ITER_D0, 5120, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(2,
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_78_in_0"),
            TCArg(CNN_ArgDataType(1, 1, 1), "expr_78_out_0")
        ),
        Calls(1,
            Call("s349_kernel", LOC_D0,
                Bindings(3,
                    K_ArgPar("expr_78_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_Arg("expr_78_in_0", KER_ARG_TILE),
                    K_Arg("expr_78_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_78_out_0 axes: (0,)
        // var: expr_78_in_0 axes: (0,)
        KerArgs(2,
            KerArg("expr_78_out_0", KerArgSpace(1, KER_ITER_D0), O_OUT|O_DB, 1, 1, 1, 0, 0, 0, "expr_78_out_0"),
            KerArg("expr_78_in_0",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 1, 0, 0, 0, "expr_78_in_0")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 5120, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 10240, 0);
        AddKernelArgDim(Name, "expr_78_in_0",  4, 64, 8, 10, 1);
        AddKernelArgDim(Name, "expr_78_out_0", 4, 64, 8, 10, 1);
    }
    return (Kernel!=0);
}

void modelModel(unsigned int L1Memory, unsigned int L2Memory, unsigned int L3Memory, unsigned int L3Flash)
{
    KernelOper_T Cop = KOP_CONV;

    // SetKernelOpts(KER_OPT_NONE, KER_OPT_BUFFER_PROMOTE);
    SetSymbolDynamics();

    SetUsedFilesNames(0, 7, "Gap.h", "model.h", "CNN_BasicKernels_SQ8.h", "ResizeBasicKernels.h", "CNN_BasicKernels_SQ8.h", "Expression_Kernels.h", "CNN_Copy.h");
    SetGeneratedFilesNames("modelKernels.c", "modelKernels.h");
    AT_SetGraphCtrl(AT_GRAPH_MONITOR_CYCLES, AT_OPT_ON);
    AT_SetGraphCtrl(AT_GRAPH_PRODUCE_NODE_NAMES, AT_OPT_ON);
    AT_SetGraphCtrl(AT_GRAPH_PRODUCE_OPERINFOS, AT_OPT_ON);
    // AT_SetGraphCtrl(AT_GRAPH_DUMP_TENSOR, AT_OPT_VAL(6));

    SetMemoryDeviceInfos(4,
        AT_MEM_L1, L1Memory, "model_L1_Memory", 0, 0,
        AT_MEM_L2, L2Memory, "model_L2_Memory", 0, 1,
        AT_MEM_L3_HRAM, L3Memory, "model_L3_Memory", 0, 0,
        AT_MEM_L3_HFLASH, L3Flash, "model_L3_Flash", "model_L3_Flash_Const.dat", 0
    );

    LoadCNN_SQ8_Library();
    LoadResizeLibrary();
    LoadCNN_Copy_Library();
    load_expressions_kernels();

    CNN_GenControl_T gen_ctrl_S3_Conv2d_16x12x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S3_Conv2d_16x12x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S3_Conv2d_16x12x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "custom_0");
    CNN_SetGenCtrl(&gen_ctrl_S3_Conv2d_16x12x3x3_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_0_fusion
    CNN_ConvolutionPoolAct_SQ8("S3_Conv2d_16x12x3x3_Custom", &gen_ctrl_S3_Conv2d_16x12x3x3_Custom,
                               4, 1,
                               12, 16, 160, 128,
                               KOP_CONV, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S6_Conv2d_16x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S6_Conv2d_16x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S6_Conv2d_16x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "custom_1");
    CNN_SetGenCtrl(&gen_ctrl_S6_Conv2d_16x1x3x3_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(16));
    // generator for Conv_3_fusion
    CNN_ConvolutionPoolAct_SQ8("S6_Conv2d_16x1x3x3_Custom", &gen_ctrl_S6_Conv2d_16x1x3x3_Custom,
                               4, 1,
                               16, 16, 160, 128,
                               KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S9_Conv2d_32x16x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S9_Conv2d_32x16x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S9_Conv2d_32x16x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S9_Conv2d_32x16x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_1");
    CNN_SetGenCtrl(&gen_ctrl_S9_Conv2d_32x16x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(16));
    // generator for Conv_6_fusion
    CNN_ConvolutionPoolAct_SQ8("S9_Conv2d_32x16x1x1_Custom", &gen_ctrl_S9_Conv2d_32x16x1x1_Custom,
                               4, 1,
                               16, 32, 80, 64,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S12_Conv2d_32x32x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S12_Conv2d_32x32x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S12_Conv2d_32x32x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S12_Conv2d_32x32x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_1");
    CNN_SetGenCtrl(&gen_ctrl_S12_Conv2d_32x32x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(16));
    // generator for Conv_9_fusion
    CNN_ConvolutionPoolAct_SQ8("S12_Conv2d_32x32x1x1_Custom", &gen_ctrl_S12_Conv2d_32x32x1x1_Custom,
                               4, 1,
                               32, 32, 80, 64,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    
    // generator for Conv_9_split_copy
    CNN_Copy("S14_Op_Conv_9_split_copy", 0, 81920, 1);
    
    CNN_GenControl_T gen_ctrl_S17_Conv2d_16x16x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S17_Conv2d_16x16x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S17_Conv2d_16x16x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S17_Conv2d_16x16x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_2");
    CNN_SetGenCtrl(&gen_ctrl_S17_Conv2d_16x16x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_15_fusion
    CNN_ConvolutionPoolAct_SQ8("S17_Conv2d_16x16x1x1_Custom", &gen_ctrl_S17_Conv2d_16x16x1x1_Custom,
                               4, 1,
                               16, 16, 80, 64,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S20_Conv2d_16x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S20_Conv2d_16x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S20_Conv2d_16x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "custom_1");
    CNN_SetGenCtrl(&gen_ctrl_S20_Conv2d_16x1x3x3_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(16));
    // generator for Conv_18_fusion
    CNN_ConvolutionPoolAct_SQ8("S20_Conv2d_16x1x3x3_Custom", &gen_ctrl_S20_Conv2d_16x1x3x3_Custom,
                               4, 1,
                               16, 16, 80, 64,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S23_Conv2d_16x16x1x1;
    CNN_InitGenCtrl(&gen_ctrl_S23_Conv2d_16x16x1x1);
    CNN_SetGenCtrl(&gen_ctrl_S23_Conv2d_16x16x1x1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for Conv_21
    CNN_ConvolutionPoolAct_SQ8("S23_Conv2d_16x16x1x1", &gen_ctrl_S23_Conv2d_16x16x1x1,
                               4, 1,
                               16, 16, 80, 64,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    
    // generator for expr_0
    s24_kernel_gen("S24_Op_expr_0");
    
    CNN_GenControl_T gen_ctrl_S28_Conv2d_32x32x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S28_Conv2d_32x32x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S28_Conv2d_32x32x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S28_Conv2d_32x32x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_0");
    CNN_SetGenCtrl(&gen_ctrl_S28_Conv2d_32x32x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_26_fusion
    CNN_ConvolutionPoolAct_SQ8("S28_Conv2d_32x32x1x1_Custom", &gen_ctrl_S28_Conv2d_32x32x1x1_Custom,
                               4, 1,
                               32, 32, 80, 64,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S31_Conv2d_32x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S31_Conv2d_32x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S31_Conv2d_32x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "custom_0");
    CNN_SetGenCtrl(&gen_ctrl_S31_Conv2d_32x1x3x3_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_29_fusion
    CNN_ConvolutionPoolAct_SQ8("S31_Conv2d_32x1x3x3_Custom", &gen_ctrl_S31_Conv2d_32x1x3x3_Custom,
                               4, 1,
                               32, 32, 80, 64,
                               KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S34_Conv2d_64x32x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S34_Conv2d_64x32x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S34_Conv2d_64x32x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S34_Conv2d_64x32x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_0");
    CNN_SetGenCtrl(&gen_ctrl_S34_Conv2d_64x32x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_32_fusion
    CNN_ConvolutionPoolAct_SQ8("S34_Conv2d_64x32x1x1_Custom", &gen_ctrl_S34_Conv2d_64x32x1x1_Custom,
                               4, 1,
                               32, 64, 40, 32,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S37_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S37_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S37_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S37_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_2");
    CNN_SetGenCtrl(&gen_ctrl_S37_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_35_fusion
    CNN_ConvolutionPoolAct_SQ8("S37_Conv2d_64x64x1x1_Custom", &gen_ctrl_S37_Conv2d_64x64x1x1_Custom,
                               4, 1,
                               64, 64, 40, 32,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    
    // generator for Conv_41_fusion_qin0
    CNN_Convert("S39_Op_Conv_41_fusion_qin0", 1, 1, 40960, KOP_CONVERT_FP_FP_SCALE);
    
    
    // generator for Conv_35_split_copy
    CNN_Copy("S40_Op_Conv_35_split_copy", 0, 40960, 1);
    
    CNN_GenControl_T gen_ctrl_S43_Conv2d_32x32x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S43_Conv2d_32x32x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S43_Conv2d_32x32x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S43_Conv2d_32x32x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S43_Conv2d_32x32x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_41_fusion
    CNN_ConvolutionPoolAct_SQ8("S43_Conv2d_32x32x1x1_Custom", &gen_ctrl_S43_Conv2d_32x32x1x1_Custom,
                               4, 1,
                               32, 32, 40, 32,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S46_Conv2d_32x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S46_Conv2d_32x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S46_Conv2d_32x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "custom_0");
    CNN_SetGenCtrl(&gen_ctrl_S46_Conv2d_32x1x3x3_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_44_fusion
    CNN_ConvolutionPoolAct_SQ8("S46_Conv2d_32x1x3x3_Custom", &gen_ctrl_S46_Conv2d_32x1x3x3_Custom,
                               4, 1,
                               32, 32, 40, 32,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S49_Conv2d_32x32x1x1;
    CNN_InitGenCtrl(&gen_ctrl_S49_Conv2d_32x32x1x1);
    CNN_SetGenCtrl(&gen_ctrl_S49_Conv2d_32x32x1x1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for Conv_47
    CNN_ConvolutionPoolAct_SQ8("S49_Conv2d_32x32x1x1", &gen_ctrl_S49_Conv2d_32x32x1x1,
                               4, 1,
                               32, 32, 40, 32,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    
    // generator for expr_1
    s50_kernel_gen("S50_Op_expr_1");
    
    CNN_GenControl_T gen_ctrl_S53_Conv2d_32x32x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S53_Conv2d_32x32x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S53_Conv2d_32x32x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S53_Conv2d_32x32x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_2");
    CNN_SetGenCtrl(&gen_ctrl_S53_Conv2d_32x32x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_51_fusion
    CNN_ConvolutionPoolAct_SQ8("S53_Conv2d_32x32x1x1_Custom", &gen_ctrl_S53_Conv2d_32x32x1x1_Custom,
                               4, 1,
                               32, 32, 40, 32,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S56_Conv2d_32x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S56_Conv2d_32x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S56_Conv2d_32x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "custom_2");
    CNN_SetGenCtrl(&gen_ctrl_S56_Conv2d_32x1x3x3_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_54_fusion
    CNN_ConvolutionPoolAct_SQ8("S56_Conv2d_32x1x3x3_Custom", &gen_ctrl_S56_Conv2d_32x1x3x3_Custom,
                               4, 1,
                               32, 32, 40, 32,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S59_Conv2d_32x32x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S59_Conv2d_32x32x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S59_Conv2d_32x32x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S59_Conv2d_32x32x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S59_Conv2d_32x32x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_57_fusion
    CNN_ConvolutionPoolAct_SQ8("S59_Conv2d_32x32x1x1_Custom", &gen_ctrl_S59_Conv2d_32x32x1x1_Custom,
                               4, 1,
                               32, 32, 40, 32,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    
    // generator for Add_60
    CNN_MatAddAct_SQ8("S60_MatAdd_32x32x40", 0, 32, 32, 40, KOP_MATADD, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S63_Conv2d_32x32x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S63_Conv2d_32x32x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S63_Conv2d_32x32x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S63_Conv2d_32x32x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S63_Conv2d_32x32x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_61_fusion
    CNN_ConvolutionPoolAct_SQ8("S63_Conv2d_32x32x1x1_Custom", &gen_ctrl_S63_Conv2d_32x32x1x1_Custom,
                               4, 1,
                               32, 32, 40, 32,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S66_Conv2d_32x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S66_Conv2d_32x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S66_Conv2d_32x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S66_Conv2d_32x1x3x3_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_64_fusion
    CNN_ConvolutionPoolAct_SQ8("S66_Conv2d_32x1x3x3_Custom", &gen_ctrl_S66_Conv2d_32x1x3x3_Custom,
                               4, 1,
                               32, 32, 40, 32,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S69_Conv2d_32x32x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S69_Conv2d_32x32x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S69_Conv2d_32x32x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S69_Conv2d_32x32x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_2");
    CNN_SetGenCtrl(&gen_ctrl_S69_Conv2d_32x32x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_67_fusion
    CNN_ConvolutionPoolAct_SQ8("S69_Conv2d_32x32x1x1_Custom", &gen_ctrl_S69_Conv2d_32x32x1x1_Custom,
                               4, 1,
                               32, 32, 40, 32,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    
    // generator for Add_70
    CNN_MatAddAct_SQ8("S70_MatAdd_32x32x40", 0, 32, 32, 40, KOP_MATADD, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S74_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S74_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S74_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S74_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_2");
    CNN_SetGenCtrl(&gen_ctrl_S74_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_72_fusion
    CNN_ConvolutionPoolAct_SQ8("S74_Conv2d_64x64x1x1_Custom", &gen_ctrl_S74_Conv2d_64x64x1x1_Custom,
                               4, 1,
                               64, 64, 40, 32,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S77_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S77_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S77_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "custom_2");
    CNN_SetGenCtrl(&gen_ctrl_S77_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_75_fusion
    CNN_ConvolutionPoolAct_SQ8("S77_Conv2d_64x1x3x3_Custom", &gen_ctrl_S77_Conv2d_64x1x3x3_Custom,
                               4, 1,
                               64, 64, 40, 32,
                               KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S80_Conv2d_128x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S80_Conv2d_128x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S80_Conv2d_128x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S80_Conv2d_128x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_2");
    CNN_SetGenCtrl(&gen_ctrl_S80_Conv2d_128x64x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_78_fusion
    CNN_ConvolutionPoolAct_SQ8("S80_Conv2d_128x64x1x1_Custom", &gen_ctrl_S80_Conv2d_128x64x1x1_Custom,
                               4, 1,
                               64, 128, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S83_Conv2d_128x128x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S83_Conv2d_128x128x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S83_Conv2d_128x128x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S83_Conv2d_128x128x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_2");
    CNN_SetGenCtrl(&gen_ctrl_S83_Conv2d_128x128x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_81_fusion
    CNN_ConvolutionPoolAct_SQ8("S83_Conv2d_128x128x1x1_Custom", &gen_ctrl_S83_Conv2d_128x128x1x1_Custom,
                               4, 1,
                               128, 128, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    
    // generator for Conv_87_fusion_qin0
    CNN_Convert("S85_Op_Conv_87_fusion_qin0", 1, 1, 20480, KOP_CONVERT_FP_FP_SCALE);
    
    
    // generator for Conv_81_split_copy
    CNN_Copy("S86_Op_Conv_81_split_copy", 0, 20480, 1);
    
    CNN_GenControl_T gen_ctrl_S89_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S89_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S89_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S89_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S89_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_87_fusion
    CNN_ConvolutionPoolAct_SQ8("S89_Conv2d_64x64x1x1_Custom", &gen_ctrl_S89_Conv2d_64x64x1x1_Custom,
                               4, 1,
                               64, 64, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S92_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S92_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S92_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "custom_2");
    CNN_SetGenCtrl(&gen_ctrl_S92_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_90_fusion
    CNN_ConvolutionPoolAct_SQ8("S92_Conv2d_64x1x3x3_Custom", &gen_ctrl_S92_Conv2d_64x1x3x3_Custom,
                               4, 1,
                               64, 64, 20, 16,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S95_Conv2d_64x64x1x1;
    CNN_InitGenCtrl(&gen_ctrl_S95_Conv2d_64x64x1x1);
    CNN_SetGenCtrl(&gen_ctrl_S95_Conv2d_64x64x1x1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for Conv_93
    CNN_ConvolutionPoolAct_SQ8("S95_Conv2d_64x64x1x1", &gen_ctrl_S95_Conv2d_64x64x1x1,
                               4, 1,
                               64, 64, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    
    // generator for expr_2
    s96_kernel_gen("S96_Op_expr_2");
    
    CNN_GenControl_T gen_ctrl_S99_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S99_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S99_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S99_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S99_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_97_fusion
    CNN_ConvolutionPoolAct_SQ8("S99_Conv2d_64x64x1x1_Custom", &gen_ctrl_S99_Conv2d_64x64x1x1_Custom,
                               4, 1,
                               64, 64, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S102_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S102_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S102_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "custom_2");
    CNN_SetGenCtrl(&gen_ctrl_S102_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_100_fusion
    CNN_ConvolutionPoolAct_SQ8("S102_Conv2d_64x1x3x3_Custom", &gen_ctrl_S102_Conv2d_64x1x3x3_Custom,
                               4, 1,
                               64, 64, 20, 16,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S105_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S105_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S105_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S105_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S105_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_103_fusion
    CNN_ConvolutionPoolAct_SQ8("S105_Conv2d_64x64x1x1_Custom", &gen_ctrl_S105_Conv2d_64x64x1x1_Custom,
                               4, 1,
                               64, 64, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    
    // generator for Add_106
    CNN_MatAddAct_SQ8("S106_MatAdd_64x16x20", 0, 64, 16, 20, KOP_MATADD, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S109_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S109_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S109_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S109_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S109_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_107_fusion
    CNN_ConvolutionPoolAct_SQ8("S109_Conv2d_64x64x1x1_Custom", &gen_ctrl_S109_Conv2d_64x64x1x1_Custom,
                               4, 1,
                               64, 64, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S112_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S112_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S112_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "custom_2");
    CNN_SetGenCtrl(&gen_ctrl_S112_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_110_fusion
    CNN_ConvolutionPoolAct_SQ8("S112_Conv2d_64x1x3x3_Custom", &gen_ctrl_S112_Conv2d_64x1x3x3_Custom,
                               4, 1,
                               64, 64, 20, 16,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S115_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S115_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S115_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S115_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_2");
    CNN_SetGenCtrl(&gen_ctrl_S115_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_113_fusion
    CNN_ConvolutionPoolAct_SQ8("S115_Conv2d_64x64x1x1_Custom", &gen_ctrl_S115_Conv2d_64x64x1x1_Custom,
                               4, 1,
                               64, 64, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    
    // generator for Add_116
    CNN_MatAddAct_SQ8("S116_MatAdd_64x16x20", 0, 64, 16, 20, KOP_MATADD, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S120_Conv2d_128x128x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S120_Conv2d_128x128x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S120_Conv2d_128x128x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S120_Conv2d_128x128x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_2");
    CNN_SetGenCtrl(&gen_ctrl_S120_Conv2d_128x128x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_118_fusion
    CNN_ConvolutionPoolAct_SQ8("S120_Conv2d_128x128x1x1_Custom", &gen_ctrl_S120_Conv2d_128x128x1x1_Custom,
                               4, 1,
                               128, 128, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S123_Conv2d_128x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S123_Conv2d_128x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S123_Conv2d_128x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "custom_2");
    CNN_SetGenCtrl(&gen_ctrl_S123_Conv2d_128x1x3x3_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_121_fusion
    CNN_ConvolutionPoolAct_SQ8("S123_Conv2d_128x1x3x3_Custom", &gen_ctrl_S123_Conv2d_128x1x3x3_Custom,
                               4, 1,
                               128, 128, 20, 16,
                               KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S126_Conv2d_256x128x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S126_Conv2d_256x128x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S126_Conv2d_256x128x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S126_Conv2d_256x128x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_2");
    CNN_SetGenCtrl(&gen_ctrl_S126_Conv2d_256x128x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_124_fusion
    CNN_ConvolutionPoolAct_SQ8("S126_Conv2d_256x128x1x1_Custom", &gen_ctrl_S126_Conv2d_256x128x1x1_Custom,
                               4, 1,
                               128, 256, 10, 8,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S129_Conv2d_128x256x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S129_Conv2d_128x256x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S129_Conv2d_128x256x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S129_Conv2d_128x256x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S129_Conv2d_128x256x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_127_fusion
    CNN_ConvolutionPoolAct_SQ8("S129_Conv2d_128x256x1x1_Custom", &gen_ctrl_S129_Conv2d_128x256x1x1_Custom,
                               4, 1,
                               256, 128, 10, 8,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    // generator for MaxPool_132
    CNN_PoolAct_SQ8("S130_MaxPool_13x13", 0,
                    128, 10, 8,
                    KOP_MAXPOOL, 13, 13, 1, 1, 1, 1, 1,
                    KOP_NONE);
    
    // generator for MaxPool_130
    CNN_PoolAct_SQ8("S131_MaxPool_5x5", 0,
                    128, 10, 8,
                    KOP_MAXPOOL, 5, 5, 1, 1, 1, 1, 1,
                    KOP_NONE);
    
    // generator for MaxPool_131
    CNN_PoolAct_SQ8("S132_MaxPool_9x9", 0,
                    128, 10, 8,
                    KOP_MAXPOOL, 9, 9, 1, 1, 1, 1, 1,
                    KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S136_Conv2d_256x512x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S136_Conv2d_256x512x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S136_Conv2d_256x512x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S136_Conv2d_256x512x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_2");
    CNN_SetGenCtrl(&gen_ctrl_S136_Conv2d_256x512x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_134_fusion
    CNN_ConvolutionPoolAct_SQ8("S136_Conv2d_256x512x1x1_Custom", &gen_ctrl_S136_Conv2d_256x512x1x1_Custom,
                               4, 1,
                               512, 256, 10, 8,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S139_Conv2d_256x256x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S139_Conv2d_256x256x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S139_Conv2d_256x256x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S139_Conv2d_256x256x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S139_Conv2d_256x256x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_137_fusion
    CNN_ConvolutionPoolAct_SQ8("S139_Conv2d_256x256x1x1_Custom", &gen_ctrl_S139_Conv2d_256x256x1x1_Custom,
                               4, 1,
                               256, 256, 10, 8,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    
    // generator for Conv_143_fusion_qin0
    CNN_Convert("S141_Op_Conv_143_fusion_qin0", 1, 1, 10240, KOP_CONVERT_FP_FP_SCALE);
    
    
    // generator for Conv_137_split_copy
    CNN_Copy("S142_Op_Conv_137_split_copy", 0, 10240, 1);
    
    CNN_GenControl_T gen_ctrl_S145_Conv2d_128x128x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S145_Conv2d_128x128x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S145_Conv2d_128x128x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S145_Conv2d_128x128x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S145_Conv2d_128x128x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_143_fusion
    CNN_ConvolutionPoolAct_SQ8("S145_Conv2d_128x128x1x1_Custom", &gen_ctrl_S145_Conv2d_128x128x1x1_Custom,
                               4, 1,
                               128, 128, 10, 8,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S148_Conv2d_128x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S148_Conv2d_128x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S148_Conv2d_128x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "custom_2");
    CNN_SetGenCtrl(&gen_ctrl_S148_Conv2d_128x1x3x3_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_146_fusion
    CNN_ConvolutionPoolAct_SQ8("S148_Conv2d_128x1x3x3_Custom", &gen_ctrl_S148_Conv2d_128x1x3x3_Custom,
                               4, 1,
                               128, 128, 10, 8,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S151_Conv2d_128x128x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S151_Conv2d_128x128x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S151_Conv2d_128x128x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S151_Conv2d_128x128x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S151_Conv2d_128x128x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_149_fusion
    CNN_ConvolutionPoolAct_SQ8("S151_Conv2d_128x128x1x1_Custom", &gen_ctrl_S151_Conv2d_128x128x1x1_Custom,
                               4, 1,
                               128, 128, 10, 8,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S155_Conv2d_256x256x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S155_Conv2d_256x256x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S155_Conv2d_256x256x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S155_Conv2d_256x256x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_2");
    CNN_SetGenCtrl(&gen_ctrl_S155_Conv2d_256x256x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_153_fusion
    CNN_ConvolutionPoolAct_SQ8("S155_Conv2d_256x256x1x1_Custom", &gen_ctrl_S155_Conv2d_256x256x1x1_Custom,
                               4, 1,
                               256, 256, 10, 8,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S158_Conv2d_128x256x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S158_Conv2d_128x256x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S158_Conv2d_128x256x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S158_Conv2d_128x256x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S158_Conv2d_128x256x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_156_fusion
    CNN_ConvolutionPoolAct_SQ8("S158_Conv2d_128x256x1x1_Custom", &gen_ctrl_S158_Conv2d_128x256x1x1_Custom,
                               4, 1,
                               256, 128, 10, 8,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    
    // generator for Resize_160
    GenerateResizeMultiChannel("S159_Op_Resize_160", 10, 8, 20, 16, 128, SIGNED_INOUT, KOP_NEAREST_NEIGHBOR_RESIZE);
    
    CNN_GenControl_T gen_ctrl_S163_Conv2d_128x256x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S163_Conv2d_128x256x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S163_Conv2d_128x256x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S163_Conv2d_128x256x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S163_Conv2d_128x256x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_162_fusion
    CNN_ConvolutionPoolAct_SQ8("S163_Conv2d_128x256x1x1_Custom", &gen_ctrl_S163_Conv2d_128x256x1x1_Custom,
                               4, 1,
                               256, 128, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    
    // generator for Conv_168_fusion_qin0
    CNN_Convert("S165_Op_Conv_168_fusion_qin0", 1, 1, 20480, KOP_CONVERT_FP_FP_SCALE);
    
    
    // generator for Conv_162_split_copy
    CNN_Copy("S166_Op_Conv_162_split_copy", 0, 20480, 1);
    
    CNN_GenControl_T gen_ctrl_S169_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S169_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S169_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S169_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S169_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_168_fusion
    CNN_ConvolutionPoolAct_SQ8("S169_Conv2d_64x64x1x1_Custom", &gen_ctrl_S169_Conv2d_64x64x1x1_Custom,
                               4, 1,
                               64, 64, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S172_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S172_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S172_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "custom_2");
    CNN_SetGenCtrl(&gen_ctrl_S172_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_171_fusion
    CNN_ConvolutionPoolAct_SQ8("S172_Conv2d_64x1x3x3_Custom", &gen_ctrl_S172_Conv2d_64x1x3x3_Custom,
                               4, 1,
                               64, 64, 20, 16,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S175_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S175_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S175_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S175_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S175_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_174_fusion
    CNN_ConvolutionPoolAct_SQ8("S175_Conv2d_64x64x1x1_Custom", &gen_ctrl_S175_Conv2d_64x64x1x1_Custom,
                               4, 1,
                               64, 64, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S179_Conv2d_128x128x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S179_Conv2d_128x128x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S179_Conv2d_128x128x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S179_Conv2d_128x128x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_2");
    CNN_SetGenCtrl(&gen_ctrl_S179_Conv2d_128x128x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_178_fusion
    CNN_ConvolutionPoolAct_SQ8("S179_Conv2d_128x128x1x1_Custom", &gen_ctrl_S179_Conv2d_128x128x1x1_Custom,
                               4, 1,
                               128, 128, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S182_Conv2d_64x128x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S182_Conv2d_64x128x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S182_Conv2d_64x128x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S182_Conv2d_64x128x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S182_Conv2d_64x128x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_181_fusion
    CNN_ConvolutionPoolAct_SQ8("S182_Conv2d_64x128x1x1_Custom", &gen_ctrl_S182_Conv2d_64x128x1x1_Custom,
                               4, 1,
                               128, 64, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    
    // generator for Resize_185
    GenerateResizeMultiChannel("S183_Op_Resize_185", 20, 16, 40, 32, 64, SIGNED_INOUT, KOP_NEAREST_NEIGHBOR_RESIZE);
    
    
    // generator for Concat_186_qin0
    CNN_Convert("S184_Op_Concat_186_qin0", 1, 1, 81920, KOP_CONVERT_FP_FP_SCALE);
    
    CNN_GenControl_T gen_ctrl_S188_Conv2d_64x128x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S188_Conv2d_64x128x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S188_Conv2d_64x128x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S188_Conv2d_64x128x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S188_Conv2d_64x128x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_187_fusion
    CNN_ConvolutionPoolAct_SQ8("S188_Conv2d_64x128x1x1_Custom", &gen_ctrl_S188_Conv2d_64x128x1x1_Custom,
                               4, 1,
                               128, 64, 40, 32,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    
    // generator for Conv_193_fusion_qin0
    CNN_Convert("S190_Op_Conv_193_fusion_qin0", 1, 1, 40960, KOP_CONVERT_FP_FP_SCALE);
    
    
    // generator for Conv_187_split_copy
    CNN_Copy("S191_Op_Conv_187_split_copy", 0, 40960, 1);
    
    CNN_GenControl_T gen_ctrl_S194_Conv2d_32x32x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S194_Conv2d_32x32x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S194_Conv2d_32x32x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S194_Conv2d_32x32x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S194_Conv2d_32x32x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_193_fusion
    CNN_ConvolutionPoolAct_SQ8("S194_Conv2d_32x32x1x1_Custom", &gen_ctrl_S194_Conv2d_32x32x1x1_Custom,
                               4, 1,
                               32, 32, 40, 32,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S197_Conv2d_32x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S197_Conv2d_32x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S197_Conv2d_32x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "custom_2");
    CNN_SetGenCtrl(&gen_ctrl_S197_Conv2d_32x1x3x3_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_196_fusion
    CNN_ConvolutionPoolAct_SQ8("S197_Conv2d_32x1x3x3_Custom", &gen_ctrl_S197_Conv2d_32x1x3x3_Custom,
                               4, 1,
                               32, 32, 40, 32,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S200_Conv2d_32x32x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S200_Conv2d_32x32x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S200_Conv2d_32x32x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S200_Conv2d_32x32x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_2");
    CNN_SetGenCtrl(&gen_ctrl_S200_Conv2d_32x32x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_199_fusion
    CNN_ConvolutionPoolAct_SQ8("S200_Conv2d_32x32x1x1_Custom", &gen_ctrl_S200_Conv2d_32x32x1x1_Custom,
                               4, 1,
                               32, 32, 40, 32,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S204_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S204_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S204_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S204_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S204_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_203_fusion
    CNN_ConvolutionPoolAct_SQ8("S204_Conv2d_64x64x1x1_Custom", &gen_ctrl_S204_Conv2d_64x64x1x1_Custom,
                               4, 1,
                               64, 64, 40, 32,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S207_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S207_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S207_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "custom_2");
    CNN_SetGenCtrl(&gen_ctrl_S207_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_206_fusion
    CNN_ConvolutionPoolAct_SQ8("S207_Conv2d_64x1x3x3_Custom", &gen_ctrl_S207_Conv2d_64x1x3x3_Custom,
                               4, 1,
                               64, 64, 40, 32,
                               KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S210_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S210_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S210_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S210_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_2");
    CNN_SetGenCtrl(&gen_ctrl_S210_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_209_fusion
    CNN_ConvolutionPoolAct_SQ8("S210_Conv2d_64x64x1x1_Custom", &gen_ctrl_S210_Conv2d_64x64x1x1_Custom,
                               4, 1,
                               64, 64, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S214_Conv2d_128x128x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S214_Conv2d_128x128x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S214_Conv2d_128x128x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S214_Conv2d_128x128x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_39");
    // generator for Conv_213_fusion
    CNN_ConvolutionPoolAct_SQ8("S214_Conv2d_128x128x1x1_Custom", &gen_ctrl_S214_Conv2d_128x128x1x1_Custom,
                               4, 1,
                               128, 128, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    
    // generator for Conv_213_split_copy
    CNN_Copy("S216_Op_Conv_213_split_copy", 0, 20480, 1);
    
    CNN_GenControl_T gen_ctrl_S219_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S219_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S219_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S219_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S219_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_219_fusion
    CNN_ConvolutionPoolAct_SQ8("S219_Conv2d_64x64x1x1_Custom", &gen_ctrl_S219_Conv2d_64x64x1x1_Custom,
                               4, 1,
                               64, 64, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S222_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S222_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S222_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "custom_2");
    CNN_SetGenCtrl(&gen_ctrl_S222_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_222_fusion
    CNN_ConvolutionPoolAct_SQ8("S222_Conv2d_64x1x3x3_Custom", &gen_ctrl_S222_Conv2d_64x1x3x3_Custom,
                               4, 1,
                               64, 64, 20, 16,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S225_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S225_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S225_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S225_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S225_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_225_fusion
    CNN_ConvolutionPoolAct_SQ8("S225_Conv2d_64x64x1x1_Custom", &gen_ctrl_S225_Conv2d_64x64x1x1_Custom,
                               4, 1,
                               64, 64, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S229_Conv2d_128x128x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S229_Conv2d_128x128x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S229_Conv2d_128x128x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S229_Conv2d_128x128x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S229_Conv2d_128x128x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_229_fusion
    CNN_ConvolutionPoolAct_SQ8("S229_Conv2d_128x128x1x1_Custom", &gen_ctrl_S229_Conv2d_128x128x1x1_Custom,
                               4, 1,
                               128, 128, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S232_Conv2d_128x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S232_Conv2d_128x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S232_Conv2d_128x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S232_Conv2d_128x1x3x3_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_232_fusion
    CNN_ConvolutionPoolAct_SQ8("S232_Conv2d_128x1x3x3_Custom", &gen_ctrl_S232_Conv2d_128x1x3x3_Custom,
                               4, 1,
                               128, 128, 20, 16,
                               KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S235_Conv2d_128x128x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S235_Conv2d_128x128x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S235_Conv2d_128x128x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S235_Conv2d_128x128x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S235_Conv2d_128x128x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_235_fusion
    CNN_ConvolutionPoolAct_SQ8("S235_Conv2d_128x128x1x1_Custom", &gen_ctrl_S235_Conv2d_128x128x1x1_Custom,
                               4, 1,
                               128, 128, 10, 8,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    
    // generator for Conv_239_fusion_qin0
    CNN_Convert("S237_Op_Conv_239_fusion_qin0", 1, 1, 20480, KOP_CONVERT_FP_FP_SCALE);
    
    CNN_GenControl_T gen_ctrl_S240_Conv2d_256x256x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S240_Conv2d_256x256x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S240_Conv2d_256x256x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S240_Conv2d_256x256x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S240_Conv2d_256x256x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_239_fusion
    CNN_ConvolutionPoolAct_SQ8("S240_Conv2d_256x256x1x1_Custom", &gen_ctrl_S240_Conv2d_256x256x1x1_Custom,
                               4, 1,
                               256, 256, 10, 8,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    
    // generator for Conv_239_split_copy
    CNN_Copy("S242_Op_Conv_239_split_copy", 0, 10240, 1);
    
    CNN_GenControl_T gen_ctrl_S245_Conv2d_128x128x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S245_Conv2d_128x128x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S245_Conv2d_128x128x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S245_Conv2d_128x128x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S245_Conv2d_128x128x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_245_fusion
    CNN_ConvolutionPoolAct_SQ8("S245_Conv2d_128x128x1x1_Custom", &gen_ctrl_S245_Conv2d_128x128x1x1_Custom,
                               4, 1,
                               128, 128, 10, 8,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S248_Conv2d_128x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S248_Conv2d_128x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S248_Conv2d_128x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S248_Conv2d_128x1x3x3_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_248_fusion
    CNN_ConvolutionPoolAct_SQ8("S248_Conv2d_128x1x3x3_Custom", &gen_ctrl_S248_Conv2d_128x1x3x3_Custom,
                               4, 1,
                               128, 128, 10, 8,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S251_Conv2d_128x128x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S251_Conv2d_128x128x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S251_Conv2d_128x128x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S251_Conv2d_128x128x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S251_Conv2d_128x128x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_251_fusion
    CNN_ConvolutionPoolAct_SQ8("S251_Conv2d_128x128x1x1_Custom", &gen_ctrl_S251_Conv2d_128x128x1x1_Custom,
                               4, 1,
                               128, 128, 10, 8,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S255_Conv2d_256x256x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S255_Conv2d_256x256x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S255_Conv2d_256x256x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S255_Conv2d_256x256x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S255_Conv2d_256x256x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_255_fusion
    CNN_ConvolutionPoolAct_SQ8("S255_Conv2d_256x256x1x1_Custom", &gen_ctrl_S255_Conv2d_256x256x1x1_Custom,
                               4, 1,
                               256, 256, 10, 8,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S258_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S258_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S258_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S258_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_2");
    CNN_SetGenCtrl(&gen_ctrl_S258_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_258_fusion
    CNN_ConvolutionPoolAct_SQ8("S258_Conv2d_64x64x1x1_Custom", &gen_ctrl_S258_Conv2d_64x64x1x1_Custom,
                               4, 1,
                               64, 64, 40, 32,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S261_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S261_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S261_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "custom_2");
    CNN_SetGenCtrl(&gen_ctrl_S261_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_261_fusion
    CNN_ConvolutionPoolAct_SQ8("S261_Conv2d_64x1x3x3_Custom", &gen_ctrl_S261_Conv2d_64x1x3x3_Custom,
                               4, 1,
                               64, 64, 40, 32,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S264_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S264_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S264_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S264_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_2");
    CNN_SetGenCtrl(&gen_ctrl_S264_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_264_fusion
    CNN_ConvolutionPoolAct_SQ8("S264_Conv2d_64x64x1x1_Custom", &gen_ctrl_S264_Conv2d_64x64x1x1_Custom,
                               4, 1,
                               64, 64, 40, 32,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S267_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S267_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S267_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "custom_2");
    CNN_SetGenCtrl(&gen_ctrl_S267_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_267_fusion
    CNN_ConvolutionPoolAct_SQ8("S267_Conv2d_64x1x3x3_Custom", &gen_ctrl_S267_Conv2d_64x1x3x3_Custom,
                               4, 1,
                               64, 64, 40, 32,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S270_Conv2d_64x64x1x1;
    CNN_InitGenCtrl(&gen_ctrl_S270_Conv2d_64x64x1x1);
    CNN_SetGenCtrl(&gen_ctrl_S270_Conv2d_64x64x1x1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for Conv_270
    CNN_ConvolutionPoolAct_SQ8("S270_Conv2d_64x64x1x1", &gen_ctrl_S270_Conv2d_64x64x1x1,
                               4, 1,
                               64, 64, 40, 32,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    
    // generator for expr_57
    s271_kernel_gen("S271_Op_expr_57");
    
    CNN_GenControl_T gen_ctrl_S274_Conv2d_1x64x1x1_Sigmoid;
    CNN_InitGenCtrl(&gen_ctrl_S274_Conv2d_1x64x1x1_Sigmoid);
    CNN_SetGenCtrl(&gen_ctrl_S274_Conv2d_1x64x1x1_Sigmoid, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for Conv_273_fusion
    CNN_ConvolutionPoolAct_SQ8("S274_Conv2d_1x64x1x1_Sigmoid", &gen_ctrl_S274_Conv2d_1x64x1x1_Sigmoid,
                               4, 1,
                               64, 1, 40, 32,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_SIGMOID);
    
    CNN_GenControl_T gen_ctrl_S277_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S277_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S277_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "custom_0");
    CNN_SetGenCtrl(&gen_ctrl_S277_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_274_fusion
    CNN_ConvolutionPoolAct_SQ8("S277_Conv2d_64x1x3x3_Custom", &gen_ctrl_S277_Conv2d_64x1x3x3_Custom,
                               4, 1,
                               64, 64, 40, 32,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S280_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S280_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S280_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S280_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_0");
    CNN_SetGenCtrl(&gen_ctrl_S280_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_277_fusion
    CNN_ConvolutionPoolAct_SQ8("S280_Conv2d_64x64x1x1_Custom", &gen_ctrl_S280_Conv2d_64x64x1x1_Custom,
                               4, 1,
                               64, 64, 40, 32,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S283_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S283_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S283_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "custom_2");
    CNN_SetGenCtrl(&gen_ctrl_S283_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_280_fusion
    CNN_ConvolutionPoolAct_SQ8("S283_Conv2d_64x1x3x3_Custom", &gen_ctrl_S283_Conv2d_64x1x3x3_Custom,
                               4, 1,
                               64, 64, 40, 32,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S286_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S286_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S286_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S286_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_0");
    CNN_SetGenCtrl(&gen_ctrl_S286_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_283_fusion
    CNN_ConvolutionPoolAct_SQ8("S286_Conv2d_64x64x1x1_Custom", &gen_ctrl_S286_Conv2d_64x64x1x1_Custom,
                               4, 1,
                               64, 64, 40, 32,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S289_Conv2d_4x64x1x1;
    CNN_InitGenCtrl(&gen_ctrl_S289_Conv2d_4x64x1x1);
    CNN_SetGenCtrl(&gen_ctrl_S289_Conv2d_4x64x1x1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for Conv_286
    CNN_ConvolutionPoolAct_SQ8("S289_Conv2d_4x64x1x1", &gen_ctrl_S289_Conv2d_4x64x1x1,
                               4, 1,
                               64, 4, 40, 32,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S292_Conv2d_1x64x1x1_Sigmoid;
    CNN_InitGenCtrl(&gen_ctrl_S292_Conv2d_1x64x1x1_Sigmoid);
    CNN_SetGenCtrl(&gen_ctrl_S292_Conv2d_1x64x1x1_Sigmoid, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for Conv_287_fusion
    CNN_ConvolutionPoolAct_SQ8("S292_Conv2d_1x64x1x1_Sigmoid", &gen_ctrl_S292_Conv2d_1x64x1x1_Sigmoid,
                               4, 1,
                               64, 1, 40, 32,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_SIGMOID);
    
    CNN_GenControl_T gen_ctrl_S297_Conv2d_64x128x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S297_Conv2d_64x128x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S297_Conv2d_64x128x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S297_Conv2d_64x128x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S297_Conv2d_64x128x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_291_fusion
    CNN_ConvolutionPoolAct_SQ8("S297_Conv2d_64x128x1x1_Custom", &gen_ctrl_S297_Conv2d_64x128x1x1_Custom,
                               4, 1,
                               128, 64, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S300_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S300_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S300_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S300_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_294_fusion
    CNN_ConvolutionPoolAct_SQ8("S300_Conv2d_64x1x3x3_Custom", &gen_ctrl_S300_Conv2d_64x1x3x3_Custom,
                               4, 1,
                               64, 64, 20, 16,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S303_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S303_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S303_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S303_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S303_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_297_fusion
    CNN_ConvolutionPoolAct_SQ8("S303_Conv2d_64x64x1x1_Custom", &gen_ctrl_S303_Conv2d_64x64x1x1_Custom,
                               4, 1,
                               64, 64, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S306_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S306_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S306_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S306_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_300_fusion
    CNN_ConvolutionPoolAct_SQ8("S306_Conv2d_64x1x3x3_Custom", &gen_ctrl_S306_Conv2d_64x1x3x3_Custom,
                               4, 1,
                               64, 64, 20, 16,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S309_Conv2d_64x64x1x1;
    CNN_InitGenCtrl(&gen_ctrl_S309_Conv2d_64x64x1x1);
    CNN_SetGenCtrl(&gen_ctrl_S309_Conv2d_64x64x1x1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for Conv_303
    CNN_ConvolutionPoolAct_SQ8("S309_Conv2d_64x64x1x1", &gen_ctrl_S309_Conv2d_64x64x1x1,
                               4, 1,
                               64, 64, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    
    // generator for expr_68
    s310_kernel_gen("S310_Op_expr_68");
    
    CNN_GenControl_T gen_ctrl_S313_Conv2d_1x64x1x1_Sigmoid;
    CNN_InitGenCtrl(&gen_ctrl_S313_Conv2d_1x64x1x1_Sigmoid);
    CNN_SetGenCtrl(&gen_ctrl_S313_Conv2d_1x64x1x1_Sigmoid, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for Conv_306_fusion
    CNN_ConvolutionPoolAct_SQ8("S313_Conv2d_1x64x1x1_Sigmoid", &gen_ctrl_S313_Conv2d_1x64x1x1_Sigmoid,
                               4, 1,
                               64, 1, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_SIGMOID);
    
    CNN_GenControl_T gen_ctrl_S316_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S316_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S316_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "custom_2");
    CNN_SetGenCtrl(&gen_ctrl_S316_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(13));
    // generator for Conv_307_fusion
    CNN_ConvolutionPoolAct_SQ8("S316_Conv2d_64x1x3x3_Custom", &gen_ctrl_S316_Conv2d_64x1x3x3_Custom,
                               4, 1,
                               64, 64, 20, 16,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S319_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S319_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S319_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S319_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S319_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_310_fusion
    CNN_ConvolutionPoolAct_SQ8("S319_Conv2d_64x64x1x1_Custom", &gen_ctrl_S319_Conv2d_64x64x1x1_Custom,
                               4, 1,
                               64, 64, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S322_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S322_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S322_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S322_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_313_fusion
    CNN_ConvolutionPoolAct_SQ8("S322_Conv2d_64x1x3x3_Custom", &gen_ctrl_S322_Conv2d_64x1x3x3_Custom,
                               4, 1,
                               64, 64, 20, 16,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S325_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S325_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S325_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S325_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S325_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_316_fusion
    CNN_ConvolutionPoolAct_SQ8("S325_Conv2d_64x64x1x1_Custom", &gen_ctrl_S325_Conv2d_64x64x1x1_Custom,
                               4, 1,
                               64, 64, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S328_Conv2d_4x64x1x1;
    CNN_InitGenCtrl(&gen_ctrl_S328_Conv2d_4x64x1x1);
    CNN_SetGenCtrl(&gen_ctrl_S328_Conv2d_4x64x1x1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for Conv_319
    CNN_ConvolutionPoolAct_SQ8("S328_Conv2d_4x64x1x1", &gen_ctrl_S328_Conv2d_4x64x1x1,
                               4, 1,
                               64, 4, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S331_Conv2d_1x64x1x1_Sigmoid;
    CNN_InitGenCtrl(&gen_ctrl_S331_Conv2d_1x64x1x1_Sigmoid);
    CNN_SetGenCtrl(&gen_ctrl_S331_Conv2d_1x64x1x1_Sigmoid, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for Conv_320_fusion
    CNN_ConvolutionPoolAct_SQ8("S331_Conv2d_1x64x1x1_Sigmoid", &gen_ctrl_S331_Conv2d_1x64x1x1_Sigmoid,
                               4, 1,
                               64, 1, 20, 16,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_SIGMOID);
    
    CNN_GenControl_T gen_ctrl_S336_Conv2d_64x256x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S336_Conv2d_64x256x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S336_Conv2d_64x256x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S336_Conv2d_64x256x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S336_Conv2d_64x256x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_324_fusion
    CNN_ConvolutionPoolAct_SQ8("S336_Conv2d_64x256x1x1_Custom", &gen_ctrl_S336_Conv2d_64x256x1x1_Custom,
                               4, 1,
                               256, 64, 10, 8,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S339_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S339_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S339_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S339_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_327_fusion
    CNN_ConvolutionPoolAct_SQ8("S339_Conv2d_64x1x3x3_Custom", &gen_ctrl_S339_Conv2d_64x1x3x3_Custom,
                               4, 1,
                               64, 64, 10, 8,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S342_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S342_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S342_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S342_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S342_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_330_fusion
    CNN_ConvolutionPoolAct_SQ8("S342_Conv2d_64x64x1x1_Custom", &gen_ctrl_S342_Conv2d_64x64x1x1_Custom,
                               4, 1,
                               64, 64, 10, 8,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S345_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S345_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S345_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S345_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_333_fusion
    CNN_ConvolutionPoolAct_SQ8("S345_Conv2d_64x1x3x3_Custom", &gen_ctrl_S345_Conv2d_64x1x3x3_Custom,
                               4, 1,
                               64, 64, 10, 8,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S348_Conv2d_64x64x1x1;
    CNN_InitGenCtrl(&gen_ctrl_S348_Conv2d_64x64x1x1);
    CNN_SetGenCtrl(&gen_ctrl_S348_Conv2d_64x64x1x1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for Conv_336
    CNN_ConvolutionPoolAct_SQ8("S348_Conv2d_64x64x1x1", &gen_ctrl_S348_Conv2d_64x64x1x1,
                               4, 1,
                               64, 64, 10, 8,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    
    // generator for expr_78
    s349_kernel_gen("S349_Op_expr_78");
    
    CNN_GenControl_T gen_ctrl_S352_Conv2d_1x64x1x1_Sigmoid;
    CNN_InitGenCtrl(&gen_ctrl_S352_Conv2d_1x64x1x1_Sigmoid);
    CNN_SetGenCtrl(&gen_ctrl_S352_Conv2d_1x64x1x1_Sigmoid, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for Conv_339_fusion
    CNN_ConvolutionPoolAct_SQ8("S352_Conv2d_1x64x1x1_Sigmoid", &gen_ctrl_S352_Conv2d_1x64x1x1_Sigmoid,
                               4, 1,
                               64, 1, 10, 8,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_SIGMOID);
    
    CNN_GenControl_T gen_ctrl_S355_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S355_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S355_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S355_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_340_fusion
    CNN_ConvolutionPoolAct_SQ8("S355_Conv2d_64x1x3x3_Custom", &gen_ctrl_S355_Conv2d_64x1x3x3_Custom,
                               4, 1,
                               64, 64, 10, 8,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S358_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S358_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S358_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S358_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S358_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_343_fusion
    CNN_ConvolutionPoolAct_SQ8("S358_Conv2d_64x64x1x1_Custom", &gen_ctrl_S358_Conv2d_64x64x1x1_Custom,
                               4, 1,
                               64, 64, 10, 8,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S361_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S361_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S361_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S361_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_346_fusion
    CNN_ConvolutionPoolAct_SQ8("S361_Conv2d_64x1x3x3_Custom", &gen_ctrl_S361_Conv2d_64x1x3x3_Custom,
                               4, 1,
                               64, 64, 10, 8,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S364_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S364_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S364_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S364_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "custom_3");
    CNN_SetGenCtrl(&gen_ctrl_S364_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_INFOS", AT_OPT_VAL(17));
    // generator for Conv_349_fusion
    CNN_ConvolutionPoolAct_SQ8("S364_Conv2d_64x64x1x1_Custom", &gen_ctrl_S364_Conv2d_64x64x1x1_Custom,
                               4, 1,
                               64, 64, 10, 8,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S367_Conv2d_4x64x1x1;
    CNN_InitGenCtrl(&gen_ctrl_S367_Conv2d_4x64x1x1);
    CNN_SetGenCtrl(&gen_ctrl_S367_Conv2d_4x64x1x1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for Conv_352
    CNN_ConvolutionPoolAct_SQ8("S367_Conv2d_4x64x1x1", &gen_ctrl_S367_Conv2d_4x64x1x1,
                               4, 1,
                               64, 4, 10, 8,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S370_Conv2d_1x64x1x1_Sigmoid;
    CNN_InitGenCtrl(&gen_ctrl_S370_Conv2d_1x64x1x1_Sigmoid);
    CNN_SetGenCtrl(&gen_ctrl_S370_Conv2d_1x64x1x1_Sigmoid, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for Conv_353_fusion
    CNN_ConvolutionPoolAct_SQ8("S370_Conv2d_1x64x1x1_Sigmoid", &gen_ctrl_S370_Conv2d_1x64x1x1_Sigmoid,
                               4, 1,
                               64, 1, 10, 8,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_SIGMOID);
    
    
    // generator for Concat_381
    CNN_ConcatLastAxis_Generator("S373_Concat", 0, 1, 6, 1280, 320, 80, 0, KOP_CONCAT);
    

#define GRAPH
#ifdef GRAPH
    CreateGraph("modelCNN",
        /* Arguments either passed or globals */
            CArgs(630,
                TCArgInfo("signed char * __restrict__", "Input_1", ARG_SCOPE_ARG_ALLOC, ARG_DIR_IN, AT_MEM_L2, AT_MEM_L2, 0),
                TCArgInfo("signed char * __restrict__", "Conv_0_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_0_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1138", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1138.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S3_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S3_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S3_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S3_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.09741 out: 0.11345  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S3_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S3_Infos.tensor", 1, 1, 8, 0)),
                // Ref(0,0): -21 Ref(0,1): -21 Ref(0,2): 199 Ref(0,3): 220 Ref(0,4): 16
                TCArgInfo("signed char * __restrict__", "S3_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S3_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_3_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_3_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1141", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1141.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S6_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S6_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S6_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S6_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.12995 out: 0.11486  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S6_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S6_Infos.tensor", 1, 1, 8, 0)),
                // Ref(0,0): 15 Ref(0,1): 15 Ref(0,2): 133 Ref(0,3): 2 Ref(0,4): 145
                TCArgInfo("signed char * __restrict__", "S6_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S6_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_6_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_6_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1144", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1144.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S9_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S9_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S9_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S9_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.28874 out: 0.22794  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S9_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S9_Infos.tensor", 1, 1, 8, 0)),
                // Ref(1,0): 27 Ref(1,1): 27 Ref(1,2): 148 Ref(1,3): 3 Ref(1,4): 162
                TCArgInfo("signed char * __restrict__", "S9_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S9_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_9_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_9_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1147", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1147.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S12_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S12_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S12_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S12_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.14744 out: 0.12341  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S12_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S12_Infos.tensor", 1, 1, 8, 0)),
                // Ref(3,0): 21 Ref(3,1): 21 Ref(3,2): 151 Ref(3,3): 2 Ref(3,4): 153
                TCArgInfo("signed char * __restrict__", "S12_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S12_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_15_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_15_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1153", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1153.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S17_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S17_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S17_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S17_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.06004 out: 0.08192  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S17_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S17_Infos.tensor", 1, 1, 8, 0)),
                // Ref(0,0): -46 Ref(0,1): -46 Ref(0,2): 246 Ref(0,3): 188 Ref(0,4): 16
                TCArgInfo("signed char * __restrict__", "S17_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S17_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_18_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_18_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1156", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1156.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S20_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S20_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S20_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S20_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.12488 out: 0.11338  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S20_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S20_Infos.tensor", 1, 1, 8, 0)),
                // Ref(2,0): 12 Ref(2,1): 12 Ref(2,2): 256 Ref(2,3): 1 Ref(2,4): 141
                TCArgInfo("signed char * __restrict__", "S20_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S20_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_21_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_21_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1159", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1159.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S23_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S23_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S23_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S23_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S23_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S23_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_26_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_26_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1162", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1162.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S28_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S28_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S28_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S28_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.08807 out: 0.09760  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S28_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S28_Infos.tensor", 1, 1, 8, 0)),
                // Ref(1,0): -14 Ref(1,1): -14 Ref(1,2): 180 Ref(1,3): 231 Ref(1,4): 16
                TCArgInfo("signed char * __restrict__", "S28_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S28_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_29_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_29_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1165", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1165.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S31_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S31_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S31_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S31_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.07049 out: 0.06768  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S31_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S31_Infos.tensor", 1, 1, 8, 0)),
                // Ref(2,0): 5 Ref(2,1): 5 Ref(2,2): 144 Ref(2,3): 133 Ref(2,4): 15
                TCArgInfo("signed char * __restrict__", "S31_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S31_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_32_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_32_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1168", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1168.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S34_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S34_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S34_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S34_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.06324 out: 0.03846  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S34_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S34_Infos.tensor", 1, 1, 8, 0)),
                // Ref(3,0): 49 Ref(3,1): 49 Ref(3,2): 130 Ref(3,3): 210 Ref(3,4): 15
                TCArgInfo("signed char * __restrict__", "S34_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S34_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_35_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_35_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1171", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1171.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S37_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S37_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S37_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S37_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04893 out: 0.04990  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S37_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S37_Infos.tensor", 1, 1, 8, 0)),
                // Ref(30,0): -2 Ref(30,1): -2 Ref(30,2): 200 Ref(30,3): 251 Ref(30,4): 16
                TCArgInfo("signed char * __restrict__", "S37_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S37_Custom_infos.tensor", 1, 1, 8, 0)),
                // in q: -6.39<(i8-0.00)*0.04990336<6.34 forced out_q: -4.30<(i8-0.00)*0.03355826<4.26
                TCArgInfo("signed char * __restrict__", "S39_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S39_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_41_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_41_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1177", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1177.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S43_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S43_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S43_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S43_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03037 out: 0.04005  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S43_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S43_Infos.tensor", 1, 1, 8, 0)),
                // Ref(0,0): -41 Ref(0,1): -41 Ref(0,2): 249 Ref(0,3): 1 Ref(0,4): 194 Ref(0,5): 16
                TCArgInfo("signed char * __restrict__", "S43_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S43_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_44_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_44_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1180", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1180.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S46_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S46_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S46_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S46_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.06754 out: 0.05372  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S46_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S46_Infos.tensor", 1, 1, 8, 0)),
                // Ref(4,0): 26 Ref(4,1): 26 Ref(4,2): 138 Ref(4,3): 161 Ref(4,4): 15
                TCArgInfo("signed char * __restrict__", "S46_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S46_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_47_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_47_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1183", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1183.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S49_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S49_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S49_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S49_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S49_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S49_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_51_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_51_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1186", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1186.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S53_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S53_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S53_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S53_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03280 out: 0.03683  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S53_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S53_Infos.tensor", 1, 1, 8, 0)),
                // Ref(1,0): -17 Ref(1,1): -17 Ref(1,2): 134 Ref(1,3): 228 Ref(1,4): 16
                TCArgInfo("signed char * __restrict__", "S53_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S53_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_54_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_54_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1189", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1189.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S56_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S56_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S56_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S56_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03504 out: 0.03617  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S56_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S56_Infos.tensor", 1, 1, 8, 0)),
                // Ref(2,0): -5 Ref(2,1): -5 Ref(2,2): 144 Ref(2,3): 248 Ref(2,4): 16
                TCArgInfo("signed char * __restrict__", "S56_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S56_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_57_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_57_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1192", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1192.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S59_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S59_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S59_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S59_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03109 out: 0.02672  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S59_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S59_Infos.tensor", 1, 1, 8, 0)),
                // Ref(1,0): 15 Ref(1,1): 15 Ref(1,2): 255 Ref(1,3): 1 Ref(1,4): 149 Ref(1,5): 15
                TCArgInfo("signed char * __restrict__", "S59_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S59_Custom_infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [29] IN1SCALEN: [4] OUTSCALE: [145] OUTSCALEN: [8]
                TCArgInfo("signed char * __restrict__", "S60_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S60_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_61_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_61_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1195", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1195.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S63_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S63_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S63_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S63_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02772 out: 0.03317  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S63_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S63_Infos.tensor", 1, 1, 8, 0)),
                // Ref(2,0): -27 Ref(2,1): -27 Ref(2,2): 227 Ref(2,3): 1 Ref(2,4): 214 Ref(2,5): 16
                TCArgInfo("signed char * __restrict__", "S63_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S63_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_64_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_64_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1198", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1198.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S66_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S66_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S66_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S66_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03033 out: 0.03283  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S66_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S66_Infos.tensor", 1, 1, 8, 0)),
                // Ref(3,0): -12 Ref(3,1): -12 Ref(3,2): 248 Ref(3,3): 1 Ref(3,4): 237 Ref(3,5): 16
                TCArgInfo("signed char * __restrict__", "S66_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S66_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_67_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_67_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1201", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1201.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S69_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S69_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S69_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S69_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03415 out: 0.03089  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S69_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S69_Infos.tensor", 1, 1, 8, 0)),
                // Ref(3,0): 10 Ref(3,1): 10 Ref(3,2): 140 Ref(3,3): 142 Ref(3,4): 15
                TCArgInfo("signed char * __restrict__", "S69_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S69_Custom_infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [195] IN1SCALEN: [7] OUTSCALE: [79] OUTSCALEN: [7]
                TCArgInfo("signed char * __restrict__", "S70_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S70_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_72_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_72_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1204", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1204.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S74_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S74_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S74_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S74_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03310 out: 0.02932  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S74_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S74_Infos.tensor", 1, 1, 8, 0)),
                // Ref(4,0): 12 Ref(4,1): 12 Ref(4,2): 136 Ref(4,3): 144 Ref(4,4): 15
                TCArgInfo("signed char * __restrict__", "S74_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S74_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_75_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_75_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1207", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1207.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S77_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S77_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S77_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S77_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.05878 out: 0.06670  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S77_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S77_Infos.tensor", 1, 1, 8, 0)),
                // Ref(5,0): -17 Ref(5,1): -17 Ref(5,2): 241 Ref(5,3): 226 Ref(5,4): 16
                TCArgInfo("signed char * __restrict__", "S77_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S77_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_78_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_78_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1210", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1210.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S80_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S80_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S80_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S80_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04232 out: 0.04505  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S80_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S80_Infos.tensor", 1, 1, 8, 0)),
                // Ref(6,0): -9 Ref(6,1): -9 Ref(6,2): 173 Ref(6,3): 240 Ref(6,4): 16
                TCArgInfo("signed char * __restrict__", "S80_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S80_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_81_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_81_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1213", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1213.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S83_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S83_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S83_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S83_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.05513 out: 0.06245  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S83_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S83_Infos.tensor", 1, 1, 8, 0)),
                // Ref(31,0): -19 Ref(31,1): -19 Ref(31,2): 226 Ref(31,3): 226 Ref(31,4): 16
                TCArgInfo("signed char * __restrict__", "S83_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S83_Custom_infos.tensor", 1, 1, 8, 0)),
                // in q: -7.99<(i8-0.00)*0.06244617<7.93 forced out_q: -4.14<(i8-0.00)*0.03232218<4.10
                TCArgInfo("signed char * __restrict__", "S85_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S85_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_87_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_87_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1219", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1219.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S89_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S89_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S89_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S89_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02805 out: 0.02950  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S89_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S89_Infos.tensor", 1, 1, 8, 0)),
                // Ref(4,0): -9 Ref(4,1): -9 Ref(4,2): 230 Ref(4,3): 1 Ref(4,4): 243 Ref(4,5): 16
                TCArgInfo("signed char * __restrict__", "S89_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S89_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_90_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_90_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1222", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1222.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S92_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S92_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S92_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S92_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.05544 out: 0.04409  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S92_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S92_Infos.tensor", 1, 1, 8, 0)),
                // Ref(7,0): 26 Ref(7,1): 26 Ref(7,2): 227 Ref(7,3): 161 Ref(7,4): 15
                TCArgInfo("signed char * __restrict__", "S92_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S92_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_93_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_93_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1225", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1225.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S95_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S95_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S95_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S95_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S95_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S95_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_97_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_97_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1228", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1228.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S99_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S99_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S99_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S99_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02088 out: 0.02128  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S99_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S99_Infos.tensor", 1, 1, 8, 0)),
                // Ref(5,0): -10 Ref(5,1): -10 Ref(5,2): 171 Ref(5,3): 1 Ref(5,4): 251 Ref(5,5): 16
                TCArgInfo("signed char * __restrict__", "S99_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S99_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_100_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_100_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1231", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1231.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S102_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S102_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S102_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S102_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03424 out: 0.03226  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S102_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S102_Infos.tensor", 1, 1, 8, 0)),
                // Ref(8,0): 5 Ref(8,1): 5 Ref(8,2): 140 Ref(8,3): 136 Ref(8,4): 15
                TCArgInfo("signed char * __restrict__", "S102_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S102_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_103_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_103_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1234", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1234.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S105_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S105_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S105_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S105_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02932 out: 0.03287  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S105_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S105_Infos.tensor", 1, 1, 8, 0)),
                // Ref(6,0): -17 Ref(6,1): -17 Ref(6,2): 240 Ref(6,3): 1 Ref(6,4): 228 Ref(6,5): 16
                TCArgInfo("signed char * __restrict__", "S105_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S105_Custom_infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [147] IN1SCALEN: [7] OUTSCALE: [109] OUTSCALEN: [7]
                TCArgInfo("signed char * __restrict__", "S106_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S106_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_107_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_107_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1237", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1237.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S109_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S109_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S109_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S109_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01865 out: 0.01937  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S109_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S109_Infos.tensor", 1, 1, 8, 0)),
                // Ref(7,0): -14 Ref(7,1): -14 Ref(7,2): 153 Ref(7,3): 1 Ref(7,4): 246 Ref(7,5): 16
                TCArgInfo("signed char * __restrict__", "S109_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S109_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_110_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_110_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1240", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1240.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S112_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S112_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S112_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S112_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03172 out: 0.03738  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S112_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S112_Infos.tensor", 1, 1, 8, 0)),
                // Ref(9,0): -24 Ref(9,1): -24 Ref(9,2): 130 Ref(9,3): 217 Ref(9,4): 16
                TCArgInfo("signed char * __restrict__", "S112_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S112_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_113_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_113_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1243", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1243.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S115_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S115_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S115_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S115_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03690 out: 0.04884  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S115_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S115_Infos.tensor", 1, 1, 8, 0)),
                // Ref(10,0): -41 Ref(10,1): -41 Ref(10,2): 151 Ref(10,3): 193 Ref(10,4): 16
                TCArgInfo("signed char * __restrict__", "S115_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S115_Custom_infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [81] IN1SCALEN: [6] OUTSCALE: [79] OUTSCALEN: [7]
                TCArgInfo("signed char * __restrict__", "S116_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S116_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_118_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_118_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1246", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1246.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S120_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S120_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S120_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S120_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03429 out: 0.04072  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S120_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S120_Infos.tensor", 1, 1, 8, 0)),
                // Ref(11,0): -25 Ref(11,1): -25 Ref(11,2): 140 Ref(11,3): 216 Ref(11,4): 16
                TCArgInfo("signed char * __restrict__", "S120_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S120_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_121_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_121_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1249", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1249.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S123_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S123_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S123_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S123_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04094 out: 0.04790  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S123_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S123_Infos.tensor", 1, 1, 8, 0)),
                // Ref(12,0): -22 Ref(12,1): -22 Ref(12,2): 168 Ref(12,3): 219 Ref(12,4): 16
                TCArgInfo("signed char * __restrict__", "S123_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S123_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_124_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_124_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1252", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1252.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S126_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S126_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S126_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S126_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03146 out: 0.03882  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S126_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S126_Infos.tensor", 1, 1, 8, 0)),
                // Ref(13,0): -31 Ref(13,1): -31 Ref(13,2): 129 Ref(13,3): 207 Ref(13,4): 16
                TCArgInfo("signed char * __restrict__", "S126_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S126_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_127_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_127_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1255", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1255.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S129_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S129_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S129_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S129_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02697 out: 0.03483  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S129_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S129_Infos.tensor", 1, 1, 8, 0)),
                // Ref(8,0): -39 Ref(8,1): -39 Ref(8,2): 221 Ref(8,3): 1 Ref(8,4): 198 Ref(8,5): 16
                TCArgInfo("signed char * __restrict__", "S129_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S129_Custom_infos.tensor", 1, 1, 8, 0)),
                // no activation ACTSCALE: [1] ACTSCALEN: [0]
                TCArgInfo("signed char * __restrict__", "S130_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S130_Infos.tensor", 1, 1, 8, 0)),
                // no activation ACTSCALE: [1] ACTSCALEN: [0]
                TCArgInfo("signed char * __restrict__", "S131_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S131_Infos.tensor", 1, 1, 8, 0)),
                // no activation ACTSCALE: [1] ACTSCALEN: [0]
                TCArgInfo("signed char * __restrict__", "S132_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S132_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_134_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_134_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1258", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1258.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S136_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S136_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S136_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S136_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03233 out: 0.03694  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S136_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S136_Infos.tensor", 1, 1, 8, 0)),
                // Ref(14,0): -19 Ref(14,1): -19 Ref(14,2): 132 Ref(14,3): 224 Ref(14,4): 16
                TCArgInfo("signed char * __restrict__", "S136_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S136_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_137_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_137_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1261", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1261.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S139_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S139_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S139_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S139_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02893 out: 0.03261  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S139_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S139_Infos.tensor", 1, 1, 8, 0)),
                // Ref(41,0): -12 Ref(41,1): -12 Ref(41,2): 237 Ref(41,3): 1 Ref(41,4): 227 Ref(41,5): 16
                TCArgInfo("signed char * __restrict__", "S139_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S139_Custom_infos.tensor", 1, 1, 8, 0)),
                // in q: -4.17<(i8-0.00)*0.03261126<4.14 out_q: -3.97<(i8-0.00)*0.03102084<3.94
                TCArgInfo("signed char * __restrict__", "S141_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S141_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_143_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_143_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1267", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1267.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S145_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S145_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S145_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S145_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01442 out: 0.01615  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S145_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S145_Infos.tensor", 1, 1, 8, 0)),
                // Ref(9,0): -30 Ref(9,1): -30 Ref(9,2): 236 Ref(9,3): 2 Ref(9,4): 228 Ref(9,5): 16
                TCArgInfo("signed char * __restrict__", "S145_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S145_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_146_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_146_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1270", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1270.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S148_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S148_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S148_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S148_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03207 out: 0.04192  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S148_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S148_Infos.tensor", 1, 1, 8, 0)),
                // Ref(15,0): -40 Ref(15,1): -40 Ref(15,2): 131 Ref(15,3): 196 Ref(15,4): 16
                TCArgInfo("signed char * __restrict__", "S148_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S148_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_149_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_149_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1273", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1273.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S151_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S151_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S151_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S151_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02875 out: 0.03261  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S151_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S151_Infos.tensor", 1, 1, 8, 0)),
                // Ref(10,0): -19 Ref(10,1): -19 Ref(10,2): 236 Ref(10,3): 1 Ref(10,4): 226 Ref(10,5): 16
                TCArgInfo("signed char * __restrict__", "S151_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S151_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_153_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_153_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1276", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1276.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S155_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S155_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S155_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S155_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03322 out: 0.03626  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S155_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S155_Infos.tensor", 1, 1, 8, 0)),
                // Ref(16,0): -13 Ref(16,1): -13 Ref(16,2): 136 Ref(16,3): 235 Ref(16,4): 16
                TCArgInfo("signed char * __restrict__", "S155_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S155_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_156_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_156_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1279", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1279.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S158_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S158_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S158_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S158_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02685 out: 0.04072  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S158_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S158_Infos.tensor", 1, 1, 8, 0)),
                // Ref(11,0): -5 Ref(11,1): -5 Ref(11,2): 220 Ref(11,3): 1 Ref(11,4): 169 Ref(11,5): 16
                TCArgInfo("signed char * __restrict__", "S158_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S158_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_162_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_162_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1282", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1282.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S163_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S163_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S163_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S163_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02986 out: 0.02963  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S163_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S163_Infos.tensor", 1, 1, 8, 0)),
                // Ref(42,0): -9 Ref(42,1): -9 Ref(42,2): 245 Ref(42,3): 1 Ref(42,4): 129 Ref(42,5): 15
                TCArgInfo("signed char * __restrict__", "S163_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S163_Custom_infos.tensor", 1, 1, 8, 0)),
                // in q: -3.79<(i8-0.00)*0.02963090<3.76 out_q: -3.21<(i8-0.00)*0.02504119<3.18
                TCArgInfo("signed char * __restrict__", "S165_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S165_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_168_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_168_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1288", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1288.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S169_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S169_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S169_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S169_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01790 out: 0.02035  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S169_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S169_Infos.tensor", 1, 1, 8, 0)),
                // Ref(12,0): -27 Ref(12,1): -27 Ref(12,2): 147 Ref(12,3): 1 Ref(12,4): 225 Ref(12,5): 16
                TCArgInfo("signed char * __restrict__", "S169_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S169_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_171_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_171_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1291", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1291.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S172_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S172_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S172_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S172_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03248 out: 0.03477  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S172_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S172_Infos.tensor", 1, 1, 8, 0)),
                // Ref(17,0): -11 Ref(17,1): -11 Ref(17,2): 133 Ref(17,3): 239 Ref(17,4): 16
                TCArgInfo("signed char * __restrict__", "S172_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S172_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_174_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_174_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1294", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1294.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S175_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S175_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S175_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S175_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02506 out: 0.02963  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S175_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S175_Infos.tensor", 1, 1, 8, 0)),
                // Ref(13,0): -26 Ref(13,1): -26 Ref(13,2): 205 Ref(13,3): 1 Ref(13,4): 216 Ref(13,5): 16
                TCArgInfo("signed char * __restrict__", "S175_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S175_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_178_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_178_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1297", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1297.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S179_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S179_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S179_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S179_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03244 out: 0.03661  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S179_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S179_Infos.tensor", 1, 1, 8, 0)),
                // Ref(18,0): -18 Ref(18,1): -18 Ref(18,2): 133 Ref(18,3): 227 Ref(18,4): 16
                TCArgInfo("signed char * __restrict__", "S179_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S179_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_181_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_181_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1300", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1300.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S182_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S182_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S182_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S182_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02263 out: 0.04729  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S182_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S182_Infos.tensor", 1, 1, 8, 0)),
                // Ref(14,0): -26 Ref(14,1): -26 Ref(14,2): 185 Ref(14,3): 1 Ref(14,4): 245 Ref(14,5): 17
                TCArgInfo("signed char * __restrict__", "S182_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S182_Custom_infos.tensor", 1, 1, 8, 0)),
                // in q: -6.05<(i8-0.00)*0.04729335<6.01 out_q: -3.75<(i8-0.00)*0.02932421<3.72 forced
                TCArgInfo("signed char * __restrict__", "S184_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S184_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_187_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_187_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1303", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1303.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S188_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S188_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S188_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S188_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03118 out: 0.03877  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S188_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S188_Infos.tensor", 1, 1, 8, 0)),
                // Ref(43,0): 24 Ref(43,1): 24 Ref(43,2): 255 Ref(43,3): 1 Ref(43,4): 206 Ref(43,5): 16
                TCArgInfo("signed char * __restrict__", "S188_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S188_Custom_infos.tensor", 1, 1, 8, 0)),
                // in q: -4.96<(i8-0.00)*0.03877128<4.92 out_q: -2.98<(i8-0.00)*0.02329703<2.96
                TCArgInfo("signed char * __restrict__", "S190_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S190_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_193_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_193_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1309", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1309.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S194_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S194_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S194_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S194_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01793 out: 0.02303  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S194_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S194_Infos.tensor", 1, 1, 8, 0)),
                // Ref(15,0): -44 Ref(15,1): -44 Ref(15,2): 147 Ref(15,3): 1 Ref(15,4): 199 Ref(15,5): 16
                TCArgInfo("signed char * __restrict__", "S194_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S194_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_196_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_196_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1312", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1312.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S197_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S197_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S197_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S197_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04532 out: 0.04118  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S197_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S197_Infos.tensor", 1, 1, 8, 0)),
                // Ref(19,0): 11 Ref(19,1): 11 Ref(19,2): 186 Ref(19,3): 141 Ref(19,4): 15
                TCArgInfo("signed char * __restrict__", "S197_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S197_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_199_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_199_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1315", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1315.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S200_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S200_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S200_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S200_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03824 out: 0.03877  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S200_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S200_Infos.tensor", 1, 1, 8, 0)),
                // Ref(20,0): -3 Ref(20,1): -3 Ref(20,2): 157 Ref(20,3): 252 Ref(20,4): 16
                TCArgInfo("signed char * __restrict__", "S200_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S200_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_203_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_203_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1318", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1318.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S204_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S204_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S204_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S204_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03073 out: 0.03388  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S204_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S204_Infos.tensor", 1, 1, 8, 0)),
                // Ref(16,0): -15 Ref(16,1): -15 Ref(16,2): 252 Ref(16,3): 1 Ref(16,4): 232 Ref(16,5): 16
                TCArgInfo("signed char * __restrict__", "S204_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S204_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_206_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_206_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1321", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1321.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S207_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S207_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S207_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S207_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03767 out: 0.04461  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S207_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S207_Infos.tensor", 1, 1, 8, 0)),
                // Ref(21,0): -24 Ref(21,1): -24 Ref(21,2): 154 Ref(21,3): 216 Ref(21,4): 16
                TCArgInfo("signed char * __restrict__", "S207_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S207_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_209_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_209_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1324", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1324.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S210_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S210_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S210_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S210_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03974 out: 0.04729  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S210_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S210_Infos.tensor", 1, 1, 8, 0)),
                // Ref(22,0): -24 Ref(22,1): -24 Ref(22,2): 163 Ref(22,3): 215 Ref(22,4): 16
                TCArgInfo("signed char * __restrict__", "S210_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S210_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_213_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_213_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1327", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1327.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S214_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S214_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S214_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S214_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03727 out: 0.03480  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S214_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S214_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_219_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_219_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1333", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1333.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S219_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S219_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S219_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S219_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02564 out: 0.03176  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S219_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S219_Infos.tensor", 1, 1, 8, 0)),
                // Ref(17,0): -33 Ref(17,1): -33 Ref(17,2): 210 Ref(17,3): 1 Ref(17,4): 207 Ref(17,5): 16
                TCArgInfo("signed char * __restrict__", "S219_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S219_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_222_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_222_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1336", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1336.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S222_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S222_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S222_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S222_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03338 out: 0.03444  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S222_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S222_Infos.tensor", 1, 1, 8, 0)),
                // Ref(23,0): -6 Ref(23,1): -6 Ref(23,2): 137 Ref(23,3): 248 Ref(23,4): 16
                TCArgInfo("signed char * __restrict__", "S222_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S222_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_225_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_225_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1339", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1339.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S225_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S225_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S225_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S225_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02197 out: 0.03480  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S225_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S225_Infos.tensor", 1, 1, 8, 0)),
                // Ref(18,0): 8 Ref(18,1): 8 Ref(18,2): 180 Ref(18,3): 1 Ref(18,4): 162 Ref(18,5): 16
                TCArgInfo("signed char * __restrict__", "S225_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S225_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_229_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_229_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1342", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1342.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S229_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S229_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S229_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S229_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02910 out: 0.02719  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S229_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S229_Infos.tensor", 1, 1, 8, 0)),
                // Ref(19,0): 5 Ref(19,1): 5 Ref(19,2): 238 Ref(19,3): 1 Ref(19,4): 137 Ref(19,5): 15
                TCArgInfo("signed char * __restrict__", "S229_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S229_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_232_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_232_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1345", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1345.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S232_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S232_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S232_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S232_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02228 out: 0.02600  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S232_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S232_Infos.tensor", 1, 1, 8, 0)),
                // Ref(20,0): -26 Ref(20,1): -26 Ref(20,2): 183 Ref(20,3): 1 Ref(20,4): 219 Ref(20,5): 16
                TCArgInfo("signed char * __restrict__", "S232_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S232_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_235_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_235_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1348", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1348.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S235_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S235_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S235_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S235_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01768 out: 0.04072  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S235_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S235_Infos.tensor", 1, 1, 8, 0)),
                // Ref(21,0): -7 Ref(21,1): -7 Ref(21,2): 145 Ref(21,3): 1 Ref(21,4): 222 Ref(21,5): 17
                TCArgInfo("signed char * __restrict__", "S235_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S235_Custom_infos.tensor", 1, 1, 8, 0)),
                // in q: -5.21<(i8-0.00)*0.04071943<5.17 forced out_q: -3.48<(i8-0.00)*0.02722353<3.46
                TCArgInfo("signed char * __restrict__", "S237_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S237_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_239_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_239_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1351", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1351.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S240_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S240_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S240_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S240_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01930 out: 0.01934  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S240_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S240_Infos.tensor", 1, 1, 8, 0)),
                // Ref(44,0): -28 Ref(44,1): -28 Ref(44,2): 158 Ref(44,3): 1 Ref(44,4): 255 Ref(44,5): 16
                TCArgInfo("signed char * __restrict__", "S240_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S240_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_245_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_245_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1357", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1357.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S245_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S245_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S245_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S245_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01642 out: 0.01932  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S245_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S245_Infos.tensor", 1, 1, 8, 0)),
                // Ref(22,0): -33 Ref(22,1): -33 Ref(22,2): 134 Ref(22,3): 1 Ref(22,4): 218 Ref(22,5): 16
                TCArgInfo("signed char * __restrict__", "S245_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S245_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_248_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_248_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1360", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1360.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S248_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S248_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S248_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S248_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01262 out: 0.01491  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S248_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S248_Infos.tensor", 1, 1, 8, 0)),
                // Ref(23,0): -41 Ref(23,1): -41 Ref(23,2): 207 Ref(23,3): 2 Ref(23,4): 217 Ref(23,5): 16
                TCArgInfo("signed char * __restrict__", "S248_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S248_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_251_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_251_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1363", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1363.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S251_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S251_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S251_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S251_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00810 out: 0.01934  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S251_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S251_Infos.tensor", 1, 1, 8, 0)),
                // Ref(24,0): -9 Ref(24,1): -9 Ref(24,2): 133 Ref(24,3): 2 Ref(24,4): 214 Ref(24,5): 17
                TCArgInfo("signed char * __restrict__", "S251_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S251_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_255_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_255_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1366", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1366.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S255_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S255_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S255_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S255_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01481 out: 0.01764  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S255_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S255_Infos.tensor", 1, 1, 8, 0)),
                // Ref(25,0): -37 Ref(25,1): -37 Ref(25,2): 243 Ref(25,3): 2 Ref(25,4): 215 Ref(25,5): 16
                TCArgInfo("signed char * __restrict__", "S255_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S255_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_258_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_258_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1369", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1369.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S258_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S258_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S258_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S258_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03630 out: 0.04939  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S258_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S258_Infos.tensor", 1, 1, 8, 0)),
                // Ref(24,0): -46 Ref(24,1): -46 Ref(24,2): 149 Ref(24,3): 188 Ref(24,4): 16
                TCArgInfo("signed char * __restrict__", "S258_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S258_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_261_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_261_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1372", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1372.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S261_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S261_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S261_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S261_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04298 out: 0.03769  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S261_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S261_Infos.tensor", 1, 1, 8, 0)),
                // Ref(25,0): 15 Ref(25,1): 15 Ref(25,2): 176 Ref(25,3): 146 Ref(25,4): 15
                TCArgInfo("signed char * __restrict__", "S261_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S261_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_264_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_264_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1375", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1375.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S264_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S264_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S264_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S264_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03364 out: 0.02315  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S264_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S264_Infos.tensor", 1, 1, 8, 0)),
                // Ref(26,0): 36 Ref(26,1): 36 Ref(26,2): 138 Ref(26,3): 186 Ref(26,4): 15
                TCArgInfo("signed char * __restrict__", "S264_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S264_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_267_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_267_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1378", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1378.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S267_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S267_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S267_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S267_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03767 out: 0.02641  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S267_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S267_Infos.tensor", 1, 1, 8, 0)),
                // Ref(27,0): 35 Ref(27,1): 35 Ref(27,2): 154 Ref(27,3): 183 Ref(27,4): 15
                TCArgInfo("signed char * __restrict__", "S267_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S267_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_270_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_270_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1381", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1381.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S270_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S270_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S270_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S270_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S270_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S270_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_273_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_273_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_head_cls_preds_0_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant_head_cls_preds_0_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S274_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S274_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S274_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S274_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00024 out: 0.02471  actscale: [81] actscalen: [16] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S274_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S274_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_274_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_274_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1384", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1384.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S277_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S277_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S277_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S277_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.07484 out: 0.08160  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S277_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S277_Infos.tensor", 1, 1, 8, 0)),
                // Ref(5,0): -11 Ref(5,1): -11 Ref(5,2): 153 Ref(5,3): 235 Ref(5,4): 16
                TCArgInfo("signed char * __restrict__", "S277_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S277_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_277_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_277_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1387", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1387.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S280_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S280_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S280_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S280_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.07740 out: 0.07276  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S280_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S280_Infos.tensor", 1, 1, 8, 0)),
                // Ref(6,0): 8 Ref(6,1): 8 Ref(6,2): 159 Ref(6,3): 136 Ref(6,4): 15
                TCArgInfo("signed char * __restrict__", "S280_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S280_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_280_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_280_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1390", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1390.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S283_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S283_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S283_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S283_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.05798 out: 0.03523  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S283_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S283_Infos.tensor", 1, 1, 8, 0)),
                // Ref(28,0): 49 Ref(28,1): 49 Ref(28,2): 237 Ref(28,3): 211 Ref(28,4): 15
                TCArgInfo("signed char * __restrict__", "S283_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S283_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_283_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_283_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1393", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1393.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S286_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S286_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S286_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S286_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.07003 out: 0.03681  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S286_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S286_Infos.tensor", 1, 1, 8, 0)),
                // Ref(7,0): 60 Ref(7,1): 60 Ref(7,2): 143 Ref(7,3): 244 Ref(7,4): 15
                TCArgInfo("signed char * __restrict__", "S286_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S286_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_286_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_286_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_head_reg_preds_0_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant_head_reg_preds_0_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S289_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S289_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S289_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S289_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S289_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S289_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_287_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_287_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_head_obj_preds_0_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant_head_obj_preds_0_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S292_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S292_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S292_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S292_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00024 out: 0.02471  actscale: [81] actscalen: [16] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S292_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S292_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_291_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_291_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1396", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1396.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S297_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S297_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S297_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S297_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02431 out: 0.02660  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S297_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S297_Infos.tensor", 1, 1, 8, 0)),
                // Ref(26,0): -16 Ref(26,1): -16 Ref(26,2): 199 Ref(26,3): 1 Ref(26,4): 234 Ref(26,5): 16
                TCArgInfo("signed char * __restrict__", "S297_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S297_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_294_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_294_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1399", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1399.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S300_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S300_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S300_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S300_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02263 out: 0.01843  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S300_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S300_Infos.tensor", 1, 1, 8, 0)),
                // Ref(27,0): 15 Ref(27,1): 15 Ref(27,2): 185 Ref(27,3): 1 Ref(27,4): 157 Ref(27,5): 15
                TCArgInfo("signed char * __restrict__", "S300_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S300_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_297_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_297_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1402", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1402.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S303_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S303_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S303_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S303_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01383 out: 0.01204  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S303_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S303_Infos.tensor", 1, 1, 8, 0)),
                // Ref(28,0): -2 Ref(28,1): -2 Ref(28,2): 227 Ref(28,3): 2 Ref(28,4): 147 Ref(28,5): 15
                TCArgInfo("signed char * __restrict__", "S303_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S303_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_300_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_300_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1405", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1405.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S306_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S306_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S306_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S306_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01393 out: 0.01158  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S306_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S306_Infos.tensor", 1, 1, 8, 0)),
                // Ref(29,0): 3 Ref(29,1): 3 Ref(29,2): 228 Ref(29,3): 2 Ref(29,4): 154 Ref(29,5): 15
                TCArgInfo("signed char * __restrict__", "S306_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S306_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_303_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_303_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1408", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1408.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S309_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S309_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S309_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S309_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S309_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S309_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_306_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_306_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_head_cls_preds_1_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant_head_cls_preds_1_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S313_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S313_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S313_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S313_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00024 out: 0.02471  actscale: [81] actscalen: [16] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S313_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S313_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_307_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_307_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1411", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1411.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S316_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S316_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S316_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S316_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03445 out: 0.04027  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S316_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S316_Infos.tensor", 1, 1, 8, 0)),
                // Ref(29,0): -22 Ref(29,1): -22 Ref(29,2): 141 Ref(29,3): 219 Ref(29,4): 16
                TCArgInfo("signed char * __restrict__", "S316_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S316_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_310_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_310_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1414", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1414.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S319_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S319_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S319_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S319_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02513 out: 0.02206  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S319_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S319_Infos.tensor", 1, 1, 8, 0)),
                // Ref(30,0): 10 Ref(30,1): 10 Ref(30,2): 206 Ref(30,3): 1 Ref(30,4): 146 Ref(30,5): 15
                TCArgInfo("signed char * __restrict__", "S319_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S319_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_313_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_313_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1417", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1417.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S322_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S322_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S322_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S322_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02520 out: 0.02674  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S322_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S322_Infos.tensor", 1, 1, 8, 0)),
                // Ref(31,0): -12 Ref(31,1): -12 Ref(31,2): 206 Ref(31,3): 1 Ref(31,4): 241 Ref(31,5): 16
                TCArgInfo("signed char * __restrict__", "S322_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S322_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_316_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_316_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1420", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1420.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S325_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S325_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S325_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S325_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01840 out: 0.01702  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S325_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S325_Infos.tensor", 1, 1, 8, 0)),
                // Ref(32,0): -2 Ref(32,1): -2 Ref(32,2): 151 Ref(32,3): 1 Ref(32,4): 138 Ref(32,5): 15
                TCArgInfo("signed char * __restrict__", "S325_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S325_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_319_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_319_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_head_reg_preds_1_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant_head_reg_preds_1_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S328_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S328_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S328_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S328_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S328_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S328_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_320_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_320_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_head_obj_preds_1_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant_head_obj_preds_1_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S331_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S331_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S331_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S331_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00024 out: 0.02471  actscale: [81] actscalen: [16] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S331_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S331_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_324_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_324_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1423", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1423.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S336_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S336_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S336_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S336_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01253 out: 0.01313  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S336_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S336_Infos.tensor", 1, 1, 8, 0)),
                // Ref(33,0): -26 Ref(33,1): -26 Ref(33,2): 205 Ref(33,3): 2 Ref(33,4): 244 Ref(33,5): 16
                TCArgInfo("signed char * __restrict__", "S336_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S336_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_327_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_327_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1426", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1426.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S339_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S339_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S339_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S339_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01169 out: 0.01486  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S339_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S339_Infos.tensor", 1, 1, 8, 0)),
                // Ref(34,0): -54 Ref(34,1): -54 Ref(34,2): 192 Ref(34,3): 2 Ref(34,4): 201 Ref(34,5): 16
                TCArgInfo("signed char * __restrict__", "S339_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S339_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_330_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_330_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1429", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1429.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S342_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S342_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S342_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S342_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01379 out: 0.01335  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S342_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S342_Infos.tensor", 1, 1, 8, 0)),
                // Ref(35,0): -14 Ref(35,1): -14 Ref(35,2): 226 Ref(35,3): 2 Ref(35,4): 132 Ref(35,5): 15
                TCArgInfo("signed char * __restrict__", "S342_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S342_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_333_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_333_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1432", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1432.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S345_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S345_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S345_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S345_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01194 out: 0.01606  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S345_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S345_Infos.tensor", 1, 1, 8, 0)),
                // Ref(36,0): -62 Ref(36,1): -62 Ref(36,2): 196 Ref(36,3): 2 Ref(36,4): 190 Ref(36,5): 16
                TCArgInfo("signed char * __restrict__", "S345_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S345_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_336_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_336_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1435", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1435.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S348_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S348_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S348_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S348_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S348_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S348_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_339_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_339_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_head_cls_preds_2_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant_head_cls_preds_2_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S352_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S352_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S352_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S352_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00024 out: 0.02471  actscale: [81] actscalen: [16] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S352_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S352_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_340_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_340_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1438", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1438.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S355_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S355_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S355_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S355_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01217 out: 0.01252  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S355_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S355_Infos.tensor", 1, 1, 8, 0)),
                // Ref(37,0): -24 Ref(37,1): -24 Ref(37,2): 199 Ref(37,3): 2 Ref(37,4): 249 Ref(37,5): 16
                TCArgInfo("signed char * __restrict__", "S355_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S355_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_343_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_343_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1441", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1441.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S358_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S358_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S358_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S358_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01043 out: 0.00885  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S358_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S358_Infos.tensor", 1, 1, 8, 0)),
                // Ref(38,0): -7 Ref(38,1): -7 Ref(38,2): 171 Ref(38,3): 2 Ref(38,4): 151 Ref(38,5): 15
                TCArgInfo("signed char * __restrict__", "S358_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S358_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_346_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_346_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1444", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1444.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S361_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S361_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S361_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S361_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01227 out: 0.00900  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S361_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S361_Infos.tensor", 1, 1, 8, 0)),
                // Ref(39,0): 11 Ref(39,1): 11 Ref(39,2): 201 Ref(39,3): 2 Ref(39,4): 174 Ref(39,5): 15
                TCArgInfo("signed char * __restrict__", "S361_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S361_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_349_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_349_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant__1447", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant__1447.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S364_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S364_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S364_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S364_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.01296 out: 0.01174  BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S364_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S364_Infos.tensor", 1, 1, 8, 0)),
                // Ref(40,0): -8 Ref(40,1): -8 Ref(40,2): 212 Ref(40,3): 2 Ref(40,4): 141 Ref(40,5): 15
                TCArgInfo("signed char * __restrict__", "S364_Custom_infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S364_Custom_infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_352_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_352_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_head_reg_preds_2_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant_head_reg_preds_2_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S367_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S367_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S367_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S367_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S367_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S367_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv_353_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Conv_353_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_head_obj_preds_2_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/Constant_head_obj_preds_2_bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S370_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S370_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S370_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S370_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00024 out: 0.02471  actscale: [81] actscalen: [16] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S370_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S370_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Output_1", ARG_SCOPE_ARG, ARG_DIR_OUT, AT_MEM_L2, AT_MEM_L2, 0)
            ),
        /* Locals, allocated dynamically */
        CArgs(114,
            TCArgInfo("signed char * __restrict__", "S3_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S6_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S9_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S12_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S17_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S20_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S23_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S25_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S28_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S31_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S34_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S37_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S39_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S43_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S46_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S49_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S50_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S53_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S56_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S59_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S60_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S63_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S66_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S69_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S71_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S77_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S80_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S83_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S85_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S89_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S92_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S95_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S96_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S99_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S102_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S105_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S106_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S109_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S112_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S115_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S117_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S123_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S126_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S133_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S136_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S139_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S141_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S145_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S148_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S152_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S155_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S160_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S163_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S165_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S169_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S172_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S176_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S179_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S183_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S185_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S188_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S190_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S194_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S197_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S201_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S204_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S207_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S211_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S214_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S219_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S222_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S226_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S229_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S232_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S236_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S237_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S240_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S245_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S248_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S252_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S255_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S258_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S261_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S264_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S267_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S270_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S271_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S277_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S280_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S283_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S286_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S293_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S297_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S300_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S303_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S306_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S309_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S310_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S316_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S319_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S322_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S325_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S332_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S336_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S339_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S342_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S345_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S348_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S349_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S355_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S358_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S361_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S364_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S371_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0)
        )
    );

    /* Stacked tensors - Concats */
    AddStackedTensors("S25_Output", 2, "S24_Output", "S14_Output");
    AddStackedTensors("S71_Output", 2, "S70_Output", "S40_Output");
    AddStackedTensors("S117_Output", 2, "S116_Output", "S86_Output");
    AddStackedTensors("S133_Output", 4, "S129_Output", "S131_Output", "S132_Output", "S130_Output");
    AddStackedTensors("S152_Output", 2, "S151_Output", "S142_Output");
    AddStackedTensors("S160_Output", 2, "S159_Output", "S120_Output");
    AddStackedTensors("S176_Output", 2, "S175_Output", "S166_Output");
    AddStackedTensors("S185_Output", 2, "S184_Output", "S74_Output");
    AddStackedTensors("S201_Output", 2, "S200_Output", "S191_Output");
    AddStackedTensors("S211_Output", 2, "S210_Output", "S182_Output");
    AddStackedTensors("S226_Output", 2, "S225_Output", "S216_Output");
    AddStackedTensors("S236_Output", 2, "S235_Output", "S158_Output");
    AddStackedTensors("S252_Output", 2, "S251_Output", "S242_Output");
    AddStackedTensors("S293_Output", 3, "S289_Output", "S292_Output", "S274_Output");
    AddStackedTensors("S332_Output", 3, "S328_Output", "S331_Output", "S313_Output");
    AddStackedTensors("S371_Output", 3, "S367_Output", "S370_Output", "S352_Output");
    AddStackedTensors("S188_Output", 2, "S189_Output_0", "S189_Output_1");
    AddStackedTensors("S37_Output", 2, "S38_Output_0", "S38_Output_1");
    AddStackedTensors("S12_Output", 2, "S13_Output_0", "S13_Output_1");
    AddStackedTensors("S163_Output", 2, "S164_Output_0", "S164_Output_1");
    AddStackedTensors("S139_Output", 2, "S140_Output_0", "S140_Output_1");
    AddStackedTensors("S83_Output", 2, "S84_Output_0", "S84_Output_1");
    AddStackedTensors("S214_Output", 2, "S215_Output_0", "S215_Output_1");
    AddStackedTensors("S240_Output", 2, "S241_Output_0", "S241_Output_1");

    // Node S3_Conv2d_16x12x3x3_Custom inq -256.00<(i8-0.00)*2.00000000<254.00 forced weightsq chan<(i8-0.00)*chan<chan outq -14.52<(i8-0.00)*0.11344516<14.41 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S3_Conv2d_16x12x3x3_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "Input_1", 0),
            GNodeArg(GNA_IN, "Conv_0_weights", 0),
            GNodeArg(GNA_IN, "Constant__1138", 0),
            GNodeArg(GNA_OUT, "S3_Output", 0),
            GNodeArg(GNA_IN, "S3_Mul_scale", 0),
            GNodeArg(GNA_IN, "S3_Mul_shift", 0),
            GNodeArg(GNA_IN, "S3_Infos", 0),
            GNodeArg(GNA_IN, "S3_Custom_infos", 0)
        )
    );
    // Node S6_Conv2d_16x1x3x3_Custom inq -14.52<(i8-0.00)*0.11344516<14.41 forced weightsq chan<(i8-0.00)*chan<chan outq -14.70<(i8-0.00)*0.11486062<14.59 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S6_Conv2d_16x1x3x3_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S3_Output", 0),
            GNodeArg(GNA_IN, "Conv_3_weights", 0),
            GNodeArg(GNA_IN, "Constant__1141", 0),
            GNodeArg(GNA_OUT, "S6_Output", 0),
            GNodeArg(GNA_IN, "S6_Mul_scale", 0),
            GNodeArg(GNA_IN, "S6_Mul_shift", 0),
            GNodeArg(GNA_IN, "S6_Infos", 0),
            GNodeArg(GNA_IN, "S6_Custom_infos", 0)
        )
    );
    // Node S9_Conv2d_32x16x1x1_Custom inq -14.70<(i8-0.00)*0.11486062<14.59 weightsq chan<(i8-0.00)*chan<chan outq -29.18<(i8-0.00)*0.22794475<28.95 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S9_Conv2d_32x16x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S6_Output", 0),
            GNodeArg(GNA_IN, "Conv_6_weights", 0),
            GNodeArg(GNA_IN, "Constant__1144", 0),
            GNodeArg(GNA_OUT, "S9_Output", 0),
            GNodeArg(GNA_IN, "S9_Mul_scale", 0),
            GNodeArg(GNA_IN, "S9_Mul_shift", 0),
            GNodeArg(GNA_IN, "S9_Infos", 0),
            GNodeArg(GNA_IN, "S9_Custom_infos", 0)
        )
    );
    // Node S12_Conv2d_32x32x1x1_Custom inq -29.18<(i8-0.00)*0.22794475<28.95 weightsq chan<(i8-0.00)*chan<chan outq -15.80<(i8-0.00)*0.12340641<15.67 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S12_Conv2d_32x32x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S9_Output", 0),
            GNodeArg(GNA_IN, "Conv_9_weights", 0),
            GNodeArg(GNA_IN, "Constant__1147", 0),
            GNodeArg(GNA_OUT, "S12_Output", 0),
            GNodeArg(GNA_IN, "S12_Mul_scale", 0),
            GNodeArg(GNA_IN, "S12_Mul_shift", 0),
            GNodeArg(GNA_IN, "S12_Infos", 0),
            GNodeArg(GNA_IN, "S12_Custom_infos", 0)
        )
    );
    // Node Conv_9_split_copy inq -15.80<(i8-0.00)*0.12340641<15.67 outq -15.80<(i8-0.00)*0.12340641<15.67
    AddNode("S14_Op_Conv_9_split_copy",
        Bindings(2,
            GNodeArg(GNA_IN, "S13_Output_1", 0),
            GNodeArg(GNA_OUT, "S14_Output", 0)
        )
    );
    // Node S17_Conv2d_16x16x1x1_Custom inq -15.80<(i8-0.00)*0.12340641<15.67 weightsq chan<(i8-0.00)*chan<chan outq -10.49<(i8-0.00)*0.08192277<10.40 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S17_Conv2d_16x16x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S13_Output_0", 0),
            GNodeArg(GNA_IN, "Conv_15_weights", 0),
            GNodeArg(GNA_IN, "Constant__1153", 0),
            GNodeArg(GNA_OUT, "S17_Output", 0),
            GNodeArg(GNA_IN, "S17_Mul_scale", 0),
            GNodeArg(GNA_IN, "S17_Mul_shift", 0),
            GNodeArg(GNA_IN, "S17_Infos", 0),
            GNodeArg(GNA_IN, "S17_Custom_infos", 0)
        )
    );
    // Node S20_Conv2d_16x1x3x3_Custom inq -10.49<(i8-0.00)*0.08192277<10.40 forced weightsq chan<(i8-0.00)*chan<chan outq -14.51<(i8-0.00)*0.11337733<14.40 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S20_Conv2d_16x1x3x3_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S17_Output", 0),
            GNodeArg(GNA_IN, "Conv_18_weights", 0),
            GNodeArg(GNA_IN, "Constant__1156", 0),
            GNodeArg(GNA_OUT, "S20_Output", 0),
            GNodeArg(GNA_IN, "S20_Mul_scale", 0),
            GNodeArg(GNA_IN, "S20_Mul_shift", 0),
            GNodeArg(GNA_IN, "S20_Infos", 0),
            GNodeArg(GNA_IN, "S20_Custom_infos", 0)
        )
    );
    // Node S23_Conv2d_16x16x1x1 inq -14.51<(i8-0.00)*0.11337733<14.40 weightsq chan<(i8-0.00)*chan<chan outq -14.63<(i8-8.00)*0.10759941<12.80 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S23_Conv2d_16x16x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S20_Output", 0),
            GNodeArg(GNA_IN, "Conv_21_weights", 0),
            GNodeArg(GNA_IN, "Constant__1159", 0),
            GNodeArg(GNA_OUT, "S23_Output", 0),
            GNodeArg(GNA_IN, "S23_Mul_scale", 0),
            GNodeArg(GNA_IN, "S23_Mul_shift", 0),
            GNodeArg(GNA_IN, "S23_Infos", 0)
        )
    );
    // Node expr_0 in_qs [-14.63<(i8-8.00)*0.10759941<12.80,-15.80<(i8-0.00)*0.12340641<15.67] out_qs [-15.80<(i8-0.00)*0.12340641<15.67]
    AddNode("S24_Op_expr_0",
        Bindings(3,
            GNodeArg(GNA_IN, "S23_Output", 0),
            GNodeArg(GNA_IN, "S13_Output_0", 0),
            GNodeArg(GNA_OUT, "S24_Output", 0)
        )
    );
    // Node S28_Conv2d_32x32x1x1_Custom inq -15.80<(i8-0.00)*0.12340641<15.67 weightsq chan<(i8-0.00)*chan<chan outq -12.49<(i8-0.00)*0.09759504<12.39 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S28_Conv2d_32x32x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S25_Output", 0),
            GNodeArg(GNA_IN, "Conv_26_weights", 0),
            GNodeArg(GNA_IN, "Constant__1162", 0),
            GNodeArg(GNA_OUT, "S28_Output", 0),
            GNodeArg(GNA_IN, "S28_Mul_scale", 0),
            GNodeArg(GNA_IN, "S28_Mul_shift", 0),
            GNodeArg(GNA_IN, "S28_Infos", 0),
            GNodeArg(GNA_IN, "S28_Custom_infos", 0)
        )
    );
    // Node S31_Conv2d_32x1x3x3_Custom inq -12.49<(i8-0.00)*0.09759504<12.39 forced weightsq chan<(i8-0.00)*chan<chan outq -8.66<(i8-0.00)*0.06767595<8.59 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S31_Conv2d_32x1x3x3_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S28_Output", 0),
            GNodeArg(GNA_IN, "Conv_29_weights", 0),
            GNodeArg(GNA_IN, "Constant__1165", 0),
            GNodeArg(GNA_OUT, "S31_Output", 0),
            GNodeArg(GNA_IN, "S31_Mul_scale", 0),
            GNodeArg(GNA_IN, "S31_Mul_shift", 0),
            GNodeArg(GNA_IN, "S31_Infos", 0),
            GNodeArg(GNA_IN, "S31_Custom_infos", 0)
        )
    );
    // Node S34_Conv2d_64x32x1x1_Custom inq -8.66<(i8-0.00)*0.06767595<8.59 weightsq chan<(i8-0.00)*chan<chan outq -4.92<(i8-0.00)*0.03846117<4.88 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S34_Conv2d_64x32x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S31_Output", 0),
            GNodeArg(GNA_IN, "Conv_32_weights", 0),
            GNodeArg(GNA_IN, "Constant__1168", 0),
            GNodeArg(GNA_OUT, "S34_Output", 0),
            GNodeArg(GNA_IN, "S34_Mul_scale", 0),
            GNodeArg(GNA_IN, "S34_Mul_shift", 0),
            GNodeArg(GNA_IN, "S34_Infos", 0),
            GNodeArg(GNA_IN, "S34_Custom_infos", 0)
        )
    );
    // Node S37_Conv2d_64x64x1x1_Custom inq -4.92<(i8-0.00)*0.03846117<4.88 weightsq chan<(i8-0.00)*chan<chan outq -6.39<(i8-0.00)*0.04990336<6.34 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S37_Conv2d_64x64x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S34_Output", 0),
            GNodeArg(GNA_IN, "Conv_35_weights", 0),
            GNodeArg(GNA_IN, "Constant__1171", 0),
            GNodeArg(GNA_OUT, "S37_Output", 0),
            GNodeArg(GNA_IN, "S37_Mul_scale", 0),
            GNodeArg(GNA_IN, "S37_Mul_shift", 0),
            GNodeArg(GNA_IN, "S37_Infos", 0),
            GNodeArg(GNA_IN, "S37_Custom_infos", 0)
        )
    );
    // Node Conv_41_fusion_qin0 inq -6.39<(i8-0.00)*0.04990336<6.34 forced outq -4.30<(i8-0.00)*0.03355826<4.26
    AddNode("S39_Op_Conv_41_fusion_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S38_Output_0", 0),
            GNodeArg(GNA_OUT, "S39_Output", 0),
            GNodeArg(GNA_IN, "S39_Infos", 0)
        )
    );
    // Node Conv_35_split_copy inq -6.39<(i8-0.00)*0.04990336<6.34 forced outq -6.39<(i8-0.00)*0.04990336<6.34 forced
    AddNode("S40_Op_Conv_35_split_copy",
        Bindings(2,
            GNodeArg(GNA_IN, "S38_Output_1", 0),
            GNodeArg(GNA_OUT, "S40_Output", 0)
        )
    );
    // Node S43_Conv2d_32x32x1x1_Custom inq -6.39<(i8-0.00)*0.04990336<6.34 forced weightsq chan<(i8-0.00)*chan<chan outq -5.13<(i8-0.00)*0.04004702<5.09 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S43_Conv2d_32x32x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S39_Output", 0),
            GNodeArg(GNA_IN, "Conv_41_weights", 0),
            GNodeArg(GNA_IN, "Constant__1177", 0),
            GNodeArg(GNA_OUT, "S43_Output", 0),
            GNodeArg(GNA_IN, "S43_Mul_scale", 0),
            GNodeArg(GNA_IN, "S43_Mul_shift", 0),
            GNodeArg(GNA_IN, "S43_Infos", 0),
            GNodeArg(GNA_IN, "S43_Custom_infos", 0)
        )
    );
    // Node S46_Conv2d_32x1x3x3_Custom inq -5.13<(i8-0.00)*0.04004702<5.09 forced weightsq chan<(i8-0.00)*chan<chan outq -6.88<(i8-0.00)*0.05371643<6.82 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S46_Conv2d_32x1x3x3_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S43_Output", 0),
            GNodeArg(GNA_IN, "Conv_44_weights", 0),
            GNodeArg(GNA_IN, "Constant__1180", 0),
            GNodeArg(GNA_OUT, "S46_Output", 0),
            GNodeArg(GNA_IN, "S46_Mul_scale", 0),
            GNodeArg(GNA_IN, "S46_Mul_shift", 0),
            GNodeArg(GNA_IN, "S46_Infos", 0),
            GNodeArg(GNA_IN, "S46_Custom_infos", 0)
        )
    );
    // Node S49_Conv2d_32x32x1x1 inq -6.88<(i8-0.00)*0.05371643<6.82 weightsq chan<(i8-0.00)*chan<chan outq -2.97<(i8--11.00)*0.02542610<3.51 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S49_Conv2d_32x32x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S46_Output", 0),
            GNodeArg(GNA_IN, "Conv_47_weights", 0),
            GNodeArg(GNA_IN, "Constant__1183", 0),
            GNodeArg(GNA_OUT, "S49_Output", 0),
            GNodeArg(GNA_IN, "S49_Mul_scale", 0),
            GNodeArg(GNA_IN, "S49_Mul_shift", 0),
            GNodeArg(GNA_IN, "S49_Infos", 0)
        )
    );
    // Node expr_1 in_qs [-2.97<(i8--11.00)*0.02542610<3.51,-6.39<(i8-0.00)*0.04990336<6.34 forced] out_qs [-6.19<(i8-0.00)*0.04834068<6.14]
    AddNode("S50_Op_expr_1",
        Bindings(3,
            GNodeArg(GNA_IN, "S49_Output", 0),
            GNodeArg(GNA_IN, "S38_Output_0", 0),
            GNodeArg(GNA_OUT, "S50_Output", 0)
        )
    );
    // Node S53_Conv2d_32x32x1x1_Custom inq -6.19<(i8-0.00)*0.04834068<6.14 forced weightsq chan<(i8-0.00)*chan<chan outq -4.71<(i8-0.00)*0.03683466<4.68 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S53_Conv2d_32x32x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S50_Output", 0),
            GNodeArg(GNA_IN, "Conv_51_weights", 0),
            GNodeArg(GNA_IN, "Constant__1186", 0),
            GNodeArg(GNA_OUT, "S53_Output", 0),
            GNodeArg(GNA_IN, "S53_Mul_scale", 0),
            GNodeArg(GNA_IN, "S53_Mul_shift", 0),
            GNodeArg(GNA_IN, "S53_Infos", 0),
            GNodeArg(GNA_IN, "S53_Custom_infos", 0)
        )
    );
    // Node S56_Conv2d_32x1x3x3_Custom inq -4.71<(i8-0.00)*0.03683466<4.68 forced weightsq chan<(i8-0.00)*chan<chan outq -4.63<(i8-0.00)*0.03617436<4.59 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S56_Conv2d_32x1x3x3_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S53_Output", 0),
            GNodeArg(GNA_IN, "Conv_54_weights", 0),
            GNodeArg(GNA_IN, "Constant__1189", 0),
            GNodeArg(GNA_OUT, "S56_Output", 0),
            GNodeArg(GNA_IN, "S56_Mul_scale", 0),
            GNodeArg(GNA_IN, "S56_Mul_shift", 0),
            GNodeArg(GNA_IN, "S56_Infos", 0),
            GNodeArg(GNA_IN, "S56_Custom_infos", 0)
        )
    );
    // Node S59_Conv2d_32x32x1x1_Custom inq -4.63<(i8-0.00)*0.03617436<4.59 weightsq chan<(i8-0.00)*chan<chan outq -3.42<(i8-0.00)*0.02671523<3.39 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S59_Conv2d_32x32x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S56_Output", 0),
            GNodeArg(GNA_IN, "Conv_57_weights", 0),
            GNodeArg(GNA_IN, "Constant__1192", 0),
            GNodeArg(GNA_OUT, "S59_Output", 0),
            GNodeArg(GNA_IN, "S59_Mul_scale", 0),
            GNodeArg(GNA_IN, "S59_Mul_shift", 0),
            GNodeArg(GNA_IN, "S59_Infos", 0),
            GNodeArg(GNA_IN, "S59_Custom_infos", 0)
        )
    );
    // Node S60_MatAdd_32x32x40 in1q -6.19<(i8-0.00)*0.04834068<6.14 forced
    //   in2q -3.42<(i8-0.00)*0.02671523<3.39 forced
    //   outq -6.03<(i8-0.00)*0.04713576<5.99 forced scaled input 0 is node input 1
    AddNode("S60_MatAdd_32x32x40",
        Bindings(4,
            GNodeArg(GNA_IN, "S50_Output", 0),
            GNodeArg(GNA_IN, "S59_Output", 0),
            GNodeArg(GNA_OUT, "S60_Output", 0),
            GNodeArg(GNA_IN, "S60_Infos", 0)
        )
    );
    // Node S63_Conv2d_32x32x1x1_Custom inq -6.03<(i8-0.00)*0.04713576<5.99 forced weightsq chan<(i8-0.00)*chan<chan outq -4.25<(i8-0.00)*0.03317373<4.21 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S63_Conv2d_32x32x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S60_Output", 0),
            GNodeArg(GNA_IN, "Conv_61_weights", 0),
            GNodeArg(GNA_IN, "Constant__1195", 0),
            GNodeArg(GNA_OUT, "S63_Output", 0),
            GNodeArg(GNA_IN, "S63_Mul_scale", 0),
            GNodeArg(GNA_IN, "S63_Mul_shift", 0),
            GNodeArg(GNA_IN, "S63_Infos", 0),
            GNodeArg(GNA_IN, "S63_Custom_infos", 0)
        )
    );
    // Node S66_Conv2d_32x1x3x3_Custom inq -4.25<(i8-0.00)*0.03317373<4.21 forced weightsq chan<(i8-0.00)*chan<chan outq -4.20<(i8-0.00)*0.03283278<4.17 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S66_Conv2d_32x1x3x3_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S63_Output", 0),
            GNodeArg(GNA_IN, "Conv_64_weights", 0),
            GNodeArg(GNA_IN, "Constant__1198", 0),
            GNodeArg(GNA_OUT, "S66_Output", 0),
            GNodeArg(GNA_IN, "S66_Mul_scale", 0),
            GNodeArg(GNA_IN, "S66_Mul_shift", 0),
            GNodeArg(GNA_IN, "S66_Infos", 0),
            GNodeArg(GNA_IN, "S66_Custom_infos", 0)
        )
    );
    // Node S69_Conv2d_32x32x1x1_Custom inq -4.20<(i8-0.00)*0.03283278<4.17 weightsq chan<(i8-0.00)*chan<chan outq -3.95<(i8-0.00)*0.03089452<3.92 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S69_Conv2d_32x32x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S66_Output", 0),
            GNodeArg(GNA_IN, "Conv_67_weights", 0),
            GNodeArg(GNA_IN, "Constant__1201", 0),
            GNodeArg(GNA_OUT, "S69_Output", 0),
            GNodeArg(GNA_IN, "S69_Mul_scale", 0),
            GNodeArg(GNA_IN, "S69_Mul_shift", 0),
            GNodeArg(GNA_IN, "S69_Infos", 0),
            GNodeArg(GNA_IN, "S69_Custom_infos", 0)
        )
    );
    // Node S70_MatAdd_32x32x40 in1q -6.03<(i8-0.00)*0.04713576<5.99 forced
    //   in2q -3.95<(i8-0.00)*0.03089452<3.92 forced
    //   outq -6.39<(i8-0.00)*0.04990336<6.34 forced scaled input 0 is node input 1
    AddNode("S70_MatAdd_32x32x40",
        Bindings(4,
            GNodeArg(GNA_IN, "S60_Output", 0),
            GNodeArg(GNA_IN, "S69_Output", 0),
            GNodeArg(GNA_OUT, "S70_Output", 0),
            GNodeArg(GNA_IN, "S70_Infos", 0)
        )
    );
    // Node S74_Conv2d_64x64x1x1_Custom inq -6.39<(i8-0.00)*0.04990336<6.34 forced weightsq chan<(i8-0.00)*chan<chan outq -3.75<(i8-0.00)*0.02932421<3.72 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S74_Conv2d_64x64x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S71_Output", 0),
            GNodeArg(GNA_IN, "Conv_72_weights", 0),
            GNodeArg(GNA_IN, "Constant__1204", 0),
            GNodeArg(GNA_OUT, "S74_Output", 0),
            GNodeArg(GNA_IN, "S74_Mul_scale", 0),
            GNodeArg(GNA_IN, "S74_Mul_shift", 0),
            GNodeArg(GNA_IN, "S74_Infos", 0),
            GNodeArg(GNA_IN, "S74_Custom_infos", 0)
        )
    );
    // Node S77_Conv2d_64x1x3x3_Custom inq -3.75<(i8-0.00)*0.02932421<3.72 forced weightsq chan<(i8-0.00)*chan<chan outq -8.54<(i8-0.00)*0.06669504<8.47 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S77_Conv2d_64x1x3x3_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S74_Output", 0),
            GNodeArg(GNA_IN, "Conv_75_weights", 0),
            GNodeArg(GNA_IN, "Constant__1207", 0),
            GNodeArg(GNA_OUT, "S77_Output", 0),
            GNodeArg(GNA_IN, "S77_Mul_scale", 0),
            GNodeArg(GNA_IN, "S77_Mul_shift", 0),
            GNodeArg(GNA_IN, "S77_Infos", 0),
            GNodeArg(GNA_IN, "S77_Custom_infos", 0)
        )
    );
    // Node S80_Conv2d_128x64x1x1_Custom inq -8.54<(i8-0.00)*0.06669504<8.47 weightsq chan<(i8-0.00)*chan<chan outq -5.77<(i8-0.00)*0.04505371<5.72 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S80_Conv2d_128x64x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S77_Output", 0),
            GNodeArg(GNA_IN, "Conv_78_weights", 0),
            GNodeArg(GNA_IN, "Constant__1210", 0),
            GNodeArg(GNA_OUT, "S80_Output", 0),
            GNodeArg(GNA_IN, "S80_Mul_scale", 0),
            GNodeArg(GNA_IN, "S80_Mul_shift", 0),
            GNodeArg(GNA_IN, "S80_Infos", 0),
            GNodeArg(GNA_IN, "S80_Custom_infos", 0)
        )
    );
    // Node S83_Conv2d_128x128x1x1_Custom inq -5.77<(i8-0.00)*0.04505371<5.72 weightsq chan<(i8-0.00)*chan<chan outq -7.99<(i8-0.00)*0.06244617<7.93 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S83_Conv2d_128x128x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S80_Output", 0),
            GNodeArg(GNA_IN, "Conv_81_weights", 0),
            GNodeArg(GNA_IN, "Constant__1213", 0),
            GNodeArg(GNA_OUT, "S83_Output", 0),
            GNodeArg(GNA_IN, "S83_Mul_scale", 0),
            GNodeArg(GNA_IN, "S83_Mul_shift", 0),
            GNodeArg(GNA_IN, "S83_Infos", 0),
            GNodeArg(GNA_IN, "S83_Custom_infos", 0)
        )
    );
    // Node Conv_87_fusion_qin0 inq -7.99<(i8-0.00)*0.06244617<7.93 forced outq -4.14<(i8-0.00)*0.03232218<4.10
    AddNode("S85_Op_Conv_87_fusion_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S84_Output_0", 0),
            GNodeArg(GNA_OUT, "S85_Output", 0),
            GNodeArg(GNA_IN, "S85_Infos", 0)
        )
    );
    // Node Conv_81_split_copy inq -7.99<(i8-0.00)*0.06244617<7.93 forced outq -7.99<(i8-0.00)*0.06244617<7.93 forced
    AddNode("S86_Op_Conv_81_split_copy",
        Bindings(2,
            GNodeArg(GNA_IN, "S84_Output_1", 0),
            GNodeArg(GNA_OUT, "S86_Output", 0)
        )
    );
    // Node S89_Conv2d_64x64x1x1_Custom inq -7.99<(i8-0.00)*0.06244617<7.93 forced weightsq chan<(i8-0.00)*chan<chan outq -3.78<(i8-0.00)*0.02949577<3.75 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S89_Conv2d_64x64x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S85_Output", 0),
            GNodeArg(GNA_IN, "Conv_87_weights", 0),
            GNodeArg(GNA_IN, "Constant__1219", 0),
            GNodeArg(GNA_OUT, "S89_Output", 0),
            GNodeArg(GNA_IN, "S89_Mul_scale", 0),
            GNodeArg(GNA_IN, "S89_Mul_shift", 0),
            GNodeArg(GNA_IN, "S89_Infos", 0),
            GNodeArg(GNA_IN, "S89_Custom_infos", 0)
        )
    );
    // Node S92_Conv2d_64x1x3x3_Custom inq -3.78<(i8-0.00)*0.02949577<3.75 forced weightsq chan<(i8-0.00)*chan<chan outq -5.64<(i8-0.00)*0.04409288<5.60 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S92_Conv2d_64x1x3x3_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S89_Output", 0),
            GNodeArg(GNA_IN, "Conv_90_weights", 0),
            GNodeArg(GNA_IN, "Constant__1222", 0),
            GNodeArg(GNA_OUT, "S92_Output", 0),
            GNodeArg(GNA_IN, "S92_Mul_scale", 0),
            GNodeArg(GNA_IN, "S92_Mul_shift", 0),
            GNodeArg(GNA_IN, "S92_Infos", 0),
            GNodeArg(GNA_IN, "S92_Custom_infos", 0)
        )
    );
    // Node S95_Conv2d_64x64x1x1 inq -5.64<(i8-0.00)*0.04409288<5.60 weightsq chan<(i8-0.00)*chan<chan outq -2.70<(i8--8.00)*0.02245940<3.03 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S95_Conv2d_64x64x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S92_Output", 0),
            GNodeArg(GNA_IN, "Conv_93_weights", 0),
            GNodeArg(GNA_IN, "Constant__1225", 0),
            GNodeArg(GNA_OUT, "S95_Output", 0),
            GNodeArg(GNA_IN, "S95_Mul_scale", 0),
            GNodeArg(GNA_IN, "S95_Mul_shift", 0),
            GNodeArg(GNA_IN, "S95_Infos", 0)
        )
    );
    // Node expr_2 in_qs [-2.70<(i8--8.00)*0.02245940<3.03,-7.99<(i8-0.00)*0.06244617<7.93 forced] out_qs [-4.83<(i8-0.00)*0.03772724<4.79]
    AddNode("S96_Op_expr_2",
        Bindings(3,
            GNodeArg(GNA_IN, "S95_Output", 0),
            GNodeArg(GNA_IN, "S84_Output_0", 0),
            GNodeArg(GNA_OUT, "S96_Output", 0)
        )
    );
    // Node S99_Conv2d_64x64x1x1_Custom inq -4.83<(i8-0.00)*0.03772724<4.79 forced weightsq chan<(i8-0.00)*chan<chan outq -2.72<(i8-0.00)*0.02127561<2.70 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S99_Conv2d_64x64x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S96_Output", 0),
            GNodeArg(GNA_IN, "Conv_97_weights", 0),
            GNodeArg(GNA_IN, "Constant__1228", 0),
            GNodeArg(GNA_OUT, "S99_Output", 0),
            GNodeArg(GNA_IN, "S99_Mul_scale", 0),
            GNodeArg(GNA_IN, "S99_Mul_shift", 0),
            GNodeArg(GNA_IN, "S99_Infos", 0),
            GNodeArg(GNA_IN, "S99_Custom_infos", 0)
        )
    );
    // Node S102_Conv2d_64x1x3x3_Custom inq -2.72<(i8-0.00)*0.02127561<2.70 forced weightsq chan<(i8-0.00)*chan<chan outq -4.13<(i8-0.00)*0.03226497<4.10 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S102_Conv2d_64x1x3x3_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S99_Output", 0),
            GNodeArg(GNA_IN, "Conv_100_weights", 0),
            GNodeArg(GNA_IN, "Constant__1231", 0),
            GNodeArg(GNA_OUT, "S102_Output", 0),
            GNodeArg(GNA_IN, "S102_Mul_scale", 0),
            GNodeArg(GNA_IN, "S102_Mul_shift", 0),
            GNodeArg(GNA_IN, "S102_Infos", 0),
            GNodeArg(GNA_IN, "S102_Custom_infos", 0)
        )
    );
    // Node S105_Conv2d_64x64x1x1_Custom inq -4.13<(i8-0.00)*0.03226497<4.10 weightsq chan<(i8-0.00)*chan<chan outq -4.21<(i8-0.00)*0.03286811<4.17 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S105_Conv2d_64x64x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S102_Output", 0),
            GNodeArg(GNA_IN, "Conv_103_weights", 0),
            GNodeArg(GNA_IN, "Constant__1234", 0),
            GNodeArg(GNA_OUT, "S105_Output", 0),
            GNodeArg(GNA_IN, "S105_Mul_scale", 0),
            GNodeArg(GNA_IN, "S105_Mul_shift", 0),
            GNodeArg(GNA_IN, "S105_Infos", 0),
            GNodeArg(GNA_IN, "S105_Custom_infos", 0)
        )
    );
    // Node S106_MatAdd_64x16x20 in1q -4.83<(i8-0.00)*0.03772724<4.79 forced
    //   in2q -4.21<(i8-0.00)*0.03286811<4.17 forced
    //   outq -4.93<(i8-0.00)*0.03851897<4.89 forced scaled input 0 is node input 1
    AddNode("S106_MatAdd_64x16x20",
        Bindings(4,
            GNodeArg(GNA_IN, "S96_Output", 0),
            GNodeArg(GNA_IN, "S105_Output", 0),
            GNodeArg(GNA_OUT, "S106_Output", 0),
            GNodeArg(GNA_IN, "S106_Infos", 0)
        )
    );
    // Node S109_Conv2d_64x64x1x1_Custom inq -4.93<(i8-0.00)*0.03851897<4.89 forced weightsq chan<(i8-0.00)*chan<chan outq -2.48<(i8-0.00)*0.01936902<2.46 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S109_Conv2d_64x64x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S106_Output", 0),
            GNodeArg(GNA_IN, "Conv_107_weights", 0),
            GNodeArg(GNA_IN, "Constant__1237", 0),
            GNodeArg(GNA_OUT, "S109_Output", 0),
            GNodeArg(GNA_IN, "S109_Mul_scale", 0),
            GNodeArg(GNA_IN, "S109_Mul_shift", 0),
            GNodeArg(GNA_IN, "S109_Infos", 0),
            GNodeArg(GNA_IN, "S109_Custom_infos", 0)
        )
    );
    // Node S112_Conv2d_64x1x3x3_Custom inq -2.48<(i8-0.00)*0.01936902<2.46 forced weightsq chan<(i8-0.00)*chan<chan outq -4.78<(i8-0.00)*0.03738197<4.75 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S112_Conv2d_64x1x3x3_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S109_Output", 0),
            GNodeArg(GNA_IN, "Conv_110_weights", 0),
            GNodeArg(GNA_IN, "Constant__1240", 0),
            GNodeArg(GNA_OUT, "S112_Output", 0),
            GNodeArg(GNA_IN, "S112_Mul_scale", 0),
            GNodeArg(GNA_IN, "S112_Mul_shift", 0),
            GNodeArg(GNA_IN, "S112_Infos", 0),
            GNodeArg(GNA_IN, "S112_Custom_infos", 0)
        )
    );
    // Node S115_Conv2d_64x64x1x1_Custom inq -4.78<(i8-0.00)*0.03738197<4.75 weightsq chan<(i8-0.00)*chan<chan outq -6.25<(i8-0.00)*0.04884294<6.20 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S115_Conv2d_64x64x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S112_Output", 0),
            GNodeArg(GNA_IN, "Conv_113_weights", 0),
            GNodeArg(GNA_IN, "Constant__1243", 0),
            GNodeArg(GNA_OUT, "S115_Output", 0),
            GNodeArg(GNA_IN, "S115_Mul_scale", 0),
            GNodeArg(GNA_IN, "S115_Mul_shift", 0),
            GNodeArg(GNA_IN, "S115_Infos", 0),
            GNodeArg(GNA_IN, "S115_Custom_infos", 0)
        )
    );
    // Node S116_MatAdd_64x16x20 in1q -6.25<(i8-0.00)*0.04884294<6.20 forced
    //   in2q -4.93<(i8-0.00)*0.03851897<4.89 forced
    //   outq -7.99<(i8-0.00)*0.06244617<7.93 forced scaled input 0 is node input 0
    AddNode("S116_MatAdd_64x16x20",
        Bindings(4,
            GNodeArg(GNA_IN, "S115_Output", 0),
            GNodeArg(GNA_IN, "S106_Output", 0),
            GNodeArg(GNA_OUT, "S116_Output", 0),
            GNodeArg(GNA_IN, "S116_Infos", 0)
        )
    );
    // Node S120_Conv2d_128x128x1x1_Custom inq -7.99<(i8-0.00)*0.06244617<7.93 forced weightsq chan<(i8-0.00)*chan<chan outq -5.21<(i8-0.00)*0.04071943<5.17 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S120_Conv2d_128x128x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S117_Output", 0),
            GNodeArg(GNA_IN, "Conv_118_weights", 0),
            GNodeArg(GNA_IN, "Constant__1246", 0),
            GNodeArg(GNA_OUT, "S120_Output", 0),
            GNodeArg(GNA_IN, "S120_Mul_scale", 0),
            GNodeArg(GNA_IN, "S120_Mul_shift", 0),
            GNodeArg(GNA_IN, "S120_Infos", 0),
            GNodeArg(GNA_IN, "S120_Custom_infos", 0)
        )
    );
    // Node S123_Conv2d_128x1x3x3_Custom inq -5.21<(i8-0.00)*0.04071943<5.17 forced weightsq chan<(i8-0.00)*chan<chan outq -6.13<(i8-0.00)*0.04790010<6.08 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S123_Conv2d_128x1x3x3_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S120_Output", 0),
            GNodeArg(GNA_IN, "Conv_121_weights", 0),
            GNodeArg(GNA_IN, "Constant__1249", 0),
            GNodeArg(GNA_OUT, "S123_Output", 0),
            GNodeArg(GNA_IN, "S123_Mul_scale", 0),
            GNodeArg(GNA_IN, "S123_Mul_shift", 0),
            GNodeArg(GNA_IN, "S123_Infos", 0),
            GNodeArg(GNA_IN, "S123_Custom_infos", 0)
        )
    );
    // Node S126_Conv2d_256x128x1x1_Custom inq -6.13<(i8-0.00)*0.04790010<6.08 weightsq chan<(i8-0.00)*chan<chan outq -4.97<(i8-0.00)*0.03882437<4.93 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S126_Conv2d_256x128x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S123_Output", 0),
            GNodeArg(GNA_IN, "Conv_124_weights", 0),
            GNodeArg(GNA_IN, "Constant__1252", 0),
            GNodeArg(GNA_OUT, "S126_Output", 0),
            GNodeArg(GNA_IN, "S126_Mul_scale", 0),
            GNodeArg(GNA_IN, "S126_Mul_shift", 0),
            GNodeArg(GNA_IN, "S126_Infos", 0),
            GNodeArg(GNA_IN, "S126_Custom_infos", 0)
        )
    );
    // Node S129_Conv2d_128x256x1x1_Custom inq -4.97<(i8-0.00)*0.03882437<4.93 weightsq chan<(i8-0.00)*chan<chan outq -4.46<(i8-0.00)*0.03482907<4.42 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S129_Conv2d_128x256x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S126_Output", 0),
            GNodeArg(GNA_IN, "Conv_127_weights", 0),
            GNodeArg(GNA_IN, "Constant__1255", 0),
            GNodeArg(GNA_OUT, "S129_Output", 0),
            GNodeArg(GNA_IN, "S129_Mul_scale", 0),
            GNodeArg(GNA_IN, "S129_Mul_shift", 0),
            GNodeArg(GNA_IN, "S129_Infos", 0),
            GNodeArg(GNA_IN, "S129_Custom_infos", 0)
        )
    );
    // Node MaxPool_132 inq -4.46<(i8-0.00)*0.03482907<4.42 outq -4.46<(i8-0.00)*0.03482907<4.42
    AddNode("S130_MaxPool_13x13",
        Bindings(3,
            GNodeArg(GNA_IN, "S129_Output", 0),
            GNodeArg(GNA_OUT, "S130_Output", 0),
            GNodeArg(GNA_IN, "S130_Infos", 0)
        )
    );
    // Node MaxPool_130 inq -4.46<(i8-0.00)*0.03482907<4.42 outq -4.46<(i8-0.00)*0.03482907<4.42
    AddNode("S131_MaxPool_5x5",
        Bindings(3,
            GNodeArg(GNA_IN, "S129_Output", 0),
            GNodeArg(GNA_OUT, "S131_Output", 0),
            GNodeArg(GNA_IN, "S131_Infos", 0)
        )
    );
    // Node MaxPool_131 inq -4.46<(i8-0.00)*0.03482907<4.42 outq -4.46<(i8-0.00)*0.03482907<4.42
    AddNode("S132_MaxPool_9x9",
        Bindings(3,
            GNodeArg(GNA_IN, "S129_Output", 0),
            GNodeArg(GNA_OUT, "S132_Output", 0),
            GNodeArg(GNA_IN, "S132_Infos", 0)
        )
    );
    // Node S136_Conv2d_256x512x1x1_Custom inq -4.46<(i8-0.00)*0.03482907<4.42 weightsq chan<(i8-0.00)*chan<chan outq -4.73<(i8-0.00)*0.03694303<4.69 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S136_Conv2d_256x512x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S133_Output", 0),
            GNodeArg(GNA_IN, "Conv_134_weights", 0),
            GNodeArg(GNA_IN, "Constant__1258", 0),
            GNodeArg(GNA_OUT, "S136_Output", 0),
            GNodeArg(GNA_IN, "S136_Mul_scale", 0),
            GNodeArg(GNA_IN, "S136_Mul_shift", 0),
            GNodeArg(GNA_IN, "S136_Infos", 0),
            GNodeArg(GNA_IN, "S136_Custom_infos", 0)
        )
    );
    // Node S139_Conv2d_256x256x1x1_Custom inq -4.73<(i8-0.00)*0.03694303<4.69 weightsq chan<(i8-0.00)*chan<chan outq -4.17<(i8-0.00)*0.03261126<4.14 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S139_Conv2d_256x256x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S136_Output", 0),
            GNodeArg(GNA_IN, "Conv_137_weights", 0),
            GNodeArg(GNA_IN, "Constant__1261", 0),
            GNodeArg(GNA_OUT, "S139_Output", 0),
            GNodeArg(GNA_IN, "S139_Mul_scale", 0),
            GNodeArg(GNA_IN, "S139_Mul_shift", 0),
            GNodeArg(GNA_IN, "S139_Infos", 0),
            GNodeArg(GNA_IN, "S139_Custom_infos", 0)
        )
    );
    // Node Conv_143_fusion_qin0 inq -4.17<(i8-0.00)*0.03261126<4.14 outq -3.97<(i8-0.00)*0.03102084<3.94
    AddNode("S141_Op_Conv_143_fusion_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S140_Output_0", 0),
            GNodeArg(GNA_OUT, "S141_Output", 0),
            GNodeArg(GNA_IN, "S141_Infos", 0)
        )
    );
    // Node Conv_137_split_copy inq -4.17<(i8-0.00)*0.03261126<4.14 outq -4.17<(i8-0.00)*0.03261126<4.14
    AddNode("S142_Op_Conv_137_split_copy",
        Bindings(2,
            GNodeArg(GNA_IN, "S140_Output_1", 0),
            GNodeArg(GNA_OUT, "S142_Output", 0)
        )
    );
    // Node S145_Conv2d_128x128x1x1_Custom inq -4.17<(i8-0.00)*0.03261126<4.14 weightsq chan<(i8-0.00)*chan<chan outq -2.07<(i8-0.00)*0.01615127<2.05 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S145_Conv2d_128x128x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S141_Output", 0),
            GNodeArg(GNA_IN, "Conv_143_weights", 0),
            GNodeArg(GNA_IN, "Constant__1267", 0),
            GNodeArg(GNA_OUT, "S145_Output", 0),
            GNodeArg(GNA_IN, "S145_Mul_scale", 0),
            GNodeArg(GNA_IN, "S145_Mul_shift", 0),
            GNodeArg(GNA_IN, "S145_Infos", 0),
            GNodeArg(GNA_IN, "S145_Custom_infos", 0)
        )
    );
    // Node S148_Conv2d_128x1x3x3_Custom inq -2.07<(i8-0.00)*0.01615127<2.05 forced weightsq chan<(i8-0.00)*chan<chan outq -5.37<(i8-0.00)*0.04191903<5.32 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S148_Conv2d_128x1x3x3_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S145_Output", 0),
            GNodeArg(GNA_IN, "Conv_146_weights", 0),
            GNodeArg(GNA_IN, "Constant__1270", 0),
            GNodeArg(GNA_OUT, "S148_Output", 0),
            GNodeArg(GNA_IN, "S148_Mul_scale", 0),
            GNodeArg(GNA_IN, "S148_Mul_shift", 0),
            GNodeArg(GNA_IN, "S148_Infos", 0),
            GNodeArg(GNA_IN, "S148_Custom_infos", 0)
        )
    );
    // Node S151_Conv2d_128x128x1x1_Custom inq -5.37<(i8-0.00)*0.04191903<5.32 weightsq chan<(i8-0.00)*chan<chan outq -4.17<(i8-0.00)*0.03261126<4.14 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S151_Conv2d_128x128x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S148_Output", 0),
            GNodeArg(GNA_IN, "Conv_149_weights", 0),
            GNodeArg(GNA_IN, "Constant__1273", 0),
            GNodeArg(GNA_OUT, "S151_Output", 0),
            GNodeArg(GNA_IN, "S151_Mul_scale", 0),
            GNodeArg(GNA_IN, "S151_Mul_shift", 0),
            GNodeArg(GNA_IN, "S151_Infos", 0),
            GNodeArg(GNA_IN, "S151_Custom_infos", 0)
        )
    );
    // Node S155_Conv2d_256x256x1x1_Custom inq -4.17<(i8-0.00)*0.03261126<4.14 weightsq chan<(i8-0.00)*chan<chan outq -4.64<(i8-0.00)*0.03626229<4.61 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S155_Conv2d_256x256x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S152_Output", 0),
            GNodeArg(GNA_IN, "Conv_153_weights", 0),
            GNodeArg(GNA_IN, "Constant__1276", 0),
            GNodeArg(GNA_OUT, "S155_Output", 0),
            GNodeArg(GNA_IN, "S155_Mul_scale", 0),
            GNodeArg(GNA_IN, "S155_Mul_shift", 0),
            GNodeArg(GNA_IN, "S155_Infos", 0),
            GNodeArg(GNA_IN, "S155_Custom_infos", 0)
        )
    );
    // Node S158_Conv2d_128x256x1x1_Custom inq -4.64<(i8-0.00)*0.03626229<4.61 weightsq chan<(i8-0.00)*chan<chan outq -5.21<(i8-0.00)*0.04071943<5.17 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S158_Conv2d_128x256x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S155_Output", 0),
            GNodeArg(GNA_IN, "Conv_156_weights", 0),
            GNodeArg(GNA_IN, "Constant__1279", 0),
            GNodeArg(GNA_OUT, "S158_Output", 0),
            GNodeArg(GNA_IN, "S158_Mul_scale", 0),
            GNodeArg(GNA_IN, "S158_Mul_shift", 0),
            GNodeArg(GNA_IN, "S158_Infos", 0),
            GNodeArg(GNA_IN, "S158_Custom_infos", 0)
        )
    );
    // Node Resize_160 inq -5.21<(i8-0.00)*0.04071943<5.17 forced outq -5.21<(i8-0.00)*0.04071943<5.17 forced
    AddNode("S159_Op_Resize_160",
        Bindings(2,
            GNodeArg(GNA_IN, "S158_Output", 0),
            GNodeArg(GNA_OUT, "S159_Output", 0)
        )
    );
    // Node S163_Conv2d_128x256x1x1_Custom inq -5.21<(i8-0.00)*0.04071943<5.17 forced weightsq chan<(i8-0.00)*chan<chan outq -3.79<(i8-0.00)*0.02963090<3.76 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S163_Conv2d_128x256x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S160_Output", 0),
            GNodeArg(GNA_IN, "Conv_162_weights", 0),
            GNodeArg(GNA_IN, "Constant__1282", 0),
            GNodeArg(GNA_OUT, "S163_Output", 0),
            GNodeArg(GNA_IN, "S163_Mul_scale", 0),
            GNodeArg(GNA_IN, "S163_Mul_shift", 0),
            GNodeArg(GNA_IN, "S163_Infos", 0),
            GNodeArg(GNA_IN, "S163_Custom_infos", 0)
        )
    );
    // Node Conv_168_fusion_qin0 inq -3.79<(i8-0.00)*0.02963090<3.76 outq -3.21<(i8-0.00)*0.02504119<3.18
    AddNode("S165_Op_Conv_168_fusion_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S164_Output_0", 0),
            GNodeArg(GNA_OUT, "S165_Output", 0),
            GNodeArg(GNA_IN, "S165_Infos", 0)
        )
    );
    // Node Conv_162_split_copy inq -3.79<(i8-0.00)*0.02963090<3.76 outq -3.79<(i8-0.00)*0.02963090<3.76
    AddNode("S166_Op_Conv_162_split_copy",
        Bindings(2,
            GNodeArg(GNA_IN, "S164_Output_1", 0),
            GNodeArg(GNA_OUT, "S166_Output", 0)
        )
    );
    // Node S169_Conv2d_64x64x1x1_Custom inq -3.79<(i8-0.00)*0.02963090<3.76 weightsq chan<(i8-0.00)*chan<chan outq -2.60<(i8-0.00)*0.02034514<2.58 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S169_Conv2d_64x64x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S165_Output", 0),
            GNodeArg(GNA_IN, "Conv_168_weights", 0),
            GNodeArg(GNA_IN, "Constant__1288", 0),
            GNodeArg(GNA_OUT, "S169_Output", 0),
            GNodeArg(GNA_IN, "S169_Mul_scale", 0),
            GNodeArg(GNA_IN, "S169_Mul_shift", 0),
            GNodeArg(GNA_IN, "S169_Infos", 0),
            GNodeArg(GNA_IN, "S169_Custom_infos", 0)
        )
    );
    // Node S172_Conv2d_64x1x3x3_Custom inq -2.60<(i8-0.00)*0.02034514<2.58 forced weightsq chan<(i8-0.00)*chan<chan outq -4.45<(i8-0.00)*0.03477186<4.42 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S172_Conv2d_64x1x3x3_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S169_Output", 0),
            GNodeArg(GNA_IN, "Conv_171_weights", 0),
            GNodeArg(GNA_IN, "Constant__1291", 0),
            GNodeArg(GNA_OUT, "S172_Output", 0),
            GNodeArg(GNA_IN, "S172_Mul_scale", 0),
            GNodeArg(GNA_IN, "S172_Mul_shift", 0),
            GNodeArg(GNA_IN, "S172_Infos", 0),
            GNodeArg(GNA_IN, "S172_Custom_infos", 0)
        )
    );
    // Node S175_Conv2d_64x64x1x1_Custom inq -4.45<(i8-0.00)*0.03477186<4.42 weightsq chan<(i8-0.00)*chan<chan outq -3.79<(i8-0.00)*0.02963090<3.76 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S175_Conv2d_64x64x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S172_Output", 0),
            GNodeArg(GNA_IN, "Conv_174_weights", 0),
            GNodeArg(GNA_IN, "Constant__1294", 0),
            GNodeArg(GNA_OUT, "S175_Output", 0),
            GNodeArg(GNA_IN, "S175_Mul_scale", 0),
            GNodeArg(GNA_IN, "S175_Mul_shift", 0),
            GNodeArg(GNA_IN, "S175_Infos", 0),
            GNodeArg(GNA_IN, "S175_Custom_infos", 0)
        )
    );
    // Node S179_Conv2d_128x128x1x1_Custom inq -3.79<(i8-0.00)*0.02963090<3.76 weightsq chan<(i8-0.00)*chan<chan outq -4.69<(i8-0.00)*0.03660547<4.65 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S179_Conv2d_128x128x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S176_Output", 0),
            GNodeArg(GNA_IN, "Conv_178_weights", 0),
            GNodeArg(GNA_IN, "Constant__1297", 0),
            GNodeArg(GNA_OUT, "S179_Output", 0),
            GNodeArg(GNA_IN, "S179_Mul_scale", 0),
            GNodeArg(GNA_IN, "S179_Mul_shift", 0),
            GNodeArg(GNA_IN, "S179_Infos", 0),
            GNodeArg(GNA_IN, "S179_Custom_infos", 0)
        )
    );
    // Node S182_Conv2d_64x128x1x1_Custom inq -4.69<(i8-0.00)*0.03660547<4.65 weightsq chan<(i8-0.00)*chan<chan outq -6.05<(i8-0.00)*0.04729335<6.01 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S182_Conv2d_64x128x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S179_Output", 0),
            GNodeArg(GNA_IN, "Conv_181_weights", 0),
            GNodeArg(GNA_IN, "Constant__1300", 0),
            GNodeArg(GNA_OUT, "S182_Output", 0),
            GNodeArg(GNA_IN, "S182_Mul_scale", 0),
            GNodeArg(GNA_IN, "S182_Mul_shift", 0),
            GNodeArg(GNA_IN, "S182_Infos", 0),
            GNodeArg(GNA_IN, "S182_Custom_infos", 0)
        )
    );
    // Node Resize_185 inq -6.05<(i8-0.00)*0.04729335<6.01 outq -6.05<(i8-0.00)*0.04729335<6.01
    AddNode("S183_Op_Resize_185",
        Bindings(2,
            GNodeArg(GNA_IN, "S182_Output", 0),
            GNodeArg(GNA_OUT, "S183_Output", 0)
        )
    );
    // Node Concat_186_qin0 inq -6.05<(i8-0.00)*0.04729335<6.01 outq -3.75<(i8-0.00)*0.02932421<3.72 forced
    AddNode("S184_Op_Concat_186_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S183_Output", 0),
            GNodeArg(GNA_OUT, "S184_Output", 0),
            GNodeArg(GNA_IN, "S184_Infos", 0)
        )
    );
    // Node S188_Conv2d_64x128x1x1_Custom inq -3.75<(i8-0.00)*0.02932421<3.72 forced weightsq chan<(i8-0.00)*chan<chan outq -4.96<(i8-0.00)*0.03877128<4.92 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S188_Conv2d_64x128x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S185_Output", 0),
            GNodeArg(GNA_IN, "Conv_187_weights", 0),
            GNodeArg(GNA_IN, "Constant__1303", 0),
            GNodeArg(GNA_OUT, "S188_Output", 0),
            GNodeArg(GNA_IN, "S188_Mul_scale", 0),
            GNodeArg(GNA_IN, "S188_Mul_shift", 0),
            GNodeArg(GNA_IN, "S188_Infos", 0),
            GNodeArg(GNA_IN, "S188_Custom_infos", 0)
        )
    );
    // Node Conv_193_fusion_qin0 inq -4.96<(i8-0.00)*0.03877128<4.92 outq -2.98<(i8-0.00)*0.02329703<2.96
    AddNode("S190_Op_Conv_193_fusion_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S189_Output_0", 0),
            GNodeArg(GNA_OUT, "S190_Output", 0),
            GNodeArg(GNA_IN, "S190_Infos", 0)
        )
    );
    // Node Conv_187_split_copy inq -4.96<(i8-0.00)*0.03877128<4.92 outq -4.96<(i8-0.00)*0.03877128<4.92
    AddNode("S191_Op_Conv_187_split_copy",
        Bindings(2,
            GNodeArg(GNA_IN, "S189_Output_1", 0),
            GNodeArg(GNA_OUT, "S191_Output", 0)
        )
    );
    // Node S194_Conv2d_32x32x1x1_Custom inq -4.96<(i8-0.00)*0.03877128<4.92 weightsq chan<(i8-0.00)*chan<chan outq -2.95<(i8-0.00)*0.02302548<2.92 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S194_Conv2d_32x32x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S190_Output", 0),
            GNodeArg(GNA_IN, "Conv_193_weights", 0),
            GNodeArg(GNA_IN, "Constant__1309", 0),
            GNodeArg(GNA_OUT, "S194_Output", 0),
            GNodeArg(GNA_IN, "S194_Mul_scale", 0),
            GNodeArg(GNA_IN, "S194_Mul_shift", 0),
            GNodeArg(GNA_IN, "S194_Infos", 0),
            GNodeArg(GNA_IN, "S194_Custom_infos", 0)
        )
    );
    // Node S197_Conv2d_32x1x3x3_Custom inq -2.95<(i8-0.00)*0.02302548<2.92 forced weightsq chan<(i8-0.00)*chan<chan outq -5.27<(i8-0.00)*0.04118391<5.23 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S197_Conv2d_32x1x3x3_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S194_Output", 0),
            GNodeArg(GNA_IN, "Conv_196_weights", 0),
            GNodeArg(GNA_IN, "Constant__1312", 0),
            GNodeArg(GNA_OUT, "S197_Output", 0),
            GNodeArg(GNA_IN, "S197_Mul_scale", 0),
            GNodeArg(GNA_IN, "S197_Mul_shift", 0),
            GNodeArg(GNA_IN, "S197_Infos", 0),
            GNodeArg(GNA_IN, "S197_Custom_infos", 0)
        )
    );
    // Node S200_Conv2d_32x32x1x1_Custom inq -5.27<(i8-0.00)*0.04118391<5.23 weightsq chan<(i8-0.00)*chan<chan outq -4.96<(i8-0.00)*0.03877128<4.92 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S200_Conv2d_32x32x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S197_Output", 0),
            GNodeArg(GNA_IN, "Conv_199_weights", 0),
            GNodeArg(GNA_IN, "Constant__1315", 0),
            GNodeArg(GNA_OUT, "S200_Output", 0),
            GNodeArg(GNA_IN, "S200_Mul_scale", 0),
            GNodeArg(GNA_IN, "S200_Mul_shift", 0),
            GNodeArg(GNA_IN, "S200_Infos", 0),
            GNodeArg(GNA_IN, "S200_Custom_infos", 0)
        )
    );
    // Node S204_Conv2d_64x64x1x1_Custom inq -4.96<(i8-0.00)*0.03877128<4.92 weightsq chan<(i8-0.00)*chan<chan outq -4.34<(i8-0.00)*0.03387608<4.30 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S204_Conv2d_64x64x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S201_Output", 0),
            GNodeArg(GNA_IN, "Conv_203_weights", 0),
            GNodeArg(GNA_IN, "Constant__1318", 0),
            GNodeArg(GNA_OUT, "S204_Output", 0),
            GNodeArg(GNA_IN, "S204_Mul_scale", 0),
            GNodeArg(GNA_IN, "S204_Mul_shift", 0),
            GNodeArg(GNA_IN, "S204_Infos", 0),
            GNodeArg(GNA_IN, "S204_Custom_infos", 0)
        )
    );
    // Node S207_Conv2d_64x1x3x3_Custom inq -4.34<(i8-0.00)*0.03387608<4.30 forced weightsq chan<(i8-0.00)*chan<chan outq -5.71<(i8-0.00)*0.04460969<5.67 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S207_Conv2d_64x1x3x3_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S204_Output", 0),
            GNodeArg(GNA_IN, "Conv_206_weights", 0),
            GNodeArg(GNA_IN, "Constant__1321", 0),
            GNodeArg(GNA_OUT, "S207_Output", 0),
            GNodeArg(GNA_IN, "S207_Mul_scale", 0),
            GNodeArg(GNA_IN, "S207_Mul_shift", 0),
            GNodeArg(GNA_IN, "S207_Infos", 0),
            GNodeArg(GNA_IN, "S207_Custom_infos", 0)
        )
    );
    // Node S210_Conv2d_64x64x1x1_Custom inq -5.71<(i8-0.00)*0.04460969<5.67 weightsq chan<(i8-0.00)*chan<chan outq -6.05<(i8-0.00)*0.04729335<6.01 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S210_Conv2d_64x64x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S207_Output", 0),
            GNodeArg(GNA_IN, "Conv_209_weights", 0),
            GNodeArg(GNA_IN, "Constant__1324", 0),
            GNodeArg(GNA_OUT, "S210_Output", 0),
            GNodeArg(GNA_IN, "S210_Mul_scale", 0),
            GNodeArg(GNA_IN, "S210_Mul_shift", 0),
            GNodeArg(GNA_IN, "S210_Infos", 0),
            GNodeArg(GNA_IN, "S210_Custom_infos", 0)
        )
    );
    // Node S214_Conv2d_128x128x1x1_Custom inq -6.05<(i8-0.00)*0.04729335<6.01 weightsq chan<(i8-0.00)*chan<chan outq -4.45<(i8-0.00)*0.03480387<4.42 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S214_Conv2d_128x128x1x1_Custom",
        Bindings(7,
            GNodeArg(GNA_IN, "S211_Output", 0),
            GNodeArg(GNA_IN, "Conv_213_weights", 0),
            GNodeArg(GNA_IN, "Constant__1327", 0),
            GNodeArg(GNA_OUT, "S214_Output", 0),
            GNodeArg(GNA_IN, "S214_Mul_scale", 0),
            GNodeArg(GNA_IN, "S214_Mul_shift", 0),
            GNodeArg(GNA_IN, "S214_Infos", 0)
        )
    );
    // Node Conv_213_split_copy inq -4.45<(i8-0.00)*0.03480387<4.42 outq -4.45<(i8-0.00)*0.03480387<4.42
    AddNode("S216_Op_Conv_213_split_copy",
        Bindings(2,
            GNodeArg(GNA_IN, "S215_Output_1", 0),
            GNodeArg(GNA_OUT, "S216_Output", 0)
        )
    );
    // Node S219_Conv2d_64x64x1x1_Custom inq -4.45<(i8-0.00)*0.03480387<4.42 weightsq chan<(i8-0.00)*chan<chan outq -4.06<(i8-0.00)*0.03175707<4.03 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S219_Conv2d_64x64x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S215_Output_0", 0),
            GNodeArg(GNA_IN, "Conv_219_weights", 0),
            GNodeArg(GNA_IN, "Constant__1333", 0),
            GNodeArg(GNA_OUT, "S219_Output", 0),
            GNodeArg(GNA_IN, "S219_Mul_scale", 0),
            GNodeArg(GNA_IN, "S219_Mul_shift", 0),
            GNodeArg(GNA_IN, "S219_Infos", 0),
            GNodeArg(GNA_IN, "S219_Custom_infos", 0)
        )
    );
    // Node S222_Conv2d_64x1x3x3_Custom inq -4.06<(i8-0.00)*0.03175707<4.03 forced weightsq chan<(i8-0.00)*chan<chan outq -4.41<(i8-0.00)*0.03443968<4.37 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S222_Conv2d_64x1x3x3_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S219_Output", 0),
            GNodeArg(GNA_IN, "Conv_222_weights", 0),
            GNodeArg(GNA_IN, "Constant__1336", 0),
            GNodeArg(GNA_OUT, "S222_Output", 0),
            GNodeArg(GNA_IN, "S222_Mul_scale", 0),
            GNodeArg(GNA_IN, "S222_Mul_shift", 0),
            GNodeArg(GNA_IN, "S222_Infos", 0),
            GNodeArg(GNA_IN, "S222_Custom_infos", 0)
        )
    );
    // Node S225_Conv2d_64x64x1x1_Custom inq -4.41<(i8-0.00)*0.03443968<4.37 weightsq chan<(i8-0.00)*chan<chan outq -4.45<(i8-0.00)*0.03480387<4.42 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S225_Conv2d_64x64x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S222_Output", 0),
            GNodeArg(GNA_IN, "Conv_225_weights", 0),
            GNodeArg(GNA_IN, "Constant__1339", 0),
            GNodeArg(GNA_OUT, "S225_Output", 0),
            GNodeArg(GNA_IN, "S225_Mul_scale", 0),
            GNodeArg(GNA_IN, "S225_Mul_shift", 0),
            GNodeArg(GNA_IN, "S225_Infos", 0),
            GNodeArg(GNA_IN, "S225_Custom_infos", 0)
        )
    );
    // Node S229_Conv2d_128x128x1x1_Custom inq -4.45<(i8-0.00)*0.03480387<4.42 weightsq chan<(i8-0.00)*chan<chan outq -3.48<(i8-0.00)*0.02719331<3.45 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S229_Conv2d_128x128x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S226_Output", 0),
            GNodeArg(GNA_IN, "Conv_229_weights", 0),
            GNodeArg(GNA_IN, "Constant__1342", 0),
            GNodeArg(GNA_OUT, "S229_Output", 0),
            GNodeArg(GNA_IN, "S229_Mul_scale", 0),
            GNodeArg(GNA_IN, "S229_Mul_shift", 0),
            GNodeArg(GNA_IN, "S229_Infos", 0),
            GNodeArg(GNA_IN, "S229_Custom_infos", 0)
        )
    );
    // Node S232_Conv2d_128x1x3x3_Custom inq -3.48<(i8-0.00)*0.02719331<3.45 forced weightsq chan<(i8-0.00)*chan<chan outq -3.33<(i8-0.00)*0.02600143<3.30 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S232_Conv2d_128x1x3x3_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S229_Output", 0),
            GNodeArg(GNA_IN, "Conv_232_weights", 0),
            GNodeArg(GNA_IN, "Constant__1345", 0),
            GNodeArg(GNA_OUT, "S232_Output", 0),
            GNodeArg(GNA_IN, "S232_Mul_scale", 0),
            GNodeArg(GNA_IN, "S232_Mul_shift", 0),
            GNodeArg(GNA_IN, "S232_Infos", 0),
            GNodeArg(GNA_IN, "S232_Custom_infos", 0)
        )
    );
    // Node S235_Conv2d_128x128x1x1_Custom inq -3.33<(i8-0.00)*0.02600143<3.30 weightsq chan<(i8-0.00)*chan<chan outq -5.21<(i8-0.00)*0.04071943<5.17 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S235_Conv2d_128x128x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S232_Output", 0),
            GNodeArg(GNA_IN, "Conv_235_weights", 0),
            GNodeArg(GNA_IN, "Constant__1348", 0),
            GNodeArg(GNA_OUT, "S235_Output", 0),
            GNodeArg(GNA_IN, "S235_Mul_scale", 0),
            GNodeArg(GNA_IN, "S235_Mul_shift", 0),
            GNodeArg(GNA_IN, "S235_Infos", 0),
            GNodeArg(GNA_IN, "S235_Custom_infos", 0)
        )
    );
    // Node Conv_239_fusion_qin0 inq -5.21<(i8-0.00)*0.04071943<5.17 forced outq -3.48<(i8-0.00)*0.02722353<3.46
    AddNode("S237_Op_Conv_239_fusion_qin0",
        Bindings(3,
            GNodeArg(GNA_IN, "S236_Output", 0),
            GNodeArg(GNA_OUT, "S237_Output", 0),
            GNodeArg(GNA_IN, "S237_Infos", 0)
        )
    );
    // Node S240_Conv2d_256x256x1x1_Custom inq -5.21<(i8-0.00)*0.04071943<5.17 forced weightsq chan<(i8-0.00)*chan<chan outq -2.48<(i8-0.00)*0.01934296<2.46 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S240_Conv2d_256x256x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S237_Output", 0),
            GNodeArg(GNA_IN, "Conv_239_weights", 0),
            GNodeArg(GNA_IN, "Constant__1351", 0),
            GNodeArg(GNA_OUT, "S240_Output", 0),
            GNodeArg(GNA_IN, "S240_Mul_scale", 0),
            GNodeArg(GNA_IN, "S240_Mul_shift", 0),
            GNodeArg(GNA_IN, "S240_Infos", 0),
            GNodeArg(GNA_IN, "S240_Custom_infos", 0)
        )
    );
    // Node Conv_239_split_copy inq -2.48<(i8-0.00)*0.01934296<2.46 outq -2.48<(i8-0.00)*0.01934296<2.46
    AddNode("S242_Op_Conv_239_split_copy",
        Bindings(2,
            GNodeArg(GNA_IN, "S241_Output_1", 0),
            GNodeArg(GNA_OUT, "S242_Output", 0)
        )
    );
    // Node S245_Conv2d_128x128x1x1_Custom inq -2.48<(i8-0.00)*0.01934296<2.46 weightsq chan<(i8-0.00)*chan<chan outq -2.47<(i8-0.00)*0.01932175<2.45 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S245_Conv2d_128x128x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S241_Output_0", 0),
            GNodeArg(GNA_IN, "Conv_245_weights", 0),
            GNodeArg(GNA_IN, "Constant__1357", 0),
            GNodeArg(GNA_OUT, "S245_Output", 0),
            GNodeArg(GNA_IN, "S245_Mul_scale", 0),
            GNodeArg(GNA_IN, "S245_Mul_shift", 0),
            GNodeArg(GNA_IN, "S245_Infos", 0),
            GNodeArg(GNA_IN, "S245_Custom_infos", 0)
        )
    );
    // Node S248_Conv2d_128x1x3x3_Custom inq -2.47<(i8-0.00)*0.01932175<2.45 forced weightsq chan<(i8-0.00)*chan<chan outq -1.91<(i8-0.00)*0.01490512<1.89 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S248_Conv2d_128x1x3x3_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S245_Output", 0),
            GNodeArg(GNA_IN, "Conv_248_weights", 0),
            GNodeArg(GNA_IN, "Constant__1360", 0),
            GNodeArg(GNA_OUT, "S248_Output", 0),
            GNodeArg(GNA_IN, "S248_Mul_scale", 0),
            GNodeArg(GNA_IN, "S248_Mul_shift", 0),
            GNodeArg(GNA_IN, "S248_Infos", 0),
            GNodeArg(GNA_IN, "S248_Custom_infos", 0)
        )
    );
    // Node S251_Conv2d_128x128x1x1_Custom inq -1.91<(i8-0.00)*0.01490512<1.89 weightsq chan<(i8-0.00)*chan<chan outq -2.48<(i8-0.00)*0.01934296<2.46 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S251_Conv2d_128x128x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S248_Output", 0),
            GNodeArg(GNA_IN, "Conv_251_weights", 0),
            GNodeArg(GNA_IN, "Constant__1363", 0),
            GNodeArg(GNA_OUT, "S251_Output", 0),
            GNodeArg(GNA_IN, "S251_Mul_scale", 0),
            GNodeArg(GNA_IN, "S251_Mul_shift", 0),
            GNodeArg(GNA_IN, "S251_Infos", 0),
            GNodeArg(GNA_IN, "S251_Custom_infos", 0)
        )
    );
    // Node S255_Conv2d_256x256x1x1_Custom inq -2.48<(i8-0.00)*0.01934296<2.46 weightsq chan<(i8-0.00)*chan<chan outq -2.26<(i8-0.00)*0.01763853<2.24 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S255_Conv2d_256x256x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S252_Output", 0),
            GNodeArg(GNA_IN, "Conv_255_weights", 0),
            GNodeArg(GNA_IN, "Constant__1366", 0),
            GNodeArg(GNA_OUT, "S255_Output", 0),
            GNodeArg(GNA_IN, "S255_Mul_scale", 0),
            GNodeArg(GNA_IN, "S255_Mul_shift", 0),
            GNodeArg(GNA_IN, "S255_Infos", 0),
            GNodeArg(GNA_IN, "S255_Custom_infos", 0)
        )
    );
    // Node S258_Conv2d_64x64x1x1_Custom inq -4.34<(i8-0.00)*0.03387608<4.30 forced weightsq chan<(i8-0.00)*chan<chan outq -6.32<(i8-0.00)*0.04938919<6.27 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S258_Conv2d_64x64x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S204_Output", 0),
            GNodeArg(GNA_IN, "Conv_258_weights", 0),
            GNodeArg(GNA_IN, "Constant__1369", 0),
            GNodeArg(GNA_OUT, "S258_Output", 0),
            GNodeArg(GNA_IN, "S258_Mul_scale", 0),
            GNodeArg(GNA_IN, "S258_Mul_shift", 0),
            GNodeArg(GNA_IN, "S258_Infos", 0),
            GNodeArg(GNA_IN, "S258_Custom_infos", 0)
        )
    );
    // Node S261_Conv2d_64x1x3x3_Custom inq -6.32<(i8-0.00)*0.04938919<6.27 forced weightsq chan<(i8-0.00)*chan<chan outq -4.82<(i8-0.00)*0.03768724<4.79 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S261_Conv2d_64x1x3x3_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S258_Output", 0),
            GNodeArg(GNA_IN, "Conv_261_weights", 0),
            GNodeArg(GNA_IN, "Constant__1372", 0),
            GNodeArg(GNA_OUT, "S261_Output", 0),
            GNodeArg(GNA_IN, "S261_Mul_scale", 0),
            GNodeArg(GNA_IN, "S261_Mul_shift", 0),
            GNodeArg(GNA_IN, "S261_Infos", 0),
            GNodeArg(GNA_IN, "S261_Custom_infos", 0)
        )
    );
    // Node S264_Conv2d_64x64x1x1_Custom inq -4.82<(i8-0.00)*0.03768724<4.79 weightsq chan<(i8-0.00)*chan<chan outq -2.96<(i8-0.00)*0.02315465<2.94 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S264_Conv2d_64x64x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S261_Output", 0),
            GNodeArg(GNA_IN, "Conv_264_weights", 0),
            GNodeArg(GNA_IN, "Constant__1375", 0),
            GNodeArg(GNA_OUT, "S264_Output", 0),
            GNodeArg(GNA_IN, "S264_Mul_scale", 0),
            GNodeArg(GNA_IN, "S264_Mul_shift", 0),
            GNodeArg(GNA_IN, "S264_Infos", 0),
            GNodeArg(GNA_IN, "S264_Custom_infos", 0)
        )
    );
    // Node S267_Conv2d_64x1x3x3_Custom inq -2.96<(i8-0.00)*0.02315465<2.94 forced weightsq chan<(i8-0.00)*chan<chan outq -3.38<(i8-0.00)*0.02641447<3.35 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S267_Conv2d_64x1x3x3_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S264_Output", 0),
            GNodeArg(GNA_IN, "Conv_267_weights", 0),
            GNodeArg(GNA_IN, "Constant__1378", 0),
            GNodeArg(GNA_OUT, "S267_Output", 0),
            GNodeArg(GNA_IN, "S267_Mul_scale", 0),
            GNodeArg(GNA_IN, "S267_Mul_shift", 0),
            GNodeArg(GNA_IN, "S267_Infos", 0),
            GNodeArg(GNA_IN, "S267_Custom_infos", 0)
        )
    );
    // Node S270_Conv2d_64x64x1x1 inq -3.38<(i8-0.00)*0.02641447<3.35 weightsq chan<(i8-0.00)*chan<chan outq -3.00<(i8--11.00)*0.02563244<3.54 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S270_Conv2d_64x64x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S267_Output", 0),
            GNodeArg(GNA_IN, "Conv_270_weights", 0),
            GNodeArg(GNA_IN, "Constant__1381", 0),
            GNodeArg(GNA_OUT, "S270_Output", 0),
            GNodeArg(GNA_IN, "S270_Mul_scale", 0),
            GNodeArg(GNA_IN, "S270_Mul_shift", 0),
            GNodeArg(GNA_IN, "S270_Infos", 0)
        )
    );
    // Node expr_57 in_qs [-3.00<(i8--11.00)*0.02563244<3.54] out_qs [-3.46<(i8-0.00)*0.02699266<3.43]
    AddNode("S271_Op_expr_57",
        Bindings(2,
            GNodeArg(GNA_IN, "S270_Output", 0),
            GNodeArg(GNA_OUT, "S271_Output", 0)
        )
    );
    // Node S274_Conv2d_1x64x1x1_Sigmoid inq -3.46<(i8-0.00)*0.02699266<3.43 weightsq -0.11<(i8-0.00)*0.00084325<0.11 outq -3.16<(i8-0.00)*0.02471484<3.14 forced biasesq -48880.30<(i32-0.00)*0.00002276<48880.30
    AddNode("S274_Conv2d_1x64x1x1_Sigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S271_Output", 0),
            GNodeArg(GNA_IN, "Conv_273_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_cls_preds_0_bias", 0),
            GNodeArg(GNA_OUT, "S274_Output", 0),
            GNodeArg(GNA_IN, "S274_Mul_scale", 0),
            GNodeArg(GNA_IN, "S274_Mul_shift", 0),
            GNodeArg(GNA_IN, "S274_Infos", 0)
        )
    );
    // Node S277_Conv2d_64x1x3x3_Custom inq -6.32<(i8-0.00)*0.04938919<6.27 forced weightsq chan<(i8-0.00)*chan<chan outq -10.44<(i8-0.00)*0.08159668<10.36 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S277_Conv2d_64x1x3x3_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S258_Output", 0),
            GNodeArg(GNA_IN, "Conv_274_weights", 0),
            GNodeArg(GNA_IN, "Constant__1384", 0),
            GNodeArg(GNA_OUT, "S277_Output", 0),
            GNodeArg(GNA_IN, "S277_Mul_scale", 0),
            GNodeArg(GNA_IN, "S277_Mul_shift", 0),
            GNodeArg(GNA_IN, "S277_Infos", 0),
            GNodeArg(GNA_IN, "S277_Custom_infos", 0)
        )
    );
    // Node S280_Conv2d_64x64x1x1_Custom inq -10.44<(i8-0.00)*0.08159668<10.36 weightsq chan<(i8-0.00)*chan<chan outq -9.31<(i8-0.00)*0.07275607<9.24 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S280_Conv2d_64x64x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S277_Output", 0),
            GNodeArg(GNA_IN, "Conv_277_weights", 0),
            GNodeArg(GNA_IN, "Constant__1387", 0),
            GNodeArg(GNA_OUT, "S280_Output", 0),
            GNodeArg(GNA_IN, "S280_Mul_scale", 0),
            GNodeArg(GNA_IN, "S280_Mul_shift", 0),
            GNodeArg(GNA_IN, "S280_Infos", 0),
            GNodeArg(GNA_IN, "S280_Custom_infos", 0)
        )
    );
    // Node S283_Conv2d_64x1x3x3_Custom inq -9.31<(i8-0.00)*0.07275607<9.24 forced weightsq chan<(i8-0.00)*chan<chan outq -4.51<(i8-0.00)*0.03523093<4.47 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S283_Conv2d_64x1x3x3_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S280_Output", 0),
            GNodeArg(GNA_IN, "Conv_280_weights", 0),
            GNodeArg(GNA_IN, "Constant__1390", 0),
            GNodeArg(GNA_OUT, "S283_Output", 0),
            GNodeArg(GNA_IN, "S283_Mul_scale", 0),
            GNodeArg(GNA_IN, "S283_Mul_shift", 0),
            GNodeArg(GNA_IN, "S283_Infos", 0),
            GNodeArg(GNA_IN, "S283_Custom_infos", 0)
        )
    );
    // Node S286_Conv2d_64x64x1x1_Custom inq -4.51<(i8-0.00)*0.03523093<4.47 weightsq chan<(i8-0.00)*chan<chan outq -4.71<(i8-0.00)*0.03681056<4.67 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S286_Conv2d_64x64x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S283_Output", 0),
            GNodeArg(GNA_IN, "Conv_283_weights", 0),
            GNodeArg(GNA_IN, "Constant__1393", 0),
            GNodeArg(GNA_OUT, "S286_Output", 0),
            GNodeArg(GNA_IN, "S286_Mul_scale", 0),
            GNodeArg(GNA_IN, "S286_Mul_shift", 0),
            GNodeArg(GNA_IN, "S286_Infos", 0),
            GNodeArg(GNA_IN, "S286_Custom_infos", 0)
        )
    );
    // Node S289_Conv2d_4x64x1x1 inq -4.71<(i8-0.00)*0.03681056<4.67 weightsq chan<(i8-0.00)*chan<chan outq -3.16<(i8-0.00)*0.02471484<3.14 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S289_Conv2d_4x64x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S286_Output", 0),
            GNodeArg(GNA_IN, "Conv_286_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_reg_preds_0_bias", 0),
            GNodeArg(GNA_OUT, "S289_Output", 0),
            GNodeArg(GNA_IN, "S289_Mul_scale", 0),
            GNodeArg(GNA_IN, "S289_Mul_shift", 0),
            GNodeArg(GNA_IN, "S289_Infos", 0)
        )
    );
    // Node S292_Conv2d_1x64x1x1_Sigmoid inq -4.71<(i8-0.00)*0.03681056<4.67 weightsq -0.99<(i8-0.00)*0.00775827<0.99 outq -3.16<(i8-0.00)*0.02471484<3.14 forced biasesq -613291.81<(i32-0.00)*0.00028559<613291.81
    AddNode("S292_Conv2d_1x64x1x1_Sigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S286_Output", 0),
            GNodeArg(GNA_IN, "Conv_287_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_obj_preds_0_bias", 0),
            GNodeArg(GNA_OUT, "S292_Output", 0),
            GNodeArg(GNA_IN, "S292_Mul_scale", 0),
            GNodeArg(GNA_IN, "S292_Mul_shift", 0),
            GNodeArg(GNA_IN, "S292_Infos", 0)
        )
    );
    // Node S297_Conv2d_64x128x1x1_Custom inq -3.48<(i8-0.00)*0.02719331<3.45 forced weightsq chan<(i8-0.00)*chan<chan outq -3.40<(i8-0.00)*0.02659554<3.38 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S297_Conv2d_64x128x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S229_Output", 0),
            GNodeArg(GNA_IN, "Conv_291_weights", 0),
            GNodeArg(GNA_IN, "Constant__1396", 0),
            GNodeArg(GNA_OUT, "S297_Output", 0),
            GNodeArg(GNA_IN, "S297_Mul_scale", 0),
            GNodeArg(GNA_IN, "S297_Mul_shift", 0),
            GNodeArg(GNA_IN, "S297_Infos", 0),
            GNodeArg(GNA_IN, "S297_Custom_infos", 0)
        )
    );
    // Node S300_Conv2d_64x1x3x3_Custom inq -3.40<(i8-0.00)*0.02659554<3.38 forced weightsq chan<(i8-0.00)*chan<chan outq -2.36<(i8-0.00)*0.01843425<2.34 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S300_Conv2d_64x1x3x3_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S297_Output", 0),
            GNodeArg(GNA_IN, "Conv_294_weights", 0),
            GNodeArg(GNA_IN, "Constant__1399", 0),
            GNodeArg(GNA_OUT, "S300_Output", 0),
            GNodeArg(GNA_IN, "S300_Mul_scale", 0),
            GNodeArg(GNA_IN, "S300_Mul_shift", 0),
            GNodeArg(GNA_IN, "S300_Infos", 0),
            GNodeArg(GNA_IN, "S300_Custom_infos", 0)
        )
    );
    // Node S303_Conv2d_64x64x1x1_Custom inq -2.36<(i8-0.00)*0.01843425<2.34 weightsq chan<(i8-0.00)*chan<chan outq -1.54<(i8-0.00)*0.01204133<1.53 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S303_Conv2d_64x64x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S300_Output", 0),
            GNodeArg(GNA_IN, "Conv_297_weights", 0),
            GNodeArg(GNA_IN, "Constant__1402", 0),
            GNodeArg(GNA_OUT, "S303_Output", 0),
            GNodeArg(GNA_IN, "S303_Mul_scale", 0),
            GNodeArg(GNA_IN, "S303_Mul_shift", 0),
            GNodeArg(GNA_IN, "S303_Infos", 0),
            GNodeArg(GNA_IN, "S303_Custom_infos", 0)
        )
    );
    // Node S306_Conv2d_64x1x3x3_Custom inq -1.54<(i8-0.00)*0.01204133<1.53 forced weightsq chan<(i8-0.00)*chan<chan outq -1.48<(i8-0.00)*0.01158015<1.47 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S306_Conv2d_64x1x3x3_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S303_Output", 0),
            GNodeArg(GNA_IN, "Conv_300_weights", 0),
            GNodeArg(GNA_IN, "Constant__1405", 0),
            GNodeArg(GNA_OUT, "S306_Output", 0),
            GNodeArg(GNA_IN, "S306_Mul_scale", 0),
            GNodeArg(GNA_IN, "S306_Mul_shift", 0),
            GNodeArg(GNA_IN, "S306_Infos", 0),
            GNodeArg(GNA_IN, "S306_Custom_infos", 0)
        )
    );
    // Node S309_Conv2d_64x64x1x1 inq -1.48<(i8-0.00)*0.01158015<1.47 weightsq chan<(i8-0.00)*chan<chan outq -1.08<(i8--28.00)*0.01080235<1.67 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S309_Conv2d_64x64x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S306_Output", 0),
            GNodeArg(GNA_IN, "Conv_303_weights", 0),
            GNodeArg(GNA_IN, "Constant__1408", 0),
            GNodeArg(GNA_OUT, "S309_Output", 0),
            GNodeArg(GNA_IN, "S309_Mul_scale", 0),
            GNodeArg(GNA_IN, "S309_Mul_shift", 0),
            GNodeArg(GNA_IN, "S309_Infos", 0)
        )
    );
    // Node expr_68 in_qs [-1.08<(i8--28.00)*0.01080235<1.67] out_qs [-1.42<(i8-0.00)*0.01113129<1.41]
    AddNode("S310_Op_expr_68",
        Bindings(2,
            GNodeArg(GNA_IN, "S309_Output", 0),
            GNodeArg(GNA_OUT, "S310_Output", 0)
        )
    );
    // Node S313_Conv2d_1x64x1x1_Sigmoid inq -1.42<(i8-0.00)*0.01113129<1.41 weightsq -0.12<(i8-0.00)*0.00092315<0.12 outq -3.16<(i8-0.00)*0.02471484<3.14 forced biasesq -22067.16<(i32-0.00)*0.00001028<22067.16
    AddNode("S313_Conv2d_1x64x1x1_Sigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S310_Output", 0),
            GNodeArg(GNA_IN, "Conv_306_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_cls_preds_1_bias", 0),
            GNodeArg(GNA_OUT, "S313_Output", 0),
            GNodeArg(GNA_IN, "S313_Mul_scale", 0),
            GNodeArg(GNA_IN, "S313_Mul_shift", 0),
            GNodeArg(GNA_IN, "S313_Infos", 0)
        )
    );
    // Node S316_Conv2d_64x1x3x3_Custom inq -3.40<(i8-0.00)*0.02659554<3.38 forced weightsq chan<(i8-0.00)*chan<chan outq -5.15<(i8-0.00)*0.04027301<5.11 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S316_Conv2d_64x1x3x3_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S297_Output", 0),
            GNodeArg(GNA_IN, "Conv_307_weights", 0),
            GNodeArg(GNA_IN, "Constant__1411", 0),
            GNodeArg(GNA_OUT, "S316_Output", 0),
            GNodeArg(GNA_IN, "S316_Mul_scale", 0),
            GNodeArg(GNA_IN, "S316_Mul_shift", 0),
            GNodeArg(GNA_IN, "S316_Infos", 0),
            GNodeArg(GNA_IN, "S316_Custom_infos", 0)
        )
    );
    // Node S319_Conv2d_64x64x1x1_Custom inq -5.15<(i8-0.00)*0.04027301<5.11 weightsq chan<(i8-0.00)*chan<chan outq -2.82<(i8-0.00)*0.02205693<2.80 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S319_Conv2d_64x64x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S316_Output", 0),
            GNodeArg(GNA_IN, "Conv_310_weights", 0),
            GNodeArg(GNA_IN, "Constant__1414", 0),
            GNodeArg(GNA_OUT, "S319_Output", 0),
            GNodeArg(GNA_IN, "S319_Mul_scale", 0),
            GNodeArg(GNA_IN, "S319_Mul_shift", 0),
            GNodeArg(GNA_IN, "S319_Infos", 0),
            GNodeArg(GNA_IN, "S319_Custom_infos", 0)
        )
    );
    // Node S322_Conv2d_64x1x3x3_Custom inq -2.82<(i8-0.00)*0.02205693<2.80 forced weightsq chan<(i8-0.00)*chan<chan outq -3.42<(i8-0.00)*0.02673931<3.40 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S322_Conv2d_64x1x3x3_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S319_Output", 0),
            GNodeArg(GNA_IN, "Conv_313_weights", 0),
            GNodeArg(GNA_IN, "Constant__1417", 0),
            GNodeArg(GNA_OUT, "S322_Output", 0),
            GNodeArg(GNA_IN, "S322_Mul_scale", 0),
            GNodeArg(GNA_IN, "S322_Mul_shift", 0),
            GNodeArg(GNA_IN, "S322_Infos", 0),
            GNodeArg(GNA_IN, "S322_Custom_infos", 0)
        )
    );
    // Node S325_Conv2d_64x64x1x1_Custom inq -3.42<(i8-0.00)*0.02673931<3.40 weightsq chan<(i8-0.00)*chan<chan outq -2.18<(i8-0.00)*0.01702390<2.16 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S325_Conv2d_64x64x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S322_Output", 0),
            GNodeArg(GNA_IN, "Conv_316_weights", 0),
            GNodeArg(GNA_IN, "Constant__1420", 0),
            GNodeArg(GNA_OUT, "S325_Output", 0),
            GNodeArg(GNA_IN, "S325_Mul_scale", 0),
            GNodeArg(GNA_IN, "S325_Mul_shift", 0),
            GNodeArg(GNA_IN, "S325_Infos", 0),
            GNodeArg(GNA_IN, "S325_Custom_infos", 0)
        )
    );
    // Node S328_Conv2d_4x64x1x1 inq -2.18<(i8-0.00)*0.01702390<2.16 weightsq chan<(i8-0.00)*chan<chan outq -3.16<(i8-0.00)*0.02471484<3.14 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S328_Conv2d_4x64x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S325_Output", 0),
            GNodeArg(GNA_IN, "Conv_319_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_reg_preds_1_bias", 0),
            GNodeArg(GNA_OUT, "S328_Output", 0),
            GNodeArg(GNA_IN, "S328_Mul_scale", 0),
            GNodeArg(GNA_IN, "S328_Mul_shift", 0),
            GNodeArg(GNA_IN, "S328_Infos", 0)
        )
    );
    // Node S331_Conv2d_1x64x1x1_Sigmoid inq -2.18<(i8-0.00)*0.01702390<2.16 weightsq -1.08<(i8-0.00)*0.00850352<1.08 outq -3.16<(i8-0.00)*0.02471484<3.14 forced biasesq -310876.16<(i32-0.00)*0.00014476<310876.16
    AddNode("S331_Conv2d_1x64x1x1_Sigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S325_Output", 0),
            GNodeArg(GNA_IN, "Conv_320_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_obj_preds_1_bias", 0),
            GNodeArg(GNA_OUT, "S331_Output", 0),
            GNodeArg(GNA_IN, "S331_Mul_scale", 0),
            GNodeArg(GNA_IN, "S331_Mul_shift", 0),
            GNodeArg(GNA_IN, "S331_Infos", 0)
        )
    );
    // Node S336_Conv2d_64x256x1x1_Custom inq -2.26<(i8-0.00)*0.01763853<2.24 weightsq chan<(i8-0.00)*chan<chan outq -1.68<(i8-0.00)*0.01313226<1.67 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S336_Conv2d_64x256x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S255_Output", 0),
            GNodeArg(GNA_IN, "Conv_324_weights", 0),
            GNodeArg(GNA_IN, "Constant__1423", 0),
            GNodeArg(GNA_OUT, "S336_Output", 0),
            GNodeArg(GNA_IN, "S336_Mul_scale", 0),
            GNodeArg(GNA_IN, "S336_Mul_shift", 0),
            GNodeArg(GNA_IN, "S336_Infos", 0),
            GNodeArg(GNA_IN, "S336_Custom_infos", 0)
        )
    );
    // Node S339_Conv2d_64x1x3x3_Custom inq -1.68<(i8-0.00)*0.01313226<1.67 forced weightsq chan<(i8-0.00)*chan<chan outq -1.90<(i8-0.00)*0.01485778<1.89 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S339_Conv2d_64x1x3x3_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S336_Output", 0),
            GNodeArg(GNA_IN, "Conv_327_weights", 0),
            GNodeArg(GNA_IN, "Constant__1426", 0),
            GNodeArg(GNA_OUT, "S339_Output", 0),
            GNodeArg(GNA_IN, "S339_Mul_scale", 0),
            GNodeArg(GNA_IN, "S339_Mul_shift", 0),
            GNodeArg(GNA_IN, "S339_Infos", 0),
            GNodeArg(GNA_IN, "S339_Custom_infos", 0)
        )
    );
    // Node S342_Conv2d_64x64x1x1_Custom inq -1.90<(i8-0.00)*0.01485778<1.89 weightsq chan<(i8-0.00)*chan<chan outq -1.71<(i8-0.00)*0.01335468<1.70 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S342_Conv2d_64x64x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S339_Output", 0),
            GNodeArg(GNA_IN, "Conv_330_weights", 0),
            GNodeArg(GNA_IN, "Constant__1429", 0),
            GNodeArg(GNA_OUT, "S342_Output", 0),
            GNodeArg(GNA_IN, "S342_Mul_scale", 0),
            GNodeArg(GNA_IN, "S342_Mul_shift", 0),
            GNodeArg(GNA_IN, "S342_Infos", 0),
            GNodeArg(GNA_IN, "S342_Custom_infos", 0)
        )
    );
    // Node S345_Conv2d_64x1x3x3_Custom inq -1.71<(i8-0.00)*0.01335468<1.70 forced weightsq chan<(i8-0.00)*chan<chan outq -2.06<(i8-0.00)*0.01606062<2.04 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S345_Conv2d_64x1x3x3_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S342_Output", 0),
            GNodeArg(GNA_IN, "Conv_333_weights", 0),
            GNodeArg(GNA_IN, "Constant__1432", 0),
            GNodeArg(GNA_OUT, "S345_Output", 0),
            GNodeArg(GNA_IN, "S345_Mul_scale", 0),
            GNodeArg(GNA_IN, "S345_Mul_shift", 0),
            GNodeArg(GNA_IN, "S345_Infos", 0),
            GNodeArg(GNA_IN, "S345_Custom_infos", 0)
        )
    );
    // Node S348_Conv2d_64x64x1x1 inq -2.06<(i8-0.00)*0.01606062<2.04 weightsq chan<(i8-0.00)*chan<chan outq -0.72<(i8--62.00)*0.01085656<2.05 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S348_Conv2d_64x64x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S345_Output", 0),
            GNodeArg(GNA_IN, "Conv_336_weights", 0),
            GNodeArg(GNA_IN, "Constant__1435", 0),
            GNodeArg(GNA_OUT, "S348_Output", 0),
            GNodeArg(GNA_IN, "S348_Mul_scale", 0),
            GNodeArg(GNA_IN, "S348_Mul_shift", 0),
            GNodeArg(GNA_IN, "S348_Infos", 0)
        )
    );
    // Node expr_78 in_qs [-0.72<(i8--62.00)*0.01085656<2.05] out_qs [-1.83<(i8-0.00)*0.01428580<1.81]
    AddNode("S349_Op_expr_78",
        Bindings(2,
            GNodeArg(GNA_IN, "S348_Output", 0),
            GNodeArg(GNA_OUT, "S349_Output", 0)
        )
    );
    // Node S352_Conv2d_1x64x1x1_Sigmoid inq -1.83<(i8-0.00)*0.01428580<1.81 weightsq -0.08<(i8-0.00)*0.00059674<0.08 outq -3.16<(i8-0.00)*0.02471484<3.14 forced biasesq -18307.23<(i32-0.00)*0.00000852<18307.23
    AddNode("S352_Conv2d_1x64x1x1_Sigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S349_Output", 0),
            GNodeArg(GNA_IN, "Conv_339_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_cls_preds_2_bias", 0),
            GNodeArg(GNA_OUT, "S352_Output", 0),
            GNodeArg(GNA_IN, "S352_Mul_scale", 0),
            GNodeArg(GNA_IN, "S352_Mul_shift", 0),
            GNodeArg(GNA_IN, "S352_Infos", 0)
        )
    );
    // Node S355_Conv2d_64x1x3x3_Custom inq -1.68<(i8-0.00)*0.01313226<1.67 forced weightsq chan<(i8-0.00)*chan<chan outq -1.60<(i8-0.00)*0.01251750<1.59 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S355_Conv2d_64x1x3x3_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S336_Output", 0),
            GNodeArg(GNA_IN, "Conv_340_weights", 0),
            GNodeArg(GNA_IN, "Constant__1438", 0),
            GNodeArg(GNA_OUT, "S355_Output", 0),
            GNodeArg(GNA_IN, "S355_Mul_scale", 0),
            GNodeArg(GNA_IN, "S355_Mul_shift", 0),
            GNodeArg(GNA_IN, "S355_Infos", 0),
            GNodeArg(GNA_IN, "S355_Custom_infos", 0)
        )
    );
    // Node S358_Conv2d_64x64x1x1_Custom inq -1.60<(i8-0.00)*0.01251750<1.59 weightsq chan<(i8-0.00)*chan<chan outq -1.13<(i8-0.00)*0.00884513<1.12 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S358_Conv2d_64x64x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S355_Output", 0),
            GNodeArg(GNA_IN, "Conv_343_weights", 0),
            GNodeArg(GNA_IN, "Constant__1441", 0),
            GNodeArg(GNA_OUT, "S358_Output", 0),
            GNodeArg(GNA_IN, "S358_Mul_scale", 0),
            GNodeArg(GNA_IN, "S358_Mul_shift", 0),
            GNodeArg(GNA_IN, "S358_Infos", 0),
            GNodeArg(GNA_IN, "S358_Custom_infos", 0)
        )
    );
    // Node S361_Conv2d_64x1x3x3_Custom inq -1.13<(i8-0.00)*0.00884513<1.12 forced weightsq chan<(i8-0.00)*chan<chan outq -1.15<(i8-0.00)*0.00900488<1.14 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S361_Conv2d_64x1x3x3_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S358_Output", 0),
            GNodeArg(GNA_IN, "Conv_346_weights", 0),
            GNodeArg(GNA_IN, "Constant__1444", 0),
            GNodeArg(GNA_OUT, "S361_Output", 0),
            GNodeArg(GNA_IN, "S361_Mul_scale", 0),
            GNodeArg(GNA_IN, "S361_Mul_shift", 0),
            GNodeArg(GNA_IN, "S361_Infos", 0),
            GNodeArg(GNA_IN, "S361_Custom_infos", 0)
        )
    );
    // Node S364_Conv2d_64x64x1x1_Custom inq -1.15<(i8-0.00)*0.00900488<1.14 weightsq chan<(i8-0.00)*chan<chan outq -1.50<(i8-0.00)*0.01173902<1.49 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S364_Conv2d_64x64x1x1_Custom",
        Bindings(8,
            GNodeArg(GNA_IN, "S361_Output", 0),
            GNodeArg(GNA_IN, "Conv_349_weights", 0),
            GNodeArg(GNA_IN, "Constant__1447", 0),
            GNodeArg(GNA_OUT, "S364_Output", 0),
            GNodeArg(GNA_IN, "S364_Mul_scale", 0),
            GNodeArg(GNA_IN, "S364_Mul_shift", 0),
            GNodeArg(GNA_IN, "S364_Infos", 0),
            GNodeArg(GNA_IN, "S364_Custom_infos", 0)
        )
    );
    // Node S367_Conv2d_4x64x1x1 inq -1.50<(i8-0.00)*0.01173902<1.49 weightsq chan<(i8-0.00)*chan<chan outq -3.16<(i8-0.00)*0.02471484<3.14 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S367_Conv2d_4x64x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S364_Output", 0),
            GNodeArg(GNA_IN, "Conv_352_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_reg_preds_2_bias", 0),
            GNodeArg(GNA_OUT, "S367_Output", 0),
            GNodeArg(GNA_IN, "S367_Mul_scale", 0),
            GNodeArg(GNA_IN, "S367_Mul_shift", 0),
            GNodeArg(GNA_IN, "S367_Infos", 0)
        )
    );
    // Node S370_Conv2d_1x64x1x1_Sigmoid inq -1.50<(i8-0.00)*0.01173902<1.49 weightsq -0.80<(i8-0.00)*0.00630339<0.80 outq -3.16<(i8-0.00)*0.02471484<3.14 forced biasesq -158904.44<(i32-0.00)*0.00007400<158904.44
    AddNode("S370_Conv2d_1x64x1x1_Sigmoid",
        Bindings(7,
            GNodeArg(GNA_IN, "S364_Output", 0),
            GNodeArg(GNA_IN, "Conv_353_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_obj_preds_2_bias", 0),
            GNodeArg(GNA_OUT, "S370_Output", 0),
            GNodeArg(GNA_IN, "S370_Mul_scale", 0),
            GNodeArg(GNA_IN, "S370_Mul_shift", 0),
            GNodeArg(GNA_IN, "S370_Infos", 0)
        )
    );
    // Node Concat_381 inq ['-3.16<(i8-0.00)*0.02471484<3.14 forced', '-3.16<(i8-0.00)*0.02471484<3.14 forced', '-3.16<(i8-0.00)*0.02471484<3.14 forced'] outq ['-3.16<(i8-0.00)*0.02471484<3.14 forced']
    AddNode("S373_Concat",
        Bindings(4,
            GNodeArg(GNA_IN, "S293_Output", 0),
            GNodeArg(GNA_IN, "S332_Output", 0),
            GNodeArg(GNA_IN, "S371_Output", 0),
            GNodeArg(GNA_OUT, "Output_1", 0)
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
    modelModel(128000, 1000000, 8000000, 64*1024*1024);
    GenerateTilingCode();
    return 0;
}
