#include <stdint.h>
#include <stdio.h>
#include "AutoTilerLib.h"
#include "CNN_Generators_fp16.h"
#include "ResizeGenerator.h"

#include "CNN_Copy_Generators.h"

void load_expressions_kernels() {
    LibKernelTemplate(
        "s25_kernel_args_t",
        CArgs(4,
            TCArg("unsigned int", "I0"),
            TCArg("F16 *__restrict__ ", "expr_0_in_0"),
            TCArg("F16 *__restrict__ ", "expr_0_in_1"),
            TCArg("F16 *__restrict__ ", "expr_0_out_0")
        )
    );
    
    LibKernel(
        "s25_kernel",
        CALL_PARALLEL,
        0,
        "s25_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s51_kernel_args_t",
        CArgs(4,
            TCArg("unsigned int", "I0"),
            TCArg("F16 *__restrict__ ", "expr_1_in_0"),
            TCArg("F16 *__restrict__ ", "expr_1_in_1"),
            TCArg("F16 *__restrict__ ", "expr_1_out_0")
        )
    );
    
    LibKernel(
        "s51_kernel",
        CALL_PARALLEL,
        0,
        "s51_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s97_kernel_args_t",
        CArgs(4,
            TCArg("unsigned int", "I0"),
            TCArg("F16 *__restrict__ ", "expr_2_in_0"),
            TCArg("F16 *__restrict__ ", "expr_2_in_1"),
            TCArg("F16 *__restrict__ ", "expr_2_out_0")
        )
    );
    
    LibKernel(
        "s97_kernel",
        CALL_PARALLEL,
        0,
        "s97_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "expr_66_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_66_in_0"),
            TCArg("F16 *__restrict__ ", "expr_66_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_66",
        CALL_PARALLEL,
        0,
        "expr_66_args_t",
        0
    );
    LibKernelTemplate(
        "expr_91_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_91_in_0"),
            TCArg("F16 *__restrict__ ", "expr_91_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_91",
        CALL_PARALLEL,
        0,
        "expr_91_args_t",
        0
    );
    LibKernelTemplate(
        "expr_101_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_101_in_0"),
            TCArg("F16 *__restrict__ ", "expr_101_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_101",
        CALL_PARALLEL,
        0,
        "expr_101_args_t",
        0
    );
    LibKernelTemplate(
        "expr_8_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_8_in_0"),
            TCArg("F16 *__restrict__ ", "expr_8_out_0"),
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
        "expr_18_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_18_in_0"),
            TCArg("F16 *__restrict__ ", "expr_18_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_18",
        CALL_PARALLEL,
        0,
        "expr_18_args_t",
        0
    );
    LibKernelTemplate(
        "expr_27_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_27_in_0"),
            TCArg("F16 *__restrict__ ", "expr_27_out_0"),
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
        "expr_37_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_37_in_0"),
            TCArg("F16 *__restrict__ ", "expr_37_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_37",
        CALL_PARALLEL,
        0,
        "expr_37_args_t",
        0
    );
    LibKernelTemplate(
        "expr_62_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_62_in_0"),
            TCArg("F16 *__restrict__ ", "expr_62_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_62",
        CALL_PARALLEL,
        0,
        "expr_62_args_t",
        0
    );
    LibKernelTemplate(
        "expr_73_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_73_in_0"),
            TCArg("F16 *__restrict__ ", "expr_73_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_73",
        CALL_PARALLEL,
        0,
        "expr_73_args_t",
        0
    );
    LibKernelTemplate(
        "expr_82_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_82_in_0"),
            TCArg("F16 *__restrict__ ", "expr_82_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_82",
        CALL_PARALLEL,
        0,
        "expr_82_args_t",
        0
    );
    LibKernelTemplate(
        "expr_84_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_84_in_0"),
            TCArg("F16 *__restrict__ ", "expr_84_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_84",
        CALL_PARALLEL,
        0,
        "expr_84_args_t",
        0
    );
    LibKernelTemplate(
        "expr_85_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_85_in_0"),
            TCArg("F16 *__restrict__ ", "expr_85_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_85",
        CALL_PARALLEL,
        0,
        "expr_85_args_t",
        0
    );
    LibKernelTemplate(
        "expr_86_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_86_in_0"),
            TCArg("F16 *__restrict__ ", "expr_86_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_86",
        CALL_PARALLEL,
        0,
        "expr_86_args_t",
        0
    );
    LibKernelTemplate(
        "expr_87_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_87_in_0"),
            TCArg("F16 *__restrict__ ", "expr_87_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_87",
        CALL_PARALLEL,
        0,
        "expr_87_args_t",
        0
    );
    LibKernelTemplate(
        "expr_88_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_88_in_0"),
            TCArg("F16 *__restrict__ ", "expr_88_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_88",
        CALL_PARALLEL,
        0,
        "expr_88_args_t",
        0
    );
    LibKernelTemplate(
        "expr_89_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_89_in_0"),
            TCArg("F16 *__restrict__ ", "expr_89_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_89",
        CALL_PARALLEL,
        0,
        "expr_89_args_t",
        0
    );
    LibKernelTemplate(
        "expr_90_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_90_in_0"),
            TCArg("F16 *__restrict__ ", "expr_90_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_90",
        CALL_PARALLEL,
        0,
        "expr_90_args_t",
        0
    );
    LibKernelTemplate(
        "expr_92_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_92_in_0"),
            TCArg("F16 *__restrict__ ", "expr_92_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_92",
        CALL_PARALLEL,
        0,
        "expr_92_args_t",
        0
    );
    LibKernelTemplate(
        "expr_93_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_93_in_0"),
            TCArg("F16 *__restrict__ ", "expr_93_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_93",
        CALL_PARALLEL,
        0,
        "expr_93_args_t",
        0
    );
    LibKernelTemplate(
        "expr_94_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_94_in_0"),
            TCArg("F16 *__restrict__ ", "expr_94_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_94",
        CALL_PARALLEL,
        0,
        "expr_94_args_t",
        0
    );
    LibKernelTemplate(
        "expr_95_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_95_in_0"),
            TCArg("F16 *__restrict__ ", "expr_95_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_95",
        CALL_PARALLEL,
        0,
        "expr_95_args_t",
        0
    );
    LibKernelTemplate(
        "expr_96_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_96_in_0"),
            TCArg("F16 *__restrict__ ", "expr_96_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_96",
        CALL_PARALLEL,
        0,
        "expr_96_args_t",
        0
    );
    LibKernelTemplate(
        "expr_97_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_97_in_0"),
            TCArg("F16 *__restrict__ ", "expr_97_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_97",
        CALL_PARALLEL,
        0,
        "expr_97_args_t",
        0
    );
    LibKernelTemplate(
        "expr_98_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_98_in_0"),
            TCArg("F16 *__restrict__ ", "expr_98_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_98",
        CALL_PARALLEL,
        0,
        "expr_98_args_t",
        0
    );
    LibKernelTemplate(
        "expr_99_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_99_in_0"),
            TCArg("F16 *__restrict__ ", "expr_99_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_99",
        CALL_PARALLEL,
        0,
        "expr_99_args_t",
        0
    );
    LibKernelTemplate(
        "expr_100_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_100_in_0"),
            TCArg("F16 *__restrict__ ", "expr_100_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_100",
        CALL_PARALLEL,
        0,
        "expr_100_args_t",
        0
    );
    LibKernelTemplate(
        "expr_102_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_102_in_0"),
            TCArg("F16 *__restrict__ ", "expr_102_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_102",
        CALL_PARALLEL,
        0,
        "expr_102_args_t",
        0
    );
    LibKernelTemplate(
        "expr_103_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_103_in_0"),
            TCArg("F16 *__restrict__ ", "expr_103_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_103",
        CALL_PARALLEL,
        0,
        "expr_103_args_t",
        0
    );
    LibKernelTemplate(
        "expr_3_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_3_in_0"),
            TCArg("F16 *__restrict__ ", "expr_3_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_3",
        CALL_PARALLEL,
        0,
        "expr_3_args_t",
        0
    );
    LibKernelTemplate(
        "expr_4_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_4_in_0"),
            TCArg("F16 *__restrict__ ", "expr_4_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_4",
        CALL_PARALLEL,
        0,
        "expr_4_args_t",
        0
    );
    LibKernelTemplate(
        "expr_5_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_5_in_0"),
            TCArg("F16 *__restrict__ ", "expr_5_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_5",
        CALL_PARALLEL,
        0,
        "expr_5_args_t",
        0
    );
    LibKernelTemplate(
        "expr_6_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_6_in_0"),
            TCArg("F16 *__restrict__ ", "expr_6_out_0"),
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
        "expr_7_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_7_in_0"),
            TCArg("F16 *__restrict__ ", "expr_7_out_0"),
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
        "expr_9_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_9_in_0"),
            TCArg("F16 *__restrict__ ", "expr_9_out_0"),
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
            TCArg("F16 *__restrict__ ", "expr_10_in_0"),
            TCArg("F16 *__restrict__ ", "expr_10_out_0"),
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
        "expr_11_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_11_in_0"),
            TCArg("F16 *__restrict__ ", "expr_11_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_11",
        CALL_PARALLEL,
        0,
        "expr_11_args_t",
        0
    );
    LibKernelTemplate(
        "expr_12_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_12_in_0"),
            TCArg("F16 *__restrict__ ", "expr_12_out_0"),
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
            TCArg("F16 *__restrict__ ", "expr_13_in_0"),
            TCArg("F16 *__restrict__ ", "expr_13_out_0"),
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
        "expr_14_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_14_in_0"),
            TCArg("F16 *__restrict__ ", "expr_14_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_14",
        CALL_PARALLEL,
        0,
        "expr_14_args_t",
        0
    );
    LibKernelTemplate(
        "expr_15_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_15_in_0"),
            TCArg("F16 *__restrict__ ", "expr_15_out_0"),
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
        "expr_16_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_16_in_0"),
            TCArg("F16 *__restrict__ ", "expr_16_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_16",
        CALL_PARALLEL,
        0,
        "expr_16_args_t",
        0
    );
    LibKernelTemplate(
        "expr_17_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_17_in_0"),
            TCArg("F16 *__restrict__ ", "expr_17_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_17",
        CALL_PARALLEL,
        0,
        "expr_17_args_t",
        0
    );
    LibKernelTemplate(
        "expr_19_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_19_in_0"),
            TCArg("F16 *__restrict__ ", "expr_19_out_0"),
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
            TCArg("F16 *__restrict__ ", "expr_20_in_0"),
            TCArg("F16 *__restrict__ ", "expr_20_out_0"),
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
            TCArg("F16 *__restrict__ ", "expr_21_in_0"),
            TCArg("F16 *__restrict__ ", "expr_21_out_0"),
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
            TCArg("F16 *__restrict__ ", "expr_22_in_0"),
            TCArg("F16 *__restrict__ ", "expr_22_out_0"),
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
            TCArg("F16 *__restrict__ ", "expr_23_in_0"),
            TCArg("F16 *__restrict__ ", "expr_23_out_0"),
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
            TCArg("F16 *__restrict__ ", "expr_24_in_0"),
            TCArg("F16 *__restrict__ ", "expr_24_out_0"),
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
        "expr_25_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_25_in_0"),
            TCArg("F16 *__restrict__ ", "expr_25_out_0"),
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
        "expr_26_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_26_in_0"),
            TCArg("F16 *__restrict__ ", "expr_26_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_26",
        CALL_PARALLEL,
        0,
        "expr_26_args_t",
        0
    );
    LibKernelTemplate(
        "expr_28_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_28_in_0"),
            TCArg("F16 *__restrict__ ", "expr_28_out_0"),
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
        "expr_29_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_29_in_0"),
            TCArg("F16 *__restrict__ ", "expr_29_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_29",
        CALL_PARALLEL,
        0,
        "expr_29_args_t",
        0
    );
    LibKernelTemplate(
        "expr_30_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_30_in_0"),
            TCArg("F16 *__restrict__ ", "expr_30_out_0"),
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
            TCArg("F16 *__restrict__ ", "expr_31_in_0"),
            TCArg("F16 *__restrict__ ", "expr_31_out_0"),
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
        "expr_32_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_32_in_0"),
            TCArg("F16 *__restrict__ ", "expr_32_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_32",
        CALL_PARALLEL,
        0,
        "expr_32_args_t",
        0
    );
    LibKernelTemplate(
        "expr_33_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_33_in_0"),
            TCArg("F16 *__restrict__ ", "expr_33_out_0"),
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
        "expr_34_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_34_in_0"),
            TCArg("F16 *__restrict__ ", "expr_34_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_34",
        CALL_PARALLEL,
        0,
        "expr_34_args_t",
        0
    );
    LibKernelTemplate(
        "expr_35_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_35_in_0"),
            TCArg("F16 *__restrict__ ", "expr_35_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_35",
        CALL_PARALLEL,
        0,
        "expr_35_args_t",
        0
    );
    LibKernelTemplate(
        "expr_36_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_36_in_0"),
            TCArg("F16 *__restrict__ ", "expr_36_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_36",
        CALL_PARALLEL,
        0,
        "expr_36_args_t",
        0
    );
    LibKernelTemplate(
        "expr_38_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_38_in_0"),
            TCArg("F16 *__restrict__ ", "expr_38_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_38",
        CALL_PARALLEL,
        0,
        "expr_38_args_t",
        0
    );
    LibKernelTemplate(
        "expr_39_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_39_in_0"),
            TCArg("F16 *__restrict__ ", "expr_39_out_0"),
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
    LibKernelTemplate(
        "expr_40_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_40_in_0"),
            TCArg("F16 *__restrict__ ", "expr_40_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_40",
        CALL_PARALLEL,
        0,
        "expr_40_args_t",
        0
    );
    LibKernelTemplate(
        "expr_41_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_41_in_0"),
            TCArg("F16 *__restrict__ ", "expr_41_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_41",
        CALL_PARALLEL,
        0,
        "expr_41_args_t",
        0
    );
    LibKernelTemplate(
        "expr_42_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_42_in_0"),
            TCArg("F16 *__restrict__ ", "expr_42_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_42",
        CALL_PARALLEL,
        0,
        "expr_42_args_t",
        0
    );
    LibKernelTemplate(
        "expr_43_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_43_in_0"),
            TCArg("F16 *__restrict__ ", "expr_43_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_43",
        CALL_PARALLEL,
        0,
        "expr_43_args_t",
        0
    );
    LibKernelTemplate(
        "expr_44_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_44_in_0"),
            TCArg("F16 *__restrict__ ", "expr_44_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_44",
        CALL_PARALLEL,
        0,
        "expr_44_args_t",
        0
    );
    LibKernelTemplate(
        "expr_45_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_45_in_0"),
            TCArg("F16 *__restrict__ ", "expr_45_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_45",
        CALL_PARALLEL,
        0,
        "expr_45_args_t",
        0
    );
    LibKernelTemplate(
        "expr_46_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_46_in_0"),
            TCArg("F16 *__restrict__ ", "expr_46_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_46",
        CALL_PARALLEL,
        0,
        "expr_46_args_t",
        0
    );
    LibKernelTemplate(
        "expr_47_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_47_in_0"),
            TCArg("F16 *__restrict__ ", "expr_47_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_47",
        CALL_PARALLEL,
        0,
        "expr_47_args_t",
        0
    );
    LibKernelTemplate(
        "expr_48_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_48_in_0"),
            TCArg("F16 *__restrict__ ", "expr_48_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_48",
        CALL_PARALLEL,
        0,
        "expr_48_args_t",
        0
    );
    LibKernelTemplate(
        "expr_49_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_49_in_0"),
            TCArg("F16 *__restrict__ ", "expr_49_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_49",
        CALL_PARALLEL,
        0,
        "expr_49_args_t",
        0
    );
    LibKernelTemplate(
        "expr_50_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_50_in_0"),
            TCArg("F16 *__restrict__ ", "expr_50_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_50",
        CALL_PARALLEL,
        0,
        "expr_50_args_t",
        0
    );
    LibKernelTemplate(
        "expr_51_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_51_in_0"),
            TCArg("F16 *__restrict__ ", "expr_51_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_51",
        CALL_PARALLEL,
        0,
        "expr_51_args_t",
        0
    );
    LibKernelTemplate(
        "expr_52_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_52_in_0"),
            TCArg("F16 *__restrict__ ", "expr_52_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_52",
        CALL_PARALLEL,
        0,
        "expr_52_args_t",
        0
    );
    LibKernelTemplate(
        "expr_53_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_53_in_0"),
            TCArg("F16 *__restrict__ ", "expr_53_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_53",
        CALL_PARALLEL,
        0,
        "expr_53_args_t",
        0
    );
    LibKernelTemplate(
        "expr_54_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_54_in_0"),
            TCArg("F16 *__restrict__ ", "expr_54_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_54",
        CALL_PARALLEL,
        0,
        "expr_54_args_t",
        0
    );
    LibKernelTemplate(
        "expr_55_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_55_in_0"),
            TCArg("F16 *__restrict__ ", "expr_55_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_55",
        CALL_PARALLEL,
        0,
        "expr_55_args_t",
        0
    );
    LibKernelTemplate(
        "expr_56_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_56_in_0"),
            TCArg("F16 *__restrict__ ", "expr_56_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_56",
        CALL_PARALLEL,
        0,
        "expr_56_args_t",
        0
    );
    LibKernelTemplate(
        "expr_57_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_57_in_0"),
            TCArg("F16 *__restrict__ ", "expr_57_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_57",
        CALL_PARALLEL,
        0,
        "expr_57_args_t",
        0
    );
    LibKernelTemplate(
        "expr_58_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_58_in_0"),
            TCArg("F16 *__restrict__ ", "expr_58_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_58",
        CALL_PARALLEL,
        0,
        "expr_58_args_t",
        0
    );
    LibKernelTemplate(
        "expr_59_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_59_in_0"),
            TCArg("F16 *__restrict__ ", "expr_59_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_59",
        CALL_PARALLEL,
        0,
        "expr_59_args_t",
        0
    );
    LibKernelTemplate(
        "expr_60_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_60_in_0"),
            TCArg("F16 *__restrict__ ", "expr_60_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_60",
        CALL_PARALLEL,
        0,
        "expr_60_args_t",
        0
    );
    LibKernelTemplate(
        "expr_61_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_61_in_0"),
            TCArg("F16 *__restrict__ ", "expr_61_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_61",
        CALL_PARALLEL,
        0,
        "expr_61_args_t",
        0
    );
    LibKernelTemplate(
        "expr_63_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_63_in_0"),
            TCArg("F16 *__restrict__ ", "expr_63_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_63",
        CALL_PARALLEL,
        0,
        "expr_63_args_t",
        0
    );
    LibKernelTemplate(
        "expr_64_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_64_in_0"),
            TCArg("F16 *__restrict__ ", "expr_64_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_64",
        CALL_PARALLEL,
        0,
        "expr_64_args_t",
        0
    );
    LibKernelTemplate(
        "expr_65_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_65_in_0"),
            TCArg("F16 *__restrict__ ", "expr_65_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_65",
        CALL_PARALLEL,
        0,
        "expr_65_args_t",
        0
    );
    LibKernelTemplate(
        "expr_67_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_67_in_0"),
            TCArg("F16 *__restrict__ ", "expr_67_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_67",
        CALL_PARALLEL,
        0,
        "expr_67_args_t",
        0
    );
    LibKernelTemplate(
        "expr_68_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_68_in_0"),
            TCArg("F16 *__restrict__ ", "expr_68_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_68",
        CALL_PARALLEL,
        0,
        "expr_68_args_t",
        0
    );
    LibKernelTemplate(
        "expr_69_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_69_in_0"),
            TCArg("F16 *__restrict__ ", "expr_69_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_69",
        CALL_PARALLEL,
        0,
        "expr_69_args_t",
        0
    );
    LibKernelTemplate(
        "expr_70_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_70_in_0"),
            TCArg("F16 *__restrict__ ", "expr_70_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_70",
        CALL_PARALLEL,
        0,
        "expr_70_args_t",
        0
    );
    LibKernelTemplate(
        "expr_71_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_71_in_0"),
            TCArg("F16 *__restrict__ ", "expr_71_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_71",
        CALL_PARALLEL,
        0,
        "expr_71_args_t",
        0
    );
    LibKernelTemplate(
        "expr_72_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_72_in_0"),
            TCArg("F16 *__restrict__ ", "expr_72_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_72",
        CALL_PARALLEL,
        0,
        "expr_72_args_t",
        0
    );
    LibKernelTemplate(
        "expr_74_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_74_in_0"),
            TCArg("F16 *__restrict__ ", "expr_74_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_74",
        CALL_PARALLEL,
        0,
        "expr_74_args_t",
        0
    );
    LibKernelTemplate(
        "expr_75_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_75_in_0"),
            TCArg("F16 *__restrict__ ", "expr_75_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_75",
        CALL_PARALLEL,
        0,
        "expr_75_args_t",
        0
    );
    LibKernelTemplate(
        "expr_76_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_76_in_0"),
            TCArg("F16 *__restrict__ ", "expr_76_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_76",
        CALL_PARALLEL,
        0,
        "expr_76_args_t",
        0
    );
    LibKernelTemplate(
        "expr_77_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_77_in_0"),
            TCArg("F16 *__restrict__ ", "expr_77_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_77",
        CALL_PARALLEL,
        0,
        "expr_77_args_t",
        0
    );
    LibKernelTemplate(
        "expr_78_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_78_in_0"),
            TCArg("F16 *__restrict__ ", "expr_78_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_78",
        CALL_PARALLEL,
        0,
        "expr_78_args_t",
        0
    );
    LibKernelTemplate(
        "expr_79_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_79_in_0"),
            TCArg("F16 *__restrict__ ", "expr_79_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_79",
        CALL_PARALLEL,
        0,
        "expr_79_args_t",
        0
    );
    LibKernelTemplate(
        "expr_80_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_80_in_0"),
            TCArg("F16 *__restrict__ ", "expr_80_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_80",
        CALL_PARALLEL,
        0,
        "expr_80_args_t",
        0
    );
    LibKernelTemplate(
        "expr_81_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_81_in_0"),
            TCArg("F16 *__restrict__ ", "expr_81_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_81",
        CALL_PARALLEL,
        0,
        "expr_81_args_t",
        0
    );
    LibKernelTemplate(
        "expr_83_args_t",
        CArgs(5,
            TCArg("F16 *__restrict__ ", "expr_83_in_0"),
            TCArg("F16 *__restrict__ ", "expr_83_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_83",
        CALL_PARALLEL,
        0,
        "expr_83_args_t",
        0
    );
}



int s25_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (16, 64, 80) spaces: ((0, 1, 2),) 
        // parametric_spaces: ((0, 1, 2),) 
        // exterior_shape: (81920.0,) 
        KernelIterSpace(2, IterParSpace(KER_ITER_D0, 81920, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(3,
            TCArg(CNN_ArgDataTypeF(2, 1, 1), "expr_0_in_0"),
            TCArg(CNN_ArgDataTypeF(2, 1, 1), "expr_0_in_1"),
            TCArg(CNN_ArgDataTypeF(2, 1, 1), "expr_0_out_0")
        ),
        Calls(1,
            Call("s25_kernel", LOC_D0,
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
            KerArg("expr_0_out_0", KerArgSpace(1, KER_ITER_D0), O_OUT|O_DB, 1, 1, 2, 0, 0, 0, "expr_0_out_0"),
            KerArg("expr_0_in_0",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_0_in_0"),
            KerArg("expr_0_in_1",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_0_in_1")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 81920, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 245760, 0);
        AddKernelFloatArgDim(Name, "expr_0_in_0",  4, 16, 64, 80, 2);
        AddKernelFloatArgDim(Name, "expr_0_in_1",  4, 16, 64, 80, 2);
        AddKernelFloatArgDim(Name, "expr_0_out_0", 4, 16, 64, 80, 2);
    }
    return (Kernel!=0);
}
int s51_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (32, 32, 40) spaces: ((0, 1, 2),) 
        // parametric_spaces: ((0, 1, 2),) 
        // exterior_shape: (40960.0,) 
        KernelIterSpace(2, IterParSpace(KER_ITER_D0, 40960, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(3,
            TCArg(CNN_ArgDataTypeF(2, 1, 1), "expr_1_in_0"),
            TCArg(CNN_ArgDataTypeF(2, 1, 1), "expr_1_in_1"),
            TCArg(CNN_ArgDataTypeF(2, 1, 1), "expr_1_out_0")
        ),
        Calls(1,
            Call("s51_kernel", LOC_D0,
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
            KerArg("expr_1_out_0", KerArgSpace(1, KER_ITER_D0), O_OUT|O_DB, 1, 1, 2, 0, 0, 0, "expr_1_out_0"),
            KerArg("expr_1_in_0",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_1_in_0"),
            KerArg("expr_1_in_1",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_1_in_1")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 40960, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 122880, 0);
        AddKernelFloatArgDim(Name, "expr_1_in_0",  4, 32, 32, 40, 2);
        AddKernelFloatArgDim(Name, "expr_1_in_1",  4, 32, 32, 40, 2);
        AddKernelFloatArgDim(Name, "expr_1_out_0", 4, 32, 32, 40, 2);
    }
    return (Kernel!=0);
}
int s97_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (64, 16, 20) spaces: ((0, 1, 2),) 
        // parametric_spaces: ((0, 1, 2),) 
        // exterior_shape: (20480.0,) 
        KernelIterSpace(2, IterParSpace(KER_ITER_D0, 20480, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(3,
            TCArg(CNN_ArgDataTypeF(2, 1, 1), "expr_2_in_0"),
            TCArg(CNN_ArgDataTypeF(2, 1, 1), "expr_2_in_1"),
            TCArg(CNN_ArgDataTypeF(2, 1, 1), "expr_2_out_0")
        ),
        Calls(1,
            Call("s97_kernel", LOC_D0,
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
            KerArg("expr_2_out_0", KerArgSpace(1, KER_ITER_D0), O_OUT|O_DB, 1, 1, 2, 0, 0, 0, "expr_2_out_0"),
            KerArg("expr_2_in_0",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_2_in_0"),
            KerArg("expr_2_in_1",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_2_in_1")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 20480, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 61440, 0);
        AddKernelFloatArgDim(Name, "expr_2_in_0",  4, 64, 16, 20, 2);
        AddKernelFloatArgDim(Name, "expr_2_in_1",  4, 64, 16, 20, 2);
        AddKernelFloatArgDim(Name, "expr_2_out_0", 4, 64, 16, 20, 2);
    }
    return (Kernel!=0);
}

void modelModel(unsigned int L1Memory, unsigned int L2Memory, unsigned int L3Memory, unsigned int L3Flash)
{
    KernelOper_T Cop = KOP_CONV;

    // SetKernelOpts(KER_OPT_NONE, KER_OPT_BUFFER_PROMOTE);
    SetSymbolDynamics();

    SetUsedFilesNames(0, 6, "Gap.h", "model.h", "CNN_BasicKernels_fp16.h", "ResizeBasicKernels.h", "CNN_BasicKernels_SQ8.h", "Expression_Kernels.h");
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

    LoadCNNLibrary_fp16();
    LoadResizeLibrary();
    LoadCNN_Copy_Library();
    load_expressions_kernels();

    CNN_GenControl_T gen_ctrl_S3_Conv2d_16x12x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S3_Conv2d_16x12x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S3_Conv2d_16x12x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_66");
    // generator for Conv_0_fusion
    CNN_ConvolutionPoolAct_fp16("S3_Conv2d_16x12x3x3_Custom", &gen_ctrl_S3_Conv2d_16x12x3x3_Custom,
                                 12, 16, 160, 128,
                                 KOP_CONV, 3, 3, 1, 1, 1, 1, 1,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S6_Conv2d_16x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S6_Conv2d_16x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S6_Conv2d_16x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_91");
    // generator for Conv_3_fusion
    CNN_ConvolutionPoolAct_fp16("S6_Conv2d_16x1x3x3_Custom", &gen_ctrl_S6_Conv2d_16x1x3x3_Custom,
                                 16, 16, 160, 128,
                                 KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S9_Conv2d_32x16x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S9_Conv2d_32x16x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S9_Conv2d_32x16x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S9_Conv2d_32x16x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_101");
    // generator for Conv_6_fusion
    CNN_ConvolutionPoolAct_fp16("S9_Conv2d_32x16x1x1_Custom", &gen_ctrl_S9_Conv2d_32x16x1x1_Custom,
                                 16, 32, 80, 64,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S12_Conv2d_16x32x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S12_Conv2d_16x32x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S12_Conv2d_16x32x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S12_Conv2d_16x32x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_8");
    // generator for Conv_9_fusion
    CNN_ConvolutionPoolAct_fp16("S12_Conv2d_16x32x1x1_Custom", &gen_ctrl_S12_Conv2d_16x32x1x1_Custom,
                                 32, 16, 80, 64,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S15_Conv2d_16x32x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S15_Conv2d_16x32x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S15_Conv2d_16x32x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S15_Conv2d_16x32x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_18");
    // generator for Conv_12_fusion
    CNN_ConvolutionPoolAct_fp16("S15_Conv2d_16x32x1x1_Custom", &gen_ctrl_S15_Conv2d_16x32x1x1_Custom,
                                 32, 16, 80, 64,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S18_Conv2d_16x16x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S18_Conv2d_16x16x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S18_Conv2d_16x16x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S18_Conv2d_16x16x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_27");
    // generator for Conv_15_fusion
    CNN_ConvolutionPoolAct_fp16("S18_Conv2d_16x16x1x1_Custom", &gen_ctrl_S18_Conv2d_16x16x1x1_Custom,
                                 16, 16, 80, 64,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S21_Conv2d_16x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S21_Conv2d_16x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S21_Conv2d_16x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_37");
    // generator for Conv_18_fusion
    CNN_ConvolutionPoolAct_fp16("S21_Conv2d_16x1x3x3_Custom", &gen_ctrl_S21_Conv2d_16x1x3x3_Custom,
                                 16, 16, 80, 64,
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S24_Conv2d_16x16x1x1;
    CNN_InitGenCtrl(&gen_ctrl_S24_Conv2d_16x16x1x1);
    CNN_SetGenCtrl(&gen_ctrl_S24_Conv2d_16x16x1x1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for Conv_21
    CNN_ConvolutionPoolAct_fp16("S24_Conv2d_16x16x1x1", &gen_ctrl_S24_Conv2d_16x16x1x1,
                                 16, 16, 80, 64,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    
    // generator for expr_0
    s25_kernel_gen("S25_Op_expr_0");
    
    CNN_GenControl_T gen_ctrl_S29_Conv2d_32x32x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S29_Conv2d_32x32x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S29_Conv2d_32x32x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S29_Conv2d_32x32x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_62");
    // generator for Conv_26_fusion
    CNN_ConvolutionPoolAct_fp16("S29_Conv2d_32x32x1x1_Custom", &gen_ctrl_S29_Conv2d_32x32x1x1_Custom,
                                 32, 32, 80, 64,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S32_Conv2d_32x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S32_Conv2d_32x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S32_Conv2d_32x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_73");
    // generator for Conv_29_fusion
    CNN_ConvolutionPoolAct_fp16("S32_Conv2d_32x1x3x3_Custom", &gen_ctrl_S32_Conv2d_32x1x3x3_Custom,
                                 32, 32, 80, 64,
                                 KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S35_Conv2d_64x32x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S35_Conv2d_64x32x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S35_Conv2d_64x32x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S35_Conv2d_64x32x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_82");
    // generator for Conv_32_fusion
    CNN_ConvolutionPoolAct_fp16("S35_Conv2d_64x32x1x1_Custom", &gen_ctrl_S35_Conv2d_64x32x1x1_Custom,
                                 32, 64, 40, 32,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S38_Conv2d_32x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S38_Conv2d_32x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S38_Conv2d_32x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S38_Conv2d_32x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_84");
    // generator for Conv_35_fusion
    CNN_ConvolutionPoolAct_fp16("S38_Conv2d_32x64x1x1_Custom", &gen_ctrl_S38_Conv2d_32x64x1x1_Custom,
                                 64, 32, 40, 32,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S41_Conv2d_32x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S41_Conv2d_32x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S41_Conv2d_32x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S41_Conv2d_32x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_85");
    // generator for Conv_38_fusion
    CNN_ConvolutionPoolAct_fp16("S41_Conv2d_32x64x1x1_Custom", &gen_ctrl_S41_Conv2d_32x64x1x1_Custom,
                                 64, 32, 40, 32,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S44_Conv2d_32x32x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S44_Conv2d_32x32x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S44_Conv2d_32x32x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S44_Conv2d_32x32x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_86");
    // generator for Conv_41_fusion
    CNN_ConvolutionPoolAct_fp16("S44_Conv2d_32x32x1x1_Custom", &gen_ctrl_S44_Conv2d_32x32x1x1_Custom,
                                 32, 32, 40, 32,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S47_Conv2d_32x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S47_Conv2d_32x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S47_Conv2d_32x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_87");
    // generator for Conv_44_fusion
    CNN_ConvolutionPoolAct_fp16("S47_Conv2d_32x1x3x3_Custom", &gen_ctrl_S47_Conv2d_32x1x3x3_Custom,
                                 32, 32, 40, 32,
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S50_Conv2d_32x32x1x1;
    CNN_InitGenCtrl(&gen_ctrl_S50_Conv2d_32x32x1x1);
    CNN_SetGenCtrl(&gen_ctrl_S50_Conv2d_32x32x1x1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for Conv_47
    CNN_ConvolutionPoolAct_fp16("S50_Conv2d_32x32x1x1", &gen_ctrl_S50_Conv2d_32x32x1x1,
                                 32, 32, 40, 32,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    
    // generator for expr_1
    s51_kernel_gen("S51_Op_expr_1");
    
    CNN_GenControl_T gen_ctrl_S54_Conv2d_32x32x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S54_Conv2d_32x32x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S54_Conv2d_32x32x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S54_Conv2d_32x32x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_88");
    // generator for Conv_51_fusion
    CNN_ConvolutionPoolAct_fp16("S54_Conv2d_32x32x1x1_Custom", &gen_ctrl_S54_Conv2d_32x32x1x1_Custom,
                                 32, 32, 40, 32,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S57_Conv2d_32x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S57_Conv2d_32x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S57_Conv2d_32x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_89");
    // generator for Conv_54_fusion
    CNN_ConvolutionPoolAct_fp16("S57_Conv2d_32x1x3x3_Custom", &gen_ctrl_S57_Conv2d_32x1x3x3_Custom,
                                 32, 32, 40, 32,
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S60_Conv2d_32x32x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S60_Conv2d_32x32x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S60_Conv2d_32x32x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S60_Conv2d_32x32x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_90");
    // generator for Conv_57_fusion
    CNN_ConvolutionPoolAct_fp16("S60_Conv2d_32x32x1x1_Custom", &gen_ctrl_S60_Conv2d_32x32x1x1_Custom,
                                 32, 32, 40, 32,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    
    // generator for Add_60
    CNN_MatAddAct_fp16("S61_MatAdd_32x32x40", 0, 32, 32, 32, 40, KOP_MATADD, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S64_Conv2d_32x32x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S64_Conv2d_32x32x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S64_Conv2d_32x32x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S64_Conv2d_32x32x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_92");
    // generator for Conv_61_fusion
    CNN_ConvolutionPoolAct_fp16("S64_Conv2d_32x32x1x1_Custom", &gen_ctrl_S64_Conv2d_32x32x1x1_Custom,
                                 32, 32, 40, 32,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S67_Conv2d_32x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S67_Conv2d_32x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S67_Conv2d_32x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_93");
    // generator for Conv_64_fusion
    CNN_ConvolutionPoolAct_fp16("S67_Conv2d_32x1x3x3_Custom", &gen_ctrl_S67_Conv2d_32x1x3x3_Custom,
                                 32, 32, 40, 32,
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S70_Conv2d_32x32x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S70_Conv2d_32x32x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S70_Conv2d_32x32x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S70_Conv2d_32x32x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_94");
    // generator for Conv_67_fusion
    CNN_ConvolutionPoolAct_fp16("S70_Conv2d_32x32x1x1_Custom", &gen_ctrl_S70_Conv2d_32x32x1x1_Custom,
                                 32, 32, 40, 32,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    
    // generator for Add_70
    CNN_MatAddAct_fp16("S71_MatAdd_32x32x40", 0, 32, 32, 32, 40, KOP_MATADD, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S75_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S75_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S75_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S75_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_95");
    // generator for Conv_72_fusion
    CNN_ConvolutionPoolAct_fp16("S75_Conv2d_64x64x1x1_Custom", &gen_ctrl_S75_Conv2d_64x64x1x1_Custom,
                                 64, 64, 40, 32,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S78_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S78_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S78_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_96");
    // generator for Conv_75_fusion
    CNN_ConvolutionPoolAct_fp16("S78_Conv2d_64x1x3x3_Custom", &gen_ctrl_S78_Conv2d_64x1x3x3_Custom,
                                 64, 64, 40, 32,
                                 KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S81_Conv2d_128x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S81_Conv2d_128x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S81_Conv2d_128x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S81_Conv2d_128x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_97");
    // generator for Conv_78_fusion
    CNN_ConvolutionPoolAct_fp16("S81_Conv2d_128x64x1x1_Custom", &gen_ctrl_S81_Conv2d_128x64x1x1_Custom,
                                 64, 128, 20, 16,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S84_Conv2d_64x128x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S84_Conv2d_64x128x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S84_Conv2d_64x128x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S84_Conv2d_64x128x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_98");
    // generator for Conv_81_fusion
    CNN_ConvolutionPoolAct_fp16("S84_Conv2d_64x128x1x1_Custom", &gen_ctrl_S84_Conv2d_64x128x1x1_Custom,
                                 128, 64, 20, 16,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S87_Conv2d_64x128x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S87_Conv2d_64x128x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S87_Conv2d_64x128x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S87_Conv2d_64x128x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_99");
    // generator for Conv_84_fusion
    CNN_ConvolutionPoolAct_fp16("S87_Conv2d_64x128x1x1_Custom", &gen_ctrl_S87_Conv2d_64x128x1x1_Custom,
                                 128, 64, 20, 16,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S90_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S90_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S90_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S90_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_100");
    // generator for Conv_87_fusion
    CNN_ConvolutionPoolAct_fp16("S90_Conv2d_64x64x1x1_Custom", &gen_ctrl_S90_Conv2d_64x64x1x1_Custom,
                                 64, 64, 20, 16,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S93_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S93_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S93_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_102");
    // generator for Conv_90_fusion
    CNN_ConvolutionPoolAct_fp16("S93_Conv2d_64x1x3x3_Custom", &gen_ctrl_S93_Conv2d_64x1x3x3_Custom,
                                 64, 64, 20, 16,
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S96_Conv2d_64x64x1x1;
    CNN_InitGenCtrl(&gen_ctrl_S96_Conv2d_64x64x1x1);
    CNN_SetGenCtrl(&gen_ctrl_S96_Conv2d_64x64x1x1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for Conv_93
    CNN_ConvolutionPoolAct_fp16("S96_Conv2d_64x64x1x1", &gen_ctrl_S96_Conv2d_64x64x1x1,
                                 64, 64, 20, 16,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    
    // generator for expr_2
    s97_kernel_gen("S97_Op_expr_2");
    
    CNN_GenControl_T gen_ctrl_S100_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S100_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S100_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S100_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_103");
    // generator for Conv_97_fusion
    CNN_ConvolutionPoolAct_fp16("S100_Conv2d_64x64x1x1_Custom", &gen_ctrl_S100_Conv2d_64x64x1x1_Custom,
                                 64, 64, 20, 16,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S103_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S103_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S103_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_3");
    // generator for Conv_100_fusion
    CNN_ConvolutionPoolAct_fp16("S103_Conv2d_64x1x3x3_Custom", &gen_ctrl_S103_Conv2d_64x1x3x3_Custom,
                                 64, 64, 20, 16,
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S106_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S106_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S106_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S106_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_4");
    // generator for Conv_103_fusion
    CNN_ConvolutionPoolAct_fp16("S106_Conv2d_64x64x1x1_Custom", &gen_ctrl_S106_Conv2d_64x64x1x1_Custom,
                                 64, 64, 20, 16,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    
    // generator for Add_106
    CNN_MatAddAct_fp16("S107_MatAdd_64x16x20", 0, 64, 64, 16, 20, KOP_MATADD, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S110_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S110_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S110_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S110_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_5");
    // generator for Conv_107_fusion
    CNN_ConvolutionPoolAct_fp16("S110_Conv2d_64x64x1x1_Custom", &gen_ctrl_S110_Conv2d_64x64x1x1_Custom,
                                 64, 64, 20, 16,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S113_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S113_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S113_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_6");
    // generator for Conv_110_fusion
    CNN_ConvolutionPoolAct_fp16("S113_Conv2d_64x1x3x3_Custom", &gen_ctrl_S113_Conv2d_64x1x3x3_Custom,
                                 64, 64, 20, 16,
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S116_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S116_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S116_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S116_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_7");
    // generator for Conv_113_fusion
    CNN_ConvolutionPoolAct_fp16("S116_Conv2d_64x64x1x1_Custom", &gen_ctrl_S116_Conv2d_64x64x1x1_Custom,
                                 64, 64, 20, 16,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    
    // generator for Add_116
    CNN_MatAddAct_fp16("S117_MatAdd_64x16x20", 0, 64, 64, 16, 20, KOP_MATADD, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S121_Conv2d_128x128x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S121_Conv2d_128x128x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S121_Conv2d_128x128x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S121_Conv2d_128x128x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_9");
    // generator for Conv_118_fusion
    CNN_ConvolutionPoolAct_fp16("S121_Conv2d_128x128x1x1_Custom", &gen_ctrl_S121_Conv2d_128x128x1x1_Custom,
                                 128, 128, 20, 16,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S124_Conv2d_128x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S124_Conv2d_128x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S124_Conv2d_128x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_10");
    // generator for Conv_121_fusion
    CNN_ConvolutionPoolAct_fp16("S124_Conv2d_128x1x3x3_Custom", &gen_ctrl_S124_Conv2d_128x1x3x3_Custom,
                                 128, 128, 20, 16,
                                 KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S127_Conv2d_256x128x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S127_Conv2d_256x128x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S127_Conv2d_256x128x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S127_Conv2d_256x128x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_11");
    // generator for Conv_124_fusion
    CNN_ConvolutionPoolAct_fp16("S127_Conv2d_256x128x1x1_Custom", &gen_ctrl_S127_Conv2d_256x128x1x1_Custom,
                                 128, 256, 10, 8,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S130_Conv2d_128x256x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S130_Conv2d_128x256x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S130_Conv2d_128x256x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S130_Conv2d_128x256x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_12");
    // generator for Conv_127_fusion
    CNN_ConvolutionPoolAct_fp16("S130_Conv2d_128x256x1x1_Custom", &gen_ctrl_S130_Conv2d_128x256x1x1_Custom,
                                 256, 128, 10, 8,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    // generator for MaxPool_131
    CNN_PoolAct_fp16("S131_MaxPool_9x9", 0,
                     128, 128, 10, 8,
                     KOP_MAXPOOL, 9, 9, 1, 1, 1, 1, 1,
                     KOP_NONE);
    
    // generator for MaxPool_132
    CNN_PoolAct_fp16("S132_MaxPool_13x13", 0,
                     128, 128, 10, 8,
                     KOP_MAXPOOL, 13, 13, 1, 1, 1, 1, 1,
                     KOP_NONE);
    
    // generator for MaxPool_130
    CNN_PoolAct_fp16("S133_MaxPool_5x5", 0,
                     128, 128, 10, 8,
                     KOP_MAXPOOL, 5, 5, 1, 1, 1, 1, 1,
                     KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S137_Conv2d_256x512x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S137_Conv2d_256x512x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S137_Conv2d_256x512x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S137_Conv2d_256x512x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_13");
    // generator for Conv_134_fusion
    CNN_ConvolutionPoolAct_fp16("S137_Conv2d_256x512x1x1_Custom", &gen_ctrl_S137_Conv2d_256x512x1x1_Custom,
                                 512, 256, 10, 8,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S140_Conv2d_128x256x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S140_Conv2d_128x256x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S140_Conv2d_128x256x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S140_Conv2d_128x256x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_14");
    // generator for Conv_137_fusion
    CNN_ConvolutionPoolAct_fp16("S140_Conv2d_128x256x1x1_Custom", &gen_ctrl_S140_Conv2d_128x256x1x1_Custom,
                                 256, 128, 10, 8,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S143_Conv2d_128x256x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S143_Conv2d_128x256x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S143_Conv2d_128x256x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S143_Conv2d_128x256x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_15");
    // generator for Conv_140_fusion
    CNN_ConvolutionPoolAct_fp16("S143_Conv2d_128x256x1x1_Custom", &gen_ctrl_S143_Conv2d_128x256x1x1_Custom,
                                 256, 128, 10, 8,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S146_Conv2d_128x128x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S146_Conv2d_128x128x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S146_Conv2d_128x128x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S146_Conv2d_128x128x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_16");
    // generator for Conv_143_fusion
    CNN_ConvolutionPoolAct_fp16("S146_Conv2d_128x128x1x1_Custom", &gen_ctrl_S146_Conv2d_128x128x1x1_Custom,
                                 128, 128, 10, 8,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S149_Conv2d_128x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S149_Conv2d_128x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S149_Conv2d_128x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_17");
    // generator for Conv_146_fusion
    CNN_ConvolutionPoolAct_fp16("S149_Conv2d_128x1x3x3_Custom", &gen_ctrl_S149_Conv2d_128x1x3x3_Custom,
                                 128, 128, 10, 8,
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S152_Conv2d_128x128x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S152_Conv2d_128x128x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S152_Conv2d_128x128x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S152_Conv2d_128x128x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_19");
    // generator for Conv_149_fusion
    CNN_ConvolutionPoolAct_fp16("S152_Conv2d_128x128x1x1_Custom", &gen_ctrl_S152_Conv2d_128x128x1x1_Custom,
                                 128, 128, 10, 8,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S156_Conv2d_256x256x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S156_Conv2d_256x256x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S156_Conv2d_256x256x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S156_Conv2d_256x256x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_20");
    // generator for Conv_153_fusion
    CNN_ConvolutionPoolAct_fp16("S156_Conv2d_256x256x1x1_Custom", &gen_ctrl_S156_Conv2d_256x256x1x1_Custom,
                                 256, 256, 10, 8,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S159_Conv2d_128x256x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S159_Conv2d_128x256x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S159_Conv2d_128x256x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S159_Conv2d_128x256x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_21");
    // generator for Conv_156_fusion
    CNN_ConvolutionPoolAct_fp16("S159_Conv2d_128x256x1x1_Custom", &gen_ctrl_S159_Conv2d_128x256x1x1_Custom,
                                 256, 128, 10, 8,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    
    // generator for Resize_160
    GenerateResizeMultiChannel_fp16("S160_Op_Resize_160", 10, 8, 20, 16, 128, SIGNED_INOUT, KOP_NEAREST_NEIGHBOR_RESIZE);
    
    CNN_GenControl_T gen_ctrl_S164_Conv2d_64x256x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S164_Conv2d_64x256x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S164_Conv2d_64x256x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S164_Conv2d_64x256x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_22");
    // generator for Conv_162_fusion
    CNN_ConvolutionPoolAct_fp16("S164_Conv2d_64x256x1x1_Custom", &gen_ctrl_S164_Conv2d_64x256x1x1_Custom,
                                 256, 64, 20, 16,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S167_Conv2d_64x256x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S167_Conv2d_64x256x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S167_Conv2d_64x256x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S167_Conv2d_64x256x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_23");
    // generator for Conv_165_fusion
    CNN_ConvolutionPoolAct_fp16("S167_Conv2d_64x256x1x1_Custom", &gen_ctrl_S167_Conv2d_64x256x1x1_Custom,
                                 256, 64, 20, 16,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S170_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S170_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S170_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S170_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_24");
    // generator for Conv_168_fusion
    CNN_ConvolutionPoolAct_fp16("S170_Conv2d_64x64x1x1_Custom", &gen_ctrl_S170_Conv2d_64x64x1x1_Custom,
                                 64, 64, 20, 16,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S173_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S173_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S173_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_25");
    // generator for Conv_171_fusion
    CNN_ConvolutionPoolAct_fp16("S173_Conv2d_64x1x3x3_Custom", &gen_ctrl_S173_Conv2d_64x1x3x3_Custom,
                                 64, 64, 20, 16,
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S176_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S176_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S176_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S176_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_26");
    // generator for Conv_174_fusion
    CNN_ConvolutionPoolAct_fp16("S176_Conv2d_64x64x1x1_Custom", &gen_ctrl_S176_Conv2d_64x64x1x1_Custom,
                                 64, 64, 20, 16,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S180_Conv2d_128x128x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S180_Conv2d_128x128x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S180_Conv2d_128x128x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S180_Conv2d_128x128x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_28");
    // generator for Conv_178_fusion
    CNN_ConvolutionPoolAct_fp16("S180_Conv2d_128x128x1x1_Custom", &gen_ctrl_S180_Conv2d_128x128x1x1_Custom,
                                 128, 128, 20, 16,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S183_Conv2d_64x128x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S183_Conv2d_64x128x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S183_Conv2d_64x128x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S183_Conv2d_64x128x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_29");
    // generator for Conv_181_fusion
    CNN_ConvolutionPoolAct_fp16("S183_Conv2d_64x128x1x1_Custom", &gen_ctrl_S183_Conv2d_64x128x1x1_Custom,
                                 128, 64, 20, 16,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    
    // generator for Resize_185
    GenerateResizeMultiChannel_fp16("S184_Op_Resize_185", 20, 16, 40, 32, 64, SIGNED_INOUT, KOP_NEAREST_NEIGHBOR_RESIZE);
    
    CNN_GenControl_T gen_ctrl_S188_Conv2d_32x128x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S188_Conv2d_32x128x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S188_Conv2d_32x128x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S188_Conv2d_32x128x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_30");
    // generator for Conv_187_fusion
    CNN_ConvolutionPoolAct_fp16("S188_Conv2d_32x128x1x1_Custom", &gen_ctrl_S188_Conv2d_32x128x1x1_Custom,
                                 128, 32, 40, 32,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S191_Conv2d_32x128x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S191_Conv2d_32x128x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S191_Conv2d_32x128x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S191_Conv2d_32x128x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_31");
    // generator for Conv_190_fusion
    CNN_ConvolutionPoolAct_fp16("S191_Conv2d_32x128x1x1_Custom", &gen_ctrl_S191_Conv2d_32x128x1x1_Custom,
                                 128, 32, 40, 32,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S194_Conv2d_32x32x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S194_Conv2d_32x32x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S194_Conv2d_32x32x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S194_Conv2d_32x32x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_32");
    // generator for Conv_193_fusion
    CNN_ConvolutionPoolAct_fp16("S194_Conv2d_32x32x1x1_Custom", &gen_ctrl_S194_Conv2d_32x32x1x1_Custom,
                                 32, 32, 40, 32,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S197_Conv2d_32x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S197_Conv2d_32x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S197_Conv2d_32x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_33");
    // generator for Conv_196_fusion
    CNN_ConvolutionPoolAct_fp16("S197_Conv2d_32x1x3x3_Custom", &gen_ctrl_S197_Conv2d_32x1x3x3_Custom,
                                 32, 32, 40, 32,
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S200_Conv2d_32x32x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S200_Conv2d_32x32x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S200_Conv2d_32x32x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S200_Conv2d_32x32x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_34");
    // generator for Conv_199_fusion
    CNN_ConvolutionPoolAct_fp16("S200_Conv2d_32x32x1x1_Custom", &gen_ctrl_S200_Conv2d_32x32x1x1_Custom,
                                 32, 32, 40, 32,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S204_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S204_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S204_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S204_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_35");
    // generator for Conv_203_fusion
    CNN_ConvolutionPoolAct_fp16("S204_Conv2d_64x64x1x1_Custom", &gen_ctrl_S204_Conv2d_64x64x1x1_Custom,
                                 64, 64, 40, 32,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S207_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S207_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S207_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_36");
    // generator for Conv_206_fusion
    CNN_ConvolutionPoolAct_fp16("S207_Conv2d_64x1x3x3_Custom", &gen_ctrl_S207_Conv2d_64x1x3x3_Custom,
                                 64, 64, 40, 32,
                                 KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S210_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S210_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S210_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S210_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_38");
    // generator for Conv_209_fusion
    CNN_ConvolutionPoolAct_fp16("S210_Conv2d_64x64x1x1_Custom", &gen_ctrl_S210_Conv2d_64x64x1x1_Custom,
                                 64, 64, 20, 16,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S214_Conv2d_64x128x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S214_Conv2d_64x128x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S214_Conv2d_64x128x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S214_Conv2d_64x128x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_39");
    // generator for Conv_213_fusion
    CNN_ConvolutionPoolAct_fp16("S214_Conv2d_64x128x1x1_Custom", &gen_ctrl_S214_Conv2d_64x128x1x1_Custom,
                                 128, 64, 20, 16,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S217_Conv2d_64x128x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S217_Conv2d_64x128x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S217_Conv2d_64x128x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S217_Conv2d_64x128x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_40");
    // generator for Conv_216_fusion
    CNN_ConvolutionPoolAct_fp16("S217_Conv2d_64x128x1x1_Custom", &gen_ctrl_S217_Conv2d_64x128x1x1_Custom,
                                 128, 64, 20, 16,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S220_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S220_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S220_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S220_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_41");
    // generator for Conv_219_fusion
    CNN_ConvolutionPoolAct_fp16("S220_Conv2d_64x64x1x1_Custom", &gen_ctrl_S220_Conv2d_64x64x1x1_Custom,
                                 64, 64, 20, 16,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S223_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S223_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S223_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_42");
    // generator for Conv_222_fusion
    CNN_ConvolutionPoolAct_fp16("S223_Conv2d_64x1x3x3_Custom", &gen_ctrl_S223_Conv2d_64x1x3x3_Custom,
                                 64, 64, 20, 16,
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S226_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S226_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S226_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S226_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_43");
    // generator for Conv_225_fusion
    CNN_ConvolutionPoolAct_fp16("S226_Conv2d_64x64x1x1_Custom", &gen_ctrl_S226_Conv2d_64x64x1x1_Custom,
                                 64, 64, 20, 16,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S230_Conv2d_128x128x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S230_Conv2d_128x128x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S230_Conv2d_128x128x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S230_Conv2d_128x128x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_44");
    // generator for Conv_229_fusion
    CNN_ConvolutionPoolAct_fp16("S230_Conv2d_128x128x1x1_Custom", &gen_ctrl_S230_Conv2d_128x128x1x1_Custom,
                                 128, 128, 20, 16,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S233_Conv2d_128x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S233_Conv2d_128x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S233_Conv2d_128x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_45");
    // generator for Conv_232_fusion
    CNN_ConvolutionPoolAct_fp16("S233_Conv2d_128x1x3x3_Custom", &gen_ctrl_S233_Conv2d_128x1x3x3_Custom,
                                 128, 128, 20, 16,
                                 KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S236_Conv2d_128x128x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S236_Conv2d_128x128x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S236_Conv2d_128x128x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S236_Conv2d_128x128x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_46");
    // generator for Conv_235_fusion
    CNN_ConvolutionPoolAct_fp16("S236_Conv2d_128x128x1x1_Custom", &gen_ctrl_S236_Conv2d_128x128x1x1_Custom,
                                 128, 128, 10, 8,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S240_Conv2d_128x256x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S240_Conv2d_128x256x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S240_Conv2d_128x256x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S240_Conv2d_128x256x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_47");
    // generator for Conv_239_fusion
    CNN_ConvolutionPoolAct_fp16("S240_Conv2d_128x256x1x1_Custom", &gen_ctrl_S240_Conv2d_128x256x1x1_Custom,
                                 256, 128, 10, 8,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S243_Conv2d_128x256x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S243_Conv2d_128x256x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S243_Conv2d_128x256x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S243_Conv2d_128x256x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_48");
    // generator for Conv_242_fusion
    CNN_ConvolutionPoolAct_fp16("S243_Conv2d_128x256x1x1_Custom", &gen_ctrl_S243_Conv2d_128x256x1x1_Custom,
                                 256, 128, 10, 8,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S246_Conv2d_128x128x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S246_Conv2d_128x128x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S246_Conv2d_128x128x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S246_Conv2d_128x128x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_49");
    // generator for Conv_245_fusion
    CNN_ConvolutionPoolAct_fp16("S246_Conv2d_128x128x1x1_Custom", &gen_ctrl_S246_Conv2d_128x128x1x1_Custom,
                                 128, 128, 10, 8,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S249_Conv2d_128x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S249_Conv2d_128x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S249_Conv2d_128x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_50");
    // generator for Conv_248_fusion
    CNN_ConvolutionPoolAct_fp16("S249_Conv2d_128x1x3x3_Custom", &gen_ctrl_S249_Conv2d_128x1x3x3_Custom,
                                 128, 128, 10, 8,
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S252_Conv2d_128x128x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S252_Conv2d_128x128x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S252_Conv2d_128x128x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S252_Conv2d_128x128x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_51");
    // generator for Conv_251_fusion
    CNN_ConvolutionPoolAct_fp16("S252_Conv2d_128x128x1x1_Custom", &gen_ctrl_S252_Conv2d_128x128x1x1_Custom,
                                 128, 128, 10, 8,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S256_Conv2d_256x256x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S256_Conv2d_256x256x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S256_Conv2d_256x256x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S256_Conv2d_256x256x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_52");
    // generator for Conv_255_fusion
    CNN_ConvolutionPoolAct_fp16("S256_Conv2d_256x256x1x1_Custom", &gen_ctrl_S256_Conv2d_256x256x1x1_Custom,
                                 256, 256, 10, 8,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S259_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S259_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S259_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S259_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_53");
    // generator for Conv_258_fusion
    CNN_ConvolutionPoolAct_fp16("S259_Conv2d_64x64x1x1_Custom", &gen_ctrl_S259_Conv2d_64x64x1x1_Custom,
                                 64, 64, 40, 32,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S262_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S262_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S262_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_54");
    // generator for Conv_261_fusion
    CNN_ConvolutionPoolAct_fp16("S262_Conv2d_64x1x3x3_Custom", &gen_ctrl_S262_Conv2d_64x1x3x3_Custom,
                                 64, 64, 40, 32,
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S265_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S265_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S265_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S265_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_55");
    // generator for Conv_264_fusion
    CNN_ConvolutionPoolAct_fp16("S265_Conv2d_64x64x1x1_Custom", &gen_ctrl_S265_Conv2d_64x64x1x1_Custom,
                                 64, 64, 40, 32,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S268_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S268_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S268_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_56");
    // generator for Conv_267_fusion
    CNN_ConvolutionPoolAct_fp16("S268_Conv2d_64x1x3x3_Custom", &gen_ctrl_S268_Conv2d_64x1x3x3_Custom,
                                 64, 64, 40, 32,
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S271_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S271_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S271_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S271_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_57");
    // generator for Conv_270_fusion
    CNN_ConvolutionPoolAct_fp16("S271_Conv2d_64x64x1x1_Custom", &gen_ctrl_S271_Conv2d_64x64x1x1_Custom,
                                 64, 64, 40, 32,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S274_Conv2d_1x64x1x1_Sigmoid;
    CNN_InitGenCtrl(&gen_ctrl_S274_Conv2d_1x64x1x1_Sigmoid);
    CNN_SetGenCtrl(&gen_ctrl_S274_Conv2d_1x64x1x1_Sigmoid, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for Conv_273_fusion
    CNN_ConvolutionPoolAct_fp16("S274_Conv2d_1x64x1x1_Sigmoid", &gen_ctrl_S274_Conv2d_1x64x1x1_Sigmoid,
                                 64, 1, 40, 32,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_SIGMOID);
    
    CNN_GenControl_T gen_ctrl_S277_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S277_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S277_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_58");
    // generator for Conv_274_fusion
    CNN_ConvolutionPoolAct_fp16("S277_Conv2d_64x1x3x3_Custom", &gen_ctrl_S277_Conv2d_64x1x3x3_Custom,
                                 64, 64, 40, 32,
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S280_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S280_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S280_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S280_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_59");
    // generator for Conv_277_fusion
    CNN_ConvolutionPoolAct_fp16("S280_Conv2d_64x64x1x1_Custom", &gen_ctrl_S280_Conv2d_64x64x1x1_Custom,
                                 64, 64, 40, 32,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S283_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S283_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S283_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_60");
    // generator for Conv_280_fusion
    CNN_ConvolutionPoolAct_fp16("S283_Conv2d_64x1x3x3_Custom", &gen_ctrl_S283_Conv2d_64x1x3x3_Custom,
                                 64, 64, 40, 32,
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S286_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S286_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S286_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S286_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_61");
    // generator for Conv_283_fusion
    CNN_ConvolutionPoolAct_fp16("S286_Conv2d_64x64x1x1_Custom", &gen_ctrl_S286_Conv2d_64x64x1x1_Custom,
                                 64, 64, 40, 32,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S289_Conv2d_4x64x1x1;
    CNN_InitGenCtrl(&gen_ctrl_S289_Conv2d_4x64x1x1);
    CNN_SetGenCtrl(&gen_ctrl_S289_Conv2d_4x64x1x1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for Conv_286
    CNN_ConvolutionPoolAct_fp16("S289_Conv2d_4x64x1x1", &gen_ctrl_S289_Conv2d_4x64x1x1,
                                 64, 4, 40, 32,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S292_Conv2d_1x64x1x1_Sigmoid;
    CNN_InitGenCtrl(&gen_ctrl_S292_Conv2d_1x64x1x1_Sigmoid);
    CNN_SetGenCtrl(&gen_ctrl_S292_Conv2d_1x64x1x1_Sigmoid, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for Conv_287_fusion
    CNN_ConvolutionPoolAct_fp16("S292_Conv2d_1x64x1x1_Sigmoid", &gen_ctrl_S292_Conv2d_1x64x1x1_Sigmoid,
                                 64, 1, 40, 32,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_SIGMOID);
    
    CNN_GenControl_T gen_ctrl_S297_Conv2d_64x128x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S297_Conv2d_64x128x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S297_Conv2d_64x128x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S297_Conv2d_64x128x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_63");
    // generator for Conv_291_fusion
    CNN_ConvolutionPoolAct_fp16("S297_Conv2d_64x128x1x1_Custom", &gen_ctrl_S297_Conv2d_64x128x1x1_Custom,
                                 128, 64, 20, 16,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S300_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S300_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S300_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_64");
    // generator for Conv_294_fusion
    CNN_ConvolutionPoolAct_fp16("S300_Conv2d_64x1x3x3_Custom", &gen_ctrl_S300_Conv2d_64x1x3x3_Custom,
                                 64, 64, 20, 16,
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S303_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S303_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S303_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S303_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_65");
    // generator for Conv_297_fusion
    CNN_ConvolutionPoolAct_fp16("S303_Conv2d_64x64x1x1_Custom", &gen_ctrl_S303_Conv2d_64x64x1x1_Custom,
                                 64, 64, 20, 16,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S306_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S306_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S306_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_67");
    // generator for Conv_300_fusion
    CNN_ConvolutionPoolAct_fp16("S306_Conv2d_64x1x3x3_Custom", &gen_ctrl_S306_Conv2d_64x1x3x3_Custom,
                                 64, 64, 20, 16,
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S309_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S309_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S309_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S309_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_68");
    // generator for Conv_303_fusion
    CNN_ConvolutionPoolAct_fp16("S309_Conv2d_64x64x1x1_Custom", &gen_ctrl_S309_Conv2d_64x64x1x1_Custom,
                                 64, 64, 20, 16,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S312_Conv2d_1x64x1x1_Sigmoid;
    CNN_InitGenCtrl(&gen_ctrl_S312_Conv2d_1x64x1x1_Sigmoid);
    CNN_SetGenCtrl(&gen_ctrl_S312_Conv2d_1x64x1x1_Sigmoid, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for Conv_306_fusion
    CNN_ConvolutionPoolAct_fp16("S312_Conv2d_1x64x1x1_Sigmoid", &gen_ctrl_S312_Conv2d_1x64x1x1_Sigmoid,
                                 64, 1, 20, 16,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_SIGMOID);
    
    CNN_GenControl_T gen_ctrl_S315_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S315_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S315_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_69");
    // generator for Conv_307_fusion
    CNN_ConvolutionPoolAct_fp16("S315_Conv2d_64x1x3x3_Custom", &gen_ctrl_S315_Conv2d_64x1x3x3_Custom,
                                 64, 64, 20, 16,
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S318_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S318_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S318_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S318_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_70");
    // generator for Conv_310_fusion
    CNN_ConvolutionPoolAct_fp16("S318_Conv2d_64x64x1x1_Custom", &gen_ctrl_S318_Conv2d_64x64x1x1_Custom,
                                 64, 64, 20, 16,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S321_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S321_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S321_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_71");
    // generator for Conv_313_fusion
    CNN_ConvolutionPoolAct_fp16("S321_Conv2d_64x1x3x3_Custom", &gen_ctrl_S321_Conv2d_64x1x3x3_Custom,
                                 64, 64, 20, 16,
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S324_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S324_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S324_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S324_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_72");
    // generator for Conv_316_fusion
    CNN_ConvolutionPoolAct_fp16("S324_Conv2d_64x64x1x1_Custom", &gen_ctrl_S324_Conv2d_64x64x1x1_Custom,
                                 64, 64, 20, 16,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S327_Conv2d_4x64x1x1;
    CNN_InitGenCtrl(&gen_ctrl_S327_Conv2d_4x64x1x1);
    CNN_SetGenCtrl(&gen_ctrl_S327_Conv2d_4x64x1x1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for Conv_319
    CNN_ConvolutionPoolAct_fp16("S327_Conv2d_4x64x1x1", &gen_ctrl_S327_Conv2d_4x64x1x1,
                                 64, 4, 20, 16,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S330_Conv2d_1x64x1x1_Sigmoid;
    CNN_InitGenCtrl(&gen_ctrl_S330_Conv2d_1x64x1x1_Sigmoid);
    CNN_SetGenCtrl(&gen_ctrl_S330_Conv2d_1x64x1x1_Sigmoid, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for Conv_320_fusion
    CNN_ConvolutionPoolAct_fp16("S330_Conv2d_1x64x1x1_Sigmoid", &gen_ctrl_S330_Conv2d_1x64x1x1_Sigmoid,
                                 64, 1, 20, 16,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_SIGMOID);
    
    CNN_GenControl_T gen_ctrl_S335_Conv2d_64x256x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S335_Conv2d_64x256x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S335_Conv2d_64x256x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S335_Conv2d_64x256x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_74");
    // generator for Conv_324_fusion
    CNN_ConvolutionPoolAct_fp16("S335_Conv2d_64x256x1x1_Custom", &gen_ctrl_S335_Conv2d_64x256x1x1_Custom,
                                 256, 64, 10, 8,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S338_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S338_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S338_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_75");
    // generator for Conv_327_fusion
    CNN_ConvolutionPoolAct_fp16("S338_Conv2d_64x1x3x3_Custom", &gen_ctrl_S338_Conv2d_64x1x3x3_Custom,
                                 64, 64, 10, 8,
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S341_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S341_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S341_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S341_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_76");
    // generator for Conv_330_fusion
    CNN_ConvolutionPoolAct_fp16("S341_Conv2d_64x64x1x1_Custom", &gen_ctrl_S341_Conv2d_64x64x1x1_Custom,
                                 64, 64, 10, 8,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S344_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S344_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S344_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_77");
    // generator for Conv_333_fusion
    CNN_ConvolutionPoolAct_fp16("S344_Conv2d_64x1x3x3_Custom", &gen_ctrl_S344_Conv2d_64x1x3x3_Custom,
                                 64, 64, 10, 8,
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S347_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S347_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S347_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S347_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_78");
    // generator for Conv_336_fusion
    CNN_ConvolutionPoolAct_fp16("S347_Conv2d_64x64x1x1_Custom", &gen_ctrl_S347_Conv2d_64x64x1x1_Custom,
                                 64, 64, 10, 8,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S350_Conv2d_1x64x1x1_Sigmoid;
    CNN_InitGenCtrl(&gen_ctrl_S350_Conv2d_1x64x1x1_Sigmoid);
    CNN_SetGenCtrl(&gen_ctrl_S350_Conv2d_1x64x1x1_Sigmoid, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for Conv_339_fusion
    CNN_ConvolutionPoolAct_fp16("S350_Conv2d_1x64x1x1_Sigmoid", &gen_ctrl_S350_Conv2d_1x64x1x1_Sigmoid,
                                 64, 1, 10, 8,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_SIGMOID);
    
    CNN_GenControl_T gen_ctrl_S353_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S353_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S353_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_79");
    // generator for Conv_340_fusion
    CNN_ConvolutionPoolAct_fp16("S353_Conv2d_64x1x3x3_Custom", &gen_ctrl_S353_Conv2d_64x1x3x3_Custom,
                                 64, 64, 10, 8,
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S356_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S356_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S356_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S356_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_80");
    // generator for Conv_343_fusion
    CNN_ConvolutionPoolAct_fp16("S356_Conv2d_64x64x1x1_Custom", &gen_ctrl_S356_Conv2d_64x64x1x1_Custom,
                                 64, 64, 10, 8,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S359_Conv2d_64x1x3x3_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S359_Conv2d_64x1x3x3_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S359_Conv2d_64x1x3x3_Custom, "CUSTOM_ACTIVATION_NAME", "expr_81");
    // generator for Conv_346_fusion
    CNN_ConvolutionPoolAct_fp16("S359_Conv2d_64x1x3x3_Custom", &gen_ctrl_S359_Conv2d_64x1x3x3_Custom,
                                 64, 64, 10, 8,
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S362_Conv2d_64x64x1x1_Custom;
    CNN_InitGenCtrl(&gen_ctrl_S362_Conv2d_64x64x1x1_Custom);
    CNN_SetGenCtrl(&gen_ctrl_S362_Conv2d_64x64x1x1_Custom, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S362_Conv2d_64x64x1x1_Custom, "CUSTOM_ACTIVATION_NAME", "expr_83");
    // generator for Conv_349_fusion
    CNN_ConvolutionPoolAct_fp16("S362_Conv2d_64x64x1x1_Custom", &gen_ctrl_S362_Conv2d_64x64x1x1_Custom,
                                 64, 64, 10, 8,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S365_Conv2d_4x64x1x1;
    CNN_InitGenCtrl(&gen_ctrl_S365_Conv2d_4x64x1x1);
    CNN_SetGenCtrl(&gen_ctrl_S365_Conv2d_4x64x1x1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for Conv_352
    CNN_ConvolutionPoolAct_fp16("S365_Conv2d_4x64x1x1", &gen_ctrl_S365_Conv2d_4x64x1x1,
                                 64, 4, 10, 8,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S368_Conv2d_1x64x1x1_Sigmoid;
    CNN_InitGenCtrl(&gen_ctrl_S368_Conv2d_1x64x1x1_Sigmoid);
    CNN_SetGenCtrl(&gen_ctrl_S368_Conv2d_1x64x1x1_Sigmoid, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for Conv_353_fusion
    CNN_ConvolutionPoolAct_fp16("S368_Conv2d_1x64x1x1_Sigmoid", &gen_ctrl_S368_Conv2d_1x64x1x1_Sigmoid,
                                 64, 1, 10, 8,
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_SIGMOID);
    
    CNN_GenControl_T gen_ctrl_S371_Concat;
    CNN_InitGenCtrl(&gen_ctrl_S371_Concat);
    CNN_SetGenCtrl(&gen_ctrl_S371_Concat, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for Concat_381
    CNN_ConcatLastAxis_Generator("S371_Concat", &gen_ctrl_S371_Concat, 2, 6, 1280, 320, 80, 0, KOP_CONCAT);
    
    CNN_GenControl_T gen_ctrl_S372_Op_Transpose_382;
    CNN_InitGenCtrl(&gen_ctrl_S372_Op_Transpose_382);
    CNN_SetGenCtrl(&gen_ctrl_S372_Op_Transpose_382, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for Transpose_382 Transpose 6x1680 -> 1680x6 ((1, 0))
    CNN_MatTranspose("S372_Op_Transpose_382", &gen_ctrl_S372_Op_Transpose_382, 2,
                      1, 1680, 6, KOP_MATTRANSP);
    

#define GRAPH
#ifdef GRAPH
    CreateGraph("modelCNN",
        /* Arguments either passed or globals */
            CArgs(228,
                TCArgInfo("F16 * __restrict__", "Input_1", ARG_SCOPE_ARG_ALLOC, ARG_DIR_IN, AT_MEM_L2, AT_MEM_L2, 0),
                TCArgInfo("F16 * __restrict__", "Conv_0_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_0_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1138", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1138.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_3_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_3_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1141", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1141.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_6_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_6_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1144", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1144.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_9_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_9_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1147", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1147.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_12_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_12_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1150", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1150.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_15_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_15_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1153", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1153.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_18_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_18_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1156", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1156.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_21_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_21_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1159", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1159.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_26_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_26_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1162", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1162.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_29_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_29_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1165", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1165.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_32_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_32_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1168", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1168.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_35_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_35_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1171", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1171.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_38_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_38_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1174", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1174.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_41_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_41_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1177", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1177.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_44_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_44_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1180", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1180.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_47_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_47_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1183", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1183.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_51_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_51_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1186", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1186.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_54_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_54_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1189", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1189.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_57_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_57_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1192", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1192.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_61_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_61_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1195", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1195.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_64_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_64_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1198", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1198.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_67_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_67_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1201", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1201.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_72_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_72_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1204", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1204.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_75_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_75_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1207", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1207.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_78_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_78_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1210", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1210.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_81_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_81_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1213", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1213.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_84_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_84_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1216", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1216.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_87_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_87_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1219", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1219.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_90_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_90_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1222", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1222.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_93_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_93_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1225", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1225.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_97_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_97_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1228", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1228.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_100_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_100_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1231", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1231.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_103_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_103_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1234", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1234.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_107_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_107_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1237", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1237.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_110_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_110_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1240", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1240.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_113_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_113_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1243", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1243.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_118_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_118_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1246", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1246.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_121_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_121_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1249", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1249.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_124_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_124_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1252", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1252.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_127_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_127_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1255", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1255.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_134_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_134_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1258", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1258.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_137_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_137_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1261", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1261.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_140_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_140_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1264", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1264.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_143_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_143_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1267", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1267.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_146_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_146_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1270", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1270.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_149_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_149_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1273", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1273.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_153_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_153_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1276", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1276.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_156_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_156_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1279", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1279.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_162_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_162_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1282", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1282.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_165_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_165_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1285", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1285.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_168_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_168_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1288", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1288.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_171_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_171_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1291", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1291.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_174_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_174_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1294", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1294.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_178_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_178_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1297", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1297.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_181_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_181_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1300", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1300.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_187_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_187_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1303", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1303.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_190_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_190_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1306", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1306.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_193_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_193_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1309", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1309.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_196_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_196_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1312", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1312.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_199_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_199_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1315", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1315.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_203_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_203_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1318", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1318.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_206_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_206_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1321", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1321.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_209_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_209_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1324", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1324.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_213_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_213_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1327", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1327.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_216_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_216_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1330", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1330.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_219_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_219_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1333", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1333.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_222_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_222_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1336", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1336.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_225_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_225_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1339", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1339.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_229_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_229_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1342", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1342.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_232_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_232_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1345", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1345.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_235_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_235_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1348", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1348.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_239_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_239_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1351", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1351.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_242_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_242_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1354", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1354.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_245_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_245_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1357", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1357.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_248_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_248_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1360", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1360.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_251_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_251_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1363", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1363.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_255_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_255_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1366", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1366.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_258_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_258_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1369", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1369.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_261_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_261_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1372", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1372.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_264_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_264_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1375", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1375.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_267_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_267_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1378", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1378.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_270_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_270_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1381", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1381.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_273_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_273_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant_head_cls_preds_0_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant_head_cls_preds_0_bias.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_274_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_274_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1384", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1384.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_277_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_277_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1387", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1387.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_280_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_280_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1390", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1390.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_283_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_283_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1393", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1393.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_286_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_286_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant_head_reg_preds_0_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant_head_reg_preds_0_bias.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_287_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_287_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant_head_obj_preds_0_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant_head_obj_preds_0_bias.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_291_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_291_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1396", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1396.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_294_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_294_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1399", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1399.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_297_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_297_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1402", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1402.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_300_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_300_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1405", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1405.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_303_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_303_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1408", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1408.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_306_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_306_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant_head_cls_preds_1_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant_head_cls_preds_1_bias.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_307_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_307_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1411", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1411.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_310_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_310_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1414", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1414.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_313_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_313_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1417", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1417.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_316_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_316_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1420", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1420.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_319_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_319_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant_head_reg_preds_1_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant_head_reg_preds_1_bias.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_320_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_320_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant_head_obj_preds_1_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant_head_obj_preds_1_bias.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_324_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_324_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1423", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1423.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_327_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_327_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1426", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1426.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_330_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_330_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1429", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1429.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_333_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_333_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1432", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1432.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_336_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_336_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1435", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1435.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_339_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_339_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant_head_cls_preds_2_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant_head_cls_preds_2_bias.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_340_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_340_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1438", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1438.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_343_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_343_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1441", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1441.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_346_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_346_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1444", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1444.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_349_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_349_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant__1447", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant__1447.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_352_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_352_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant_head_reg_preds_2_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant_head_reg_preds_2_bias.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Conv_353_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Conv_353_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Constant_head_obj_preds_2_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./weights_tensors/Constant_head_obj_preds_2_bias.tensor", 1, 1, 16, 0)),
                TCArgInfo("F16 * __restrict__", "Output_1", ARG_SCOPE_ARG, ARG_DIR_OUT, AT_MEM_L2, AT_MEM_L2, 0)
            ),
        /* Locals, allocated dynamically */
        CArgs(105,
            TCArgInfo("F16 * __restrict__", "S3_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S6_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S9_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S12_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S18_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S21_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S24_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S26_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S29_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S32_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S35_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S38_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S44_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S47_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S50_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S51_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S54_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S57_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S60_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S61_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S64_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S67_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S70_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S72_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S78_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S81_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S84_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S90_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S93_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S96_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S97_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S100_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S103_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S106_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S107_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S110_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S113_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S116_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S118_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S124_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S127_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S134_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S137_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S140_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S146_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S149_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S153_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S156_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S161_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S164_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S170_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S173_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S177_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S180_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S185_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S188_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S194_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S197_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S201_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S204_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S207_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S211_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S214_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S220_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S223_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S227_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S230_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S233_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S237_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S240_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S246_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S249_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S253_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S256_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S259_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S262_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S265_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S268_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S271_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S277_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S280_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S283_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S286_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S293_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S297_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S300_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S303_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S306_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S309_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S315_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S318_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S321_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S324_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S331_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S335_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S338_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S341_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S344_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S347_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S353_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S356_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S359_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S362_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S369_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("F16 * __restrict__", "S371_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0)
        )
    );

    /* Stacked tensors - Concats */
    AddStackedTensors("S26_Output", 2, "S25_Output", "S15_Output");
    AddStackedTensors("S72_Output", 2, "S71_Output", "S41_Output");
    AddStackedTensors("S118_Output", 2, "S117_Output", "S87_Output");
    AddStackedTensors("S134_Output", 4, "S130_Output", "S133_Output", "S131_Output", "S132_Output");
    AddStackedTensors("S153_Output", 2, "S152_Output", "S143_Output");
    AddStackedTensors("S161_Output", 2, "S160_Output", "S121_Output");
    AddStackedTensors("S177_Output", 2, "S176_Output", "S167_Output");
    AddStackedTensors("S185_Output", 2, "S184_Output", "S75_Output");
    AddStackedTensors("S201_Output", 2, "S200_Output", "S191_Output");
    AddStackedTensors("S211_Output", 2, "S210_Output", "S183_Output");
    AddStackedTensors("S227_Output", 2, "S226_Output", "S217_Output");
    AddStackedTensors("S237_Output", 2, "S236_Output", "S159_Output");
    AddStackedTensors("S253_Output", 2, "S252_Output", "S243_Output");
    AddStackedTensors("S293_Output", 3, "S289_Output", "S292_Output", "S274_Output");
    AddStackedTensors("S331_Output", 3, "S327_Output", "S330_Output", "S312_Output");
    AddStackedTensors("S369_Output", 3, "S365_Output", "S368_Output", "S350_Output");

    // Node S3_Conv2d_16x12x3x3_Custom
    AddNode("S3_Conv2d_16x12x3x3_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "Input_1", 0),
            GNodeArg(GNA_IN, "Conv_0_weights", 0),
            GNodeArg(GNA_IN, "Constant__1138", 0),
            GNodeArg(GNA_OUT, "S3_Output", 0)
        )
    );
    // Node S6_Conv2d_16x1x3x3_Custom
    AddNode("S6_Conv2d_16x1x3x3_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S3_Output", 0),
            GNodeArg(GNA_IN, "Conv_3_weights", 0),
            GNodeArg(GNA_IN, "Constant__1141", 0),
            GNodeArg(GNA_OUT, "S6_Output", 0)
        )
    );
    // Node S9_Conv2d_32x16x1x1_Custom
    AddNode("S9_Conv2d_32x16x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S6_Output", 0),
            GNodeArg(GNA_IN, "Conv_6_weights", 0),
            GNodeArg(GNA_IN, "Constant__1144", 0),
            GNodeArg(GNA_OUT, "S9_Output", 0)
        )
    );
    // Node S12_Conv2d_16x32x1x1_Custom
    AddNode("S12_Conv2d_16x32x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S9_Output", 0),
            GNodeArg(GNA_IN, "Conv_9_weights", 0),
            GNodeArg(GNA_IN, "Constant__1147", 0),
            GNodeArg(GNA_OUT, "S12_Output", 0)
        )
    );
    // Node S15_Conv2d_16x32x1x1_Custom
    AddNode("S15_Conv2d_16x32x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S9_Output", 0),
            GNodeArg(GNA_IN, "Conv_12_weights", 0),
            GNodeArg(GNA_IN, "Constant__1150", 0),
            GNodeArg(GNA_OUT, "S15_Output", 0)
        )
    );
    // Node S18_Conv2d_16x16x1x1_Custom
    AddNode("S18_Conv2d_16x16x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S12_Output", 0),
            GNodeArg(GNA_IN, "Conv_15_weights", 0),
            GNodeArg(GNA_IN, "Constant__1153", 0),
            GNodeArg(GNA_OUT, "S18_Output", 0)
        )
    );
    // Node S21_Conv2d_16x1x3x3_Custom
    AddNode("S21_Conv2d_16x1x3x3_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S18_Output", 0),
            GNodeArg(GNA_IN, "Conv_18_weights", 0),
            GNodeArg(GNA_IN, "Constant__1156", 0),
            GNodeArg(GNA_OUT, "S21_Output", 0)
        )
    );
    // Node S24_Conv2d_16x16x1x1
    AddNode("S24_Conv2d_16x16x1x1",
        Bindings(4,
            GNodeArg(GNA_IN, "S21_Output", 0),
            GNodeArg(GNA_IN, "Conv_21_weights", 0),
            GNodeArg(GNA_IN, "Constant__1159", 0),
            GNodeArg(GNA_OUT, "S24_Output", 0)
        )
    );
    // Node expr_0 in_qs [f16,f16] out_qs [f16]
    AddNode("S25_Op_expr_0",
        Bindings(3,
            GNodeArg(GNA_IN, "S24_Output", 0),
            GNodeArg(GNA_IN, "S12_Output", 0),
            GNodeArg(GNA_OUT, "S25_Output", 0)
        )
    );
    // Node S29_Conv2d_32x32x1x1_Custom
    AddNode("S29_Conv2d_32x32x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S26_Output", 0),
            GNodeArg(GNA_IN, "Conv_26_weights", 0),
            GNodeArg(GNA_IN, "Constant__1162", 0),
            GNodeArg(GNA_OUT, "S29_Output", 0)
        )
    );
    // Node S32_Conv2d_32x1x3x3_Custom
    AddNode("S32_Conv2d_32x1x3x3_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S29_Output", 0),
            GNodeArg(GNA_IN, "Conv_29_weights", 0),
            GNodeArg(GNA_IN, "Constant__1165", 0),
            GNodeArg(GNA_OUT, "S32_Output", 0)
        )
    );
    // Node S35_Conv2d_64x32x1x1_Custom
    AddNode("S35_Conv2d_64x32x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S32_Output", 0),
            GNodeArg(GNA_IN, "Conv_32_weights", 0),
            GNodeArg(GNA_IN, "Constant__1168", 0),
            GNodeArg(GNA_OUT, "S35_Output", 0)
        )
    );
    // Node S38_Conv2d_32x64x1x1_Custom
    AddNode("S38_Conv2d_32x64x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S35_Output", 0),
            GNodeArg(GNA_IN, "Conv_35_weights", 0),
            GNodeArg(GNA_IN, "Constant__1171", 0),
            GNodeArg(GNA_OUT, "S38_Output", 0)
        )
    );
    // Node S41_Conv2d_32x64x1x1_Custom
    AddNode("S41_Conv2d_32x64x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S35_Output", 0),
            GNodeArg(GNA_IN, "Conv_38_weights", 0),
            GNodeArg(GNA_IN, "Constant__1174", 0),
            GNodeArg(GNA_OUT, "S41_Output", 0)
        )
    );
    // Node S44_Conv2d_32x32x1x1_Custom
    AddNode("S44_Conv2d_32x32x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S38_Output", 0),
            GNodeArg(GNA_IN, "Conv_41_weights", 0),
            GNodeArg(GNA_IN, "Constant__1177", 0),
            GNodeArg(GNA_OUT, "S44_Output", 0)
        )
    );
    // Node S47_Conv2d_32x1x3x3_Custom
    AddNode("S47_Conv2d_32x1x3x3_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S44_Output", 0),
            GNodeArg(GNA_IN, "Conv_44_weights", 0),
            GNodeArg(GNA_IN, "Constant__1180", 0),
            GNodeArg(GNA_OUT, "S47_Output", 0)
        )
    );
    // Node S50_Conv2d_32x32x1x1
    AddNode("S50_Conv2d_32x32x1x1",
        Bindings(4,
            GNodeArg(GNA_IN, "S47_Output", 0),
            GNodeArg(GNA_IN, "Conv_47_weights", 0),
            GNodeArg(GNA_IN, "Constant__1183", 0),
            GNodeArg(GNA_OUT, "S50_Output", 0)
        )
    );
    // Node expr_1 in_qs [f16,f16] out_qs [f16]
    AddNode("S51_Op_expr_1",
        Bindings(3,
            GNodeArg(GNA_IN, "S50_Output", 0),
            GNodeArg(GNA_IN, "S38_Output", 0),
            GNodeArg(GNA_OUT, "S51_Output", 0)
        )
    );
    // Node S54_Conv2d_32x32x1x1_Custom
    AddNode("S54_Conv2d_32x32x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S51_Output", 0),
            GNodeArg(GNA_IN, "Conv_51_weights", 0),
            GNodeArg(GNA_IN, "Constant__1186", 0),
            GNodeArg(GNA_OUT, "S54_Output", 0)
        )
    );
    // Node S57_Conv2d_32x1x3x3_Custom
    AddNode("S57_Conv2d_32x1x3x3_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S54_Output", 0),
            GNodeArg(GNA_IN, "Conv_54_weights", 0),
            GNodeArg(GNA_IN, "Constant__1189", 0),
            GNodeArg(GNA_OUT, "S57_Output", 0)
        )
    );
    // Node S60_Conv2d_32x32x1x1_Custom
    AddNode("S60_Conv2d_32x32x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S57_Output", 0),
            GNodeArg(GNA_IN, "Conv_57_weights", 0),
            GNodeArg(GNA_IN, "Constant__1192", 0),
            GNodeArg(GNA_OUT, "S60_Output", 0)
        )
    );
    // Node S61_MatAdd_32x32x40 in1q f16 in2q f16 outq f16
    AddNode("S61_MatAdd_32x32x40",
        Bindings(3,
            GNodeArg(GNA_IN, "S60_Output", 0),
            GNodeArg(GNA_IN, "S51_Output", 0),
            GNodeArg(GNA_OUT, "S61_Output", 0)
        )
    );
    // Node S64_Conv2d_32x32x1x1_Custom
    AddNode("S64_Conv2d_32x32x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S61_Output", 0),
            GNodeArg(GNA_IN, "Conv_61_weights", 0),
            GNodeArg(GNA_IN, "Constant__1195", 0),
            GNodeArg(GNA_OUT, "S64_Output", 0)
        )
    );
    // Node S67_Conv2d_32x1x3x3_Custom
    AddNode("S67_Conv2d_32x1x3x3_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S64_Output", 0),
            GNodeArg(GNA_IN, "Conv_64_weights", 0),
            GNodeArg(GNA_IN, "Constant__1198", 0),
            GNodeArg(GNA_OUT, "S67_Output", 0)
        )
    );
    // Node S70_Conv2d_32x32x1x1_Custom
    AddNode("S70_Conv2d_32x32x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S67_Output", 0),
            GNodeArg(GNA_IN, "Conv_67_weights", 0),
            GNodeArg(GNA_IN, "Constant__1201", 0),
            GNodeArg(GNA_OUT, "S70_Output", 0)
        )
    );
    // Node S71_MatAdd_32x32x40 in1q f16 in2q f16 outq f16
    AddNode("S71_MatAdd_32x32x40",
        Bindings(3,
            GNodeArg(GNA_IN, "S70_Output", 0),
            GNodeArg(GNA_IN, "S61_Output", 0),
            GNodeArg(GNA_OUT, "S71_Output", 0)
        )
    );
    // Node S75_Conv2d_64x64x1x1_Custom
    AddNode("S75_Conv2d_64x64x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S72_Output", 0),
            GNodeArg(GNA_IN, "Conv_72_weights", 0),
            GNodeArg(GNA_IN, "Constant__1204", 0),
            GNodeArg(GNA_OUT, "S75_Output", 0)
        )
    );
    // Node S78_Conv2d_64x1x3x3_Custom
    AddNode("S78_Conv2d_64x1x3x3_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S75_Output", 0),
            GNodeArg(GNA_IN, "Conv_75_weights", 0),
            GNodeArg(GNA_IN, "Constant__1207", 0),
            GNodeArg(GNA_OUT, "S78_Output", 0)
        )
    );
    // Node S81_Conv2d_128x64x1x1_Custom
    AddNode("S81_Conv2d_128x64x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S78_Output", 0),
            GNodeArg(GNA_IN, "Conv_78_weights", 0),
            GNodeArg(GNA_IN, "Constant__1210", 0),
            GNodeArg(GNA_OUT, "S81_Output", 0)
        )
    );
    // Node S84_Conv2d_64x128x1x1_Custom
    AddNode("S84_Conv2d_64x128x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S81_Output", 0),
            GNodeArg(GNA_IN, "Conv_81_weights", 0),
            GNodeArg(GNA_IN, "Constant__1213", 0),
            GNodeArg(GNA_OUT, "S84_Output", 0)
        )
    );
    // Node S87_Conv2d_64x128x1x1_Custom
    AddNode("S87_Conv2d_64x128x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S81_Output", 0),
            GNodeArg(GNA_IN, "Conv_84_weights", 0),
            GNodeArg(GNA_IN, "Constant__1216", 0),
            GNodeArg(GNA_OUT, "S87_Output", 0)
        )
    );
    // Node S90_Conv2d_64x64x1x1_Custom
    AddNode("S90_Conv2d_64x64x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S84_Output", 0),
            GNodeArg(GNA_IN, "Conv_87_weights", 0),
            GNodeArg(GNA_IN, "Constant__1219", 0),
            GNodeArg(GNA_OUT, "S90_Output", 0)
        )
    );
    // Node S93_Conv2d_64x1x3x3_Custom
    AddNode("S93_Conv2d_64x1x3x3_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S90_Output", 0),
            GNodeArg(GNA_IN, "Conv_90_weights", 0),
            GNodeArg(GNA_IN, "Constant__1222", 0),
            GNodeArg(GNA_OUT, "S93_Output", 0)
        )
    );
    // Node S96_Conv2d_64x64x1x1
    AddNode("S96_Conv2d_64x64x1x1",
        Bindings(4,
            GNodeArg(GNA_IN, "S93_Output", 0),
            GNodeArg(GNA_IN, "Conv_93_weights", 0),
            GNodeArg(GNA_IN, "Constant__1225", 0),
            GNodeArg(GNA_OUT, "S96_Output", 0)
        )
    );
    // Node expr_2 in_qs [f16,f16] out_qs [f16]
    AddNode("S97_Op_expr_2",
        Bindings(3,
            GNodeArg(GNA_IN, "S96_Output", 0),
            GNodeArg(GNA_IN, "S84_Output", 0),
            GNodeArg(GNA_OUT, "S97_Output", 0)
        )
    );
    // Node S100_Conv2d_64x64x1x1_Custom
    AddNode("S100_Conv2d_64x64x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S97_Output", 0),
            GNodeArg(GNA_IN, "Conv_97_weights", 0),
            GNodeArg(GNA_IN, "Constant__1228", 0),
            GNodeArg(GNA_OUT, "S100_Output", 0)
        )
    );
    // Node S103_Conv2d_64x1x3x3_Custom
    AddNode("S103_Conv2d_64x1x3x3_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S100_Output", 0),
            GNodeArg(GNA_IN, "Conv_100_weights", 0),
            GNodeArg(GNA_IN, "Constant__1231", 0),
            GNodeArg(GNA_OUT, "S103_Output", 0)
        )
    );
    // Node S106_Conv2d_64x64x1x1_Custom
    AddNode("S106_Conv2d_64x64x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S103_Output", 0),
            GNodeArg(GNA_IN, "Conv_103_weights", 0),
            GNodeArg(GNA_IN, "Constant__1234", 0),
            GNodeArg(GNA_OUT, "S106_Output", 0)
        )
    );
    // Node S107_MatAdd_64x16x20 in1q f16 in2q f16 outq f16
    AddNode("S107_MatAdd_64x16x20",
        Bindings(3,
            GNodeArg(GNA_IN, "S106_Output", 0),
            GNodeArg(GNA_IN, "S97_Output", 0),
            GNodeArg(GNA_OUT, "S107_Output", 0)
        )
    );
    // Node S110_Conv2d_64x64x1x1_Custom
    AddNode("S110_Conv2d_64x64x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S107_Output", 0),
            GNodeArg(GNA_IN, "Conv_107_weights", 0),
            GNodeArg(GNA_IN, "Constant__1237", 0),
            GNodeArg(GNA_OUT, "S110_Output", 0)
        )
    );
    // Node S113_Conv2d_64x1x3x3_Custom
    AddNode("S113_Conv2d_64x1x3x3_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S110_Output", 0),
            GNodeArg(GNA_IN, "Conv_110_weights", 0),
            GNodeArg(GNA_IN, "Constant__1240", 0),
            GNodeArg(GNA_OUT, "S113_Output", 0)
        )
    );
    // Node S116_Conv2d_64x64x1x1_Custom
    AddNode("S116_Conv2d_64x64x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S113_Output", 0),
            GNodeArg(GNA_IN, "Conv_113_weights", 0),
            GNodeArg(GNA_IN, "Constant__1243", 0),
            GNodeArg(GNA_OUT, "S116_Output", 0)
        )
    );
    // Node S117_MatAdd_64x16x20 in1q f16 in2q f16 outq f16
    AddNode("S117_MatAdd_64x16x20",
        Bindings(3,
            GNodeArg(GNA_IN, "S116_Output", 0),
            GNodeArg(GNA_IN, "S107_Output", 0),
            GNodeArg(GNA_OUT, "S117_Output", 0)
        )
    );
    // Node S121_Conv2d_128x128x1x1_Custom
    AddNode("S121_Conv2d_128x128x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S118_Output", 0),
            GNodeArg(GNA_IN, "Conv_118_weights", 0),
            GNodeArg(GNA_IN, "Constant__1246", 0),
            GNodeArg(GNA_OUT, "S121_Output", 0)
        )
    );
    // Node S124_Conv2d_128x1x3x3_Custom
    AddNode("S124_Conv2d_128x1x3x3_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S121_Output", 0),
            GNodeArg(GNA_IN, "Conv_121_weights", 0),
            GNodeArg(GNA_IN, "Constant__1249", 0),
            GNodeArg(GNA_OUT, "S124_Output", 0)
        )
    );
    // Node S127_Conv2d_256x128x1x1_Custom
    AddNode("S127_Conv2d_256x128x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S124_Output", 0),
            GNodeArg(GNA_IN, "Conv_124_weights", 0),
            GNodeArg(GNA_IN, "Constant__1252", 0),
            GNodeArg(GNA_OUT, "S127_Output", 0)
        )
    );
    // Node S130_Conv2d_128x256x1x1_Custom
    AddNode("S130_Conv2d_128x256x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S127_Output", 0),
            GNodeArg(GNA_IN, "Conv_127_weights", 0),
            GNodeArg(GNA_IN, "Constant__1255", 0),
            GNodeArg(GNA_OUT, "S130_Output", 0)
        )
    );
    // Node MaxPool_131 inq f16 outq f16
    AddNode("S131_MaxPool_9x9",
        Bindings(2,
            GNodeArg(GNA_IN, "S130_Output", 0),
            GNodeArg(GNA_OUT, "S131_Output", 0)
        )
    );
    // Node MaxPool_132 inq f16 outq f16
    AddNode("S132_MaxPool_13x13",
        Bindings(2,
            GNodeArg(GNA_IN, "S130_Output", 0),
            GNodeArg(GNA_OUT, "S132_Output", 0)
        )
    );
    // Node MaxPool_130 inq f16 outq f16
    AddNode("S133_MaxPool_5x5",
        Bindings(2,
            GNodeArg(GNA_IN, "S130_Output", 0),
            GNodeArg(GNA_OUT, "S133_Output", 0)
        )
    );
    // Node S137_Conv2d_256x512x1x1_Custom
    AddNode("S137_Conv2d_256x512x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S134_Output", 0),
            GNodeArg(GNA_IN, "Conv_134_weights", 0),
            GNodeArg(GNA_IN, "Constant__1258", 0),
            GNodeArg(GNA_OUT, "S137_Output", 0)
        )
    );
    // Node S140_Conv2d_128x256x1x1_Custom
    AddNode("S140_Conv2d_128x256x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S137_Output", 0),
            GNodeArg(GNA_IN, "Conv_137_weights", 0),
            GNodeArg(GNA_IN, "Constant__1261", 0),
            GNodeArg(GNA_OUT, "S140_Output", 0)
        )
    );
    // Node S143_Conv2d_128x256x1x1_Custom
    AddNode("S143_Conv2d_128x256x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S137_Output", 0),
            GNodeArg(GNA_IN, "Conv_140_weights", 0),
            GNodeArg(GNA_IN, "Constant__1264", 0),
            GNodeArg(GNA_OUT, "S143_Output", 0)
        )
    );
    // Node S146_Conv2d_128x128x1x1_Custom
    AddNode("S146_Conv2d_128x128x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S140_Output", 0),
            GNodeArg(GNA_IN, "Conv_143_weights", 0),
            GNodeArg(GNA_IN, "Constant__1267", 0),
            GNodeArg(GNA_OUT, "S146_Output", 0)
        )
    );
    // Node S149_Conv2d_128x1x3x3_Custom
    AddNode("S149_Conv2d_128x1x3x3_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S146_Output", 0),
            GNodeArg(GNA_IN, "Conv_146_weights", 0),
            GNodeArg(GNA_IN, "Constant__1270", 0),
            GNodeArg(GNA_OUT, "S149_Output", 0)
        )
    );
    // Node S152_Conv2d_128x128x1x1_Custom
    AddNode("S152_Conv2d_128x128x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S149_Output", 0),
            GNodeArg(GNA_IN, "Conv_149_weights", 0),
            GNodeArg(GNA_IN, "Constant__1273", 0),
            GNodeArg(GNA_OUT, "S152_Output", 0)
        )
    );
    // Node S156_Conv2d_256x256x1x1_Custom
    AddNode("S156_Conv2d_256x256x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S153_Output", 0),
            GNodeArg(GNA_IN, "Conv_153_weights", 0),
            GNodeArg(GNA_IN, "Constant__1276", 0),
            GNodeArg(GNA_OUT, "S156_Output", 0)
        )
    );
    // Node S159_Conv2d_128x256x1x1_Custom
    AddNode("S159_Conv2d_128x256x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S156_Output", 0),
            GNodeArg(GNA_IN, "Conv_156_weights", 0),
            GNodeArg(GNA_IN, "Constant__1279", 0),
            GNodeArg(GNA_OUT, "S159_Output", 0)
        )
    );
    // Node Resize_160 inq f16 outq f16
    AddNode("S160_Op_Resize_160",
        Bindings(2,
            GNodeArg(GNA_IN, "S159_Output", 0),
            GNodeArg(GNA_OUT, "S160_Output", 0)
        )
    );
    // Node S164_Conv2d_64x256x1x1_Custom
    AddNode("S164_Conv2d_64x256x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S161_Output", 0),
            GNodeArg(GNA_IN, "Conv_162_weights", 0),
            GNodeArg(GNA_IN, "Constant__1282", 0),
            GNodeArg(GNA_OUT, "S164_Output", 0)
        )
    );
    // Node S167_Conv2d_64x256x1x1_Custom
    AddNode("S167_Conv2d_64x256x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S161_Output", 0),
            GNodeArg(GNA_IN, "Conv_165_weights", 0),
            GNodeArg(GNA_IN, "Constant__1285", 0),
            GNodeArg(GNA_OUT, "S167_Output", 0)
        )
    );
    // Node S170_Conv2d_64x64x1x1_Custom
    AddNode("S170_Conv2d_64x64x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S164_Output", 0),
            GNodeArg(GNA_IN, "Conv_168_weights", 0),
            GNodeArg(GNA_IN, "Constant__1288", 0),
            GNodeArg(GNA_OUT, "S170_Output", 0)
        )
    );
    // Node S173_Conv2d_64x1x3x3_Custom
    AddNode("S173_Conv2d_64x1x3x3_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S170_Output", 0),
            GNodeArg(GNA_IN, "Conv_171_weights", 0),
            GNodeArg(GNA_IN, "Constant__1291", 0),
            GNodeArg(GNA_OUT, "S173_Output", 0)
        )
    );
    // Node S176_Conv2d_64x64x1x1_Custom
    AddNode("S176_Conv2d_64x64x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S173_Output", 0),
            GNodeArg(GNA_IN, "Conv_174_weights", 0),
            GNodeArg(GNA_IN, "Constant__1294", 0),
            GNodeArg(GNA_OUT, "S176_Output", 0)
        )
    );
    // Node S180_Conv2d_128x128x1x1_Custom
    AddNode("S180_Conv2d_128x128x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S177_Output", 0),
            GNodeArg(GNA_IN, "Conv_178_weights", 0),
            GNodeArg(GNA_IN, "Constant__1297", 0),
            GNodeArg(GNA_OUT, "S180_Output", 0)
        )
    );
    // Node S183_Conv2d_64x128x1x1_Custom
    AddNode("S183_Conv2d_64x128x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S180_Output", 0),
            GNodeArg(GNA_IN, "Conv_181_weights", 0),
            GNodeArg(GNA_IN, "Constant__1300", 0),
            GNodeArg(GNA_OUT, "S183_Output", 0)
        )
    );
    // Node Resize_185 inq f16 outq f16
    AddNode("S184_Op_Resize_185",
        Bindings(2,
            GNodeArg(GNA_IN, "S183_Output", 0),
            GNodeArg(GNA_OUT, "S184_Output", 0)
        )
    );
    // Node S188_Conv2d_32x128x1x1_Custom
    AddNode("S188_Conv2d_32x128x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S185_Output", 0),
            GNodeArg(GNA_IN, "Conv_187_weights", 0),
            GNodeArg(GNA_IN, "Constant__1303", 0),
            GNodeArg(GNA_OUT, "S188_Output", 0)
        )
    );
    // Node S191_Conv2d_32x128x1x1_Custom
    AddNode("S191_Conv2d_32x128x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S185_Output", 0),
            GNodeArg(GNA_IN, "Conv_190_weights", 0),
            GNodeArg(GNA_IN, "Constant__1306", 0),
            GNodeArg(GNA_OUT, "S191_Output", 0)
        )
    );
    // Node S194_Conv2d_32x32x1x1_Custom
    AddNode("S194_Conv2d_32x32x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S188_Output", 0),
            GNodeArg(GNA_IN, "Conv_193_weights", 0),
            GNodeArg(GNA_IN, "Constant__1309", 0),
            GNodeArg(GNA_OUT, "S194_Output", 0)
        )
    );
    // Node S197_Conv2d_32x1x3x3_Custom
    AddNode("S197_Conv2d_32x1x3x3_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S194_Output", 0),
            GNodeArg(GNA_IN, "Conv_196_weights", 0),
            GNodeArg(GNA_IN, "Constant__1312", 0),
            GNodeArg(GNA_OUT, "S197_Output", 0)
        )
    );
    // Node S200_Conv2d_32x32x1x1_Custom
    AddNode("S200_Conv2d_32x32x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S197_Output", 0),
            GNodeArg(GNA_IN, "Conv_199_weights", 0),
            GNodeArg(GNA_IN, "Constant__1315", 0),
            GNodeArg(GNA_OUT, "S200_Output", 0)
        )
    );
    // Node S204_Conv2d_64x64x1x1_Custom
    AddNode("S204_Conv2d_64x64x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S201_Output", 0),
            GNodeArg(GNA_IN, "Conv_203_weights", 0),
            GNodeArg(GNA_IN, "Constant__1318", 0),
            GNodeArg(GNA_OUT, "S204_Output", 0)
        )
    );
    // Node S207_Conv2d_64x1x3x3_Custom
    AddNode("S207_Conv2d_64x1x3x3_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S204_Output", 0),
            GNodeArg(GNA_IN, "Conv_206_weights", 0),
            GNodeArg(GNA_IN, "Constant__1321", 0),
            GNodeArg(GNA_OUT, "S207_Output", 0)
        )
    );
    // Node S210_Conv2d_64x64x1x1_Custom
    AddNode("S210_Conv2d_64x64x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S207_Output", 0),
            GNodeArg(GNA_IN, "Conv_209_weights", 0),
            GNodeArg(GNA_IN, "Constant__1324", 0),
            GNodeArg(GNA_OUT, "S210_Output", 0)
        )
    );
    // Node S214_Conv2d_64x128x1x1_Custom
    AddNode("S214_Conv2d_64x128x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S211_Output", 0),
            GNodeArg(GNA_IN, "Conv_213_weights", 0),
            GNodeArg(GNA_IN, "Constant__1327", 0),
            GNodeArg(GNA_OUT, "S214_Output", 0)
        )
    );
    // Node S217_Conv2d_64x128x1x1_Custom
    AddNode("S217_Conv2d_64x128x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S211_Output", 0),
            GNodeArg(GNA_IN, "Conv_216_weights", 0),
            GNodeArg(GNA_IN, "Constant__1330", 0),
            GNodeArg(GNA_OUT, "S217_Output", 0)
        )
    );
    // Node S220_Conv2d_64x64x1x1_Custom
    AddNode("S220_Conv2d_64x64x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S214_Output", 0),
            GNodeArg(GNA_IN, "Conv_219_weights", 0),
            GNodeArg(GNA_IN, "Constant__1333", 0),
            GNodeArg(GNA_OUT, "S220_Output", 0)
        )
    );
    // Node S223_Conv2d_64x1x3x3_Custom
    AddNode("S223_Conv2d_64x1x3x3_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S220_Output", 0),
            GNodeArg(GNA_IN, "Conv_222_weights", 0),
            GNodeArg(GNA_IN, "Constant__1336", 0),
            GNodeArg(GNA_OUT, "S223_Output", 0)
        )
    );
    // Node S226_Conv2d_64x64x1x1_Custom
    AddNode("S226_Conv2d_64x64x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S223_Output", 0),
            GNodeArg(GNA_IN, "Conv_225_weights", 0),
            GNodeArg(GNA_IN, "Constant__1339", 0),
            GNodeArg(GNA_OUT, "S226_Output", 0)
        )
    );
    // Node S230_Conv2d_128x128x1x1_Custom
    AddNode("S230_Conv2d_128x128x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S227_Output", 0),
            GNodeArg(GNA_IN, "Conv_229_weights", 0),
            GNodeArg(GNA_IN, "Constant__1342", 0),
            GNodeArg(GNA_OUT, "S230_Output", 0)
        )
    );
    // Node S233_Conv2d_128x1x3x3_Custom
    AddNode("S233_Conv2d_128x1x3x3_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S230_Output", 0),
            GNodeArg(GNA_IN, "Conv_232_weights", 0),
            GNodeArg(GNA_IN, "Constant__1345", 0),
            GNodeArg(GNA_OUT, "S233_Output", 0)
        )
    );
    // Node S236_Conv2d_128x128x1x1_Custom
    AddNode("S236_Conv2d_128x128x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S233_Output", 0),
            GNodeArg(GNA_IN, "Conv_235_weights", 0),
            GNodeArg(GNA_IN, "Constant__1348", 0),
            GNodeArg(GNA_OUT, "S236_Output", 0)
        )
    );
    // Node S240_Conv2d_128x256x1x1_Custom
    AddNode("S240_Conv2d_128x256x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S237_Output", 0),
            GNodeArg(GNA_IN, "Conv_239_weights", 0),
            GNodeArg(GNA_IN, "Constant__1351", 0),
            GNodeArg(GNA_OUT, "S240_Output", 0)
        )
    );
    // Node S243_Conv2d_128x256x1x1_Custom
    AddNode("S243_Conv2d_128x256x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S237_Output", 0),
            GNodeArg(GNA_IN, "Conv_242_weights", 0),
            GNodeArg(GNA_IN, "Constant__1354", 0),
            GNodeArg(GNA_OUT, "S243_Output", 0)
        )
    );
    // Node S246_Conv2d_128x128x1x1_Custom
    AddNode("S246_Conv2d_128x128x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S240_Output", 0),
            GNodeArg(GNA_IN, "Conv_245_weights", 0),
            GNodeArg(GNA_IN, "Constant__1357", 0),
            GNodeArg(GNA_OUT, "S246_Output", 0)
        )
    );
    // Node S249_Conv2d_128x1x3x3_Custom
    AddNode("S249_Conv2d_128x1x3x3_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S246_Output", 0),
            GNodeArg(GNA_IN, "Conv_248_weights", 0),
            GNodeArg(GNA_IN, "Constant__1360", 0),
            GNodeArg(GNA_OUT, "S249_Output", 0)
        )
    );
    // Node S252_Conv2d_128x128x1x1_Custom
    AddNode("S252_Conv2d_128x128x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S249_Output", 0),
            GNodeArg(GNA_IN, "Conv_251_weights", 0),
            GNodeArg(GNA_IN, "Constant__1363", 0),
            GNodeArg(GNA_OUT, "S252_Output", 0)
        )
    );
    // Node S256_Conv2d_256x256x1x1_Custom
    AddNode("S256_Conv2d_256x256x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S253_Output", 0),
            GNodeArg(GNA_IN, "Conv_255_weights", 0),
            GNodeArg(GNA_IN, "Constant__1366", 0),
            GNodeArg(GNA_OUT, "S256_Output", 0)
        )
    );
    // Node S259_Conv2d_64x64x1x1_Custom
    AddNode("S259_Conv2d_64x64x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S204_Output", 0),
            GNodeArg(GNA_IN, "Conv_258_weights", 0),
            GNodeArg(GNA_IN, "Constant__1369", 0),
            GNodeArg(GNA_OUT, "S259_Output", 0)
        )
    );
    // Node S262_Conv2d_64x1x3x3_Custom
    AddNode("S262_Conv2d_64x1x3x3_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S259_Output", 0),
            GNodeArg(GNA_IN, "Conv_261_weights", 0),
            GNodeArg(GNA_IN, "Constant__1372", 0),
            GNodeArg(GNA_OUT, "S262_Output", 0)
        )
    );
    // Node S265_Conv2d_64x64x1x1_Custom
    AddNode("S265_Conv2d_64x64x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S262_Output", 0),
            GNodeArg(GNA_IN, "Conv_264_weights", 0),
            GNodeArg(GNA_IN, "Constant__1375", 0),
            GNodeArg(GNA_OUT, "S265_Output", 0)
        )
    );
    // Node S268_Conv2d_64x1x3x3_Custom
    AddNode("S268_Conv2d_64x1x3x3_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S265_Output", 0),
            GNodeArg(GNA_IN, "Conv_267_weights", 0),
            GNodeArg(GNA_IN, "Constant__1378", 0),
            GNodeArg(GNA_OUT, "S268_Output", 0)
        )
    );
    // Node S271_Conv2d_64x64x1x1_Custom
    AddNode("S271_Conv2d_64x64x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S268_Output", 0),
            GNodeArg(GNA_IN, "Conv_270_weights", 0),
            GNodeArg(GNA_IN, "Constant__1381", 0),
            GNodeArg(GNA_OUT, "S271_Output", 0)
        )
    );
    // Node S274_Conv2d_1x64x1x1_Sigmoid
    AddNode("S274_Conv2d_1x64x1x1_Sigmoid",
        Bindings(4,
            GNodeArg(GNA_IN, "S271_Output", 0),
            GNodeArg(GNA_IN, "Conv_273_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_cls_preds_0_bias", 0),
            GNodeArg(GNA_OUT, "S274_Output", 0)
        )
    );
    // Node S277_Conv2d_64x1x3x3_Custom
    AddNode("S277_Conv2d_64x1x3x3_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S259_Output", 0),
            GNodeArg(GNA_IN, "Conv_274_weights", 0),
            GNodeArg(GNA_IN, "Constant__1384", 0),
            GNodeArg(GNA_OUT, "S277_Output", 0)
        )
    );
    // Node S280_Conv2d_64x64x1x1_Custom
    AddNode("S280_Conv2d_64x64x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S277_Output", 0),
            GNodeArg(GNA_IN, "Conv_277_weights", 0),
            GNodeArg(GNA_IN, "Constant__1387", 0),
            GNodeArg(GNA_OUT, "S280_Output", 0)
        )
    );
    // Node S283_Conv2d_64x1x3x3_Custom
    AddNode("S283_Conv2d_64x1x3x3_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S280_Output", 0),
            GNodeArg(GNA_IN, "Conv_280_weights", 0),
            GNodeArg(GNA_IN, "Constant__1390", 0),
            GNodeArg(GNA_OUT, "S283_Output", 0)
        )
    );
    // Node S286_Conv2d_64x64x1x1_Custom
    AddNode("S286_Conv2d_64x64x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S283_Output", 0),
            GNodeArg(GNA_IN, "Conv_283_weights", 0),
            GNodeArg(GNA_IN, "Constant__1393", 0),
            GNodeArg(GNA_OUT, "S286_Output", 0)
        )
    );
    // Node S289_Conv2d_4x64x1x1
    AddNode("S289_Conv2d_4x64x1x1",
        Bindings(4,
            GNodeArg(GNA_IN, "S286_Output", 0),
            GNodeArg(GNA_IN, "Conv_286_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_reg_preds_0_bias", 0),
            GNodeArg(GNA_OUT, "S289_Output", 0)
        )
    );
    // Node S292_Conv2d_1x64x1x1_Sigmoid
    AddNode("S292_Conv2d_1x64x1x1_Sigmoid",
        Bindings(4,
            GNodeArg(GNA_IN, "S286_Output", 0),
            GNodeArg(GNA_IN, "Conv_287_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_obj_preds_0_bias", 0),
            GNodeArg(GNA_OUT, "S292_Output", 0)
        )
    );
    // Node S297_Conv2d_64x128x1x1_Custom
    AddNode("S297_Conv2d_64x128x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S230_Output", 0),
            GNodeArg(GNA_IN, "Conv_291_weights", 0),
            GNodeArg(GNA_IN, "Constant__1396", 0),
            GNodeArg(GNA_OUT, "S297_Output", 0)
        )
    );
    // Node S300_Conv2d_64x1x3x3_Custom
    AddNode("S300_Conv2d_64x1x3x3_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S297_Output", 0),
            GNodeArg(GNA_IN, "Conv_294_weights", 0),
            GNodeArg(GNA_IN, "Constant__1399", 0),
            GNodeArg(GNA_OUT, "S300_Output", 0)
        )
    );
    // Node S303_Conv2d_64x64x1x1_Custom
    AddNode("S303_Conv2d_64x64x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S300_Output", 0),
            GNodeArg(GNA_IN, "Conv_297_weights", 0),
            GNodeArg(GNA_IN, "Constant__1402", 0),
            GNodeArg(GNA_OUT, "S303_Output", 0)
        )
    );
    // Node S306_Conv2d_64x1x3x3_Custom
    AddNode("S306_Conv2d_64x1x3x3_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S303_Output", 0),
            GNodeArg(GNA_IN, "Conv_300_weights", 0),
            GNodeArg(GNA_IN, "Constant__1405", 0),
            GNodeArg(GNA_OUT, "S306_Output", 0)
        )
    );
    // Node S309_Conv2d_64x64x1x1_Custom
    AddNode("S309_Conv2d_64x64x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S306_Output", 0),
            GNodeArg(GNA_IN, "Conv_303_weights", 0),
            GNodeArg(GNA_IN, "Constant__1408", 0),
            GNodeArg(GNA_OUT, "S309_Output", 0)
        )
    );
    // Node S312_Conv2d_1x64x1x1_Sigmoid
    AddNode("S312_Conv2d_1x64x1x1_Sigmoid",
        Bindings(4,
            GNodeArg(GNA_IN, "S309_Output", 0),
            GNodeArg(GNA_IN, "Conv_306_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_cls_preds_1_bias", 0),
            GNodeArg(GNA_OUT, "S312_Output", 0)
        )
    );
    // Node S315_Conv2d_64x1x3x3_Custom
    AddNode("S315_Conv2d_64x1x3x3_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S297_Output", 0),
            GNodeArg(GNA_IN, "Conv_307_weights", 0),
            GNodeArg(GNA_IN, "Constant__1411", 0),
            GNodeArg(GNA_OUT, "S315_Output", 0)
        )
    );
    // Node S318_Conv2d_64x64x1x1_Custom
    AddNode("S318_Conv2d_64x64x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S315_Output", 0),
            GNodeArg(GNA_IN, "Conv_310_weights", 0),
            GNodeArg(GNA_IN, "Constant__1414", 0),
            GNodeArg(GNA_OUT, "S318_Output", 0)
        )
    );
    // Node S321_Conv2d_64x1x3x3_Custom
    AddNode("S321_Conv2d_64x1x3x3_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S318_Output", 0),
            GNodeArg(GNA_IN, "Conv_313_weights", 0),
            GNodeArg(GNA_IN, "Constant__1417", 0),
            GNodeArg(GNA_OUT, "S321_Output", 0)
        )
    );
    // Node S324_Conv2d_64x64x1x1_Custom
    AddNode("S324_Conv2d_64x64x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S321_Output", 0),
            GNodeArg(GNA_IN, "Conv_316_weights", 0),
            GNodeArg(GNA_IN, "Constant__1420", 0),
            GNodeArg(GNA_OUT, "S324_Output", 0)
        )
    );
    // Node S327_Conv2d_4x64x1x1
    AddNode("S327_Conv2d_4x64x1x1",
        Bindings(4,
            GNodeArg(GNA_IN, "S324_Output", 0),
            GNodeArg(GNA_IN, "Conv_319_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_reg_preds_1_bias", 0),
            GNodeArg(GNA_OUT, "S327_Output", 0)
        )
    );
    // Node S330_Conv2d_1x64x1x1_Sigmoid
    AddNode("S330_Conv2d_1x64x1x1_Sigmoid",
        Bindings(4,
            GNodeArg(GNA_IN, "S324_Output", 0),
            GNodeArg(GNA_IN, "Conv_320_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_obj_preds_1_bias", 0),
            GNodeArg(GNA_OUT, "S330_Output", 0)
        )
    );
    // Node S335_Conv2d_64x256x1x1_Custom
    AddNode("S335_Conv2d_64x256x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S256_Output", 0),
            GNodeArg(GNA_IN, "Conv_324_weights", 0),
            GNodeArg(GNA_IN, "Constant__1423", 0),
            GNodeArg(GNA_OUT, "S335_Output", 0)
        )
    );
    // Node S338_Conv2d_64x1x3x3_Custom
    AddNode("S338_Conv2d_64x1x3x3_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S335_Output", 0),
            GNodeArg(GNA_IN, "Conv_327_weights", 0),
            GNodeArg(GNA_IN, "Constant__1426", 0),
            GNodeArg(GNA_OUT, "S338_Output", 0)
        )
    );
    // Node S341_Conv2d_64x64x1x1_Custom
    AddNode("S341_Conv2d_64x64x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S338_Output", 0),
            GNodeArg(GNA_IN, "Conv_330_weights", 0),
            GNodeArg(GNA_IN, "Constant__1429", 0),
            GNodeArg(GNA_OUT, "S341_Output", 0)
        )
    );
    // Node S344_Conv2d_64x1x3x3_Custom
    AddNode("S344_Conv2d_64x1x3x3_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S341_Output", 0),
            GNodeArg(GNA_IN, "Conv_333_weights", 0),
            GNodeArg(GNA_IN, "Constant__1432", 0),
            GNodeArg(GNA_OUT, "S344_Output", 0)
        )
    );
    // Node S347_Conv2d_64x64x1x1_Custom
    AddNode("S347_Conv2d_64x64x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S344_Output", 0),
            GNodeArg(GNA_IN, "Conv_336_weights", 0),
            GNodeArg(GNA_IN, "Constant__1435", 0),
            GNodeArg(GNA_OUT, "S347_Output", 0)
        )
    );
    // Node S350_Conv2d_1x64x1x1_Sigmoid
    AddNode("S350_Conv2d_1x64x1x1_Sigmoid",
        Bindings(4,
            GNodeArg(GNA_IN, "S347_Output", 0),
            GNodeArg(GNA_IN, "Conv_339_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_cls_preds_2_bias", 0),
            GNodeArg(GNA_OUT, "S350_Output", 0)
        )
    );
    // Node S353_Conv2d_64x1x3x3_Custom
    AddNode("S353_Conv2d_64x1x3x3_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S335_Output", 0),
            GNodeArg(GNA_IN, "Conv_340_weights", 0),
            GNodeArg(GNA_IN, "Constant__1438", 0),
            GNodeArg(GNA_OUT, "S353_Output", 0)
        )
    );
    // Node S356_Conv2d_64x64x1x1_Custom
    AddNode("S356_Conv2d_64x64x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S353_Output", 0),
            GNodeArg(GNA_IN, "Conv_343_weights", 0),
            GNodeArg(GNA_IN, "Constant__1441", 0),
            GNodeArg(GNA_OUT, "S356_Output", 0)
        )
    );
    // Node S359_Conv2d_64x1x3x3_Custom
    AddNode("S359_Conv2d_64x1x3x3_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S356_Output", 0),
            GNodeArg(GNA_IN, "Conv_346_weights", 0),
            GNodeArg(GNA_IN, "Constant__1444", 0),
            GNodeArg(GNA_OUT, "S359_Output", 0)
        )
    );
    // Node S362_Conv2d_64x64x1x1_Custom
    AddNode("S362_Conv2d_64x64x1x1_Custom",
        Bindings(4,
            GNodeArg(GNA_IN, "S359_Output", 0),
            GNodeArg(GNA_IN, "Conv_349_weights", 0),
            GNodeArg(GNA_IN, "Constant__1447", 0),
            GNodeArg(GNA_OUT, "S362_Output", 0)
        )
    );
    // Node S365_Conv2d_4x64x1x1
    AddNode("S365_Conv2d_4x64x1x1",
        Bindings(4,
            GNodeArg(GNA_IN, "S362_Output", 0),
            GNodeArg(GNA_IN, "Conv_352_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_reg_preds_2_bias", 0),
            GNodeArg(GNA_OUT, "S365_Output", 0)
        )
    );
    // Node S368_Conv2d_1x64x1x1_Sigmoid
    AddNode("S368_Conv2d_1x64x1x1_Sigmoid",
        Bindings(4,
            GNodeArg(GNA_IN, "S362_Output", 0),
            GNodeArg(GNA_IN, "Conv_353_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_obj_preds_2_bias", 0),
            GNodeArg(GNA_OUT, "S368_Output", 0)
        )
    );
    // Node Concat_381 inq ['f16', 'f16', 'f16'] outq ['f16']
    AddNode("S371_Concat",
        Bindings(4,
            GNodeArg(GNA_IN, "S293_Output", 0),
            GNodeArg(GNA_IN, "S331_Output", 0),
            GNodeArg(GNA_IN, "S369_Output", 0),
            GNodeArg(GNA_OUT, "S371_Output", 0)
        )
    );
    // Node Transpose_382 inq f16 outq f16
    AddNode("S372_Op_Transpose_382",
        Bindings(2,
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
