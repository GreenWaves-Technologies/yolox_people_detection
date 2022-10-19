cd /home/abduragim/projects/tracking/nntool_inference/inference_gvsoc_fp16

source /home/abduragim/projects/gap_sdk/configs/gap9_v2.sh
export TILER_PATH=/home/abduragim/projects/gap_sdk/tools/autotiler_v3
export TILER_LIB=/home/abduragim/projects/gap_sdk/tools/autotiler_v3/Autotiler/LibTile.a
export TILER_INC=/home/abduragim/projects/gap_sdk/tools/autotiler_v3/Autotiler
export TILER_EMU_INC=/home/abduragim/projects/gap_sdk/tools/autotiler_v3/Emulation
export TILER_GENERATOR_PATH=/home/abduragim/projects/gap_sdk/tools/autotiler_v3/Generators
export TILER_BILINEAR_RESIZE_GENERATOR_PATH=/home/abduragim/projects/gap_sdk/tools/autotiler_v3/Generators/BilinearResizes
export TILER_BILINEAR_RESIZE_KERNEL_PATH=/home/abduragim/projects/gap_sdk/tools/autotiler_v3/Generators/BilinearResizes
export TILER_INTEGRAL_GENERATOR_PATH=/home/abduragim/projects/gap_sdk/tools/autotiler_v3/Generators/IntegralImage
export TILER_INTEGRAL_KERNEL_PATH=/home/abduragim/projects/gap_sdk/tools/autotiler_v3/Generators/IntegralImage
export TILER_FFT2D_GENERATOR_PATH=/home/abduragim/projects/gap_sdk/tools/autotiler_v3/Generators/FFT2DModel
export TILER_FFT2D_KERNEL_PATH=/home/abduragim/projects/gap_sdk/tools/autotiler_v3/Generators/FFT2DModel
export TILER_FFT2D_TWIDDLE_PATH=/home/abduragim/projects/gap_sdk/tools/autotiler_v3/Generators/FFT2DModel
export TILER_CNN_KERNEL_PATH=/home/abduragim/projects/gap_sdk/tools/autotiler_v3/CNN_Libraries
export TILER_CNN_GENERATOR_PATH=/home/abduragim/projects/gap_sdk/tools/autotiler_v3/CNN_Generators
export TILER_CNN_KERNEL_PATH_SQ8=/home/abduragim/projects/gap_sdk/tools/autotiler_v3/CNN_Libraries_SQ8
export TILER_CNN_KERNEL_PATH_NE16=/home/abduragim/projects/gap_sdk/tools/autotiler_v3/CNN_Libraries_NE16
export TILER_CNN_KERNEL_PATH_FP16=/home/abduragim/projects/gap_sdk/tools/autotiler_v3/CNN_Libraries_fp16
export TILER_CNN_GENERATOR_PATH_SQ8=/home/abduragim/projects/gap_sdk/tools/autotiler_v3/CNN_Generators_SQ8
export TILER_CNN_GENERATOR_PATH_NE16=/home/abduragim/projects/gap_sdk/tools/autotiler_v3/CNN_Generators_NE16
export TILER_CNN_GENERATOR_PATH_FP16=/home/abduragim/projects/gap_sdk/tools/autotiler_v3/CNN_Generators_fp16
export TILER_DSP_GENERATOR_PATH=/home/abduragim/projects/gap_sdk/tools/autotiler_v3/DSP_Generators
export TILER_DSP_KERNEL_PATH=/home/abduragim/projects/gap_sdk/tools/autotiler_v3/DSP_Libraries
export TILER_FFT_LUT_PATH=/home/abduragim/projects/gap_sdk/tools/autotiler_v3/DSP_Libraries/LUT_Tables
export TILER_MFCC_GEN_LUT_SCRIPT=/home/abduragim/projects/gap_sdk/tools/autotiler_v3/DSP_Libraries/LUT_Tables/gen_scripts/GenMFCCLUT.py
export TILER_FFT_GEN_LUT_SCRIPT=/home/abduragim/projects/gap_sdk/tools/autotiler_v3/DSP_Libraries/LUT_Tables/gen_scripts/GenFFTLUT.py
export TILER_WIN_GEN_LUT_SCRIPT=/home/abduragim/projects/gap_sdk/tools/autotiler_v3/DSP_Libraries/LUT_Tables/gen_scripts/GenWinLUT.py
export PATH=/home/abduragim/projects/gap_sdk/tools/nntool/scripts:$PATH
export PYTHONPATH=/home/abduragim/projects/gap_sdk/tools/nntool:$PYTHONPATH
make -C /home/abduragim/projects/tracking/nntool_inference/inference_gvsoc_fp16  model PMSIS_OS=freertos
retVal=$?
if [ $retVal -ne 0 ]; then
    exit $retVal
fi
make -C /home/abduragim/projects/tracking/nntool_inference/inference_gvsoc_fp16 -j 12 all platform=gvsoc PMSIS_OS=freertos MODEL_GEN_AT_LOGLEVEL=0
retVal=$?
if [ $retVal -ne 0 ]; then
    exit $retVal
fi
make -C /home/abduragim/projects/tracking/nntool_inference/inference_gvsoc_fp16 run platform=gvsoc PMSIS_OS=freertos
exit $?
