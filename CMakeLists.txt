cmake_minimum_required(VERSION 3.16)

set(TARGET_NAME         main)

###############################################################################
# CMake pre initialization
###############################################################################

include($ENV{GAP_SDK_HOME}/utils/cmake/setup.cmake)
set(APP_MSG_PREFIX "[${BoldRed}YoloX${ColourReset}]")

set(ONNX_TRAINED_FILE         ${CMAKE_CURRENT_SOURCE_DIR}/weights/yolox_nano_rgb_240x320_wfocus.onnx)
set(QUANT_STATS_FILE          ${CMAKE_CURRENT_SOURCE_DIR}/weights/precalculated_stats_yolox_nano_rgb_240x320_wfocus.pickle)
set(TEST_OUTPUT_FILE_NAME     ${CMAKE_CURRENT_SOURCE_DIR}/test_data/gt_boxes.bin)
set(TEST_INPUT_FILE_NAME      ${CMAKE_CURRENT_SOURCE_DIR}/test_data/input.ppm)
set(OUTPUT_FILE_NAME          ${CMAKE_CURRENT_BINARY_DIR}/output.ppm)
set(OUTPUT_BIN_FILE_NAME      ${CMAKE_CURRENT_BINARY_DIR}/output.bin)
set(OUTPUT_JPEG_FILE_NAME     ${CMAKE_CURRENT_BINARY_DIR}/output_compressed.jpg)

if(CONFIG_INFERENCE)
  set(CONF_THRESH 0.35)
  if(CONFIG_CI)
    message(STATUS "${APP_MSG_PREFIX} CI MODE")
    set(INPUT_FILE_NAME ${TEST_INPUT_FILE_NAME})
    list(APPEND TARGET_PREPROCESSOR -DCI)
  else()
    if (CONFIG_IMAGE_ABS_PATH STREQUAL "")
      set(INPUT_FILE_NAME           ${CMAKE_CURRENT_SOURCE_DIR}/${CONFIG_IMAGE_REL_PATH})
    else()
      message(STATUS "${APP_MSG_PREFIX} Absolute path image passed")
      set(INPUT_FILE_NAME           ${CONFIG_IMAGE_ABS_PATH})
    endif()
  endif()
  message(STATUS "${APP_MSG_PREFIX} Building App in INFERENCE Mode: with image ${INPUT_FILE_NAME}")
  set(MAIN                main_inference.c)

elseif(CONFIG_DEMO)
  message(STATUS "${APP_MSG_PREFIX} Building App in DEMO Mode")
  set(CONFIG_PLATFORM_GVSOC n)
  set(CONFIG_PLATFORM_BOARD y)
  set(CONF_THRESH 0.35)
  list(APPEND TARGET_PREPROCESSOR -DDEMO -DSILENT)
  set(MAIN                main_demo.c)

endif()

###############################################################################
# Autotiler Variables
###############################################################################
set(NN_DIR                ${CMAKE_BINARY_DIR}/BUILD_MODEL_NN)
set(NN_AT_MODEL_PATH      ${NN_DIR}/yoloxModel.c)
set(NN_AT_TENSORS_DIR     ${NN_DIR}/tensors)
set(NN_EXPRESSION_KERNELS ${NN_DIR}/Expression_Kernels.c)
autotiler_add_model(
  TARGET_NAME "yolox"
  MODEL_PATH ${NN_AT_MODEL_PATH}
  MODEL_TENSORS_DIR ${NN_AT_TENSORS_DIR}
  MODEL_BUILDDIR ${NN_DIR}
  PRI1_USED TRUE
  PRI1_FLASH mram
  PRI1_FILENAME "L3_Flash_Const.dat"
  L1_MEMORY ${CONFIG_MODEL_L1_MEMORY}
  L2_MEMORY ${CONFIG_MODEL_L2_MEMORY}
  L3_MEMORY ${CONFIG_MODEL_L3_MEMORY}
  L2STATIC_MEMORY ${CONFIG_MODEL_L2STATIC_MEMORY}
  L3STATIC_MEMORY ${CONFIG_MODEL_L3STATIC_MEMORY}
)

add_custom_command(
  OUTPUT ${NN_AT_MODEL_PATH} ${NN_AT_TENSORS_DIR} ${NN_EXPRESSION_KERNELS}
  COMMENT "Generating the autotiler model ${NN_AT_MODEL_PATH} ${NN_AT_TENSORS_DIR} ${NN_EXPRESSION_KERNELS}"
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/deployment_scripts/nntool_script.py
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/deployment_scripts/nntool_script.py
           --trained_model=${ONNX_TRAINED_FILE}
           --stats_path=${QUANT_STATS_FILE}
           --mode="generate"
           --at_model_path=${NN_AT_MODEL_PATH}
           --tensors_dir=${NN_AT_TENSORS_DIR}
           --flash_type="AT_MEM_L3_MRAMFLASH"
)
add_custom_target(${NNTOOL_TARGET_PREFIX}clean_model
  COMMENT "clean_model: removing ${COMMON_MODEL_BUILDDIR}/${MODEL_GEN}"
  COMMAND rm -rf ${AT_MODEL_PATH}
)

add_custom_command(
  OUTPUT ${TEST_OUTPUT_FILE_NAME}
  COMMENT "Generating the ground truth bounding boxes"
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/deployment_scripts/nntool_script.py
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/deployment_scripts/nntool_script.py
           --trained_model=${ONNX_TRAINED_FILE}
           --stats_path=${QUANT_STATS_FILE}
           --mode="inference"
           --input_image=${TEST_INPUT_FILE_NAME}
           --gt_file=${TEST_OUTPUT_FILE_NAME}
           --conf_thresh=${CONF_THRESH}
)
add_custom_target(gen_gt_file
  COMMENT "gen_gt_file: ${TEST_OUTPUT_FILE_NAME}"
  DEPENDS ${TEST_OUTPUT_FILE_NAME}
)
add_custom_target(clean_gt_file
  COMMENT "clean_gt_file: removing ${TEST_OUTPUT_FILE_NAME}"
  COMMAND rm -rf ${TEST_OUTPUT_FILE_NAME}
)

if(CONFIG_DEMO)
  set(ISP_GEN_PATH $ENV{TILER_ISP_GENERATOR_PATH})
  set(ISP_KER_PATH $ENV{TILER_ISP_KERNEL_PATH})
  set(MODEL_SRC_ISP ${CMAKE_SOURCE_DIR}/gap_functions/ISP_model.c)
  set(MODEL_GEN_ISP_C ${CMAKE_BINARY_DIR}/ISP_Kernels.c)

  add_custom_command(
    OUTPUT  ${CMAKE_BINARY_DIR}/GenTileISP
    DEPENDS ${MODEL_SRC_ISP} ${ISP_GEN_PATH}/ISP_Generators.c
    COMMAND gcc -o ${CMAKE_BINARY_DIR}/GenTileISP -I$ENV{TILER_INC} -I${ISP_GEN_PATH} ${MODEL_SRC_ISP} ${ISP_GEN_PATH}/ISP_Generators.c $ENV{TILER_LIB} ${SDL2_LDFLAGS}
  )

  add_custom_command(
    OUTPUT              ${MODEL_GEN_ISP_C}
    DEPENDS             ${CMAKE_BINARY_DIR}/GenTileISP
    COMMAND             ${CMAKE_BINARY_DIR}/GenTileISP
  )
  list(APPEND TARGET_SRCS ${ISP_KER_PATH}/ISP_BasicKernels.c ${MODEL_GEN_ISP_C})
endif()

###############################################################################
# Application dependencies
###############################################################################

list(APPEND TARGET_SRCS ${MAIN}
                        ${CNN_LIB}
                        ${KERNEL_C_PATHS}
                        ${NN_EXPRESSION_KERNELS}
                        ${CMAKE_CURRENT_SOURCE_DIR}/gap_functions/detection/slicing.c
                        ${CMAKE_CURRENT_SOURCE_DIR}/gap_functions/detection/decoding.c
                        ${CMAKE_CURRENT_SOURCE_DIR}/gap_functions/detection/postprocessing.c
                        ${CMAKE_CURRENT_SOURCE_DIR}/gap_functions/utils/draw.c
                        ${CMAKE_CURRENT_SOURCE_DIR}/gap_functions/utils/jpeg_compress.c
                        ${CMAKE_CURRENT_SOURCE_DIR}/gap_functions/utils/img_flush.c
                        ${CMAKE_CURRENT_SOURCE_DIR}/gap_functions/utils/camera.c
                        ${CMAKE_CURRENT_SOURCE_DIR}/gap_functions/utils/uart_comm.c
)

list(APPEND TARGET_INCS -I.
                        -I$ENV{GAP_H_INC}
                        -I${TILER_INC}
                        -I${TILER_EMU_INC}
                        -I${NN_DIR}
                        -I$ENV{GAP_SDK_HOME}/libs/gap_lib/include
                        -I${CNN_LIB_INCLUDE}
                        -I${CMAKE_CURRENT_SOURCE_DIR}/gap_functions/detection
                        -I${CMAKE_CURRENT_SOURCE_DIR}/gap_functions/utils
                        -I${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_MODEL_BUILDDIR}
                        -I${CMAKE_CURRENT_SOURCE_DIR}
                        -I$ENV{GAP_SDK_HOME}/utils/power_meas_utils
)

set(MAX_PERFORMANCE y)
if (${MAX_PERFORMANCE})
  set(FREQ    370)
  set(VOLTAGE 800)
else()
  set(FREQ    240)
  set(VOLTAGE 650)
endif()

list(APPEND TARGET_CFLAGS   -O3)

list(APPEND TARGET_PREPROCESSOR -DPERF
                                -DCOMPRESS
                                -DAT_MODEL_PREFIX=${MODEL_PREFIX}
                                -DSTACK_SIZE=${CONFIG_CL_MASTER_CORE_STACK_SIZE}
                                -DSLAVE_STACK_SIZE=${CONFIG_CL_SLAVE_CORE_STACK_SIZE}
                                -DFREQ_CL=${FREQ}
                                -DFREQ_FC=${FREQ}
                                -DFREQ_PE=${FREQ}
                                -DVOLTAGE=${VOLTAGE}
                                -DINPUT_FILE_NAME=${INPUT_FILE_NAME}
                                -DOUTPUT_FILE_NAME=${OUTPUT_FILE_NAME}
                                -DOUTPUT_BIN_FILE_NAME=${OUTPUT_BIN_FILE_NAME}
                                -DOUTPUT_JPEG_FILE_NAME=${OUTPUT_JPEG_FILE_NAME}
                                -DTEST_OUTPUT_FILE_NAME=${TEST_OUTPUT_FILE_NAME}
                                -DCONF_THRESH=${CONF_THRESH}
                                -DGPIO_MEAS
)

project(${TARGET_NAME} C ASM)
add_executable(${TARGET_NAME} ${TARGET_SRCS})
target_compile_options(${TARGET_NAME} PUBLIC ${TARGET_CFLAGS}
                                             ${TARGET_INCS}
                                             ${TARGET_PREPROCESSOR}
)

if (CONFIG_CI)
  add_dependencies(${TARGET_NAME} gen_gt_file)
endif()
###############################################################################
# CMake post initialization
###############################################################################
setupos(${TARGET_NAME})
