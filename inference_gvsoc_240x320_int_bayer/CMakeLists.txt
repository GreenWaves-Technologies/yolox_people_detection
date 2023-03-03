cmake_minimum_required(VERSION 3.16)

set(TARGET_NAME         main)

###############################################################################
# CMake pre initialization
###############################################################################

include($ENV{GAP_SDK_HOME}/utils/cmake/setup.cmake)
set(APP_MSG_PREFIX "[${BoldRed}YoloX${ColourReset}]")

set(TEST_INPUT_FILE_NAME      ${CMAKE_CURRENT_SOURCE_DIR}/test_data/input.ppm)
set(TEST_OUTPUT_FILE_NAME     ${CMAKE_CURRENT_SOURCE_DIR}/test_data/gt_boxes.bin)
set(INPUT_FILE_NAME           ${CMAKE_CURRENT_SOURCE_DIR}/input_rgb.ppm)
set(OUTPUT_FILE_NAME          ${CMAKE_CURRENT_SOURCE_DIR}/output.ppm)
set(OUTPUT_BIN_FILE_NAME      ${CMAKE_CURRENT_SOURCE_DIR}/output.bin)
set(OUTPUT_JPEG_FILE_NAME      ${CMAKE_CURRENT_SOURCE_DIR}/output_compressed.jpg)

if(CONFIG_CI)
  message(STATUS "${APP_MSG_PREFIX} Building App in CI Mode")
  list(APPEND TARGET_PREPROCESSOR -DCI)
  set(MAIN                main_inference.c)

elseif(CONFIG_INFERENCE)
  message(STATUS "${APP_MSG_PREFIX} Building App in INFERENCE Mode")
  list(APPEND TARGET_PREPROCESSOR -DINFERENCE)
  set(MAIN                main_inference.c)

elseif(CONFIG_DEMO)
  message(STATUS "${APP_MSG_PREFIX} Building App in DEMO Mode")
  list(APPEND TARGET_PREPROCESSOR -DDEMO -DSILENT)
  set(MAIN                main_demo.c)

endif()

###############################################################################
# Autotiler Variables
###############################################################################

autotiler_define_gen_names(${TARGET_NAME})
set(AT_TENSORS_DIR BUILD_MODEL/weights_tensors)
autotiler_setup(${TARGET_NAME})
add_custom_command(
  OUTPUT ${AT_MODEL_PATH} ${AT_TENSORS_DIR}
  COMMENT "Generating the autotiler model"
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../model_generator/nntool_generate_model.py
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../model_generator/nntool_generate_model.py --at_model_path=${AT_MODEL_PATH}
                                                                 --tensors_dir=${AT_TENSORS_DIR}
                                                                 --trained_model=${CMAKE_CURRENT_SOURCE_DIR}/../weights/yolox-QVGA-bayer-BGR.onnx
                                                                 --stats_path=${CMAKE_CURRENT_SOURCE_DIR}/../weights/precalculated_stats_bayer_v2_qvga_trainset_nonpadded.pickle
                                                                 --flash_type=${MODEL_L3_FLASH}
                                                                 --ram_type=${MODEL_L3_RAM}
)
add_custom_target(${NNTOOL_TARGET_PREFIX}clean_model
  COMMENT "clean_model: removing ${COMMON_MODEL_BUILDDIR}/${MODEL_GEN}"
  COMMAND rm -rf ${AT_MODEL_PATH}
)

###############################################################################
# Application dependencies
###############################################################################

list(APPEND TARGET_SRCS ${MAIN}
                        ${CNN_LIB}
                        ${CONFIG_MODEL_BUILDDIR}/${KERNEL_C}
                        ../custom_layers/slicing.c
                        ../custom_layers/decoding.c
                        ../custom_layers/postprocessing.c
                        ../custom_layers/draw.c
                        ../custom_layers/jpeg_compress.c
                        ../custom_layers/img_flush.c
                        ../custom_layers/camera.c
)

list(APPEND TARGET_INCS -I.
                        -I${TILER_INC}
                        -I${TILER_EMU_INC}
                        -I$ENV{GAP_SDK_HOME}/libs/gap_lib/include
                          ${CNN_LIB_INCLUDE}
                        -I${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_MODEL_BUILDDIR}
                        -I${CMAKE_CURRENT_SOURCE_DIR})

if (CONFIG_MODEL_EMUL)
list(REMOVE_ITEM TARGET_CFLAGS
  -mno-memcpy)
list(APPEND TARGET_CFLAGS
  -g
  -m32
  -O0
  -fcommon
  -D__EMUL__)
else()
list(APPEND TARGET_CFLAGS   -O3)
endif()

set(FREQ_CL 370)
set(FREQ_FC 370)
set(FREQ_PE 370)

message(STATUS "Clock speeds - Cluster ${FREQ_CL} FC ${FREQ_FC} Peripheral ${FREQ_PE}")
# math(EXPR FREQ_CL ${CONFIG_FREQUENCY_CLUSTER}/1000000 OUTPUT_FORMAT DECIMAL)
# math(EXPR FREQ_FC ${CONFIG_FREQUENCY_FC}/1000000 OUTPUT_FORMAT DECIMAL)
# math(EXPR FREQ_PE ${CONFIG_FREQUENCY_PERIPH}/1000000 OUTPUT_FORMAT DECIMAL)

list(APPEND TARGET_PREPROCESSOR -DPERF
                                -DCOMPRESS
                                -DAT_MODEL_PREFIX=${MODEL_PREFIX}
                                -DSTACK_SIZE=${CONFIG_CL_MASTER_CORE_STACK_SIZE}
                                -DSLAVE_STACK_SIZE=${CONFIG_CL_SLAVE_CORE_STACK_SIZE}
                                -DFREQ_CL=${FREQ_CL}
                                -DFREQ_FC=${FREQ_FC}
                                -DFREQ_PE=${FREQ_PE}
                                -DINPUT_FILE_NAME=${INPUT_FILE_NAME}
                                -DOUTPUT_FILE_NAME=${OUTPUT_FILE_NAME}
                                -DOUTPUT_BIN_FILE_NAME=${OUTPUT_BIN_FILE_NAME}
                                -DOUTPUT_JPEG_FILE_NAME=${OUTPUT_JPEG_FILE_NAME}
                                -DTEST_INPUT_FILE_NAME=${TEST_INPUT_FILE_NAME}
                                -DTEST_OUTPUT_FILE_NAME=${TEST_OUTPUT_FILE_NAME}
)

project(${TARGET_NAME} C ASM)

add_executable(${TARGET_NAME} ${TARGET_SRCS})

if (CONFIG_MODEL_EMUL)
find_program(CCACHE_FOUND ccache)
if (CCACHE_FOUND)
    set(CMAKE_C_COMPILER_LAUNCHER   ccache)
    set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
endif()

target_link_libraries(${TARGET_NAME} PUBLIC m)

set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_FLAGS "-m32" LINK_FLAGS "-m32")

if (NOT CONFIG_MODEL_HAS_NO_CONSTANTS)
  add_custom_command(TARGET ${TARGET_NAME}
    COMMENT "Copy tensors to build directory"
    POST_BUILD
    COMMAND cp ${MODEL_TENSORS} ${CMAKE_CURRENT_BINARY_DIR}
  )
endif()
add_custom_target(run ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME})
endif()

target_compile_options(${TARGET_NAME} PUBLIC   ${TARGET_CFLAGS}
                                                ${TARGET_INCS}
                                                ${TARGET_PREPROCESSOR})

###############################################################################
# CMake post initialization
###############################################################################
if (NOT CONFIG_MODEL_EMUL)
  setupos(${TARGET_NAME})
endif ()
