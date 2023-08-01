cmake_minimum_required(VERSION 3.16)

set(TARGET_NAME         main)

###############################################################################
# CMake pre initialization
###############################################################################

include($ENV{GAP_SDK_HOME}/utils/cmake/setup.cmake)
set(APP_MSG_PREFIX "[${BoldRed}YoloX${ColourReset}]")

set(TEST_OUTPUT_FILE_NAME     ${CMAKE_CURRENT_SOURCE_DIR}/test_data/gt_boxes.bin)
set(OUTPUT_FILE_NAME          ${CMAKE_CURRENT_SOURCE_DIR}/output.ppm)
set(OUTPUT_BIN_FILE_NAME      ${CMAKE_CURRENT_SOURCE_DIR}/output.bin)
set(OUTPUT_JPEG_FILE_NAME     ${CMAKE_CURRENT_SOURCE_DIR}/output_compressed.jpg)

if(CONFIG_INFERENCE)
  set(CONF_THRESH 0.35)
  set(INPUT_FILE_NAME           ${CMAKE_CURRENT_SOURCE_DIR}/${CONFIG_IMAGE_REL_PATH})
  message(STATUS "${APP_MSG_PREFIX} Building App in INFERENCE Mode: with image ${INPUT_FILE_NAME}")
  if(DEFINED CI)
    message(STATUS "${APP_MSG_PREFIX} CI MODE")
    list(APPEND TARGET_PREPROCESSOR -DCI)
  endif()

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

autotiler_define_gen_names(${TARGET_NAME})
set(AT_TENSORS_DIR BUILD_MODEL/weights_tensors)
autotiler_setup(${TARGET_NAME})
add_custom_command(
  OUTPUT ${AT_MODEL_PATH} ${AT_TENSORS_DIR} ${CONFIG_MODEL_BUILDDIR}/Expression_Kernels.c
  COMMENT "Generating the autotiler model"
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/model_generator/nntool_generate_model.py
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/model_generator/nntool_generate_model.py
           --trained_model=${CMAKE_CURRENT_SOURCE_DIR}/weights/yolox-QVGA-bayer-BGR.onnx
           --stats_path=${CMAKE_CURRENT_SOURCE_DIR}/weights/precalculated_stats_bayer_v2_qvga_trainset_nonpadded.pickle
           --mode="generate"
           --at_model_path=${AT_MODEL_PATH}
           --tensors_dir=${AT_TENSORS_DIR}
           --flash_type=${MODEL_L3_FLASH}
           --ram_type=${MODEL_L3_RAM}
)
add_custom_target(${NNTOOL_TARGET_PREFIX}clean_model
  COMMENT "clean_model: removing ${COMMON_MODEL_BUILDDIR}/${MODEL_GEN}"
  COMMAND rm -rf ${AT_MODEL_PATH}
)

# add_custom_command(
#   OUTPUT ${TEST_OUTPUT_FILE_NAME}
#   COMMENT "Generating the ground truth bounding boxes"
#   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/model_generator/nntool_generate_model.py
#   COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/model_generator/nntool_generate_model.py
#            --trained_model=${CMAKE_CURRENT_SOURCE_DIR}/weights/yolox-QVGA-bayer-BGR.onnx
#            --stats_path=${CMAKE_CURRENT_SOURCE_DIR}/weights/precalculated_stats_bayer_v2_qvga_trainset_nonpadded.pickle
#            --mode="inference"
#            --input_image=${TEST_INPUT_FILE_NAME}
#            --gt_file=${TEST_OUTPUT_FILE_NAME}
#            --conf_thresh=${CONF_THRESH}
# )
# add_custom_target(gen_gt_file
#   COMMENT "gen_gt_file: ${TEST_OUTPUT_FILE_NAME}"
#   DEPENDS ${TEST_OUTPUT_FILE_NAME}
# )
# add_custom_target(clean_gt_file
#   COMMENT "clean_gt_file: removing ${TEST_OUTPUT_FILE_NAME}"
#   COMMAND rm -rf ${TEST_OUTPUT_FILE_NAME}
# )
###############################################################################
# Application dependencies
###############################################################################

list(APPEND TARGET_SRCS ${MAIN}
                        ${CNN_LIB}
                        ${CONFIG_MODEL_BUILDDIR}/Expression_Kernels.c
                        ${CONFIG_MODEL_BUILDDIR}/${KERNEL_C}
                        ${CMAKE_CURRENT_SOURCE_DIR}/custom_layers/slicing.c
                        ${CMAKE_CURRENT_SOURCE_DIR}/custom_layers/decoding.c
                        ${CMAKE_CURRENT_SOURCE_DIR}/custom_layers/postprocessing.c
                        ${CMAKE_CURRENT_SOURCE_DIR}/custom_layers/draw.c
                        ${CMAKE_CURRENT_SOURCE_DIR}/custom_layers/jpeg_compress.c
                        ${CMAKE_CURRENT_SOURCE_DIR}/custom_layers/img_flush.c
                        ${CMAKE_CURRENT_SOURCE_DIR}/custom_layers/camera.c
)

list(APPEND TARGET_INCS -I.
                        -I${TILER_INC}
                        -I${TILER_EMU_INC}
                        -I$ENV{GAP_SDK_HOME}/libs/gap_lib/include
                          ${CNN_LIB_INCLUDE}
                        -I${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_MODEL_BUILDDIR}
                        -I${CMAKE_CURRENT_SOURCE_DIR})

list(APPEND TARGET_CFLAGS   -O3)

set(FREQ_CL 370)
set(FREQ_FC 370)
set(FREQ_PE 360)

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
                                -DTEST_OUTPUT_FILE_NAME=${TEST_OUTPUT_FILE_NAME}
                                -DCONF_THRESH=${CONF_THRESH}
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
