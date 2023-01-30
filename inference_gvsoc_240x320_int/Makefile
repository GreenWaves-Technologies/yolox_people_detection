# Copyright (C) 2017 GreenWaves Technologies
# All rights reserved.

# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE file for details.

ifndef GAP_SDK_HOME
  $(error Source sourceme in gap_sdk first)
endif

include common.mk
include $(RULES_DIR)/at_common_decl.mk

io?=host

$(info Building NNTOOL model)
NNTOOL_EXTRA_FLAGS ?=

include common/model_decl.mk

# pulpChip = GAP
# PULP_APP = $(MODEL_PREFIX)

## input file names 
input_file_name=../../../input_rgb.ppm
output_file_name=../../../output.ppm
output_bin_file_name=../../../output.bin


APP = $(MODEL_PREFIX)
APP_SRCS += $(MODEL_PREFIX).c $(MODEL_GEN_C) $(MODEL_EXPRESSIONS) $(MODEL_COMMON_SRCS) $(CNN_LIB)
APP_SRCS += slicing.c
APP_SRCS += decoding.c
APP_SRCS += postprocessing.c
APP_SRCS += draw.c


APP_CFLAGS += -g -O3 -mno-memcpy -fno-tree-loop-distribute-patterns
APP_CFLAGS += -I. -I$(MODEL_COMMON_INC) -I$(TILER_EMU_INC) -I$(TILER_INC) $(CNN_LIB_INCLUDE) -I$(MODEL_BUILD)
APP_CFLAGS += -DPERF -DAT_MODEL_PREFIX=$(MODEL_PREFIX) $(MODEL_SIZE_CFLAGS)
APP_CFLAGS += -DSTACK_SIZE=$(CLUSTER_STACK_SIZE) -DSLAVE_STACK_SIZE=$(CLUSTER_SLAVE_STACK_SIZE)
APP_CFLAGS += -DAT_IMAGE=$(IMAGE) -DFREQ_FC=$(FREQ_FC) -DFREQ_CL=$(FREQ_CL) -DFREQ_PE=$(FREQ_PE)
APP_CFLAGS += -DINPUT_FILE_NAME=$(input_file_name)
APP_CFLAGS += -DOUTPUT_FILE_NAME=$(output_file_name) 
APP_CFLAGS += -DOUTPUT_BIN_FILE_NAME=$(output_bin_file_name)

READFS_FILES=$(abspath $(MODEL_TENSORS))


# Application modes:
#  - CI        Compare newtwork outputs with expected results on test samples
#  - DEMO      Run the network on a .ppm image and save the output in a .ppm file with drawn boxes 
#  - INFERENCE Run in Inference on static image  

mode?=CI
ifeq (,$(filter $(mode), CI DEMO INFERENCE))
    $(error Invalid mode $(mode))
endif
APP_CFLAGS += -D$(mode) 

#uncomment this line to hide printfs
ifeq '$(mode)' 'DEMO'
  APP_CFLAGS += -DSILENT=1
else ifeq '$(mode)' 'INFERENCE'
    APP_CLFAGS += -DPERF
endif



# build depends on the model
build:: model

clean:: clean_model

include common/model_rules.mk
$(info APP_SRCS... $(APP_SRCS))
$(info APP_CFLAGS... $(APP_CFLAGS))
include $(RULES_DIR)/pmsis_rules.mk

