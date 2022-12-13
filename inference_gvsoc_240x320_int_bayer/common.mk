NNTOOL=nntool
MODEL_SQ8=1
# MODEL_POW2=1
MODEL_FP16=1
MODEL_NE16=1


MODEL_SUFFIX?=
MODEL_PREFIX?=main
MODEL_PYTHON=python3
MODEL_BUILD=BUILD_MODEL$(MODEL_SUFFIX)

AT_MODEL_PATH=./model.c
AT_TENSORS_DIR=./weights_tensors

MODEL_EXPRESSIONS = Expression_Kernels.c



# Memory sizes for cluster L1, SoC L2 and Flash
TARGET_L1_SIZE = 128000
TARGET_L2_SIZE = 1000000
TARGET_L3_SIZE = 8000000

# Options for the memory settings: will require
# set l3_flash_device $(MODEL_L3_FLASH)
# set l3_ram_device $(MODEL_L3_RAM)
# in the nntool_script
# FLASH and RAM type
FLASH_TYPE = MRAM
RAM_TYPE   = DEFAULT

ifeq '$(FLASH_TYPE)' 'HYPER'
    MODEL_L3_FLASH=AT_MEM_L3_HFLASH
else ifeq '$(FLASH_TYPE)' 'MRAM'
    MODEL_L3_FLASH=AT_MEM_L3_MRAMFLASH
    READFS_FLASH = target/chip/soc/mram
else ifeq '$(FLASH_TYPE)' 'QSPI'
    MODEL_L3_FLASH=AT_MEM_L3_QSPIFLASH
    READFS_FLASH = target/board/devices/spiflash
else ifeq '$(FLASH_TYPE)' 'OSPI'
    MODEL_L3_FLASH=AT_MEM_L3_OSPIFLASH
else ifeq '$(FLASH_TYPE)' 'DEFAULT'
    MODEL_L3_FLASH=AT_MEM_L3_DEFAULTFLASH
endif

ifeq '$(RAM_TYPE)' 'HYPER'
    MODEL_L3_RAM=AT_MEM_L3_HRAM
else ifeq '$(RAM_TYPE)' 'QSPI'
    MODEL_L3_RAM=AT_MEM_L3_QSPIRAM
else ifeq '$(RAM_TYPE)' 'OSPI'
    MODEL_L3_RAM=AT_MEM_L3_OSPIRAM
else ifeq '$(RAM_TYPE)' 'DEFAULT'
    MODEL_L3_RAM=AT_MEM_L3_DEFAULTRAM
endif

ifeq '$(TARGET_CHIP_FAMILY)' 'GAP9'
    FREQ_CL?=370
    FREQ_FC?=370
    #360 is for mipi camera which needs to be multiple of 20 mhz
    FREQ_PE?=360 
else
    ifeq '$(TARGET_CHIP)' 'GAP8_V3'
    FREQ_CL?=175
    else
    FREQ_CL?=50
    endif
    FREQ_FC?=250
    FREQ_PE?=250
endif

# Cluster stack size for master core and other cores
CLUSTER_STACK_SIZE=2048
CLUSTER_SLAVE_STACK_SIZE=512

# define STD_FLOAT if float16 in use

# load math library if float expressions in use


NNTOOL_SCRIPT = nntool_script
$(info GEN ... $(CNN_GEN))
