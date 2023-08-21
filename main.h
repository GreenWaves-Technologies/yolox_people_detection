#ifndef __main_H__
#define __main_H__

/* ---------- parameters ----------*/

#define QUOTE(name) #name
#define STR(macro) QUOTE(macro)

#ifndef STACK_SIZE
#define STACK_SIZE      1024
#endif

// CI test parameters
#define CI_BOX_TYPE_SIZE 6
#define CI_TEST_BOX_NUM 2 

// parameters needed for slicing layer
#define H_INP 240
#define W_INP 320
#define CHANNELS 3

#define H_CAM 480
#define W_CAM 640
#define BYTES_CAM 2


// parameters needed for decoding layer
#define STRIDE_SIZE 3

// parameters needed for xywh2xyxy layer
#define RAWS 1580

// parameters needed for function to_boxes
#define top_k_boxes 400 

// parameters needed for nms
#define NMS_THRESH 0.65

//parameters for drawing boxes
#define MAX(a, b)        (((a)>(b))?(a):(b))
#define MIN(a, b)        (((a)<(b))?(a):(b))

/* ---------- end ----------*/

#define __PREFIX(x) main ## x
// Include basic GAP builtins defined in the Autotiler
#include "Gap.h"

#ifdef __EMUL__
#include <sys/types.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/param.h>
#include <string.h>
#endif

#include "mainKernels.h"
#include "gaplib/ImgIO.h"
#include "gaplib/fs_switch.h"
#include "bsp/camera/ov5647.h"
#include "custom_layers/draw.h"
#include "custom_layers/slicing.h"
#include "custom_layers/decoding.h"
#include "custom_layers/postprocessing.h"
#include "custom_layers/camera.h"
#include "custom_layers/jpeg_compress.h"
#include "custom_layers/img_flush.h"

#if SILENT
#define PRINTF(...) ((void) 0)
#else
#define PRINTF printf
#endif

extern AT_DEFAULTFLASH_EXT_ADDR_TYPE main_L3_Flash;
#endif
