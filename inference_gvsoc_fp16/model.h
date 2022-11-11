#ifndef __model_H__
#define __model_H__


#define QUOTE(name) #name
#define STR(macro) QUOTE(macro)

#ifndef STACK_SIZE
#define STACK_SIZE      1024
#endif

// ------------------------- PARAMETERS -------------------------
// parameters needed for slicing layer
#define H_INP 256
#define W_INP 320
#define CHANNELS 3

// parameters needed for decoding layer
#define STRIDE_SIZE 3

// parameters needed for xywh2xyxy layer
#define RAWS 1680

// parameters needed for postprocessing layer
#define CONF_THRESH 0.30

// parameters needed for function to_boxes
#define top_k_boxes 70 

// parameters needed for nms
#define NMS_THRESH 0.30

//parameters for drawing boxes
#define MAX(a, b)        (((a)>(b))?(a):(b))
#define MIN(a, b)        (((a)<(b))?(a):(b))
// -----------------------------------------------------------------

#define __PREFIX(x) model ## x
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

extern AT_HYPERFLASH_EXT_ADDR_TYPE model_L3_Flash;
#endif