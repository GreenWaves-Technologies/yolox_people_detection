#ifndef __main_H__
#define __main_H__

/* ---------- parameters ----------*/

#define QUOTE(name) #name
#define STR(macro) QUOTE(macro)

#ifndef STACK_SIZE
#define STACK_SIZE      1024
#endif

// parameters needed for slicing layer
#define H_INP 240
#define W_INP 320
#define CHANNELS 3

// parameters needed for decoding layer
#define STRIDE_SIZE 3

// parameters needed for xywh2xyxy layer
#define RAWS 1580

// parameters needed for postprocessing layer
#define CONF_THRESH 0.5

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

extern AT_DEFAULTFLASH_EXT_ADDR_TYPE main_L3_Flash;
#endif