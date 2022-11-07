#ifndef __model_H__
#define __model_H__

// parameters needed for slicing layer
#define H_INP 256
#define W_INP 320
#define CHANNELS 3

//parameters for drawing boxes
#define MAX(a, b)        (((a)>(b))?(a):(b))
#define MIN(a, b)        (((a)<(b))?(a):(b))

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