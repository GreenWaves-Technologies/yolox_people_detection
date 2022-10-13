#include "Gap.h"

// void slicing_chw_channel(signed char * Input, signed char * Output, int h, int w, int chnls);
void slicing_chw_channel(f16 * Input, f16 * Output, int h, int w, int chnls);
void slicing_hwc_channel(signed char * Input, signed char * Output, int h, int w, int chnls);