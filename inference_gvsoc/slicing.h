
// void slicing(unsigned char * array, unsigned char * tmp, int h, int w);
void slicing_chw_channel(signed char * Input, signed char * Output, int h, int w, int chnls);
void slicing_hwc_channel(signed char * Input, signed char * Output, int h, int w, int chnls);