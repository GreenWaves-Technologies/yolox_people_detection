#include "pmsis.h"
#include "bsp/bsp.h"
#include "bsp/ram.h"

void init_uart_communication(pi_device_t* uart_dev, uint32_t baudrate);
void send_image_to_uart(pi_device_t* uart_dev,uint8_t* img,int img_w,int img_h,int pixel_size, unsigned int *perf_array);
void send_jpeg_to_uart(pi_device_t* uart_dev, uint8_t* img, int img_size, unsigned int *perf_array);
