#include "uart_comm.h"

uint8_t UART_START_COM[] = {0xF1,0x1F};
uint8_t UART_START_JPEG[] = {0xAB,0xBA};

void init_uart_communication(pi_device_t* uart_dev,uint32_t baudrate ){
    pi_pad_function_set(PI_PAD_065, PI_PAD_FUNC0);
    pi_pad_function_set(PI_PAD_066, PI_PAD_FUNC0);
    pi_pad_function_set(PI_PAD_044, PI_PAD_FUNC0);
    pi_pad_mux_group_set(PI_PAD_044, PI_PAD_MUX_GROUP_UART1_RX);
    pi_pad_function_set(PI_PAD_045, PI_PAD_FUNC0);
    pi_pad_mux_group_set(PI_PAD_045, PI_PAD_MUX_GROUP_UART1_TX);

    struct pi_uart_conf config = {0};
    /* Init & open uart. */
    pi_uart_conf_init(&config);
    config.uart_id = 1;
    config.use_fast_clk = 0;              // Enable the fast clk for uart
    config.baudrate_bps = baudrate;
    config.enable_tx = 1;
    config.enable_rx = 0;
    pi_open_from_conf(uart_dev, &config);

    if (pi_uart_open(uart_dev))
    {
        pmsis_exit(-117);
    }
}

void send_image_to_uart(pi_device_t* uart_dev,uint8_t* img,int img_w,int img_h,int pixel_size, unsigned int *perf_array){

    pi_uart_write(uart_dev,UART_START_COM,2);
    //Write Image row by row
    for(int i=0;i<img_h;i++) pi_uart_write(uart_dev,&(img[i*img_w*pixel_size]),img_w*pixel_size);
}

void send_jpeg_to_uart(pi_device_t* uart_dev, uint8_t* img, int img_size, unsigned int *perf_array){

    pi_uart_write(uart_dev, UART_START_JPEG, 2);
    pi_uart_write(uart_dev, &img_size, 4);
    pi_uart_write(uart_dev, &perf_array[0], 4*8);
    //Write Image row by row
    int size = img_size;
    int idx = 0;
    while (size > 0) {
        int size_to_write = (size > 256) ? 256 : size;
        pi_uart_write(uart_dev, &(img[idx]), size_to_write);
        size -= size_to_write;
        idx += size_to_write;
    }
}
