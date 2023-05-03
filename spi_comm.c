#include "spi_comm.h"

void spi_slave_init(pi_device_t* spi_slave, struct pi_spi_conf* spi_slave_conf)
{
    pi_pad_function_set(SPI_SLAVE_PAD_SCK, SPI_SLAVE_PAD_FUNC);
    pi_pad_function_set(SPI_SLAVE_PAD_CS0, SPI_SLAVE_PAD_FUNC);
    pi_pad_function_set(SPI_SLAVE_PAD_SDO, SPI_SLAVE_PAD_FUNC);
    pi_pad_function_set(SPI_SLAVE_PAD_SDI, SPI_SLAVE_PAD_FUNC);

    pi_assert(spi_slave);
    pi_assert(spi_slave_conf);

    pi_spi_conf_init(spi_slave_conf);
    spi_slave_conf->wordsize = SPI_SLAVE_WORDSIZE;
    spi_slave_conf->big_endian = SPI_SLAVE_ENDIANESS;
    spi_slave_conf->max_baudrate = SPI_SLAVE_BAUDRATE;
    spi_slave_conf->polarity = SPI_SLAVE_POLARITY;
    spi_slave_conf->phase = SPI_SLAVE_PHASE;
    spi_slave_conf->itf = SPI_SLAVE_ITF;
    spi_slave_conf->cs = SPI_SLAVE_CS;
    spi_slave_conf->dummy_clk_cycle = SPI_MASTER_DUMMY_CYCLE;
    spi_slave_conf->dummy_clk_cycle_mode = SPI_MASTER_DUMMY_CYCLE_MODE;
    spi_slave_conf->is_slave = SPI_SLAVE_IS_SLAVE;
    pi_open_from_conf(spi_slave, spi_slave_conf);
    if (pi_spi_open(spi_slave))
    {
        printf("ERROR: Failed to open SPI peripheral\n");
        return;
    }
}

// Test *******************************************************************************************

static void send_callback(void* context)
{
    pi_evt_push((pi_evt_t*)context);
}


void send_image_spi(pi_device_t* spi_slave,uint8_t*img, uint16_t img_w, uint16_t img_h){

    uint8_t tx_buffer[4];
    pi_evt_t send_task, end_task;


    do{
        pi_spi_receive(spi_slave, tx_buffer, 3 << 3, SPI_NO_OPTION);
    }
    while(tx_buffer[0]!=MAGIC_NUMBER_H && tx_buffer[1]!=MAGIC_NUMBER_L && tx_buffer[2]!=  CMD_STATUS_GET);

    tx_buffer[0]= MAGIC_NUMBER_H;
    tx_buffer[1]= MAGIC_NUMBER_L;
    tx_buffer[2]= STATUS_RDY;
    pi_spi_send(spi_slave, tx_buffer, 3 << 3, SPI_NO_OPTION);
    uint16_t* img_size = (uint16_t*) tx_buffer;
    img_size[0] = img_h;
    img_size[1] = img_w;
    pi_spi_send(spi_slave, img_size,   2 << 3, SPI_NO_OPTION);
    pi_spi_send(spi_slave, img_size+1, 2 << 3, SPI_NO_OPTION);
    
    for(int i=0;i<img_h;i++){
        pi_evt_sig_init(&end_task);
        pi_spi_send_async(spi_slave, img+(i*img_w), (img_w) << 3, SPI_NO_OPTION, pi_evt_callback_irq_init(&send_task, &send_callback, &end_task));
        pi_evt_wait(&end_task);
    }
}

void send_jpeg_spi(pi_device_t* spi_slave,  uint8_t* img, int img_size, unsigned int *perf_array){

    uint8_t tx_buffer[4];
    uint32_t chunk_size = 1024;
    pi_evt_t send_task, end_task;


    do{
        pi_spi_receive(spi_slave, tx_buffer, 3 << 3, SPI_NO_OPTION);
    }
    while(tx_buffer[0]!=MAGIC_NUMBER_H && tx_buffer[1]!=MAGIC_NUMBER_L && tx_buffer[2]!=  CMD_STATUS_GET);

    tx_buffer[0]= MAGIC_NUMBER_H;
    tx_buffer[1]= MAGIC_NUMBER_L;
    tx_buffer[2]= STATUS_RDY;
    //Sending ready to tranfer
    pi_spi_send(spi_slave, tx_buffer, 3 << 3, SPI_NO_OPTION);
    uint32_t* img_size_b = (uint32_t*) tx_buffer;
    img_size_b[0] = img_size;
    //Send size to send
    pi_spi_send(spi_slave, img_size_b,   4 << 3, SPI_NO_OPTION);
    
    //Send performance array:
    pi_spi_send(spi_slave, &perf_array[0],   (4*8) << 3, SPI_NO_OPTION);

    int size = img_size;
    int idx = 0;
    while (size > 0) {
        int size_to_write = (size > 256) ? 256 : size;
        pi_evt_sig_init(&end_task);
        pi_spi_send_async(spi_slave, &(img[idx]), (size_to_write) << 3, SPI_NO_OPTION, pi_evt_callback_irq_init(&send_task, &send_callback, &end_task));
        pi_evt_wait(&end_task);
        size -= size_to_write;
        idx += size_to_write;
    }
}

#if 0

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
#endif


void send_image_spi_ram(pi_device_t* spi_slave,pi_device_t* ram, uint32_t img, uint16_t img_w, uint16_t img_h){

    uint8_t tx_buffer[4];
    pi_evt_t send_task, end_task;
    int wait=0;

    do{
        pi_spi_receive(spi_slave, tx_buffer, 3 << 3, SPI_NO_OPTION);
    }
    while(tx_buffer[0]!=MAGIC_NUMBER_H && tx_buffer[1]!=MAGIC_NUMBER_L && tx_buffer[2]!=  CMD_STATUS_GET);

    uint8_t*buffer = pi_l2_malloc(img_w*2);

    tx_buffer[0]= MAGIC_NUMBER_H;
    tx_buffer[1]= MAGIC_NUMBER_L;
    tx_buffer[2]= STATUS_RDY;
    pi_spi_send(spi_slave, tx_buffer, 3 << 3, SPI_NO_OPTION);
    uint16_t* img_size = (uint16_t*) tx_buffer;
    img_size[0] = img_h;
    img_size[1] = img_w;
    pi_spi_send(spi_slave, img_size,   2 << 3, SPI_NO_OPTION);
    pi_spi_send(spi_slave, img_size+1, 2 << 3, SPI_NO_OPTION);
    
    for(int i=0;i<img_h;i++) 
    {
        pi_ram_read(ram, img+(i*img_w*2), buffer, (uint32_t) img_w*2);
        for (int a = 0; a < img_w; a++) { // Put pixels on 8 bits instead of 10 to go on 1 byte encoding only
                // Shifts bits to delete the 2 LSB, on the 10 useful bits
                buffer[a] = (buffer[a*2+1] << 6) | (buffer[a*2] >> 2);
        }

        pi_evt_sig_init(&end_task);
        pi_spi_send_async(spi_slave, buffer, (img_w) << 3, SPI_NO_OPTION, pi_evt_callback_irq_init(&send_task, &send_callback, &end_task));
        pi_evt_wait(&end_task);

    }


    pi_l2_free(buffer,img_w*2);
}