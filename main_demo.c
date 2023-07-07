
/*
 * Copyright (C) 2017 GreenWaves Technologies
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 */


/* Autotiler includes. */
#include "main.h"
#include "spi_comm.h"
#include "ISP_Kernels.h"

#define STREAM_OVER_UART

// parameters needed for decoding layer
// !!! do not forget to change the stride sizes accordint to the input size !!!  
tTuple feature_maps[STRIDE_SIZE] = {{30.0, 40.0}, {15.0, 20.0}, {8.0, 10.0}};
float strides[STRIDE_SIZE] = {8.0, 16.0, 32.0};

// parameters needed for postprocessing layer
unsigned int * num_val_boxes;

// parameters needed for function to_boxes
Box bboxes[top_k_boxes];

// parameters needed for nms
int final_valid_boxes;

// cycles count variables
unsigned int performances[8];


AT_DEFAULTFLASH_EXT_ADDR_TYPE main_L3_Flash = 0;


/* Inputs */
/* Outputs */
L2_MEM float Output_1[9480];

//pi_device_t Ram;
AT_DEFAULTRAM_T DefaultRam;
static struct pi_default_ram_conf ram_conf;
uint8_t ext_ram_buf;

static pi_evt_t ctrl_tasks[2];
static pi_evt_t ram_tasks[2];
static int remaining_size;
static int saved_size = 0;
static volatile int done = 0;
static int nb_transfers = 0;
static int count_transfers = 0;
static unsigned char current_buff = 0;
static int current_task = 0;
static int current_size[2];
static void handle_transfer_end(void *arg);
static void handle_ram_end(void *arg);
PI_L2 unsigned char *iter_buff[2];

// 2 Rows
#define ITER_SIZE (W_CAM*2*BYTES_CAM)
#define RAW_SIZE (W_CAM*H_CAM*BYTES_CAM) // For now only 10 bits config works

static pi_evt_t proc_task;

pi_device_t* camera;
PI_L2 unsigned char *buff[2];

static void cluster()
{
    PRINTF("\t\t***Start CLUSTER ***\n");
    mainCNN(Output_1);
}


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

extern L1_CL_MEM AT_L1_POINTER DeMosaic_L1_Memory;
extern L2_MEM AT_L2_POINTER DeMosaic_L2_Memory;

static void pre_filtering(){
    unsigned char * ram_pointer= (unsigned char *) ext_ram_buf;
    
    demosaic_image_HWC_L3((unsigned char *)main_L2_Memory_Dyn, (unsigned char *)ram_pointer);
    
    white_balance_HWC_L3((unsigned char *)ram_pointer,(unsigned char *) ram_pointer,  95);
}

int test_main(void)
{
    PRINTF("Entering main controller\n");

    /* Configure And open cluster. */
    struct pi_device cluster_dev;
    struct pi_cluster_conf cl_conf;
    pi_cluster_conf_init(&cl_conf);
    cl_conf.cc_stack_size = STACK_SIZE;

    cl_conf.id = 0; /* Set cluster ID. */
                    // Enable the special icache for the master core
    cl_conf.icache_conf = PI_CLUSTER_MASTER_CORE_ICACHE_ENABLE |
                    // Enable the prefetch for all the cores, it's a 9bits mask (from bit 2 to bit 10), each bit correspond to 1 core
                    PI_CLUSTER_ICACHE_PREFETCH_ENABLE |
                    // Enable the icache for all the cores
                    PI_CLUSTER_ICACHE_ENABLE;

    pi_open_from_conf(&cluster_dev, (void *) &cl_conf);
    if (pi_cluster_open(&cluster_dev))
    {
        PRINTF("Cluster open failed !\n");
        pmsis_exit(-4);
    }


    if(FREQ_FC >240 || FREQ_CL >240 || FREQ_PE >240)
        pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP, 800);

    /* Frequency Settings: defined in the Makefile */
    int cur_fc_freq = pi_freq_set(PI_FREQ_DOMAIN_FC, FREQ_FC*1000*1000);
    int cur_cl_freq = pi_freq_set(PI_FREQ_DOMAIN_CL, FREQ_CL*1000*1000);
    int cur_pe_freq = pi_freq_set(PI_FREQ_DOMAIN_PERIPH, FREQ_PE*1000*1000);
    if (cur_fc_freq == -1 || cur_cl_freq == -1 || cur_pe_freq == -1)
    {
        PRINTF("Error changing frequency !\nTest failed...\n");
        pmsis_exit(-4);
    }
    if(FREQ_FC <=240 && FREQ_CL <=240 && FREQ_PE <=240)
        pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP, 650);
    
    PRINTF("FC Frequency as %d Hz, CL Frequency = %d Hz, PERIIPH Frequency = %d Hz\n", 
            pi_freq_get(PI_FREQ_DOMAIN_FC), pi_freq_get(PI_FREQ_DOMAIN_CL), pi_freq_get(PI_FREQ_DOMAIN_PERIPH));

    // IMPORTANT - MUST BE CALLED AFTER THE CLUSTER IS SWITCHED ON!!!!
    PRINTF("Constructor\n");
    int ConstructorErr = mainCNN_Construct();
    if (ConstructorErr)
    {
        PRINTF("Graph constructor exited with error: %d\n(check the generated file mainKernels.c to see which memory have failed to be allocated)\n", ConstructorErr);
        pmsis_exit(-6);
    }
    
    struct pi_cluster_task task;
    pi_cluster_task(&task, (void (*)(void *))cluster, NULL);
    pi_cluster_task_stacks(&task, NULL, SLAVE_STACK_SIZE);

    
    #ifdef STREAM_OVER_UART
    pi_device_t uart_dev;
    init_uart_communication(&uart_dev,1000000);
    #else
    // Initialize SPI
    pi_device_t spi_slave;
    struct pi_spi_conf spi_slave_conf;
    spi_slave_init(&spi_slave, &spi_slave_conf);
    #endif


    //Open camera
    if (pi_open(PI_CAMERA_HM0360, &camera))
    {
        PRINTF("Failed to open camera\n");
        return -1;
    }
    PRINTF("Turning camera on...\n");
    //turn on camera
    pi_camera_control(camera, PI_CAMERA_CMD_ON, 0);
    // Allocate ping pong buffers for Camera read
    iter_buff[0] = pi_l2_malloc(ITER_SIZE);
    if (iter_buff[0] == NULL) return -1;
    iter_buff[1] = pi_l2_malloc(ITER_SIZE);
    if (iter_buff[1] == NULL) return -1;

    /* Init & open ram. */
    pi_default_ram_conf_init(&ram_conf);
    pi_open_from_conf(&DefaultRam, &ram_conf);

    pi_evt_sig_init(&proc_task);
    PRINTF("open ram\n");
    if (pi_ram_open(&DefaultRam)) {
        printf("Error ram open !\n");
        pmsis_exit(-5);
    }
    PRINTF("ram opened\n");
    if (pi_ram_alloc(&DefaultRam, (uint32_t *)(uint32_t ) ext_ram_buf, H_INP * W_INP * CHANNELS) != 0) {
        printf("Failed to allocate memory in external ram (%ld bytes)\n", H_INP * W_INP * CHANNELS);
        pmsis_exit(-1);
    }

    /* Init and opend jpeg encoder */
    jpeg_encoder_t enc;

    pi_gpio_flags_e flags = PI_GPIO_OUTPUT;
    //This is CN6 pin5
    pi_gpio_pin_configure(PI_PAD_048, flags);
    //This is GPIO
    //pi_gpio_pin_configure(PI_PAD_086, flags);

    int iter=0;    
    while(1){
        //pi_gpio_pin_toggle(PI_PAD_086);
        pi_gpio_pin_toggle(PI_PAD_048);
        // remaining_size = RAW_SIZE;
        // saved_size=0;
        // nb_transfers=0;
        // count_transfers=0;
        // current_buff=0;
        // done=0;
        // current_task = 0;

        // enqueue_transfer();
        // pi_camera_control(camera, PI_CAMERA_CMD_START, 0);
        // pi_evt_wait(&proc_task);
        // pi_camera_control(camera, PI_CAMERA_CMD_STOP, 0);

        pi_camera_control(camera, PI_CAMERA_CMD_START, 0);
        pi_camera_capture(camera, (unsigned char*)main_L2_Memory_Dyn, 640*480);
        pi_camera_control(camera, PI_CAMERA_CMD_STOP, 0);

        DeMosaic_L1_Memory = main_L1_Memory;
        DeMosaic_L2_Memory = main_L2_Memory_Dyn + 640*480;

        pi_cluster_task(&task, (void (*)(void *))pre_filtering, NULL);

        pi_cluster_send_task(&cluster_dev, &task);
        
        pi_cluster_task(&task, (void (*)(void *))cluster, NULL);

        //Copy from ram to: main_L2_Memory_Dyn + (H_INP * W_INP * CHANNELS)
        //The image is saved onto External
        pi_ram_read(&DefaultRam, ext_ram_buf, main_L2_Memory_Dyn + (H_INP * W_INP * CHANNELS), (uint32_t)(H_INP * W_INP * CHANNELS));

        PRINTF("\t\t***Start FC timer***\n");
        gap_fc_starttimer();
        gap_fc_resethwtimer();
        int perf_idx = 0;

        /* ------ SLICING ------*/
        PRINTF("\t\t***Start slicing***\n");
        performances[perf_idx] = pi_time_get_us();
        slicing_hwc_channel(
            ((unsigned char *)main_L2_Memory_Dyn) + (H_INP * W_INP * CHANNELS), 
            (unsigned char *) Input_1, 
            H_INP, 
            W_INP,
            CHANNELS
            );
        performances[perf_idx] = pi_time_get_us() - performances[perf_idx]; perf_idx++;

        /* ------ INFERENCE ------*/
        PRINTF("\t\t***Call CLUSTER***\n");
        performances[perf_idx] = pi_time_get_us();
        pi_cluster_send_task_to_cl(&cluster_dev, &task);
        performances[perf_idx] = pi_time_get_us() - performances[perf_idx]; perf_idx++;


        /* ------ DECODING ------*/
        PRINTF("\t\t***Start decoding***\n");
        performances[perf_idx] = pi_time_get_us();
        decoding(
            Output_1,
            feature_maps, 
            strides, 
            STRIDE_SIZE
        );
        performances[perf_idx] = pi_time_get_us() - performances[perf_idx]; perf_idx++;

        /* ------ POST PROCESSING ------*/
        /* ------ xywh2xyxy ------*/
        PRINTF("\t\t***Start xywh2xyxy***\n");
        performances[perf_idx] = pi_time_get_us();
        xywh2xyxy(Output_1, (int) (RAWS));
        performances[perf_idx] = pi_time_get_us() - performances[perf_idx]; perf_idx++;

        /* ------ filter boxes ------*/
        PRINTF("\t\t***Start filter boxes ***\n");
        //cast model_L2_Memory_Dyn to float16
        float * main_L2_Memory_Dyn_casted = (float *) main_L2_Memory_Dyn;
        *num_val_boxes = 0;
        performances[perf_idx] = pi_time_get_us();
        filter_boxes(
            Output_1, 
            (main_L2_Memory_Dyn_casted + (RAWS * 6)), 
            CONF_THRESH, 
            RAWS, 
            num_val_boxes
            );
        performances[perf_idx] = pi_time_get_us() - performances[perf_idx]; perf_idx++;

        /* ------ conver boxes ------*/
        PRINTF("\t\t***Start conver boxes ***\n");
        performances[perf_idx] = pi_time_get_us();
        to_bboxes(
            (main_L2_Memory_Dyn_casted + (RAWS * 6)), 
            bboxes, 
            *num_val_boxes,
            top_k_boxes
            );
        performances[perf_idx] = pi_time_get_us() - performances[perf_idx]; perf_idx++;

        /* ------ nms ------*/
        PRINTF("\t\t***Start nms ***\n");
        final_valid_boxes = 0;
        performances[perf_idx] = pi_time_get_us();
        nms(
            bboxes, 
            Output_1,
            NMS_THRESH, 
            *num_val_boxes, 
            &final_valid_boxes,
            top_k_boxes
            );
        performances[perf_idx] = pi_time_get_us() - performances[perf_idx]; perf_idx++;


        //Draw rectangles and send trought UART
        pi_ram_read(&DefaultRam, ext_ram_buf, main_L2_Memory_Dyn, (uint32_t) H_INP*W_INP*3);
        draw_boxes((unsigned char *) main_L2_Memory_Dyn, Output_1, final_valid_boxes, H_INP, W_INP, 3);

        /* ------ JPEG COMPRESSION ------ */
        
        PRINTF("\t\t***Start JPEG compression ***\n");
        int bitstream_size;
        //uint8_t * jpeg_image = (uint8_t *) pi_l2_malloc(80*2048);
        uint8_t * jpeg_image = (uint8_t *)(main_L2_Memory_Dyn + H_INP*W_INP*3);
        if (jpeg_image == 0) {
            printf("Error allocating jpeg buffer\n");
            return -1;
        }
        if(final_valid_boxes){
            performances[perf_idx] = pi_time_get_us();
            jpeg_init(&enc, H_INP, W_INP, cluster_dev, main_L1_Memory);
            int jpeg_ret = compress(
                &enc,
                (uint8_t *) main_L2_Memory_Dyn,
                jpeg_image,
                &bitstream_size,
                H_INP,
                W_INP,
                CHANNELS);
            performances[perf_idx] = pi_time_get_us() - performances[perf_idx]; perf_idx++;
        }else{
            performances[perf_idx] = 0; perf_idx++;
        
        }
        #ifdef STREAM_OVER_UART
        if(final_valid_boxes)
            send_jpeg_to_uart(&uart_dev, jpeg_image, bitstream_size, performances);
        else
            send_jpeg_to_uart(&uart_dev, jpeg_image, 0, performances);
        #else
        //send_jpeg_spi(&spi_slave,jpeg_image, bitstream_size,performances);
        #endif
        iter++;
        pi_evt_sig_init(&proc_task);

    } //end of while 1
    pi_l2_free(iter_buff[0], ITER_SIZE);
    pi_l2_free(iter_buff[1], ITER_SIZE);
    jpeg_deinit(&enc);

    /* ------ END ------*/
    PRINTF("\t\t***Runner completed***\n");

    mainCNN_Destruct();

    PRINTF("Ended\n");
    pmsis_exit(0);
    return 0;
}

int main(int argc, char *argv[])
{
    PRINTF("\n\n\t *** NNTOOL main_demo ***\n\n");
    return test_main();
}
