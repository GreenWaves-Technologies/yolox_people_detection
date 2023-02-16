
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
#include "mainKernels.h"
#include "gaplib/ImgIO.h"
#include "gaplib/fs_switch.h"
#include "bsp/camera/ov5647.h"
#include "../custom_layers/draw.h"
#include "../custom_layers/slicing.h"
#include "../custom_layers/decoding.h"
#include "../custom_layers/postprocessing.h"
#include "../custom_layers/camera.h"
#include "../custom_layers/jpeg_compress.h"
#include "../custom_layers/img_flush.h"

#if SILENT
#define PRINTF(...) ((void) 0)
#else
#define PRINTF printf
#endif

// parameters needed for decoding layer
// !!! do not forget to change the stride sizes accordint to the input size !!!  
tTuple feature_maps[STRIDE_SIZE] = {{45.0, 60.0}, {23.0, 30.0}, {12.0, 15.0}};
float strides[STRIDE_SIZE] = {8.0, 16.0, 32.0};

// parameters needed for postprocessing layer
unsigned int * num_val_boxes;

// parameters needed for function to_boxes
Box bboxes[top_k_boxes];

// parameters needed for nms
int final_valid_boxes;

// cycles count variables
unsigned int slicing_cycles;
unsigned int decoding_cycles;
unsigned int xywh2xyxy_cycles;
unsigned int filter_boxes_cycles;
unsigned int bbox_cycles;
unsigned int nms_cycles;


AT_DEFAULTFLASH_EXT_ADDR_TYPE main_L3_Flash = 0;


/* Inputs */
/* Outputs */
L2_MEM float Output_1[21420];

/* Copy inputs function */
void copy_inputs() {
    int status;
#ifdef CI
    /* ------------------- reading data for test ----------------------*/

    PRINTF("\n\t\t*** READING TEST INPUT ***\n");
    status = ReadImageFromFile(
        STR(TEST_INPUT_FILE_NAME),
        W_INP, 
        H_INP, 
        CHANNELS, 
        Input_1,
        W_INP * H_INP * CHANNELS * sizeof(char), 
        IMGIO_OUTPUT_CHAR,
        0
    );
#else
    PRINTF("\n\t\t*** READING INPUT FROM PPM FILE ***\n");
    status = ReadImageFromFile(
        STR(INPUT_FILE_NAME),
        W_INP, 
        H_INP, 
        CHANNELS, 
        Input_1,
        W_INP * H_INP * CHANNELS * sizeof(char), 
        IMGIO_OUTPUT_CHAR,
        0 // transpose from HWC to CHW 
    );

    if (status != 0) {
        PRINTF("Error reading image from file %s (error: %d) \n", STR(INPUT_FILE_NAME), status);
        exit(-1);
    } 
#endif 
}


void ci_output_test(float * model_output, char * GT_file_name, float * GT_buffer){

    switch_fs_t fs;
    __FS_INIT(fs); 

    void *File_GT;
    int ret_GT = 0;

    File_GT = __OPEN_READ(fs, GT_file_name);
    ret_GT = __READ(File_GT, GT_buffer, CI_TEST_BOX_NUM * OUTPUT_BOX_SIZE * sizeof(float));

    __CLOSE(File_GT);
    __FS_DEINIT(fs);

    //check the difference between the model output and the ground truth
    float diff = 0;
    for (int i = 0; i < CI_TEST_BOX_NUM * OUTPUT_BOX_SIZE; i++){
        diff += Abs(model_output[i] - GT_buffer[i]);
    }

    if (diff > 0.01){
        PRINTF("CI test failed, the difference between the model output and the ground truth is %f\n", diff);
        exit(-1);
    }
    else{
        PRINTF("CI test passed, the difference between the model output and the ground truth is %f\n", diff);
    }
}



/* Copy inputs function */
void write_outputs() {

#ifdef CI
    PRINTF("\t\t***Start CI output test***\n");
    char GT_file[] = STR(TEST_OUTPUT_FILE_NAME); 
    ci_output_test(Output_1, GT_file, (float *) main_L2_Memory_Dyn);

#else
    /* ------ SAVE ------*/
    PRINTF("\t\t***Start saving output***\n");

    switch_fs_t fs;
    __FS_INIT(fs);

    void *File_Output_1;
    int ret_Output_1 = 0;

    File_Output_1 = __OPEN_WRITE(fs, STR(OUTPUT_BIN_FILE_NAME));
    ret_Output_1 = __WRITE(File_Output_1, Output_1, final_valid_boxes * 7 * sizeof(float));

    __CLOSE(File_Output_1);
    __FS_DEINIT(fs);

#endif

}


static void cluster()
{
    #ifdef PERF
    PRINTF("\t\t***Start CLUSTER timer***\n");
    gap_cl_starttimer();
    gap_cl_resethwtimer();
    #endif

    mainCNN(Output_1);
}


static int open_camera(struct pi_device *device)
{
    PRINTF("Opening CSI2 camera\n");

    struct pi_ov5647_conf cam_conf;
    pi_ov5647_conf_init(&cam_conf);

    cam_conf.format=PI_CAMERA_QVGA;
    pi_open_from_conf(device, &cam_conf);
    if (pi_camera_open(device))
        return -1;

    return 0;
}

uint8_t UART_START_COM[] = {0xF1,0x1F};

void init_uart_communication(pi_device_t* uart_dev,uint32_t baudrate ){
    pi_pad_function_set(PI_PAD_065, PI_PAD_FUNC0);
    pi_pad_function_set(PI_PAD_066, PI_PAD_FUNC0);

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

void send_image_to_uart(pi_device_t* uart_dev,uint8_t* img,int img_w,int img_h,int pixel_size){

    pi_uart_write(uart_dev,UART_START_COM,2);
    //Write Image row by row
    for(int i=0;i<img_h;i++) pi_uart_write(uart_dev,&(img[i*img_w*pixel_size]),img_w*pixel_size);
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

    /* Frequency Settings: defined in the Makefile */
    int cur_fc_freq = pi_freq_set(PI_FREQ_DOMAIN_FC, FREQ_FC*1000*1000);
    int cur_cl_freq = pi_freq_set(PI_FREQ_DOMAIN_CL, FREQ_CL*1000*1000);
    int cur_pe_freq = pi_freq_set(PI_FREQ_DOMAIN_PERIPH, FREQ_PE*1000*1000);
    if (cur_fc_freq == -1 || cur_cl_freq == -1 || cur_pe_freq == -1)
    {
        PRINTF("Error changing frequency !\nTest failed...\n");
        pmsis_exit(-4);
    }
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

    /*
     * Put here Your input settings
    */
    #if defined CI || defined INFERENCE
    copy_inputs();
    #elif defined DEMO

    pi_device_t uart_dev;
    init_uart_communication(&uart_dev,3000000);
    
    //Open camera
    struct pi_device camera;
    pi_evt_t cam_task;

    if (open_camera(&camera))
    {
        printf("Failed to open camera\n");
        return -1;
    }
    //turn on camera
    pi_camera_control(&camera, PI_CAMERA_CMD_ON, 0);

    uint8_t * cam_image = (uint8_t *) Input_1; // Input_1 is the buffer that is sent to the cluster
    if(cam_image==NULL){
        printf("Error allocating image");
        return -1;
    }

    //Loop through images coming from camera
    while(1){
        pi_camera_control(&camera, PI_CAMERA_CMD_START, 0);
        pi_camera_capture_async(&camera, cam_image, H_CAM*W_CAM*BYTES_CAM, pi_evt_sig_init(&cam_task));
        pi_evt_wait(&cam_task);
        pi_camera_control(&camera, PI_CAMERA_CMD_STOP, 0);

        // shift image bit by LSB to make 8 bit image from 10 bit 
        shift_bits(cam_image, H_CAM, W_CAM);

    #endif


    #ifdef PERF
    PRINTF("\t\t***Start FC timer***\n");
    gap_fc_starttimer();
    gap_fc_resethwtimer();
    #endif

    /* ------ SLICING ------*/
    PRINTF("\t\t***Start slicing***\n");
    slicing_cycles = gap_fc_readhwtimer();

    #ifdef DEMO

    char * cam_image = (char *) cam_image; //TODO: check if pointer points to Input_1 
    slicing_hwc_channel_less_buffer(
        cam_image, 
        main_L2_Memory_Dyn + (H_INP * W_INP * CHANNELS),  // buffer used for slicing 
        H_INP, 
        W_INP,
        CHANNELS
        );
    #else

    slicing_hwc_channel_less_buffer(
        Input_1, 
        main_L2_Memory_Dyn + (H_INP * W_INP * CHANNELS), 
        H_INP,
        W_INP,
        CHANNELS
    );
    #endif

    slicing_cycles = gap_fc_readhwtimer() - slicing_cycles;


    /* ------ INFERENCE ------*/
    PRINTF("\t\t***Call CLUSTER***\n");
    pi_cluster_send_task_to_cl(&cluster_dev, &task);

    /* ------ DECODING ------*/
    PRINTF("\t\t***Start decoding***\n");
    decoding_cycles = gap_fc_readhwtimer();
    decoding(
        Output_1,
        feature_maps, 
        strides, 
        STRIDE_SIZE
    );
    decoding_cycles = gap_fc_readhwtimer() - decoding_cycles;


    /* ------ POST PROCESSING ------*/
    /* ------ xywh2xyxy ------*/
    PRINTF("\t\t***Start xywh2xyxy***\n");
    xywh2xyxy_cycles = gap_fc_readhwtimer();
    xywh2xyxy(Output_1, (int) (RAWS));
    xywh2xyxy_cycles = gap_fc_readhwtimer() - xywh2xyxy_cycles;

    /* ------ filter boxes ------*/
    PRINTF("\t\t***Start filter boxes ***\n");
    //cast model_L2_Memory_Dyn to float16
    float * main_L2_Memory_Dyn_casted = (float *) main_L2_Memory_Dyn;
    *num_val_boxes = 0;
    filter_boxes_cycles = gap_fc_readhwtimer();
    filter_boxes(
        Output_1, 
        (main_L2_Memory_Dyn_casted + (RAWS * 6)), 
        CONF_THRESH, 
        RAWS, 
        num_val_boxes
        );
    filter_boxes_cycles = gap_fc_readhwtimer() - filter_boxes_cycles;

    /* ------ conver boxes ------*/
    PRINTF("\t\t***Start conver boxes ***\n");
    bbox_cycles = gap_fc_readhwtimer();
    to_bboxes(
        (main_L2_Memory_Dyn_casted + (RAWS * 6)), 
        bboxes, 
        *num_val_boxes,
        top_k_boxes
        );
    bbox_cycles = gap_fc_readhwtimer() - bbox_cycles;
    
    /* ------ nms ------*/
    PRINTF("\t\t***Start nms ***\n");
    final_valid_boxes = 0;
    nms_cycles = gap_fc_readhwtimer();
    nms(
        bboxes, 
        Output_1,
        NMS_THRESH, 
        *num_val_boxes, 
        &final_valid_boxes,
        top_k_boxes
        );
    nms_cycles = gap_fc_readhwtimer() - nms_cycles;

    #ifdef DEMO 

    //Draw rectangles and send trought UART
    draw_boxes(cam_image, Output_1, final_valid_boxes, H_INP, W_INP, CHANNELS);
    send_image_to_uart(&uart_dev,cam_image,W_CAM,H_CAM,1);

    } //end of while 1
    #endif

    #ifdef INFERENCE
        /* ------ DRAW REATANGLES ------*/
        // first read image
        if (ReadImageFromFile(
            STR(INPUT_FILE_NAME),
            W_INP, 
            H_INP, 
            CHANNELS, 
            (unsigned char *) main_L2_Memory_Dyn_casted,
            W_INP * H_INP * CHANNELS * sizeof(char), 
            IMGIO_OUTPUT_CHAR,
            0) != 0){ 
            PRINTF("Error reading image\n");
            }

        PRINTF("\t\t***Start draw reactangles ***\n");
        draw_boxes(
            (unsigned char *) main_L2_Memory_Dyn_casted,
            Output_1,
            final_valid_boxes,
            H_INP,
            W_INP,
            CHANNELS);

        #ifdef COMPRESS 
        /* ------ JPEG COMPRESSION ------ */
        PRINTF("\t\t***Start JPEG compression ***\n");

        int bitstream_size;
        char * jpeg_image;
        jpeg_image = compress(
            (uint8_t *) main_L2_Memory_Dyn_casted, 
            &bitstream_size,
            H_INP,
            W_INP,
            CHANNELS);
        
        /* ------ FLUSH COMPRESSED IMAGE  ------ */
        PRINTF("\t\t***Start flushing compressed image ***\n");

        if (flush_iamge(jpeg_image, bitstream_size)){
            PRINTF("Error flushing image\n");
            return -1;
        }

        #else 
        /* ------ WRITE IMAGE -------- */
        int status = WriteImageToFile(
            STR(OUTPUT_FILE_NAME),
            W_INP, 
            H_INP, 
            CHANNELS, 
            (unsigned char *) main_L2_Memory_Dyn_casted,
            RGB888_IO // GRAY_SCALE_IO
        ); 
        #endif

    #endif
    /* ------ END ------*/
    PRINTF("\t\t***Runner completed***\n");

    write_outputs();

    mainCNN_Destruct();

#ifdef PERF
    {
      unsigned int TotalCycles = 0, TotalOper = 0;
      printf("\n");
      for (unsigned int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
        TotalCycles += AT_GraphPerf[i]; TotalOper += AT_GraphOperInfosNames[i];
      }
      TotalCycles += slicing_cycles + decoding_cycles + xywh2xyxy_cycles + filter_boxes_cycles + bbox_cycles + nms_cycles;
      for (unsigned int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
        printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", AT_GraphNodeNames[i], AT_GraphPerf[i], 100*((float) (AT_GraphPerf[i]) / TotalCycles), AT_GraphOperInfosNames[i], 100*((float) (AT_GraphOperInfosNames[i]) / TotalOper), ((float) AT_GraphOperInfosNames[i])/ AT_GraphPerf[i]);
      }
        printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", "Slicing", slicing_cycles, 100 * ((float) (slicing_cycles) / TotalCycles), NULL, NULL, NULL);
        
        // decoding cycles
        printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", "Decoding", decoding_cycles, 100 * ((float) (decoding_cycles) / TotalCycles), NULL, NULL, NULL);

        // xywh2xyxy cycles
        printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", "xywh2xyxy", xywh2xyxy_cycles, 100 * ((float) (xywh2xyxy_cycles) / TotalCycles), NULL, NULL, NULL);

        // filter boxes cycles
        printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", "Filter boxes", filter_boxes_cycles, 100 * ((float) (filter_boxes_cycles) / TotalCycles), NULL, NULL, NULL);

        // bbox cycles
        printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", "seq2bboxes", bbox_cycles, 100 * ((float) (bbox_cycles) / TotalCycles), NULL, NULL, NULL);

        // nms cycles
        printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", "NMS", nms_cycles, 100 * ((float) (nms_cycles) / TotalCycles), NULL, NULL, NULL);

        printf("\n");
        printf("%45s: Cycles: %12u, Cyc%%: 100.0%%, Operations: %12u, Op%%: 100.0%%, Operations/Cycle: %f\n", "Total Inference", TotalCycles, TotalOper, ((float) TotalOper)/ TotalCycles);
        printf("\n");
    }
#endif

    PRINTF("Ended\n");
    pmsis_exit(0);
    return 0;
}

int main(int argc, char *argv[])
{
    PRINTF("\n\n\t *** NNTOOL main Example ***\n\n");
    return test_main();
}