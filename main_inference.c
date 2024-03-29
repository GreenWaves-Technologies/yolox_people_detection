
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
unsigned int slicing_cycles      = 0;
unsigned int jpeg_cycles         = 0;
unsigned int decoding_cycles     = 0;
unsigned int xywh2xyxy_cycles    = 0;
unsigned int filter_boxes_cycles = 0;
unsigned int bbox_cycles         = 0;
unsigned int nms_cycles          = 0;


AT_DEFAULTFLASH_EXT_ADDR_TYPE main_L3_Flash = 0;


/* Inputs */
/* Outputs */
L2_MEM float Output_1[9480];

/* Copy inputs function */
void copy_inputs() {
    int status;
    PRINTF("\n\t\t*** READING INPUT FROM PPM FILE ***\n");
    status = ReadImageFromFile(
        STR(INPUT_FILE_NAME),
        W_INP, 
        H_INP, 
        CHANNELS, 
        main_L2_Memory_Dyn + (H_INP * W_INP * CHANNELS),
        W_INP * H_INP * CHANNELS * sizeof(char), 
        IMGIO_OUTPUT_CHAR,
        0 // transpose from HWC to CHW 
    );

    if (status != 0) {
        PRINTF("Error reading image from file %s (error: %d) \n", STR(INPUT_FILE_NAME), status);
        exit(-1);
    }
}


void ci_output_test(float * model_output, char * GT_file_name, float * GT_buffer, int n_valid_boxes){

    switch_fs_t fs;
    __FS_INIT(fs); 

    void *File_GT;
    int ret_GT = 0;

    File_GT = __OPEN_READ(fs, GT_file_name);
    ret_GT = __READ(File_GT, GT_buffer, n_valid_boxes * CI_BOX_TYPE_SIZE * sizeof(float));

    __CLOSE(File_GT);
    __FS_DEINIT(fs);

    //check the difference between the model output and the ground truth
    float diff = 0;
    for (int i = 0; i < n_valid_boxes; i++){
        for (int j = 0; j < CI_BOX_TYPE_SIZE; j++) {
            diff += Abs(model_output[i * 7 + j] - GT_buffer[i * CI_BOX_TYPE_SIZE + j]);
            // printf("[%d %d] %10f - %10f\n", i, j, model_output[i * 7 + j], GT_buffer[i * CI_BOX_TYPE_SIZE + j]);
        }
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
#ifdef CI
    PRINTF("\t\t***Start CI output test***\n");
    char GT_file[] = STR(TEST_OUTPUT_FILE_NAME); 
    ci_output_test(Output_1, GT_file, (float *) main_L2_Memory_Dyn, final_valid_boxes);
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
    // for (int i=0; i<1580; i++) {
    //     printf("[%d] %.2f %.2f %.2f %.2f %.2f %.2f\n", i,  Output_1[i*6+0], Output_1[i*6+1], Output_1[i*6+2], Output_1[i*6+3], Output_1[i*6+4], Output_1[i*6+5]);
    // }
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
    printf("FC Frequency as %d Hz, CL Frequency = %d Hz, PERIIPH Frequency = %d Hz\n", 
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
    copy_inputs();

    #ifdef PERF
    PRINTF("\t\t***Start FC timer***\n");
    gap_fc_starttimer();
    gap_fc_resethwtimer();
    #endif

    /* ------ SLICING ------*/
    PRINTF("\t\t***Start slicing***\n");
    slicing_cycles = gap_fc_readhwtimer();

    slicing_hwc_channel(
        (unsigned char *) main_L2_Memory_Dyn + (H_INP * W_INP * CHANNELS), 
        Input_1, 
        H_INP, 
        W_INP,
        CHANNELS
        );

    slicing_cycles = gap_fc_readhwtimer() - slicing_cycles;
    // for (int j=0; j<H_INP/2; j++) {
    //     for (int i=0; i<W_INP/2; i++) {
    //         for (int c=0; c<12; c++) printf("%d, ", Input_1[j*W_INP/2*12 + i*12 + c]);
    //         printf("\n");
    //     }
    // }

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

    // printf("Bounding boxes after filter\n");
    // float *filtered_boxes = (float *) (main_L2_Memory_Dyn_casted + (RAWS * 6));
    // for (unsigned int i=0; i<*num_val_boxes; i++) {
    //     printf("[%d] %.2f %.2f %.2f %.2f %.2f %.2f\n", i,  filtered_boxes[7*i], filtered_boxes[7*i+1], filtered_boxes[7*i+2], filtered_boxes[7*i+3], filtered_boxes[7*i+4], filtered_boxes[7*i+5]);
    // }

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

    printf("Bounding boxes after NMS\n");
    for (int i=0; i<final_valid_boxes; i++) {
        printf("[%d] %.2f %.2f %.2f %.2f %.2f %.2f\n", i,  Output_1[i*7], Output_1[i*7+1], Output_1[i*7+2], Output_1[i*7+3], Output_1[i*7+4], Output_1[i*7+5]);
    }

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
    jpeg_encoder_t enc;
    uint8_t * jpeg_image = (uint8_t *) pi_l2_malloc(30*2048);
    if (jpeg_image == 0) {
        printf("Error allocating jpeg buffer\n");
        return -1;
    }
    jpeg_cycles = gap_fc_readhwtimer();
    jpeg_init(&enc, H_INP, W_INP, cluster_dev, main_L1_Memory);
    int jpeg_ret = compress(
        &enc,
        (uint8_t *) main_L2_Memory_Dyn,
        jpeg_image,
        &bitstream_size,
        H_INP,
        W_INP,
        CHANNELS);
    jpeg_cycles = gap_fc_readhwtimer() - jpeg_cycles;
    
    /* ------ FLUSH COMPRESSED IMAGE  ------ */
    PRINTF("\t\t***Start flushing compressed image ***\n");

    if (write_jpeg_to_file(jpeg_image, STR(OUTPUT_JPEG_FILE_NAME), bitstream_size)){
        PRINTF("Error flushing image\n");
        return -1;
    }
    pi_l2_free(jpeg_image, 30*2048);
    jpeg_deinit(&enc);

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

    /* ------ END ------*/
    PRINTF("\t\t***Runner completed***\n");

#ifdef PERF
    {
        unsigned int NNCycles = 0, TotalCycles = 0, NNOper = 0, TotalOper = 0;
        printf("\n");
        for (unsigned int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
            NNCycles += AT_GraphPerf[i]; NNOper += AT_GraphOperInfosNames[i];
        }

        TotalOper += NNOper;
        TotalCycles += NNCycles + slicing_cycles + decoding_cycles + xywh2xyxy_cycles + filter_boxes_cycles + bbox_cycles + nms_cycles + jpeg_cycles;
        // for (unsigned int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
        //   printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", AT_GraphNodeNames[i], AT_GraphPerf[i], 100*((float) (AT_GraphPerf[i]) / TotalCycles), AT_GraphOperInfosNames[i], 100*((float) (AT_GraphOperInfosNames[i]) / TotalOper), ((float) AT_GraphOperInfosNames[i])/ AT_GraphPerf[i]);
        // }

        // Slicing
        printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", "Slicing", slicing_cycles, 100 * ((float) (slicing_cycles) / TotalCycles), 0, 0.0f, 0.0f, 0, 0.0f, 0.0f, 0, 0.0f, 0.0f);
        // NN
        printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", "NN", NNCycles, 100 * ((float) (NNCycles) / TotalCycles), NNOper, 100*((float) (NNOper) / TotalOper), ((float) NNOper)/ NNCycles);
        // decoding cycles
        printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", "Decoding", decoding_cycles, 100 * ((float) (decoding_cycles) / TotalCycles), 0, 0.0f, 0.0f, 0, 0.0f, 0.0f, 0, 0.0f, 0.0f);
        // xywh2xyxy cycles
        printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", "xywh2xyxy", xywh2xyxy_cycles, 100 * ((float) (xywh2xyxy_cycles) / TotalCycles), 0, 0.0f, 0.0f, 0, 0.0f, 0.0f, 0, 0.0f, 0.0f);
        // filter boxes cycles
        printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", "Filter boxes", filter_boxes_cycles, 100 * ((float) (filter_boxes_cycles) / TotalCycles), 0, 0.0f, 0.0f, 0, 0.0f, 0.0f, 0, 0.0f, 0.0f);
        // bbox cycles
        printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", "seq2bboxes", bbox_cycles, 100 * ((float) (bbox_cycles) / TotalCycles), 0, 0.0f, 0.0f, 0, 0.0f, 0.0f, 0, 0.0f, 0.0f);
        // nms cycles
        printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", "NMS", nms_cycles, 100 * ((float) (nms_cycles) / TotalCycles), 0, 0.0f, 0.0f, 0, 0.0f, 0.0f, 0, 0.0f, 0.0f);
        if (jpeg_cycles)
            printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", "JPEG Compress", jpeg_cycles, 100 * ((float) (jpeg_cycles) / TotalCycles), 0, 0.0f, 0.0f, 0, 0.0f, 0.0f, 0, 0.0f, 0.0f);


        printf("\n");
        printf("%45s: Cycles: %12u, Cyc%%: 100.0%%, Operations: %12u, Op%%: 100.0%%, Operations/Cycle: %f\n", "Total Inference", TotalCycles, TotalOper, ((float) TotalOper)/ TotalCycles);
        printf("\n");
    }
#endif
    write_outputs();

    mainCNN_Destruct();

    PRINTF("Ended\n");
    pmsis_exit(0);
    return 0;
}

int main(int argc, char *argv[])
{
    PRINTF("\n\n\t *** NNTOOL main_inference ***\n\n");
    return test_main();
}
