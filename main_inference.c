
/*
 * Copyright (C) 2017 GreenWaves Technologies
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 */


/* Autotiler includes. */
#include "yolox.h"

// parameters needed for decoding layer
// !!! do not forget to change the stride sizes accordint to the input size !!!  
tTuple feature_maps[STRIDE_SIZE] = {{30.0, 40.0}, {15.0, 20.0}, {8.0, 10.0}};
float strides[STRIDE_SIZE] = {8.0, 16.0, 32.0};

// cycles count variables
unsigned int performances[6] = {0};

AT_DEFAULTFLASH_EXT_ADDR_TYPE yolox_L3_Flash = 0;

/* Outputs */
L2_MEM float Output_1[9480];

/* Copy inputs function */
void ReadInput(unsigned char *input_ptr) {
    int status;
    PRINTF("\n\t\t*** READING INPUT FROM PPM FILE ***\n");
    status = ReadImageFromFile(
        STR(INPUT_FILE_NAME),
        W_INP, 
        H_INP, 
        CHANNELS, 
        input_ptr,
        W_INP * H_INP * CHANNELS * sizeof(char), 
        IMGIO_OUTPUT_CHAR,
        0 // transpose from HWC to CHW 
    );

    if (status != 0) {
        PRINTF("Error reading image from file %s (error: %d) \n", STR(INPUT_FILE_NAME), status);
        exit(-1);
    }
}

void TestOutputCI(
    float * PredictedBoxes,
    char * GTFilePath,
    float * GTTmpBuffer,
    int NValidBoxes
)
{

    switch_fs_t fs;
    __FS_INIT(fs); 

    void *File_GT;
    int ret_GT = 0;

    File_GT = __OPEN_READ(fs, GTFilePath);
    // The first thing in the GTFile is the number of expected Boxes
    ret_GT = __READ(File_GT, GTTmpBuffer, sizeof(float));
    int GtNumBoxes = (int) GTTmpBuffer[0];
    if (GtNumBoxes != NValidBoxes) {
        printf("Error (GT N boxes: %d, PredictedNBoxes: %d)\n", GtNumBoxes, NValidBoxes);
    }
    // Then you have the boxes
    ret_GT = __READ(File_GT, GTTmpBuffer, GtNumBoxes * CI_BOX_TYPE_SIZE * sizeof(float));

    __CLOSE(File_GT);
    __FS_DEINIT(fs);

    //check the difference between the model output and the ground truth
    float diff = 0;
    for (int i = 0; i < NValidBoxes; i++){
        for (int j = 0; j < CI_BOX_TYPE_SIZE; j++) {
            diff += Abs(PredictedBoxes[i * 7 + j] - GTTmpBuffer[i * CI_BOX_TYPE_SIZE + j]);
            // printf("[%d %d] %10f - %10f\n", i, j, PredictedBoxes[i * 7 + j], GTTmpBuffer[i * CI_BOX_TYPE_SIZE + j]);
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
void WriteResToFileAndTest(float *OutBoxesPtr, int NBoxes, char *OutFilePath) {

    /* ------ SAVE ------*/
    PRINTF("\t\t***Start saving output***\n");

    switch_fs_t fs;
    __FS_INIT(fs);

    void *File;
    int ret = 0;

    File = __OPEN_WRITE(fs, OutFilePath);
    ret = __WRITE(File, OutBoxesPtr, NBoxes * 7 * sizeof(float));

    __CLOSE(File);
    __FS_DEINIT(fs);
#ifdef CI
    PRINTF("\t\t***Start CI output test***\n");
    char GT_file[] = STR(TEST_OUTPUT_FILE_NAME); 
    TestOutputCI(OutBoxesPtr, GT_file, (float *) yolox_L2_Memory_Dyn, NBoxes);
#endif
}


static void NNInferenceCluster()
{
    #ifdef PERF
    gap_cl_starttimer();
    gap_cl_resethwtimer();
    #endif

    yoloxCNN(Output_1);
    // for (int i=0; i<1580; i++) {
    //     printf("[%d] %.2f %.2f %.2f %.2f %.2f %.2f\n", i,  Output_1[i*6+0], Output_1[i*6+1], Output_1[i*6+2], Output_1[i*6+3], Output_1[i*6+4], Output_1[i*6+5]);
    // }
}

int main(int argc, char *argv[])
{
    PRINTF("\n\n\t *** NNTOOL main_inference ***\n\n");

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
        return -4;
    }

    /* Frequency Settings: defined in the Makefile */
    int cur_fc_freq = pi_freq_set(PI_FREQ_DOMAIN_FC, FREQ_FC*1000*1000);
    int cur_cl_freq = pi_freq_set(PI_FREQ_DOMAIN_CL, FREQ_CL*1000*1000);
    int cur_pe_freq = pi_freq_set(PI_FREQ_DOMAIN_PERIPH, FREQ_PE*1000*1000);
    if (cur_fc_freq == -1 || cur_cl_freq == -1 || cur_pe_freq == -1)
    {
        PRINTF("Error changing frequency !\nTest failed...\n");
        return -4;
    }
    printf("FC Frequency as %d Hz, CL Frequency = %d Hz, PERIIPH Frequency = %d Hz\n", 
            pi_freq_get(PI_FREQ_DOMAIN_FC), pi_freq_get(PI_FREQ_DOMAIN_CL), pi_freq_get(PI_FREQ_DOMAIN_PERIPH));

    // IMPORTANT - MUST BE CALLED AFTER THE CLUSTER IS SWITCHED ON!!!!
    PRINTF("Constructor\n");
    int ConstructorErr = yoloxCNN_Construct();
    if (ConstructorErr)
    {
        PRINTF("Graph constructor exited with error: %d\n(check the generated file mainKernels.c to see which memory have failed to be allocated)\n", ConstructorErr);
        return -6;
    }
    
    struct pi_cluster_task task;
    pi_cluster_task(&task, (void (*)(void *))NNInferenceCluster, NULL);
    pi_cluster_task_stacks(&task, NULL, SLAVE_STACK_SIZE);

    /* --------- READ Input ---------*/
    ReadInput((unsigned char *) Input_1);

    #ifdef PERF
    PRINTF("\t\t***Start FC timer***\n");
    gap_fc_starttimer();
    gap_fc_resethwtimer();
    #endif

    /* --------- INFERENCE ---------*/
    PRINTF("\t\t***Call CLUSTER***\n");
    unsigned int start = gap_fc_readhwtimer();
    pi_cluster_send_task_to_cl(&cluster_dev, &task);
    performances[0] = gap_fc_readhwtimer() - start;

    /* ------ POST PROCESSING ------*/
    /* ---- DECODING (inplace) -----*/
    PRINTF("\t\t***Start decoding***\n");
    start = gap_fc_readhwtimer();
    decoding_and_xywh2xyxy(
        Output_1,
        feature_maps, 
        strides, 
        STRIDE_SIZE
    );
    performances[1] = gap_fc_readhwtimer() - start;

    /* ------ filter boxes ------ */
    PRINTF("\t\t***Start filter boxes ***\n");
    // cast model_L2_Memory_Dyn to float
    float * filtered_boxes = (float *) yolox_L2_Memory_Dyn;
    start = gap_fc_readhwtimer();
    int num_val_boxes = filter_boxes(
        Output_1,
        filtered_boxes, 
        CONF_THRESH, 
        RAWS
    );
    performances[2] = gap_fc_readhwtimer() - start;

    /* ------ For debugging ------*/
    // printf("Bounding boxes after filter\n");
    // for (int i=0; i<num_val_boxes; i++) {
    //     printf("[%d] %.2f %.2f %.2f %.2f %.2f %.2f\n", i,  filtered_boxes[7*i], filtered_boxes[7*i+1], filtered_boxes[7*i+2], filtered_boxes[7*i+3], filtered_boxes[7*i+4], filtered_boxes[7*i+5]);
    // }

    /* ------ Convert boxes ------*/
    PRINTF("\t\t***Start seq2boxes ***\n");
    Box *bboxes = (Box *) (filtered_boxes + (RAWS * 6));
    int nms_valid_boxes = (num_val_boxes > top_k_boxes) ? (top_k_boxes) : (num_val_boxes);
    start = gap_fc_readhwtimer();
    to_bboxes(
        filtered_boxes,
        bboxes,
        nms_valid_boxes
    );
    performances[3] = gap_fc_readhwtimer() - start;

    /* ------ nms ------*/
    PRINTF("\t\t***Start nms ***\n");
    start = gap_fc_readhwtimer();
    int final_valid_boxes = nms(
        bboxes, 
        Output_1,
        NMS_THRESH, 
        nms_valid_boxes
    );
    performances[4] = gap_fc_readhwtimer() - start;

    PRINTF("Bounding boxes after NMS\n");
    for (int i=0; i<final_valid_boxes; i++) {
        PRINTF("[%d] %.2f %.2f %.2f %.2f %.2f %.2f\n", i,  Output_1[i*7], Output_1[i*7+1], Output_1[i*7+2], Output_1[i*7+3], Output_1[i*7+4], Output_1[i*7+5]);
    }

    /* ------ DRAW REATANGLES ------*/
    // first read image
    if (ReadImageFromFile(
        STR(INPUT_FILE_NAME),
        W_INP, 
        H_INP, 
        CHANNELS, 
        (unsigned char *) yolox_L2_Memory_Dyn,
        W_INP * H_INP * CHANNELS * sizeof(char), 
        IMGIO_OUTPUT_CHAR,
        0) != 0)
    { 
        PRINTF("Error reading image\n");
    }

    PRINTF("\t\t***Start draw reactangles ***\n");
    draw_boxes(
        (unsigned char *) yolox_L2_Memory_Dyn,
        Output_1,
        final_valid_boxes,
        H_INP,
        W_INP,
        CHANNELS
    );

    #ifdef COMPRESS 
    /* ------ JPEG COMPRESSION ------ */
    PRINTF("\t\t***Start JPEG compression ***\n");

    int bitstream_size;
    jpeg_encoder_t enc;
    // uint8_t * jpeg_image = (uint8_t *) pi_l2_malloc(30*2048);
    uint8_t *jpeg_image = (uint8_t *) (yolox_L2_Memory_Dyn) + H_INP*W_INP*CHANNELS;
    start = gap_fc_readhwtimer();
    jpeg_init(&enc, H_INP, W_INP, cluster_dev, yolox_L1_Memory);
    int jpeg_ret = compress(
        &enc,
        (uint8_t *) yolox_L2_Memory_Dyn,
        jpeg_image,
        &bitstream_size,
        H_INP,
        W_INP,
        CHANNELS);
    performances[5] = gap_fc_readhwtimer() - start;

    /* ------ FLUSH COMPRESSED IMAGE  ------ */
    PRINTF("\t\t***Writing compressed image to %s ***\n", STR(OUTPUT_JPEG_FILE_NAME));

    if (WriteJPEGToFile(jpeg_image, STR(OUTPUT_JPEG_FILE_NAME), bitstream_size)){
        PRINTF("Error flushing image\n");
        return -1;
    }
    // pi_l2_free(jpeg_image, 30*2048);
    jpeg_deinit(&enc);
    #else

    /* ------ WRITE IMAGE -------- */
    int status = WriteImageToFile(
        STR(OUTPUT_FILE_NAME),
        W_INP, 
        H_INP, 
        CHANNELS, 
        (unsigned char *) yolox_L2_Memory_Dyn,
        RGB888_IO // GRAY_SCALE_IO
    );
    performances[5] = 0;
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
        TotalCycles += NNCycles + performances[1] + performances[2] + performances[3] + performances[4] + performances[5];
        // Un comment to get layer-by-layer information
        // for (unsigned int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
        //   printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", AT_GraphNodeNames[i], AT_GraphPerf[i], 100*((float) (AT_GraphPerf[i]) / TotalCycles), AT_GraphOperInfosNames[i], 100*((float) (AT_GraphOperInfosNames[i]) / TotalOper), ((float) AT_GraphOperInfosNames[i])/ AT_GraphPerf[i]);
        // }

        // NN
        printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", "NN", NNCycles, 100 * ((float) (NNCycles) / TotalCycles), NNOper, 100*((float) (NNOper) / TotalOper), ((float) NNOper)/ NNCycles);
        // decoding cycles
        printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", "Decoding", performances[1], 100 * ((float) (performances[1]) / TotalCycles), 0, 0.0f, 0.0f, 0, 0.0f, 0.0f, 0, 0.0f, 0.0f);
        // filter boxes cycles
        printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", "Filter boxes", performances[2], 100 * ((float) (performances[2]) / TotalCycles), 0, 0.0f, 0.0f, 0, 0.0f, 0.0f, 0, 0.0f, 0.0f);
        // bbox cycles
        printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", "seq2bboxes", performances[3], 100 * ((float) (performances[3]) / TotalCycles), 0, 0.0f, 0.0f, 0, 0.0f, 0.0f, 0, 0.0f, 0.0f);
        // nms cycles
        printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", "NMS", performances[4], 100 * ((float) (performances[4]) / TotalCycles), 0, 0.0f, 0.0f, 0, 0.0f, 0.0f, 0, 0.0f, 0.0f);
        if (performances[5])
            printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", "JPEG Compress", performances[5], 100 * ((float) (performances[5]) / TotalCycles), 0, 0.0f, 0.0f, 0, 0.0f, 0.0f, 0, 0.0f, 0.0f);


        printf("\n");
        printf("%45s: Cycles: %12u, Cyc%%: 100.0%%, Operations: %12u, Op%%: 100.0%%, Operations/Cycle: %f\n", "Total Inference", TotalCycles, TotalOper, ((float) TotalOper)/ TotalCycles);
        printf("\n");
    }
#endif
    WriteResToFileAndTest(Output_1, final_valid_boxes, STR(OUTPUT_BIN_FILE_NAME));

    yoloxCNN_Destruct();

    PRINTF("Ended\n");
    return 0;
}
