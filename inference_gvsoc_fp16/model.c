
/*
 * Copyright (C) 2017 GreenWaves Technologies
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 */


/* Autotiler includes. */
#include "model.h"
#include "modelKernels.h"
#include "gaplib/fs_switch.h"
#include "gaplib/ImgIO.h"
#include "slicing.h"
#include "decoding.h"
#include "postprocessing.h"
#include "draw.h"

#ifdef __EMUL__
#define pmsis_exit(n) exit(n)
#endif

#ifndef STACK_SIZE
#define STACK_SIZE      1024
#endif


// ------------------------- PARAMETERS -------------------------

//input_file_name
printf("input_file_name: %s", STR(INPUT_FILE_NAME));
printf("outptut_file_name: %s", STR(OUTPUT_FILE_NAME));

char *input_file_name   = "../../../000000001296.ppm";
char *output_file_name  = "../../../000000001296_out.ppm";

// parameters needed for decoding layer
#define STRIDE_SIZE 3
tTuple feature_maps[STACK_SIZE] = {{32.0, 40.0}, {16.0, 20.0}, {8.0, 10.0}};
float16 strides[STACK_SIZE] = {8.0, 16.0, 32.0};

// parameters needed for xywh2xyxy layer
#define RAWS 1680

// parameters needed for postprocessing layer
#define CONF_THRESH 0.30
// #define CONF_THRESH 0.02
unsigned int * num_val_boxes;

// parameters needed for function to_boxes
#define top_k_boxes 70 
Box bboxes[top_k_boxes];

// parameters needed for nms
#define NMS_THRESH 0.30
// #define NMS_THRESH 0.65
int final_valid_boxes;

// cycles count variables
unsigned int slicing_cycles;
unsigned int decoding_cycles;
unsigned int xywh2xyxy_cycles;
unsigned int filter_boxes_cycles;
unsigned int bbox_cycles;
unsigned int nms_cycles;

AT_HYPERFLASH_EXT_ADDR_TYPE model_L3_Flash = 0;

/* Inputs */
/* Outputs */
L2_MEM F16 Output_1[10080];
// ------------------------------------------------------------------------


/* Copy inputs function */
void copy_inputs() {

    // -------------------------- READ IMAGE FROM PPM FILE --------------------------
    
    printf("\n\t\t*** READING INPUT FROM PPM FILE ***\n");
    int status = ReadImageFromFile(
        input_file_name,
        W_INP, 
        H_INP, 
        CHANNELS, 
        model_L2_Memory_Dyn + (H_INP * W_INP * CHANNELS),
        W_INP * H_INP * CHANNELS * sizeof(char), 
        IMGIO_OUTPUT_CHAR,
        1
    );

    if (status != 0) {
        printf("Error reading image from file %s (error: %d) \n", input_file_name, status);
        exit(-1);
    } 



    // READ OUTPUT
    switch_fs_t fs;
    __FS_INIT(fs);

    printf("\t\t***Reading output file***\n");
    void *File_Output_1;
    int ret_Output_1 = 0;
    File_Output_1 = __OPEN_READ(fs, "../../../inputs_for_validation/000000001296.jpg.bin");
    ret_Output_1 = __READ(File_Output_1, Output_1, 10080*sizeof(F16));

    __CLOSE(File_Output_1);
    __FS_DEINIT(fs);
    
}


void draw_boxes(F16 * model_L2_Memory_Dyn_casted){

    // read image again but do not transpose it
    // cast model_L2_Memory_Dyn to back to char
    unsigned char * image = (unsigned char *) model_L2_Memory_Dyn_casted;
    int status = ReadImageFromFile(
        input_file_name,
        W_INP, 
        H_INP, 
        CHANNELS, 
        image,
        W_INP * H_INP * CHANNELS * sizeof(char), 
        IMGIO_OUTPUT_CHAR,
        0 
    );

    printf("\n");
    for (int i=0; i < final_valid_boxes; i++){

        int x1 = (int) Output_1[i*7 + 0];
        int y1 = (int) Output_1[i*7 + 1];
        int x2 = (int) Output_1[i*7 + 2];
        int y2 = (int) Output_1[i*7 + 3];
    
        float score = Output_1[i*7 + 4] * Output_1[i*7 + 5];
        int cls = (int) Output_1[i*7 + 6];

        int h = y2 - y1;
        int w = x2 - x1;
        int x = w / 2;
        int y = h / 2;
        
        draw_rectangle(image, W_INP, H_INP, x1, y1, x2, y2, 255);
    }

    /* ----------------------- SAVE IMAGE --------------------- */
    printf("\t\t***Save image***\n");
    status = WriteImageToFile(
        output_file_name,
        W_INP, 
        H_INP, 
        CHANNELS, 
        image,
        RGB888_IO
    );
}

static void cluster()
{

// ------------------------- START -------------------------

    #ifdef PERF
    printf("\t\t***Start timer***\n");
    gap_cl_starttimer();
    gap_cl_resethwtimer();
    #endif

// ------------------------- slicing -------------------------
    printf("\t\t***Start slicing***\n");

    slicing_cycles = gap_cl_readhwtimer();
    slicing_chw_channel(
        model_L2_Memory_Dyn + (H_INP * W_INP * CHANNELS), 
        Input_1, 
        H_INP, 
        W_INP,
        CHANNELS
        );
    slicing_cycles = gap_cl_readhwtimer() - slicing_cycles;

// ------------------------- INFERENCE -------------------------
    printf("\t\t***Inference***\n");
    // modelCNN(Output_1);

// ------------------------- DECODING -------------------------
    printf("\t\t***Start decoding***\n");
    decoding_cycles = gap_cl_readhwtimer();
    decoding(
        Output_1,
        feature_maps, 
        strides, 
        STRIDE_SIZE
    );
    decoding_cycles = gap_cl_readhwtimer() - decoding_cycles;
// ------------------------- POST PROCESSING -------------------------

// ------------------------- xywh2xyxy -------------------------
    printf("\t\t***Start xywh2xyxy***\n");

    xywh2xyxy_cycles = gap_cl_readhwtimer();
    xywh2xyxy(Output_1, (int) (RAWS));
    xywh2xyxy_cycles = gap_cl_readhwtimer() - xywh2xyxy_cycles;

// ------------------------- filter_boxes -------------------------
    printf("\t\t***Start filter boxes ***\n");

    //cast model_L2_Memory_Dyn to float16
    f16 * model_L2_Memory_Dyn_casted = (f16 *) model_L2_Memory_Dyn;
    *num_val_boxes = 0;
    filter_boxes_cycles = gap_cl_readhwtimer();
    filter_boxes(
        Output_1, 
        (model_L2_Memory_Dyn_casted + 10080), 
        CONF_THRESH, 
        RAWS, 
        num_val_boxes
        );
    filter_boxes_cycles = gap_cl_readhwtimer() - filter_boxes_cycles;


// ------------------------- Conver boxes -------------------------
    printf("\t\t***Start conver boxes ***\n");

    bbox_cycles = gap_cl_readhwtimer();
    to_bboxes(
        (model_L2_Memory_Dyn_casted + 10080), 
        bboxes, 
        *num_val_boxes
        );
    bbox_cycles = gap_cl_readhwtimer() - bbox_cycles;


// ------------------------- nms -------------------------
    printf("\t\t***Start nms ***\n");

    final_valid_boxes = 0;
    nms_cycles = gap_cl_readhwtimer();
    nms(
        bboxes, 
        Output_1,
        NMS_THRESH, 
        *num_val_boxes, 
        &final_valid_boxes
        );
    nms_cycles = gap_cl_readhwtimer() - nms_cycles;


// ----------------------- DRAW REACTANGLES ---------------------
    draw_boxes(model_L2_Memory_Dyn_casted);

// ------------------------- END -------------------------
    printf("\t\t***Runner completed***\n");

}

int test_model(void)
{
    printf("Entering main controller\n");
    /* ----------------> 
     * Put here Your input settings
     * <---------------
     */

#ifndef __EMUL__
    /* Configure And open cluster. */
    struct pi_device cluster_dev;
    struct pi_cluster_conf cl_conf;
    cl_conf.id = 0;
    cl_conf.cc_stack_size = STACK_SIZE;
    pi_open_from_conf(&cluster_dev, (void *) &cl_conf);
    if (pi_cluster_open(&cluster_dev))
    {
        printf("Cluster open failed !\n");
        pmsis_exit(-4);
    }

    /* Frequency Settings: defined in the Makefile */
    int cur_fc_freq = pi_freq_set(PI_FREQ_DOMAIN_FC, FREQ_FC*1000*1000);
    int cur_cl_freq = pi_freq_set(PI_FREQ_DOMAIN_CL, FREQ_CL*1000*1000);
    int cur_pe_freq = pi_freq_set(PI_FREQ_DOMAIN_PERIPH, FREQ_PE*1000*1000);
    if (cur_fc_freq == -1 || cur_cl_freq == -1 || cur_pe_freq == -1)
    {
        printf("Error changing frequency !\nTest failed...\n");
        pmsis_exit(-4);
    }
	printf("FC Frequency as %d Hz, CL Frequency = %d Hz, PERIIPH Frequency = %d Hz\n", 
            pi_freq_get(PI_FREQ_DOMAIN_FC), pi_freq_get(PI_FREQ_DOMAIN_CL), pi_freq_get(PI_FREQ_DOMAIN_PERIPH));

#endif
    

    // IMPORTANT - MUST BE CALLED AFTER THE CLUSTER IS SWITCHED ON!!!!
    printf("\t\t***Constructor***\n");
    int ConstructorErr = modelCNN_Construct();
    if (ConstructorErr)
    {
        printf("Graph constructor exited with error: %d\n(check the generated file modelKernels.c to see which memory have failed to be allocated)\n", ConstructorErr);
        pmsis_exit(-6);
    }
    

    copy_inputs();

    printf("\t\t***Call cluster***\n");
#ifndef __EMUL__
    printf("\t\t***STACK SIZE: %d \n ", STACK_SIZE);
    printf("\t\t***SLAVE STACK SIZE: %d\n", SLAVE_STACK_SIZE);
    struct pi_cluster_task task;
    pi_cluster_task(&task, (void (*)(void *))cluster, NULL);
    pi_cluster_task_stacks(&task, NULL, SLAVE_STACK_SIZE);

    pi_cluster_send_task_to_cl(&cluster_dev, &task);
#else
    cluster();
#endif

    
    printf("\t\t***Destructor***\n");
    modelCNN_Destruct();

#ifdef PERF
// #ifndef PERF
    {
        printf("\t\t***Performance***\n");
        unsigned int TotalCycles = 0, TotalOper = 0;
        printf("\n");
        for (unsigned int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
            TotalCycles += AT_GraphPerf[i]; TotalOper += AT_GraphOperInfosNames[i];
        }
        

        for (unsigned int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
            printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", AT_GraphNodeNames[i], AT_GraphPerf[i], 100*((float) (AT_GraphPerf[i]) / TotalCycles), AT_GraphOperInfosNames[i], 100*((float) (AT_GraphOperInfosNames[i]) / TotalOper), ((float) AT_GraphOperInfosNames[i])/ AT_GraphPerf[i]);
        }

        TotalCycles += 93732917;
        // slicing cycles
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

    printf("\t\t***Ended***\n");
    pmsis_exit(0);
    return 0;
}

int main(int argc, char *argv[])
{
    printf("\n\n\t *** NNTOOL model Example ***\n\n");
    #ifdef __EMUL__
    test_model();
    #else
    return pmsis_kickoff((void *) test_model);
    #endif
    return 0;
}
