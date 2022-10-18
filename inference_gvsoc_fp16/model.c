
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
#include "slicing.h"
#include "decoding.h"
#include "postprocessing.h"

#ifdef __EMUL__
#define pmsis_exit(n) exit(n)
#endif

#ifndef STACK_SIZE
#define STACK_SIZE      1024
#endif

// parameters needed for slicing layer
#define H_INP 256
#define W_INP 320
#define CHANNELS 3

// parameters needed for decoding layer
#define STRIDE_SIZE 3
tTuple feature_maps[STACK_SIZE] = {{32.0, 40.0}, {16.0, 20.0}, {8.0, 10.0}};
float16 strides[STACK_SIZE] = {8.0, 16.0, 32.0};

// parameters needed for xywh2xyxy layer
#define RAWS 1680

// parameters needed for postprocessing layer
#define CONF_THRESH 0.30
unsigned int * num_val_boxes;

// parameters needed for function to_boxes
#define top_k_boxes 70 
Box bboxes[top_k_boxes];

// parameters needed for nms
#define NMS_THRESH 0.30
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

/* Copy inputs function */
void copy_inputs() {
    switch_fs_t fs;
    __FS_INIT(fs);

    /* Reading from file Input_1 */
    void *File_Input_1;
    int ret_Input_1 = 0;
    #ifdef __EMUL__
    File_Input_1 = __OPEN_READ(fs, "Input_1.bin");
    #else
    // File_Input_1 = __OPEN_READ(fs, "../../../Input_1.bin");
    File_Input_1 = __OPEN_READ(fs, "../../../Input_1_Unsliced.bin");
    #endif

    // ret_Input_1 = __READ(File_Input_1, Input_1, 245760*sizeof(F16));
    ret_Input_1 = __READ(
        File_Input_1, 
        model_L2_Memory_Dyn + (H_INP * W_INP * CHANNELS), 
        (H_INP * W_INP * CHANNELS)*sizeof(signed char)
    );

    __CLOSE(File_Input_1);
    __FS_DEINIT(fs);

    // READ OUTPUT
    printf("\t\t***Reading output file***\n");
    void *File_Output_1;
    int ret_Output_1 = 0;
    File_Output_1 = __OPEN_READ(fs, "../../../decoding_layer/input.bin");
    ret_Output_1 = __READ(File_Output_1, Output_1, 10080*sizeof(F16));
    __CLOSE(File_Output_1);
    __FS_DEINIT(fs);

    // for (int i = 0; i < 10; i++) {
    //     printf("%f ", Output_1[i]);
    // }
    // printf("\n");
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
        (F16 *)(model_L2_Memory_Dyn + (H_INP * W_INP * CHANNELS)), 
        Input_1, 
        H_INP, 
        W_INP,
        CHANNELS
        );
    slicing_cycles = gap_cl_readhwtimer() - slicing_cycles;

// ------------------------- INFERENCE -------------------------
    printf("\t\t***Inference***\n");
    // modelCNN(Output_1);

    // polulate Output_1 with nan values
    // for (int i = 0; i < 10080; i++) {
    //     Output_1[i] = 0.0/0.0;
    // }

    // for (int i = 0; i < 10; i++) {
    //     printf("%f ", Output_1[i]);
    // }
    // printf("\n");

// ------------------------- save output -------------------------

    // switch_fs_t fs;
    // __FS_INIT(fs);
    // void *File_Output_1;
    // int ret_Output_1 = 0;
    // File_Output_1 = __OPEN_WRITE(fs, "../../../output.bin");
    // ret_Output_1 = __WRITE(File_Output_1, Output_1, 10080*sizeof(F16));
    // __CLOSE(File_Output_1);
    // __FS_DEINIT(fs);

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

    // for (int i=0; i < 10; i++){
    //     printf("%f ", Output_1[i]);
    // }
    // printf("\n");
// ------------------------- POST PROCESSING -------------------------

// ------------------------- xywh2xyxy -------------------------
    printf("\t\t***Start xywh2xyxy***\n");

    xywh2xyxy_cycles = gap_cl_readhwtimer();
    xywh2xyxy(Output_1, (int) (RAWS));
    xywh2xyxy_cycles = gap_cl_readhwtimer() - xywh2xyxy_cycles;

    // for (int i=0; i < 10; i++){
    //     printf("%f ", Output_1[i]);
    // }
    // printf("\n");

// ------------------------- filter boxes -------------------------
    printf("\t\t***Start filter boxes ***\n");

    // printf("\n%d\n", *num_val_boxes); 

    //cast model_L2_Memory_Dyn to float16
    f16 * model_L2_Memory_Dyn = (f16 *) model_L2_Memory_Dyn;
    *num_val_boxes = 0;
    filter_boxes_cycles = gap_cl_readhwtimer();
    filter_boxes(
        Output_1, 
        (model_L2_Memory_Dyn + 10080), 
        CONF_THRESH, 
        RAWS, 
        num_val_boxes
        );
    filter_boxes_cycles = gap_cl_readhwtimer() - filter_boxes_cycles;

    // for (int i=0; i < 15; i++){
    //     printf("%f ", (model_L2_Memory_Dyn + 10080)[i]);
    // }
    // printf("\n");

// ------------------------- Conver boxes -------------------------
    // printf("\n%d\n", *num_val_boxes); 
    // printf("\t\t***Start conver boxes ***\n");

    // // boxes only contains the top_k_boxes boxes with the highest confidence
    // // consider allocating bboxes dinamicly according to num_val_boxes
    // // otherwise, the function to_boxes will not work properly in all cases

    bbox_cycles = gap_cl_readhwtimer();
    to_bboxes(
        (model_L2_Memory_Dyn + 10080), 
        bboxes, 
        *num_val_boxes
        );
    bbox_cycles = gap_cl_readhwtimer() - bbox_cycles;

    // printf("\n%d\n", *num_val_boxes); 

    // for (int i=0; i < 20; i++){
    //     printf("%f ", bboxes[i].x1);
    //     printf("%f ", bboxes[i].y1);
    //     printf("%f ", bboxes[i].x2);
    //     printf("%f ", bboxes[i].y2);
    //     printf("%f ", bboxes[i].obj_conf);
    //     printf("%d ", bboxes[i].cls_conf);
    //     printf("%d ", bboxes[i].cls_id);
    //     printf("%d ", bboxes[i].alive);
    //     printf("\n");
    // }

// ------------------------- nms -------------------------

    printf("\t\t***Start nms ***\n");

    // printf("num_val_boxes points at %p\n", num_val_boxes);
    // printf("val_bboxes points at %p\n", final_valid_boxes);

    // printf("num_val_boxes value is %d\n", *num_val_boxes);
    // printf("val_bboxes value at %d\n", final_valid_boxes);


    // printf("before nms num_val_boxes: %d\n", *num_val_boxes);
    // printf("\n%d\n", *num_val_boxes); 

    final_valid_boxes = 0;
    nms_cycles = gap_cl_readhwtimer();
    nms(
        bboxes, 
        (model_L2_Memory_Dyn + 10080), 
        NMS_THRESH, 
        *num_val_boxes, 
        &final_valid_boxes
        );
    nms_cycles = gap_cl_readhwtimer() - nms_cycles;

    // printf("num_val_boxes value is %d\n", *num_val_boxes);
    // printf("val_bboxes value at %d\n", final_valid_boxes);

    printf("final_valid_boxes: %d\n", final_valid_boxes);
    for (int i=0; i < final_valid_boxes; i++){
        for(int j=0; j < 7; j++){
            printf("%f ", (model_L2_Memory_Dyn + 10080)[i*7 + j]);
        }
        printf("\n");
    }
// ------------------------- END -------------------------
    printf("\t\t***Runner completed***\n");

// ------------------------- SAVE OUTPUT -------------------------

    switch_fs_t fs;
    __FS_INIT(fs);
    void *File_Output_2;
    int ret_Output_2 = 0;
    File_Output_2 = __OPEN_WRITE(fs, "../../../output.bin");
    ret_Output_2 = __WRITE(File_Output_2, (model_L2_Memory_Dyn + 10080), final_valid_boxes*7*sizeof(F16));
    __CLOSE(File_Output_2);
    __FS_DEINIT(fs);

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

// #ifdef PERF
#ifndef PERF
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
