
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
tTuple feature_maps[STACK_SIZE] = {{32, 40}, {16, 20}, {8, 10}};
float strides[STACK_SIZE] = {8, 16, 32};

// parameters needed for postprocessing layer
#define NMS_THRESH 0.30
#define CONF_THRESH 0.30
#define OUPUTU_SIZE 10080

// cycles count variables
unsigned int slicing_cycles;
unsigned int decoding_cycles;
unsigned int filter_boxes_cycles;


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
    modelCNN(Output_1);

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
        Output_1, feature_maps, 
        strides, STRIDE_SIZE);
    decoding_cycles = gap_cl_readhwtimer() - decoding_cycles;

// ------------------------- POST PROCESSING -------------------------

// ------------------------- filter boxes -------------------------

    printf("\t\t***Start filter boxes ***\n");
    // unsigned int * num_val_boxes = (unsigned int *) __ALLOC_L2 (sizeof(unsigned int));
    unsigned int * num_val_boxes = (unsigned int *) pi_l2_malloc(sizeof(unsigned int));
    num_val_boxes = 0;
    printf("\n%d\n", num_val_boxes); 
    filter_boxes_cycles = gap_cl_readhwtimer();
    filter_boxes(
        Output_1, 
        (F16 *)(model_L2_Memory_Dyn + OUPUTU_SIZE * sizeof(F16)), 
        CONF_THRESH, 
        OUPUTU_SIZE, 
        num_val_boxes
        );
    filter_boxes_cycles = gap_cl_readhwtimer() - filter_boxes_cycles;



// ------------------------- Conver boxes -------------------------
    printf("\n%d\n", num_val_boxes); 
    printf("\t\t***Start conver boxes ***\n");

    // if (*num_val_boxes == 0){
    //     Box * bboxes = pi_l2_malloc(1 * sizeof(Box));
    // }else{
    //     Box * bboxes = pi_l2_malloc((*num_val_boxes) * sizeof(Box));
    // }

    Box * bboxes = (Box *)  pi_l2_malloc(1 * sizeof(Box));
    to_bboxes((F16 *)(model_L2_Memory_Dyn + OUPUTU_SIZE * sizeof(F16)), bboxes, *num_val_boxes);

     

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
    {
      printf("\t\t***Performance***\n");
      unsigned int TotalCycles = 0, TotalOper = 0;
      printf("\n");
      for (unsigned int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
        TotalCycles += AT_GraphPerf[i]; TotalOper += AT_GraphOperInfosNames[i];
      }
    
      // slicing cycles
      printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", "Slicing", slicing_cycles, 100 * ((float) (slicing_cycles) / TotalCycles), NULL, NULL, NULL);

      for (unsigned int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
        printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", AT_GraphNodeNames[i], AT_GraphPerf[i], 100*((float) (AT_GraphPerf[i]) / TotalCycles), AT_GraphOperInfosNames[i], 100*((float) (AT_GraphOperInfosNames[i]) / TotalOper), ((float) AT_GraphOperInfosNames[i])/ AT_GraphPerf[i]);
      }
      // decoding cycles
      printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", "Decoding", decoding_cycles, 100 * ((float) (decoding_cycles) / TotalCycles), NULL, NULL, NULL);
      // filter boxes cycles
      printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", "Filter boxes", filter_boxes_cycles, 100 * ((float) (filter_boxes_cycles) / TotalCycles), NULL, NULL, NULL);
      printf("\n");
      printf("%45s: Cycles: %12u, Cyc%%: 100.0%%, Operations: %12u, Op%%: 100.0%%, Operations/Cycle: %f\n", "Total", TotalCycles, TotalOper, ((float) TotalOper)/ TotalCycles);
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
