
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





AT_HYPERFLASH_EXT_ADDR_TYPE model_L3_Flash = 0;


/* Inputs */
/* Outputs */
L2_MEM signed char Output_1[10080];

// Count cycles
unsigned int Total_cycles = 0;

/* Copy inputs function */
void copy_inputs() {

    switch_fs_t fs;
    __FS_INIT(fs);

    /* Reading from file Input_1.bin */
    void *File_Input_1;
    int ret_Input_1 = 0;

    #ifdef __EMUL__
    File_Input_1 = __OPEN_READ(fs, "Input_1.bin");
    #else
    File_Input_1 = __OPEN_READ(fs, "../../../Input_1.bin");
    #endif

    // read input file to L2 empty buffer instead of Input_1
    ret_Input_1 = __READ(
        File_Input_1, 
        model_L2_Memory_Dyn + (H_INP * W_INP * CHANNELS), 
        (H_INP * W_INP * CHANNELS)*sizeof(signed char)
        // 245760 * sizeof(signed char)
    );

    __CLOSE(File_Input_1);
    __FS_DEINIT(fs);
}


static void cluster()
{
    #ifdef PERF
    printf("Start timer\n");
    gap_cl_starttimer();
    gap_cl_resethwtimer();
    #endif

    #ifdef PERF
    Total_cycles = gap_cl_readhwtimer(); 
    #endif

    // slice the input buffer and copy it to Input_1 
    slicing_chw_channel(
        model_L2_Memory_Dyn + (H_INP * W_INP * CHANNELS), 
        Input_1, 
        H_INP, 
        W_INP,
        CHANNELS
        );

    #ifdef PERF
    // calculate the cycle count for the slicing operation
    Total_cycles = gap_cl_readhwtimer() - Total_cycles;
    printf("Calculating cycles: %d \n", Total_cycles);
    #endif

    // run the model
    modelCNN(Output_1);

    //save Output_1 to a file named Output_1_csliced.bin
    switch_fs_t fs;
    __FS_INIT(fs);

    void *File_Output_1;
    int ret_Output_1 = 0;
    // File_Output_1 = __OPEN_WRITE(fs, "../../../Output_1_python_sliced.bin");
    File_Output_1 = __OPEN_WRITE(fs, "../../../Output_1_C_sliced.bin");
    ret_Output_1 = __WRITE(
        File_Output_1, 
        Output_1, 
        10080 * sizeof(signed char)
    );
    __CLOSE(File_Output_1);
    __FS_DEINIT(fs);


    #ifdef PERF
    // calculate the cycle count for the model operation
    Total_cycles = gap_cl_readhwtimer() - Total_cycles;
    printf("Calculating cycles: %d \n", Total_cycles);
    #endif

    printf("Runner completed\n");

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
    printf("Constructor\n");
    int ConstructorErr = modelCNN_Construct();
    if (ConstructorErr)
    {
        printf("Graph constructor exited with error: %d\n(check the generated file modelKernels.c to see which memory have failed to be allocated)\n", ConstructorErr);
        pmsis_exit(-6);
    }
    

    copy_inputs();

    printf("Call cluster\n");
#ifndef __EMUL__
    struct pi_cluster_task task;
    pi_cluster_task(&task, (void (*)(void *))cluster, NULL);
    pi_cluster_task_stacks(&task, NULL, SLAVE_STACK_SIZE);

    pi_cluster_send_task_to_cl(&cluster_dev, &task);
#else
    cluster();
#endif

    

    modelCNN_Destruct();

#ifdef PERF
    {
      unsigned int TotalCycles = 0, TotalOper = 0;
      printf("\n");
      for (unsigned int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
        TotalCycles += AT_GraphPerf[i]; TotalOper += AT_GraphOperInfosNames[i];
      }
      for (unsigned int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
        printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", AT_GraphNodeNames[i], AT_GraphPerf[i], 100*((float) (AT_GraphPerf[i]) / TotalCycles), AT_GraphOperInfosNames[i], 100*((float) (AT_GraphOperInfosNames[i]) / TotalOper), ((float) AT_GraphOperInfosNames[i])/ AT_GraphPerf[i]);
      }
      printf("\n");
      printf("%45s: Cycles: %12u, Cyc%%: 100.0%%, Operations: %12u, Op%%: 100.0%%, Operations/Cycle: %f\n", "Total", TotalCycles, TotalOper, ((float) TotalOper)/ TotalCycles);
      printf("\n");
    }
#endif

    printf("Ended\n");
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
