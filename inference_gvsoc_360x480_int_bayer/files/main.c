
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
#include "gaplib/fs_switch.h"

#ifdef __EMUL__
#define pmsis_exit(n) exit(n)
#endif

#ifndef STACK_SIZE
#define STACK_SIZE      1024
#endif





AT_DEFAULTFLASH_EXT_ADDR_TYPE main_L3_Flash = 0;


/* Inputs */
/* Outputs */
L2_MEM float Output_1[21420];

/* Copy inputs function */
void copy_inputs() {
    switch_fs_t fs;
    __FS_INIT(fs);

    /* Reading from file Input_1 */
    void *File_Input_1;
    int ret_Input_1 = 0;
    #ifdef __EMUL__
    File_Input_1 = __OPEN_READ(fs, "../Input_1.bin");
    #else
    File_Input_1 = __OPEN_READ(fs, "../Input_1.bin");
    #endif
    ret_Input_1 = __READ(File_Input_1, Input_1, 518400*sizeof(unsigned char));
    __CLOSE(File_Input_1);

    __FS_DEINIT(fs);
}

/* Copy outputs function */
void write_outputs() {
}


static void cluster()
{
    #ifdef PERF
    printf("Start timer\n");
    gap_cl_starttimer();
    gap_cl_resethwtimer();
    #endif

    mainCNN(Output_1);
    printf("Runner completed\n");

}

int test_main(void)
{
    printf("Entering main controller\n");

#ifndef __EMUL__
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
    int ConstructorErr = mainCNN_Construct();
    if (ConstructorErr)
    {
        printf("Graph constructor exited with error: %d\n(check the generated file mainKernels.c to see which memory have failed to be allocated)\n", ConstructorErr);
        pmsis_exit(-6);
    }
    
    
    
    

    /*
     * Put here Your input settings
     */
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

    write_outputs();

    

    mainCNN_Destruct();

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
    printf("\n\n\t *** NNTOOL main Example ***\n\n");
    test_main();
    return 0;
}
