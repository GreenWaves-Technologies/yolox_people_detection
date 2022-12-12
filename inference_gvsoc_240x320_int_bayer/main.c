
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
#include "gaplib/ImgIO.h"
#include "slicing.h"
#include "decoding.h"
#include "postprocessing.h"
#include "draw.h"

#ifdef __EMUL__
#define pmsis_exit(n) exit(n)
#endif

// parameters needed for decoding layer
// !!! do not forget to change the stride sizes accordint to the input size !!!  
tTuple feature_maps[STACK_SIZE] = {{30.0, 40.0}, {15.0, 20.0}, {8.0, 10.0}};
float strides[STACK_SIZE] = {8.0, 16.0, 32.0};

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
L2_MEM float Output_1[9480];

/* Copy inputs function */
void copy_inputs() {

    printf("\n\t\t*** READING INPUT FROM PPM FILE ***\n");
    printf("Number of input channels: %d\n", CHANNELS);
    int status = ReadImageFromFile(
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
        printf("Error reading image from file %s (error: %d) \n", STR(INPUT_FILE_NAME), status);
        exit(-1);
    } 

}

/* Copy inputs function */
void write_outputs() {


    /* ------ SAVE ------*/
    printf("\t\t***Start saving output***\n");

    switch_fs_t fs;
    __FS_INIT(fs);

    void *File_Output_1;
    int ret_Output_1 = 0;

    File_Output_1 = __OPEN_WRITE(fs, STR(OUTPUT_BIN_FILE_NAME));
    ret_Output_1 = __WRITE(File_Output_1, Output_1, final_valid_boxes * 7 * sizeof(float));

    __CLOSE(File_Output_1);
    __FS_DEINIT(fs);






}


static void cluster()
{
    #ifdef PERF
    printf("\t\t***Start CLUSTER timer***\n");
    gap_cl_starttimer();
    gap_cl_resethwtimer();
    #endif

    mainCNN(Output_1);
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

    #ifdef PERF
    printf("\t\t***Start FC timer***\n");
    gap_fc_starttimer();
    gap_fc_resethwtimer();
    #endif

    /* ------ SLICING ------*/
    printf("\t\t***Start slicing***\n");
    slicing_cycles = gap_fc_readhwtimer();
    slicing_hwc_channel(
        main_L2_Memory_Dyn + (H_INP * W_INP * CHANNELS), 
        Input_1, 
        H_INP, 
        W_INP,
        CHANNELS
        );
    slicing_cycles = gap_fc_readhwtimer() - slicing_cycles;


    /* ------ INFERENCE ------*/
    printf("\t\t***Call CLUSTER***\n");
#ifndef __EMUL__
    struct pi_cluster_task task;
    pi_cluster_task(&task, (void (*)(void *))cluster, NULL);
    pi_cluster_task_stacks(&task, NULL, SLAVE_STACK_SIZE);

    pi_cluster_send_task_to_cl(&cluster_dev, &task);
#else
    cluster();
#endif


    /* ------ DECODING ------*/
    printf("\t\t***Start decoding***\n");
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
    printf("\t\t***Start xywh2xyxy***\n");
    xywh2xyxy_cycles = gap_fc_readhwtimer();
    xywh2xyxy(Output_1, (int) (RAWS));
    xywh2xyxy_cycles = gap_fc_readhwtimer() - xywh2xyxy_cycles;

    /* ------ filter boxes ------*/
    printf("\t\t***Start filter boxes ***\n");
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
    printf("\t\t***Start conver boxes ***\n");
    bbox_cycles = gap_fc_readhwtimer();
    to_bboxes(
        (main_L2_Memory_Dyn_casted + (RAWS * 6)), 
        bboxes, 
        *num_val_boxes
        );
    bbox_cycles = gap_fc_readhwtimer() - bbox_cycles;
    
    /* ------ nms ------*/
    printf("\t\t***Start nms ***\n");
    final_valid_boxes = 0;
    nms_cycles = gap_fc_readhwtimer();
    nms(
        bboxes, 
        Output_1,
        NMS_THRESH, 
        *num_val_boxes, 
        &final_valid_boxes
        );
    nms_cycles = gap_fc_readhwtimer() - nms_cycles;

    #ifdef DEMO 
        /* ------ DRAW REATANGLES ------*/
        printf("\t\t***Start draw reactangles ***\n");
        draw_boxes(
            main_L2_Memory_Dyn_casted,
            Output_1,
            final_valid_boxes
        );
    #endif

    /* ------ END ------*/
    printf("\t\t***Runner completed***\n");

    #ifdef CI 
        write_outputs();
    #endif

    mainCNN_Destruct();

// #ifdef PERF
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

    printf("Ended\n");
    pmsis_exit(0);
    return 0;
}

int main(int argc, char *argv[])
{
    printf("\n\n\t *** NNTOOL main Example ***\n\n");
    return test_main();
}
