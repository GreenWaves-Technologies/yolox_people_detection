#include <stdio.h>
#include <dirent.h>
#include <stdlib.h>
#include <string.h>
#include "postprocessing.h"


typedef struct{
    char name[30];
} FilesNames;


int main(){
    char test_data_source[30] = "./post_processing_input_bin/";
    char test_data_output1[40] = "./post_processing_output1_bin/";
    char test_data_output_filtered[40] = "./post_processing_output_filtered_bin/";
    char test_data_output_nms[40] = "./post_processing_output_nms_bin/";
    FilesNames file_names[60];


    DIR *d;
    struct dirent *dir;
    d = opendir(test_data_source);
    int cout = 0;
    if (d) {
        while ((dir = readdir(d)) != NULL) {
            char * tmp; 
            tmp = dir->d_name;
            int i = 0;
            if (tmp[0] == '.') {
                continue;
            }
            // printf("%s\n", tmp);
            while(*tmp != '\0'){
                file_names[cout].name[i] = *tmp;
                tmp++;
                i++;
            }
            file_names[cout].name[i] = '\0';
            cout++;
        }
        closedir(d);
    }

    int input_size = 10080;
    for (int i = 0; i < cout; i++){

        char tmp[30];
        strcpy(tmp, file_names[i].name);

        printf(" \t\t************** %s **************", tmp);
        char source_data[100];
        char output1_data[100];

        strcpy(source_data, test_data_source);
        strcat(source_data, tmp);

        strcpy(output1_data, test_data_output1);
        strcat(output1_data, tmp);


        float * Input = malloc(input_size * sizeof(float));
        float * Output1 = malloc(input_size * sizeof(float));

        // read target data
        FILE *ptr;
        ptr = fopen(source_data, "rb");
        fread(Input, input_size, sizeof(float), ptr);

        xywh2xyxy(Input, 1680);

        ptr = fopen(output1_data, "rb");
        fread(Output1, input_size, sizeof(float), ptr);

        float mis_match = 0;
        for (int i = 0; i < input_size; i++){
            mis_match += (Input[i] - Output1[i]); 
        }

        printf("\n\n\t\t *** Mis match in hwxy2xyxy conversion: %f *** \n\n", mis_match);


        // ---------------- Filter boxes ----------------

        float * Output_tmp = malloc(input_size * sizeof(float));
        unsigned int * num_val_boxes = malloc(sizeof(unsigned int));
        *num_val_boxes = 0;

        filter_boxes(Input, Output_tmp, 0.3, 1680, num_val_boxes);

        char output_filtered[100];
        strcpy(output_filtered, test_data_output_filtered);
        strcat(output_filtered, tmp);

        ptr = fopen(output_filtered, "rb");
        fread(Output1, input_size, sizeof(float), ptr);

        printf("\n\n \t\t *** Mis match in filter boxes: %f *** \n\n", mis_match);

        // ---------------- Filter boxes end ----------------


        // ---------------- Convert to bboxes ----------------
        
        Box * bboxes = malloc((*num_val_boxes) * sizeof(Box));
        to_bboxes(Output_tmp, bboxes, *num_val_boxes);

        // ---------------- Convert to bboxes ----------------


        // ---------------- NMS ----------------

        int * val_final_boxes = malloc(sizeof(int));
        * val_final_boxes = 0;
    
        nms(bboxes, Output_tmp, 0.3, *num_val_boxes, val_final_boxes); 

        char output_final[100];
        strcpy(output_final, test_data_output_nms);
        strcat(output_final, tmp);

        ptr = fopen(output_final, "rb");
        fread(Output1, (* val_final_boxes), sizeof(float), ptr);


        Box * bboxes_target = malloc((*num_val_boxes) * sizeof(Box));
        to_bboxes(Output1, bboxes_target, *num_val_boxes);

        qsort(bboxes_target, *num_val_boxes, sizeof(Box), md_comparator);
        qsort(bboxes, *num_val_boxes, sizeof(Box), md_comparator);

        mis_match = 0;
        for (int i = 0; i < (* val_final_boxes); i++){
            if (bboxes->alive == 1){
                mis_match += (bboxes[i].x1 - bboxes_target[i].x1);
                mis_match += (bboxes[i].y1 - bboxes_target[i].y1);
                mis_match += (bboxes[i].x2 - bboxes_target[i].x2);
                mis_match += (bboxes[i].y2 - bboxes_target[i].y2);
                mis_match += (bboxes[i].obj_conf - bboxes_target[i].obj_conf);
                mis_match += (bboxes[i].cls_conf - bboxes_target[i].cls_conf);
                mis_match += (bboxes[i].cls_id - bboxes_target[i].cls_id);
                // mis_match += (bboxes[i].alive - bboxes_target[i].alive);
            }
        }

        printf("\n\n \t\t *** Mis match in NMS: %f *** \n\n", mis_match);
        printf(" \t\t ************** %s **************\n\n\n", tmp);
        // ---------------- NMS ----------------

    }
    
    return 0;
}