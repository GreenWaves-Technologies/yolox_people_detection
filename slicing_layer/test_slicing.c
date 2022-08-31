#include <stdio.h>
#include <dirent.h> 
#include <string.h>
#include <stdlib.h>
#include "../inference_gvsoc/slicing.h"


struct Names{
    char name[30]; 
};


int main(void) {

    
    struct Names test_data[60];

    char  test_data_source[100] = "./data/test_chw_source/";
    char  test_data_target[100] = "./data/test_chw_target/";

    // char  test_data_source[100] = "./data/test_hwc_source/";
    // char  test_data_target[100] = "./data/test_hwc_target/";
    
    // list dir and save file names 
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
                test_data[cout].name[i] = *tmp;
                tmp++;
                i++;
            }
            test_data[cout].name[i] = '\0';
            cout++;
        }
        closedir(d);
    }

    // // main test loop
    for (int i = 0; i < cout; i++){

        if ( test_data[i].name[0] != '.' ){

            char tmp[20];
            char tmp1[20];
            char tmp2[20];

            strcpy(tmp, test_data[i].name);
            strcpy(tmp1, test_data[i].name);
            strcpy(tmp2, test_data[i].name);

            int c, h, w; 
            if (strstr(test_data_source, "chw") != NULL){
                char * token = strtok(tmp, "_");
                c = atoi(strtok(NULL, "_"));
                h = atoi(strtok(NULL, "_"));
                w = atoi(strtok(NULL, "."));
            }
            else if (strstr(test_data_source, "hwc") != NULL){
                char * token = strtok(tmp, "_");
                h = atoi(strtok(NULL, "_"));
                w = atoi(strtok(NULL, "_"));
                c = atoi(strtok(NULL, "."));
            }
            else{
                printf("error: source data format not supported\n");
                return -1;
            }
            
            //copy test_data_source to test_path
            char test_path[100];
            strcpy(test_path, test_data_source);

            char target_path[100];
            strcpy(target_path, test_data_target);

            strcat(test_path, tmp1); 
            strcat(target_path, tmp2); 

            unsigned int data_size = w * h * c;
            unsigned char * input = malloc(w * h  * c * sizeof(unsigned char));
            unsigned char * ouput = malloc(w * h  * c * sizeof(unsigned char));
            unsigned char * target = malloc(w * h  * c * sizeof(unsigned char));

            // read input data 
            FILE *ptr;
            ptr = fopen(test_path, "rb");
            fread(input, data_size, 1, ptr);

            // reaade target data
            FILE *ptr1;
            ptr1 = fopen(target_path, "rb");
            fread(target, data_size, 1, ptr1);

            // slicing
            if (strstr(test_data_source, "chw") != NULL){
                slicing_chw_channel(input, ouput, h, w, c);
            }
            else if (strstr(test_data_source, "hwc") != NULL){
                slicing_hwc_channel(input, ouput, h, w, c);
            }
            else{
                printf("error: source data format not supported\n");
                return -1;
            }

            // slicing_chw_channel(input, ouput, h, w, c);
            // slicing_hwc_channel(input, ouput, h, w, c);

            int sum = 0;
            for(int i = 0; i < data_size; i++){
                sum += (ouput[i] - target[i]);
            }

            if (sum == 0){
                printf("test - %s : \t+\n", test_data[i].name); 
            }else{
                printf("test - %s : \t-\n", test_data[i].name); 
            }

            free(input);
            free(ouput);
            free(target);
        }

    }
    return(0);
}



