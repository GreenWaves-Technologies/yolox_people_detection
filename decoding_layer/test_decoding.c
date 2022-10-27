#include <stdio.h>
#include <dirent.h> 
#include <string.h>
#include <stdlib.h>
#include "../inference_gvsoc/decoding.h"


typedef struct{
    char name[30]; 
}FilesName;


int main(){

    tTuple feature_maps[3] = {{32, 40}, {16, 20}, {8, 10}};
    float strides[3] = {8, 16, 32};

    // list files in source dictionary
    char test_data_source[20] = "./source/";
    char test_data_target[20] = "./target/";
    FilesName file_names[60];

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

    int size = feature_maps[0].h * feature_maps[0].w * 6 + feature_maps[1].h * feature_maps[1].w * 6+ feature_maps[2].h * feature_maps[2].w * 6;

    // // main test loop
    for (int i = 0; i < cout; i++){

        if ( file_names[i].name[0] != '.' ){

            char tmp[20];
            char tmp1[20];
            char tmp2[20];

            strcpy(tmp, file_names[i].name);
            strcpy(tmp1, file_names[i].name);
            strcpy(tmp2, file_names[i].name);

            char source_data[100];
            char target_data[100];

            strcpy(source_data, test_data_source);
            strcpy(target_data, test_data_target);

            strcat(source_data, tmp);
            strcat(target_data, tmp1);

            float * Input = malloc(size * sizeof(float));
            float * Target = malloc(size * sizeof(float));

            // read target data
            FILE *ptr;
            ptr = fopen(source_data, "rb");
            fread(Input, size, sizeof(float), ptr);

            FILE *ptr1;
            ptr1 = fopen(target_data, "rb");
            fread(Target, size, sizeof(float), ptr1);

            decoding(Input, feature_maps, strides, 3);

            int sum = 0;
            for(int i = 0; i < size; i++){
                sum += (Input[i] - Target[i]);
            }
            if (sum == 0){
                printf("test %s passed \t + \n", source_data);
            }
            else{
                printf("test %s failed \t - \n", source_data);
            }
        }
    }

    return 0;
}