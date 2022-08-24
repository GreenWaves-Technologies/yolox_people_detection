#include <stdio.h>
#include <dirent.h> 
#include <string.h>
#include <stdlib.h>
#include "slicing.h"


struct Names{
    char name[30]; 
};


int main(void) {

    struct Names test_data[60];
    char  test_data_source[100] = "./data/test_chw_source";
    char  test_data_target[100] = "./data/test_chw_target";

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
    
    // main test loop
    for (int i = 0; i < cout; i++){

        if ( test_data[i].name[0] != '.' ){

            // get dimentions
            char tmp[20];
            char tmp1[20];
            char tmp2[20];

            strcpy(tmp, test_data[i].name);
            strcpy(tmp1, test_data[i].name);
            strcpy(tmp2, test_data[i].name);
            int c, h, w; 
            if (strstr(test_data_source, "chw")){
                char * token = strtok(tmp, "_");
                c = atoi(strtok(NULL, "_"));
                h = atoi(strtok(NULL, "_"));
                w = atoi(strtok(NULL, "."));
            }else{
                h = atoi(strtok(NULL, "_"));
                w = atoi(strtok(NULL, "_"));
                c = atoi(strtok(NULL, "."));
            } 
            // printf("%d %d %d\n", c, h, w);

            char test_path[100] = "./data/test_chw_source/"; 
            char target_path[100] = "./data/test_chw_target/"; 

            strcat(test_path, tmp1); 
            strcat(target_path, tmp2); 

            // printf("%s \n", test_path);
            // printf("%s \n", target_path);

            // check what append if singed char is used
            unsigned int data_size = w * h * c;
            unsigned char * data = malloc(w * h  * c * sizeof(unsigned char));
            unsigned char * sliced = malloc(w * h  * c * sizeof(unsigned char));
            unsigned char * target = malloc(w * h  * c * sizeof(unsigned char));
            // unsigned char target[w * h  * 3];

            FILE *ptr;
            ptr = fopen(test_path, "rb");
            fread(data, data_size, 1, ptr);

            FILE *ptr1;
            ptr1 = fopen(target_path, "rb");
            fread(target, data_size, 1, ptr1);

            for(int i = 0; i < data_size; i++){
                sliced[i] = data[i];
                // printf("%u ", data[i]);
            }

            slicing_3hw_channel(sliced, h, w, c);

            int sum = 0;
            for(int i = 0; i < data_size; i++){
                sum += (sliced[i] - target[i]);
            }
            if (sum == 0){
                printf("test - %s : \t+\n", test_data[i].name); 
            }else{
                printf("test - %s : \t-\n", test_data[i].name); 

            }
            free(data);
            free(sliced);
            free(target);
        }

    }
    return(0);
}



