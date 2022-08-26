#include <stdio.h>
#include <stdlib.h> 

void slicing_hwc_channel(signed char * array, int h, int w, int channels){

    signed char * tmp = malloc(h * w * channels * sizeof(signed char));

    unsigned int cur = 0; 
    unsigned int tmp1 = 0;
    unsigned int tmp2 = w * channels;
    unsigned int comp = (w * h * channels) / (h / 2);

    int larger, smaller;
    if(h > w){
        larger = h;
        smaller = w;
    }
    else{
        larger = w;
        smaller = h;
    }

    for(int j = 0; j < larger; j++){
        for(int i = 0; i < smaller; i++){
            for(int c = 0; c < channels; c++){
                if (i % 2 == 0){
                    tmp[cur] = array[tmp1];
                    tmp1++;
                }else{
                    tmp[cur] = array[tmp2];
                    tmp2++;
                }
                cur++;
                if (cur % comp == 0){
                    tmp1 += w * channels;
                    tmp2 += w * channels;
                }
            }
        }
    }
    
    for(int i = 0; i < h * w * channels; i++){
        array[i] = tmp[i];
    }
    free(tmp);
}

void slicing_chw_channel(signed char * array, int h, int w, int chnls){

    signed char * tmp = malloc(h * w * chnls * sizeof(signed char));

    for(int i = 0; i < h * w * chnls; i++){
        tmp[i] = array[i];
    }

    unsigned int count = 0;
    unsigned int k = 0;

    // top left 
    for(int c = 0; c < chnls; c++){
        for(int j = 0; j < h / 2; j++){
            k = j * ( 2 * w);
            for(int i=0; i < w / 2; i ++){
                tmp[count] = array[k + i * 2 + c * (w * h)];
                count++;
            }
        }
    }


    // bottom left 
    for(int c = 0; c < chnls; c++){
        for(int j = 0; j < h / 2; j++){
            k = j * ( 2 * w) + w;
            for(int i=0; i < w / 2; i ++){
                tmp[count] = array[k + i * 2 + c * (w * h)];
                count++;
            }
        }
    }

    // top right 
    for(int c = 0; c < chnls; c++){
        for(int j = 0; j < h / 2; j++){
            k = j * ( 2 * w) + 1;
            for(int i=0; i < w / 2; i ++){
                tmp[count] = array[k + i * 2 + c * (w * h)];
                count++;
            }
        }
    }

    // bottom right 
    for(int c = 0; c < chnls; c++){
        for(int j = 0; j < h / 2; j++){
            k = j * (2 * w) + w +  1;
            for(int i=0; i < w / 2; i ++){
                tmp[count] = array[k + i * 2 + c * (w * h)];
                count++;
            }
        }
    }

    for(int i = 0; i < h * w * chnls; i++){
        array[i] = tmp[i];
    }
    free(tmp);
}

void slicing(signed char * array, unsigned char * tmp, int h, int w){

    
    for(int i = 0; i < h * w; i++){
        tmp[i] = array[i];
    }

    unsigned int count = 0;

    // top left 
    for(int j = 0; j < h / 2; j++){
        unsigned int k = j * ( 2 * w);
        // printf("k ==: %d \n", k);
        for(int i=0; i < w / 2; i ++){
            tmp[count] = array[k + i * 2];
            count++;
        }
    }

    // bottom left 
    for(int j = 0; j < h / 2; j++){
        unsigned int k = j * ( 2 * w) + w;
        // printf("k ==: %d \n", k);
        for(int i=0; i < w / 2; i ++){
            tmp[count] = array[k + i * 2];
            count++;
        }
    }

    // top right 
    for(int j = 0; j < h / 2; j++){
        unsigned int k = j * ( 2 * w) + 1;
        // printf("k ==: %d \n", k);
        for(int i=0; i < w / 2; i ++){
            tmp[count] = array[k + i * 2];
            count++;
        }
    }

    // bottom right 
    for(int j = 0; j < h / 2; j++){
        unsigned int k = j * (2 * w) + w +  1;
        // printf("k ==: %d \n", k);
        for(int i=0; i < w / 2; i ++){
            tmp[count] = array[k + i * 2];
            count++;
        }
    }
    
    for(int i = 0; i < h * w; i++){
        array[i] = tmp[i];
    }
}

// int main(){

//     int h = 2;
//     int w = 4;
//     int c = 3;
//     unsigned int size_buff = h * w * c; 
//     unsigned int size = h * w * c;
//     unsigned char buffer[size_buff];
//     unsigned char target[size_buff];

//     printf("read it ");
//     FILE *ptr;
//     ptr = fopen("./array_2_4_3.bin", "rb");
//     fread(buffer, sizeof(buffer), 1, ptr);

//     FILE *ptr1;
//     ptr1 = fopen("./array_scliced_2_4_3.bin", "rb");
//     fread(target, sizeof(target), 1, ptr1);
    
//     printf("================== data \n");
//     for(int i = 0; i < size; i++)
//         printf("%d ", buffer[i]); // prints a series of bytes
//     printf("\n");

//     printf("================== target \n");
//     for(int i = 0; i < size; i++)
//         printf("%d ", target[i]); // prints a series of bytes
//     printf("\n");
//     printf("================== \n");

//     // slicing_3_channel(buffer, 30, 242, 3);
//     slicing_hwc_channel(buffer, h, w, c);


//     printf("================== sliced in C \n");
//     for(int i = 0; i < size; i++)
//         printf("%d ", buffer[i]); // prints a series of bytes
//     printf("\n");

//     unsigned mismatch = 0;

//     printf("================== mismatch\n");
//     for(int i = 0; i < size_buff; i++)
//         mismatch += target[i] - buffer[i];

//     printf("mismatch = %u ", mismatch); // prints a series of bytes
//     printf("\n");

// }