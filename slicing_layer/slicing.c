#include <stdio.h>
#include <stdlib.h> 

void slicing_3hw_channel(signed char * array, int h, int w, int chnls){

    // signed char tmp[h * w * chnls];
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

    // signed char tmp[h * w * 3];
    
    for(int i = 0; i < h * w; i++){
        tmp[i] = array[i];
    }

    // for(int i = 0; i < sizeof(tmp); i++)
    //     printf("%d ", tmp[i]); // prints a series of bytes
    // printf("\n");

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

    // for(int i = 0; i < sizeof(tmp); i++)
    //     printf("%d ", tmp[i]); // prints a series of bytes
    // printf("\n");
    
    for(int i = 0; i < h * w; i++){
        array[i] = tmp[i];
    }
}



// int main(){

//     unsigned int size_buff = 30 * 242 * 3;
//     unsigned int size = 10;
//     unsigned char buffer[size_buff];
//     unsigned char target[size_buff];

//     printf("read it ");
//     FILE *ptr;
//     ptr = fopen("./test_data_3c/mat_30_242.bin", "rb");
//     fread(buffer, sizeof(buffer), 1, ptr);

//     FILE *ptr1;
//     ptr1 = fopen("./test_target_3c/mat_30_242.bin", "rb");
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

//     slicing_3_channel(buffer, 30, 242, 3);

//     printf("================== sliced \n");
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