#include <stdio.h>

void slicing_3_channel(signed char * Input, signed char * Output, int h, int w, int chnls){

    // for(int i = 0; i < h * w * chnls; i++){
        // Output[i] = Input[i];
    // }

    unsigned int count = 0;
    unsigned int k = 0; 
    // top left 
    for(int c = 0; c < chnls; c++){
        for(int j = 0; j < h / 2; j++){
            k = j * ( 2 * w);
            for(int i=0; i < w / 2; i ++){
                Output[count] = Input[k + i * 2 + c * (w * h)];
                count++;
            }
        }
    }

    // bottom left 
    for(int c = 0; c < chnls; c++){
        for(int j = 0; j < h / 2; j++){
            k = j * ( 2 * w) + w;
            for(int i=0; i < w / 2; i ++){
                Output[count] = Input[k + i * 2 + c * (w * h)];
                count++;
            }
        }
    }

    // top right 
    for(int c = 0; c < chnls; c++){
        for(int j = 0; j < h / 2; j++){
            k = j * ( 2 * w) + 1;
            for(int i=0; i < w / 2; i ++){
                Output[count] = Input[k + i * 2 + c * (w * h)];
                count++;
            }
        }
    }

    // bottom right 
    for(int c = 0; c < chnls; c++){
        for(int j = 0; j < h / 2; j++){
            k = j * (2 * w) + w +  1;
            for(int i=0; i < w / 2; i ++){
                Output[count] = Input[k + i * 2 + c * (w * h)];
                count++;
            }
        }
    }

    // // explictly copying the elements 
    // for(int i = 0; i < h * w * chnls; i++){
        // Input[i] = Output[i];
    // }
    // Input = Output;
}

void slicing_1_channel(signed char * Input, signed char * Output, int h, int w){

    for(int i = 0; i < h * w; i++){
        Output[i] = Input[i];
    }

    unsigned int count = 0;
    unsigned int k = 0;

    // top left 
    for(int j = 0; j < h / 2; j++){
        k = j * ( 2 * w);
        for(int i=0; i < w / 2; i ++){
            Output[count] = Input[k + i * 2];
            count++;
        }
    }

    // bottom left 
    for(int j = 0; j < h / 2; j++){
        k = j * ( 2 * w) + w;
        for(int i=0; i < w / 2; i ++){
            Output[count] = Input[k + i * 2];
            count++;
        }
    }

    // top right 
    for(int j = 0; j < h / 2; j++){
        k = j * ( 2 * w) + 1;
        for(int i=0; i < w / 2; i ++){
            Output[count] = Input[k + i * 2];
            count++;
        }
    }

    // bottom right 
    for(int j = 0; j < h / 2; j++){
        k = j * (2 * w) + w +  1;
        for(int i=0; i < w / 2; i ++){
            Output[count] = Input[k + i * 2];
            count++;
        }
    }

    for(int i = 0; i < h * w; i++){
        Input[i] = Output[i];
    }

}

