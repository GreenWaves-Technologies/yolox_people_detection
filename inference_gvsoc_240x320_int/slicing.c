#include <stdio.h>
#include "slicing.h"

void slicing_chw_channel(char * Input, unsigned char * Output, int h, int w, int chnls){

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

}

void slicing_hwc_channel_slow(char * Input, unsigned char * Output, int h, int w, int channels){

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
                    Output[cur] = Input[tmp1];
                    tmp1++;
                }else{
                    Output[cur] = Input[tmp2];
                    tmp2++;
                }
                cur++;
                if (cur % comp == 0){
                    tmp1 += w * channels;
                    tmp2 += w * channels;
                }
                // something like this 
                // increment c by 2 
                // Ouput[i * channels + c] = Input[i * channels + c];
            }
        }
    }
}

void slicing_hwc_channel(char * Input, unsigned char * Output, int h, int w, int channels){

    unsigned int o_h = h / 2, o_w  =  w / 2; 
    unsigned int o_c = channels * 4; 

    for(int j = 0; j < o_h; j++){
        for(int i = 0; i < o_w; i++){
            for(int c = 0; c < channels; c++){
                
                Output[j * o_w * o_c + i * o_c + c]     = Input[(j * 2 * w * channels) + (i * 2 * channels) + c];
                Output[j * o_w * o_c + i * o_c + 3 + c] = Input[(j * 2 * w * channels) + (i * 2 * channels) + (channels * w) +  c];
                Output[j * o_w * o_c + i * o_c + 6 + c] = Input[(j * 2 * w * channels) + (i * 2 * channels) + channels  + c];
                Output[j * o_w * o_c + i * o_c + 9 + c] = Input[(j * 2 * w * channels) + (i * 2 * channels) + (channels * w + channels)  + c];

            }
        }
    }

}