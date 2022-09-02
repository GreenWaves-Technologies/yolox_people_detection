#include <stdio.h>

void slicing_chw_channel(signed char * Input, signed char * Output, int h, int w, int chnls){

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

void slicing_hwc_channel(signed char * Input, signed char * Output, int h, int w, int channels){

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
            }
        }
    }
}
