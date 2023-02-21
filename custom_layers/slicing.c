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


void slicing_hwc_channel(char * Input, unsigned char * Output, int h, int w, int channels){

    unsigned int o_h = h / 2, o_w  =  w / 2; 
    unsigned int o_c = channels * 4; 
    unsigned int o_idx, i_idx;
    for(unsigned int j = 0; j < o_h; j++){
        for(unsigned int i = 0; i < o_w; i++){

            o_idx = j * o_w * o_c + i * o_c;
            i_idx = (j * 2 * w * channels) + (i * 2 * channels);
            for(int c = 0; c < channels; c++){
                
                // for BGR input
                // Output[o_idx + c]     = Input[i_idx + c];
                // Output[o_idx + 3 + c] = Input[i_idx + (channels * w) +  c];
                // Output[o_idx + 6 + c] = Input[i_idx + channels  + c];
                // Output[o_idx + 9 + c] = Input[i_idx + (channels * w + channels)  + c];

                // for RGB input
                Output[o_idx + c]     = Input[i_idx + (channels - 1 - c)];
                Output[o_idx + 3 + c] = Input[i_idx + (channels * w) + (channels - 1 - c)];
                Output[o_idx + 6 + c] = Input[i_idx + channels + (channels - 1 - c)];
                Output[o_idx + 9 + c] = Input[i_idx + (channels * w + channels) + (channels - 1 - c)];
            }
        }
    }

}


void slicing_hwc_channel_less_buffer(char * Input, char * Output, int h, int w, int channels){
    
    int str_idx, end_idx, num_element = w * channels * 2;
    for(int r = 0; r < h / 2; r++){
        
        // slice the matrix  
        str_idx = r * w * channels * 2;
        slicing_hwc_channel(Input + str_idx, Output, 2, w, channels);

        // copy the element from buffer to input
        for(int i = 0; i < num_element; i++){
            Input[str_idx + i] = Output[i];
        }
    }  
}



void slicing_hw_channel(char * Input, unsigned char * Output, int h, int w, int channels){

    unsigned int o_h = h / 2, o_w  =  w / 2; 
    unsigned int o_c = channels * 4; 
    unsigned int o_idx, i_idx;

    for(unsigned int j = 0; j < o_h; j++){
        for(unsigned int i = 0; i < o_w; i++){

            o_idx = j * o_w * o_c + i * o_c;
            i_idx = (j * 2 * w * channels) + (i * 2 * channels);

            Output[o_idx + 0] = Input[i_idx];
            Output[o_idx + 1] = Input[i_idx + w];
            Output[o_idx + 2] = Input[i_idx + channels];
            Output[o_idx + 3] = Input[i_idx + w + channels];

        }
    }

}