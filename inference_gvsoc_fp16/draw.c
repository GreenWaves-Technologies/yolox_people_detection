#include "draw.h"
#include "model.h"

void draw_rectangle(
    unsigned char *Img, 
    int W, 
    int H, 
    int x, 
    int y, 
    int w, 
    int h, 
    unsigned char ColorValue
    ){

    int x0, x1, y0, y1;

    y0 = MAX(MIN(y, H - 1), 0);
    y1 = MAX(MIN(y + h - 1, H - 1), 0);

    x0 = x;
    if (x0 >= 0 && x0 < W) {
        for (int i = y0; i <= y1; i++)
            for(int c = 0; c < CHANNELS; c++)
                Img[CHANNELS*(i * W + x0) + c] = ColorValue;
    }

    x1 = w - 1;
    if (x1 >= 0 && x1 < W) {
        for (int i = y0; i <= y1; i++)
            for(int c = 0; c < CHANNELS; c++)
                Img[CHANNELS*(i * W + x1) + c] = ColorValue;
    }

    x0 = MAX(MIN(x, W - 1), 0);
    x1 = w - 1;

    y0 = y;
    if (y0 >= 0 && y0 < H) {
        for (int i = x0; i <= x1; i++)
            for(int c = 0; c < CHANNELS; c++)
                Img[CHANNELS*(y0 * W + i) + c] = ColorValue;
    }

    y1 = y + h - 1;
    if (y1 >= 0 && y1 < H) {
        for (int i = x0; i <= x1; i++)
            for(int c = 0; c < CHANNELS; c++)
                Img[CHANNELS*(y1 * W + i) + c] = ColorValue;
    }
}
