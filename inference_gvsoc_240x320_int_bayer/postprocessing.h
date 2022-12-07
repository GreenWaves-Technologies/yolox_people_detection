#include "Gap.h"


typedef struct{
    float x1, y1, x2, y2;
    float obj_conf, cls_conf;
    char cls_id;
    bool alive;
} Box;

void xywh2xyxy(float * array, unsigned int rows);

void filter_boxes(float * Input, float * Output, float conf_thresh, unsigned int rows, unsigned int  * num_val_boxes);

void to_bboxes(float * input, Box * output, int num_val_boxes);

float iou(Box * box1, Box * box2);

void nms(Box * boxes, float * Output, float nms_thresh, int num_val_boxes, int * val_final_boxes);

int md_comparator(const void *v1, const void *v2);