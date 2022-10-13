#include "Gap.h"

typedef struct{
    f16 x1, y1, x2, y2;
    f16 obj_conf, cls_conf;
    int cls_id;
    int alive;
} Box;

void xywh2xyxy(f16 * array, unsigned int rows);

void filter_boxes(f16 * Input, f16 * Output, float conf_thresh, unsigned int rows, unsigned int  * num_val_boxes);

void to_bboxes(f16 * input, Box * output, int num_val_boxes);

float iou(Box * box1, Box * box2);

void nms(Box * boxes, f16 * Output, f16 nms_thresh, int num_val_boxes, int * val_final_boxes);

int md_comparator(const void *v1, const void *v2);