
typedef struct{
    float x1, y1, x2, y2;
    float obj_conf, cls_conf;
    int cls_id;
    int alive;
} Box;

float iou(Box * box1, Box * box2);

void nms(Box * boxes, float * Output, float nms_thresh, int num_val_boxes, int * val_final_boxes);

void xywh2xyxy(float * array, unsigned int rows);

void to_bboxes(float * input, Box * output, int num_val_boxes);

void filter_boxes(float * input, float  * output, float conf_thresh, unsigned int rows, unsigned int * num_val_boxes);

int md_comparator(const void *v1, const void *v2);