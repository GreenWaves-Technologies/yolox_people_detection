#include "postprocessing.h"

void xywh2xyxy(float * input, unsigned int rows){

    int count = 0;
    float width, height, x1, y1, x2, y2;
    for (unsigned int i = 0; i < rows; i++){

        width = input[count + 2] / 2;
        height = input[count + 3] / 2;

        x1 = input[count + 0] - width; 
        y1 = input[count + 1] - height;
        x2 = input[count + 0] + width;
        y2 = input[count + 1] + height;

        input[count + 0] = x1;
        input[count + 1] = y1;
        input[count + 2] = x2;
        input[count + 3] = y2;
        count += 6;
    }
}


void filter_boxes(float * Input, float * Output, float conf_thresh, unsigned int rows, unsigned int  * num_val_boxes){

    int count = 0; 
    int last_valid_idx = 0;
    for (unsigned int i = 0; i < rows; i++){
        if (Input[count + 4] * Input[count + 5] > conf_thresh){
            Output[last_valid_idx + 0] = Input[count + 0];
            Output[last_valid_idx + 1] = Input[count + 1];
            Output[last_valid_idx + 2] = Input[count + 2];
            Output[last_valid_idx + 3] = Input[count + 3];
            Output[last_valid_idx + 4] = Input[count + 4];
            Output[last_valid_idx + 5] = Input[count + 5];
            Output[last_valid_idx + 6] = (float) 0.0; 
            last_valid_idx += 7;
            *num_val_boxes += 1;
        }
        count += 6;
    }
}

void to_bboxes(float * input, Box * output, int num_val_boxes, int top_k_b){

    int count = 0;
    for (int i = 0; i < top_k_b; i++){
        output[i].x1 = input[count + 0];
        output[i].y1 = input[count + 1];
        output[i].x2 = input[count + 2];
        output[i].y2 = input[count + 3];
        output[i].obj_conf = input[count + 4];
        output[i].cls_conf = input[count + 5];
        output[i].cls_id = (char) input[count + 6];
        output[i].alive = true;
        count += 7;
    }
}



float iou(Box * box1, Box * box2){

    float x_left    = MAX(box1->x1, box2->x1);
    float y_top     = MAX(box1->y1, box2->y1);
    float x_right   = MIN(box1->x2, box2->x2);
    float y_bottom  = MIN(box1->y2, box2->y2);

    if (x_right < x_left || y_bottom < y_top){
        return 0.0;
    }

    float intersection_area = (x_right - x_left) * (y_bottom - y_top);

    float box1_area = (box1->x2 - box1->x1) * (box1->y2 - box1->y1);
    float box2_area = (box2->x2 - box2->x1) * (box2->y2 - box2->y1);

    float iou = intersection_area / (box1_area + box2_area - intersection_area);

    if (iou < 0.0){
        printf("iou: %f < 0.0 \n", iou);
        exit(0);
    }else if (iou > 1.0 && (iou - 1.0) > 1e-3){
        printf("iou: %f > 1.0 \n", iou);
        exit(0);
    }

    return iou;

}


void nms(
    Box * boxes, 
    float * Output, 
    float nms_thresh, 
    int num_val_boxes, 
    int * val_final_boxes,
    int top_k_b
    ){

    int count = 0;
    int val_boxes;

    if (num_val_boxes > top_k_b){
        val_boxes = top_k_b;
    }
    else{
        val_boxes = num_val_boxes;
    }

    for (int i = 0; i < val_boxes; i++){
        if (boxes[i].alive == true){

            for (int j = 0; j < val_boxes; j++){
                if (i != j && boxes[j].alive == true){

                    if (iou(&boxes[i], &boxes[j]) >= nms_thresh){
                        if (boxes[i].obj_conf > boxes[j].obj_conf){
                            boxes[j].alive = false;
                        }
                        else{
                            boxes[i].alive = false;
                        }
                    }
                }
            }

            // if boxes is alive, add it to the output
            if (boxes[i].alive == 1){
                Output[count + 0] =  boxes[i].x1;
                Output[count + 1] =  boxes[i].y1;
                Output[count + 2] =  boxes[i].x2;
                Output[count + 3] =  boxes[i].y2;
                Output[count + 4] =  boxes[i].obj_conf;
                Output[count + 5] =  boxes[i].cls_conf;
                Output[count + 6] = (float) boxes[i].cls_id;
                count += 7;
                *val_final_boxes += 1;
            }
        }
    }
}



int md_comparator(const void *v1, const void *v2)
{
    const Box *p1 = (Box *)v1;
    const Box *p2 = (Box *)v2;
    if (p1->obj_conf > p2->obj_conf)
        return -1;
    else if (p1->obj_conf < p2->obj_conf)
        return +1;
    else
        return 0;
}
