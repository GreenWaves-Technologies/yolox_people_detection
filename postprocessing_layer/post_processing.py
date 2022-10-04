class Box:
    def __init__(self, x1, y1, x2, y2, obj_conf, cls_conf, cls):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.obj_conf = obj_conf 
        self.cls_conf = cls_conf 
        self.cls = cls
        self.alive = True
    
def iou(box1, box2):

    x1, y1, x2, y2 = box1.x1, box1.y1, box1.x2, box1.y2 
    x3, y3, x4, y4 = box2.x1, box2.y1, box2.x2, box2.y2 

    x_left = max(x1, x3)
    y_top = max(y1, y3)
    x_right = min(x2, x4)
    y_bottom = min(y2, y4)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def nms(boxes, num_val_boxes, nms_thresh):

    output = [] 
    for i in range(0, num_val_boxes):
        if not boxes[i].alive:
            continue

        for j in range(0, num_val_boxes):

            # if i != j or not boxes[j].alive:
            if i != j and boxes[j].alive:

                iou_val = iou(boxes[i], boxes[j]) 
                if iou_val >= nms_thresh:

                    conf1 = boxes[i].obj_conf * boxes[i].cls_conf 
                    conf2 = boxes[j].obj_conf * boxes[j].cls_conf 

                    if conf1 > conf2:
                        boxes[j].alive = False
                    else:
                        boxes[i].alive = False
    
    for i in range(0, num_val_boxes):
        if boxes[i].alive:
            output.extend([boxes[i].x1, boxes[i].y1, boxes[i].x2, boxes[i].y2, boxes[i].obj_conf, boxes[i].cls_conf, boxes[i].cls])

    return output

def xywh_to_xyxy(data):

    cout = 0
    height, width = 0, 0
    x1, y1, y2, x2 = 0, 0, 0, 0

    for r in range(0, 1680):
        width = data[cout + 2] / 2
        height = data[cout + 3] / 2
        x1 = data[cout + 0] - width 
        y1 = data[cout + 1] - height 
        x2 = data[cout + 0] + width
        y2 = data[cout + 1] + height 
        data[cout + 0] = x1
        data[cout + 1] = y1
        data[cout + 2] = x2
        data[cout + 3] = y2
        cout += 6

    return data

def filter_boxes(data, conf_thresh):
    ''' Filter boxes with confidence lower than threshold
    single class
    '''

    output = []
    cout = 0
    num_val_bboxes = 0
    for r in range(0, 1680):
        if data[cout + 4] * data[cout + 5]  >= conf_thresh:
            box_data = [data[cout + 0], data[cout + 1], data[cout + 2], data[cout + 3], data[cout + 4], data[cout + 5], 0]
            output.extend(box_data)
            num_val_bboxes += 1
        cout += 6
    return output, num_val_bboxes

def to_bboxes(data, num_val_boxes):
    bboxes = []
    for i in range(0, num_val_boxes * 7, 7):
        bboxes.append(Box(data[i], data[i + 1], data[i + 2], data[i + 3], data[i + 4], data[i + 5], data[i + 6]))
    return bboxes 

def post_processing(data, nms_thresh, conf_thresh):
    data = xywh_to_xyxy(data)
    data, num_val_boxes = filter_boxes(data, conf_thresh)
    data = to_bboxes(data, num_val_boxes) 
    data = nms(data, nms_thresh)
    return data