import tensorflow as tf
import tensorflow_probability as tfp
def intersection_over_union(boxes_preds,boxes_labels,box_format="midpoint"):
    """
    Calculates intersection over union
    :param boxes_preds (tensor): Predictions of bounding boxes (BATCH_SIZE,4)
    :param boxes_labels (tensor): Labels of bounding boxes (BATCH_SIZE,4)
    :param box_format (str) : midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    :return:
        tensor: IOU for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[...,0:1] - boxes_preds[...,2:3]//2
        box1_y1 = boxes_preds[...,1:2] - boxes_preds[...,3:4]//2
        box1_x2 = boxes_preds[...,0:1] + boxes_preds[...,2:3]//2
        box1_y2 = boxes_preds[...,1:2] + boxes_preds[...,2:3]//2

        box2_x1 = boxes_labels[...,0:1] - boxes_labels[...,2:3]//2
        box2_y1 = boxes_labels[...,1:2] - boxes_labels[...,3:4]//2
        box2_x2 = boxes_labels[...,0:1] + boxes_labels[...,2:3]//2
        box2_y2 = boxes_labels[...,1:2] + boxes_labels[...,2:3]//2
    if box_format == "corners":
        box1_x1 = boxes_preds[...,0:1]
        box1_y1 = boxes_preds[...,1:2]
        box1_x2 = boxes_preds[...,2:3]
        box1_y2 = boxes_preds[...,3:4]

        box2_x1 = boxes_labels[...,0:1]
        box2_y1 = boxes_labels[...,1:2]
        box2_x2 = boxes_labels[...,2:3]
        box2_y2 = boxes_labels[...,3:4]

    x1 = tf.maximum(box1_x1,box2_x1)
    y1 = tf.maximum(box1_y1,box2_y1)
    x2 = tf.minimum(box1_x2,box2_x2)
    y2 = tf.minimum(box1_y2,box2_y2)

    intersection_area = (x2-x1)*(y2-y1)
    if intersection_area<0:
        intersection_area = 0
    box1_area = abs((box1_x2-box1_x1)*(box1_y2-box1_y1)
    box2_area = abs((box2_x2-box2_x1)*(box2_y2-box2_y1))

    return intersection_area/(box1_area+box2_area-intersection_area+1e-6)




def non_max_suppression(bboxes,iou_threshold,conf_threshold,box_format="midpoint"):
    """

    :param bboxes: list of list of boxes with each boxes [[class,prob_score,x1,y1,x2,y2],[],[]]
    :param iou_threshold: intersection over union threshold
    :param conf_threshold: probability threshold
    :param box_format: midpoint or corners
    :return:
        list: bboxes after NMS given a specific IOU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1]>conf_threshold]
    bboxes = sorted(bboxes,key=lambda x:x[1],reverse=True) # sort the boxes in descending order of confidence
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [box for box in bboxes if box[0]!=chosen_box[0]
                  or intersection_over_union(tf.constant(chosen_box),
                tf.constant(box),box_format=box_format)<iou_threshold]

        bboxes_after_nms.append(chosen_box)
    return bboxes_after_nms

def mean_average_precision(pred_boxes,true_boxes,iou_threshold=0.5,box_format="midpoint",num_classes=20):
    """

    :param pred_boxes: list of lists predicted boxes
    like [[train_idx,class,prob,x1,y1,x2,y2],[],[]]
    :param true_boxes:list of lists true boxes
    :param iou_threshold: float value for intersection over union
    :param box_format: midpoint/corners
    :param num_classes: num of classes in dataset
    :return:
        float: mAp value for specific iou threshold
    """
    average_precisions = []

    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths= []

        for box in pred_boxes:
            if box[1] ==c:
                detections.append(box)

        for box in true_boxes:
            if box[1] ==c:
                ground_truths.append(box)

        # calculate num of gt boxes in every image.
        # train_idx will be same for boxes from same image
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        for key,val in amount_bboxes.items():
            amount_bboxes[key] = tf.zeros(val)

        detections.sort(key=lambda x: x[2],reverse=True) # sort based on prob score

        # find what is TP and what is FP
        TP = tf.zeros(len(detections))
        FP = tf.zeros(len(detections))
        total_true_boxes = len(ground_truths)

        for detection_idx,detection in enumerate(detections):
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0]==detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou=0
            for idx,gt in enumerate(ground_truth_img):
                iou = intersection_over_union(tf.constant(detection),
                                              tf.constant(gt),box_format=box_format)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            if best_iou>iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] ==0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1
        TP_cum_sum = tf.cumsum(tf.constant(TP),axis=0)
        FP_cum_sum = tf.cumsum(tf.constant(FP),axis=0)

        recalls = TP_cum_sum/(total_true_boxes+1e-6)
        precisions = TP_cum_sum/(TP_cum_sum+FP_cum_sum+1e-6)

        precisions = tf.concat(tf.constant([1]),precisions)
        recalls = tf.concat(tf.constant([0]),recalls)

        average_precisions.append(tfp.math.trapz(precisions,recalls))

        return sum(average_precisions)/len(average_precisions)









