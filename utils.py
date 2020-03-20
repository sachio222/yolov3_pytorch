"""A Study implementation of YOLOv3.

Guided from https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-
            detector-from-scratch-in-pytorch-part-2/

2019, Jacob Krajewski
"""

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2


def predict_transform(prediction, in_dim, anchors, num_classes, CUDA=True):
    """Turns a feature map into a tensor of bounding box attributes.

    Args:
        prediction:
        in_dim:
        anchors:
        num_classes:
        CUDA: (bool)

    Return:
        prediction:
    """

    batch_size = prediction.size(0)
    stride = in_dim // prediction.size(2)
    grid_size = in_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors,
                                 grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size,
                                 grid_size * grid_size * num_anchors,
                                 bbox_attrs)

    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    # Normalize between 0 and 1
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])  # Center X
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])  # Center Y
    # Object confidence

    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])
    # Add the center offsets
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset),
                           1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset

    # Anchors
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    # Sigmoid
    prediction[:, :, 5:5 + num_classes] = torch.sigmoid(
        (prediction[:, :, 5:5 + num_classes]))

    # Resize map to size of input image by size of stride
    prediction[:, :, :4] *= stride

    return prediction


def bbox_iou(box1, box2):
    """Returns intersection over union between 2 bounding boxes
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Get the corrdinates of intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1,
                             min=0) * torch.clamp(
                                 inter_rect_y2 - inter_rect_y1 + 1, min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def load_classes(namesfile):
    filepath = open(namesfile, 'r')
    names = filepath.read().split("\n")[:-1]
    return names


def letterbox_image(img, in_dim):
    """Add padding to non-square images...

    Not sure how well this works.
    """
    h, w = in_dim
    img_h, img_w = img.shape[0], img.shape[1]
    img_h = int(img_h * min(w / img_w, h / img_h))
    img_w = int(img_w * min(w / img_w, h / img_h))

    resized = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((in_dim[1], in_dim[0], 3), 128)

    canvas[(h - img_h) // 2:(h - img_h) // 2 + img_h,
           (w - img_w) // 2:(w - img_w) // 2 + img_w, :] = resized

    return canvas


def prep_image(img, in_dim):
    """Prepare imgs
    """

    img = letterbox_image(img, (in_dim, in_dim))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)

    return img


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    
    
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    #get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    #Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1,
                             min=0) * torch.clamp(
                                 inter_rect_y2 - inter_rect_y1 + 1, min=0)

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    """Helps make sure results are constrained to "True" detections

    Args:
        prediction:
        confidence: objectness score threshold.
        num_classes: number of classes in dataset.
        nms_conf: (float) non-maximal supression IoU threshold.
    """

    # Zero out every bounding box with an objectness score below threshold.
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask

    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    prediction[:, :, :4] = box_corner[:, :, :4]

    batch_size = prediction.size(0)

    write = False

    for idx in range(batch_size):
        image_pred = prediction[idx]

        # Confidence thresholding, NMS
        max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes],
                                             1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        # Remove low confidence bounding boxes
        non_zero_idx = (torch.nonzero(image_pred[:, 4]))
        try:
            image_pred_ = image_pred[non_zero_idx.squeeze(), :].view(-1, 7)
        except:
            continue

        if image_pred_.shape[0] == 0:
            continue

        # Get classes detected in image.
        img_classes = unique(image_pred_[:, -1])

        for img_cls in img_classes:
            # Setup Non Max supression
            cls_mask = image_pred_ * (
                image_pred_[:, -1] == img_cls).float().unsqueeze(1)
            cls_mask_idx = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = image_pred_[cls_mask_idx].view(-1, 7)

            # Sort with max objectness at top
            conf_sort_idx = torch.sort(image_pred_class[:, 4],
                                       descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_idx]
            pred_idx = image_pred_class.size(0)  # Number of detections

            # Perform NMS
            for i in range(pred_idx):
                # Get all IoUs
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0),
                                    image_pred_class[i + 1:])

                except ValueError:
                    break

                except IndexError:
                    break

                # Zero out where iou > threshold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i + 1:] *= iou_mask

                # Peel non-zero entries
                non_zero_idx = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_idx].view(-1, 7)

            batch_idx = image_pred_class.new(image_pred_class.size(0),
                                             1).fill_(idx)
            seq = batch_idx, image_pred_class

            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    try:
        return output
    except:
        return 0
