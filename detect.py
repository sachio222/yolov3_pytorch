# -*- coding: utf-8 -*-
"""A Study implementation of YOLOv3.

Guided from https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-
            detector-from-scratch-in-pytorch-part-2/
            
2019, Jacob Krajewski
"""

from __future__ import division

import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2

from utils import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random


def arg_parse():
    """Detect module arguments
    """

    # Set parser
    parser = argparse.ArgumentParser(description="YOLOv3 Detector")

    parser.add_argument("--img",
                        dest="image_loc",
                        help="File/dir for object detection",
                        default="img",
                        type=str)
    parser.add_argument("--det",
                        dest="detec_loc",
                        help="File/dir for storing detections",
                        default="det",
                        type=str)
    parser.add_argument("--bs",
                        dest="bs",
                        help="Batch size (default: 1)",
                        default=1)
    parser.add_argument("--confidence",
                        dest="confidence",
                        help="Object confidence filter",
                        default=0.5)
    parser.add_argument("--nms_thresh",
                        dest="nms_thresh",
                        help="Non maximum suppression threshold",
                        default=0.4)
    parser.add_argument("--cfg",
                        dest="cfg_file",
                        help="Path to cfg file. Def: cfg/yolov3.cfg",
                        default="cfg/yolov3.cfg",
                        type=str)
    parser.add_argument("--weights",
                        dest="weights_file",
                        help="Path to .weights file",
                        default="yolov3.weights",
                        type=str)
    parser.add_argument(
        "--img_res",
        dest="img_res",
        help="Input resolution. Higher: more accurate, lower: faster",
        default="416",
        type=str)

    return parser.parse_args()


args = arg_parse()
image_loc = args.image_loc
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thresh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

classes = load_classes("data/coco.names")
num_classes = len(classes)  #For COCO

# Set up network, load weights
print("Loading network.....")
model = Darknet(args.cfg_file)
model.load_weights(args.weights_file)
print("Network successfully loaded.")

model.net_info["height"] = args.img_res
in_dim = int(model.net_info["height"])

# Why?
assert in_dim % 32 == 0
assert in_dim > 32

if CUDA:
    model.cuda()

model.eval()

# Read images from disk
read_dir = time.time()  # Checkpoint

# Detection phase
try:
    img_list = [
        osp.join(osp.realpath("."), image_loc, img)
        for img in os.listdir(image_loc)
    ]
except NotADirectoryError:
    img_list = []
    img_list.append(osp.join(osp.realpath("."), image_loc))
except FileNotFoundError:
    print("\nHmmm, no file or directory found. Check spelling and try again.\n")
    exit()

if not os.path.exists(args.detec_loc):
    os.makedirs(args.detec_loc)

load_batch = time.time()  # Checkpoint!
loaded_imgs = [cv2.imread(img) for img in img_list]

img_batches = list(
    map(prep_image, loaded_imgs, [in_dim for x in range(len(img_list))]))

#List containing dimensions of original images
orig_img_dims = [(img.shape[1], img.shape[0]) for img in loaded_imgs]
orig_img_dims = torch.FloatTensor(orig_img_dims).repeat(1, 2)

# Create Batches
leftover = 0
if len(orig_img_dims) % batch_size:
    leftover = 1

if batch_size != 1:
    num_batches = len(img_list) // batch_size + leftover
    img_batches = [
        torch.cat(
            (img_batches[i * batch_size:min((i + 1) *
                                            batch_size, len(img_batches))]))
        for i in range(num_batches)
    ]

write = 0

if CUDA:
    orig_img_dims = orig_img_dims.cuda()

start_det_loop = time.time()
for i, batch in enumerate(img_batches):
    # load the img
    start = time.time()
    if CUDA:
        batch = batch.cuda()

    with torch.no_grad():
        prediction = model(batch, CUDA)

    prediction = write_results(prediction,
                               confidence,
                               num_classes,
                               nms_conf=nms_thresh)
    end = time.time()

    if type(prediction) == int:
        for img_num, img in enumerate(
                img_list[i * batch_size:min((i + 1) *
                                            batch_size, len(img_list))]):
            img_id = i * batch_size + img_num
            print("{0:20s} predicted in {1:6.3f} seconds").format(
                img.split("/")[-1], (end - start) / batch_size)
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            print("---------------------------------------------------")
        continue

    prediction[:, 0] += i * batch_size

    if not write:
        output = prediction
        write = 1
    else:
        output = torch.cat((output, prediction))

    for img_num, img in enumerate(
            img_list[i * batch_size:min((i + 1) * batch_size, len(img_list))]):

        img_id = i * batch_size + img_num
        objs = [classes[int(op[-1])] for op in output if int(op[0]) == img_id]
        print("{0:20s} predicted in {1:6.3f} seconds".format(
            img.split("/")[-1], (end - start) / batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("---------------------------------------------------")

    if CUDA:
        # Keeps time accurate due to GPU and CPU being synched.
        torch.cuda.synchronize()

# DRAW THE BOUNDING BOXES
try:
    output
except NameError:
    print(u"Womp! No detections here... ¯\\_(ツ)_/¯")
    exit()

# Resize coords to fit input image

orig_img_dims = torch.index_select(orig_img_dims, 0, output[:, 0].long())

scaling_factor = torch.min(in_dim / orig_img_dims, 1)[0].view(-1, 1)

output[:, [1, 3]] -= (in_dim -
                      scaling_factor * orig_img_dims[:, 0].view(-1, 1)) / 2
output[:, [2, 4]] -= (in_dim -
                      scaling_factor * orig_img_dims[:, 1].view(-1, 1)) / 2

# Undo letterboxing.
output[:, 1:5] /= scaling_factor

# Clip bounding boxes that go outsid the image boundaries.
for i in range(output.shape[0]):
    output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, orig_img_dims[i, 0])
    output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, orig_img_dims[i, 1])

# Load preset color palette
output_recast = time.time()
class_load = time.time()
colors = pkl.load(open("palette", "rb"))

draw = time.time()

# def write(x, results):
# """Draws rectangle of randomized color, and labels detection
# """
# c1 = tuple(x[1:3].int())
# c2 = tuple(x[3:5].int())
# img = results[int(x[0])]
# cls = int(x[-1])
# color = random.choice(colors)
# # Draw rect
# label = "{0}".format(classes[cls])
# cv2.rectangle(img, c1, c2, color, 1)
# t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
# c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
# cv2.rectangle(img, c1, c2, color, -1)
# cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4),
#             cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)

# return img


def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4),
                cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img


# This modifies images in place by mapping drawing
# to loaded images
list(map(lambda x: write(x, loaded_imgs), output))

det_names = pd.Series(img_list).apply(
    lambda x: "{}/det_{}".format(args.detec_loc,
                                 x.split("/")[-1]))

list(map(cv2.imwrite, det_names, loaded_imgs))

end = time.time()

print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
print("{:25s}: {:2.3f}".format("Detection (" + str(len(img_list)) + " images)",
                               output_recast - start_det_loop))
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
print("{:25s}: {:2.3f}".format("Average time_per_img",
                               (end - load_batch) / len(img_list)))
print("----------------------------------------------------------")

torch.cuda.empty_cache()
