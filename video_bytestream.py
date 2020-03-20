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

import urllib.request


def arg_parse():
    """Detect module arguments
    """

    # Set parser
    parser = argparse.ArgumentParser(description="YOLOv3 Detector")

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
    parser.add_argument("--video",
                        dest="video_file",
                        help="Video file for object detection",
                        default="video.avi",
                        type=str)

    return parser.parse_args()


args = arg_parse()
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thresh = float(args.nms_thresh)
video_file = args.video_file
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


def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results
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


# Detection phase
url = 'http://192.168.68.123'

frames = 0

with urllib.request.urlopen(url) as stream:

    bytes = bytearray()

    while True:
        bytes += stream.read(1024)

        # Identify start and end of jpegs
        jpg_head = bytes.find(b'\xff\xd8')
        jpg_end = bytes.find(b'\xff\xd9')

        if jpg_head != -1 and jpg_end != -1:

            jpg = bytes[jpg_head:jpg_end + 2]
            bytes = bytes[jpg_end + 2:]

            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8),
                                 cv2.IMREAD_COLOR)
            img = prep_image(frame, in_dim)

            img_dim = img.shape[1], img.shape[2]
            img_dim = torch.FloatTensor(img_dim).repeat(1, 2)

            if CUDA:
                img_dim = img_dim.cuda()
                img = img.cuda()

            with torch.no_grad():
                output = model(img, CUDA)

            output = write_results(output,
                                   confidence,
                                   num_classes,
                                   nms_conf=nms_thresh)

            if type(output) == int:
                frames += 1
                print("FPS of the video is {:5.4f}".format(
                    frames / (time.time() - start)))

                # img = cv.resize(img, (560, 420))
                cv2.imshow('Stream', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            img_dim = img_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(in_dim / img_dim, 1)[0].view(-1, 1)

            output[:,
                   [1, 3]] -= (in_dim -
                               scaling_factor * img_dim[:, 0].view(-1, 1)) / 2
            output[:,
                   [2, 4]] -= (in_dim -
                               scaling_factor * img_dim[:, 1].view(-1, 1)) / 2

            output[:, 1:5] /= scaling_factor

            for i in range(output.shape[0]):
                output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0,
                                                img_dim[i, 0])
                output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0,
                                                img_dim[i, 1])

            colors = pkl.load(open("palette", "rb"))

            list(map(lambda x: write(x, frame), output))

            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print(time.time() - start)
            print("FPS of the video is {:5.2f}".format(frames /
                                                       (time.time() - start)))
        # else:
        #     break
