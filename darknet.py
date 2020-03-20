"""A Study implementation of YOLOv3.

Guided from https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-
            detector-from-scratch-in-pytorch-part-2/

Pre-trained weights available at:
wget https://pjreddie.com/media/files/yolov3.weights

2019, Jacob Krajewski
"""
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import Variable
import numpy as np
import cv2
from utils import *

import matplotlib.pyplot as plt


def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416, 416))  #Resize to the input dimension
    img_ = img[:, :, ::-1].transpose(
        (2, 0, 1))  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[
        np.
        newaxis, :, :, :] / 255.0  #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()  #Convert to float
    return img_


def parse_cfg(cfgfile):
    """Builds blocks from a configuration file.
    
    Parses cfg and stores every block as a dict.

    Args:
        cfgfile: (Path) path to yolo3.cfg file. 

    Return:
        blocks: (list of dicts) describes each block to be built in neural network. 
    
    """

    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    # Include all lines with content
    lines = [x for x in lines if len(x) > 0]
    # Exclude comments
    lines = [x for x in lines if x[0] != '#']
    # Remove fringe whitespace
    lines = [x.rstrip().lstrip() for x in lines]

    # Loop over each block
    blocks = []
    block = {}

    for line in lines:
        # If start of new block
        if line[0] == "[":
            # If content
            if len(block) != 0:
                # Add to blocks list
                blocks.append(block)
                # Reinitialize block
                block = {}

            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


def create_modules(blocks):
    """Creates modules from blocks.

    Args:
        blocks: (list of dicts) contains instructions for module creation.

    Returns:
        module_list: (nn.ModuleList Class) used to create nn.Modules.
    """

    net_info = blocks[0]  # First block designated as info.
    module_list = nn.ModuleList()

    # Intialize filter depth to 3 for RGB (1 for black and white img).
    # Used to define the depth of conv layer during initialization.
    prev_filters = 3
    # List of all prev_filters
    output_filters = []

    # Iterate over blocks, create module for each block.
    for i, blk in enumerate(blocks[1:]):
        module = nn.Sequential()

        # Check block type
        # Create new module for block
        # Append to module_list

        if blk["type"] == "convolutional":
            activation = blk["activation"]
            try:
                batch_normalize = int(blk["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(blk["filters"])
            padding = int(blk["pad"])
            kernel_size = int(blk["size"])
            stride = int(blk["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # Add Conv layer
            conv = nn.Conv2d(prev_filters,
                             filters,
                             kernel_size,
                             stride,
                             pad,
                             bias=bias)
            module.add_module(f"conv_{i}", conv)

            # Add batch norm layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module(f"batch_norm_{i}", bn)

            # Activation layer
            if activation == "leaky":
                actvn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module(f"leaky_{i}", actvn)

        elif blk["type"] == "upsample":
            stride = int(blk["stride"])
            upsample = nn.Upsample(scale_factor=2, mode="bilinear")
            module.add_module(f"upsample_{i}", upsample)

        elif blk["type"] == "route":
            blk["layers"] = blk["layers"].split(",")
            # Route start
            start = int(blk["layers"][0])
            # Check if end
            try:
                end = int(blk["layers"][1])
            except:
                end = 0

            # Positive Annotation
            if start > 0:
                start = start - i
            if end > 0:
                end = end - i

            route = EmptyLayer()
            module.add_module(f"route_{i}", route)
            if end < 0:
                filters = output_filters[i + start] + output_filters[i + end]
            else:
                filters = output_filters[i + start]

        # a skip-connection
        elif blk["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module(f"shortcut_{i}", shortcut)

        elif blk["type"] == "yolo":
            masks = blk["mask"].split(",")
            masks = [int(mask) for mask in masks]

            anchors = blk["anchors"].split(",")
            anchors = [int(anchor) for anchor in anchors]
            anchors = [
                (anchors[j], anchors[j + 1]) for j in range(0, len(anchors), 2)
            ]
            anchors = [anchors[j] for j in masks]

            detection = DetectionLayer(anchors)
            module.add_module(f"detection_{i}", detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return net_info, module_list


class EmptyLayer(nn.Module):

    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):

    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


class Darknet(nn.Module):
    """
    Todo:
        Figure out what's going on with route layers, because I just breezed
        past that part. Same with shortcut. Something to do with recursion maybe?
    """

    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}  # To cache the outputs

        write = 0  #This is explained a bit later

        # Run inputs through each module to get outputs.
        for i, module in enumerate(modules):
            # Check the type of layer to look at.
            module_type = (module["type"])
            if module_type == "convolutional" or module_type == "upsample":
                # If conv or upsamp, pass input through layer.
                x = self.module_list[i](x)

            elif module_type == "route":

                # Get layer numbers from cfg
                layers = module["layers"]
                layers = [int(layer) for layer in layers]

                # This works by going backwards from the current layer by
                # the amount specified in cfg.
                if layers[0] > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)

            elif module_type == "shortcut":
                # Accepts outputs from layer listed in cfg.
                from_ = int(module["from"])
                x = outputs[i - 1] + outputs[i + from_]

            elif module_type == "yolo":

                anchors = self.module_list[i][0].anchors

                # Get input dims
                in_dim = int(self.net_info["height"])

                # Get num_classes
                num_classes = int(module["classes"])

                # exit()
                # Transform
                x = x.data
                x = predict_transform(x, in_dim, anchors, num_classes, CUDA)
                # Collector initialised?
                if not write:
                    detections = x
                    write = 1

                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x

        return detections

    def load_weights(self, weightfile):
        """Load pre-trained weights

        Available at:
            wget https://pjreddie.com/media/files/yolov3.weights
        """

        # Load weights:
        filepath = open(weightfile, "rb")

        # First 5 values are header info.
        # 1. Major version
        # 2. Minor version
        # 3. Patch version
        # 4, 5. Images seen by network (during training)

        header = np.fromfile(filepath, dtype=np.int32, count=5)
        # from_numpy inherits datatype.
        self.header = torch.from_numpy(header)

        weights = np.fromfile(filepath, dtype=np.float32)

        # iterate over weights, load into modules
        ptr = 0

        # Check if convolutional, otherwise ignore
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if batch_normalize:
                    bn = model[1]

                    # Get number of weights from batch norm layer.
                    num_bn_biases = bn.bias.numel()

                    # Load the weights:
                    bn_biases = torch.from_numpy(weights[ptr:ptr +
                                                         num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr:ptr +
                                                          num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr:ptr +
                                                               num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr:ptr +
                                                              num_bn_biases])
                    ptr += num_bn_biases

                    # (f, there has to be an easier way of doing this...)
                    #Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    #Number of biases
                    num_biases = conv.bias.numel()

                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr:ptr +
                                                           num_biases])
                    ptr = ptr + num_biases

                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


# model = Darknet("cfg/yolov3.cfg")
# model.load_weights("yolov3.weights")

# inp = get_test_input()
# pred = model(inp, False)
# print(pred.shape)

# blocks = parse_cfg("cfg/yolov3.cfg")
# print(create_modules(blocks))