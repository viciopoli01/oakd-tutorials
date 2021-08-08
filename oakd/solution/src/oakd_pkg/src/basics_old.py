#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import sys
import time
import yaml
import rospy
import numpy as np
import cv2
from typing import List, Dict, Callable, Optional, Any

from sensor_msgs.msg import CompressedImage, Image

import depthai as dai


# In[2]:


def create_pipeline() -> dai.Pipeline:
    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_3)
    return pipeline


# In[3]:


def create_nodes(pipeline: dai.Pipeline) -> Dict[str, dai.Node]:
    # LEFT CAMERA
    cam_left = pipeline.createMonoCamera()
    cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    cam_left.setResolution(
        dai.MonoCameraProperties.SensorResolution.THE_400_P)

    # RIGHT CAMERA
    cam_right = pipeline.createMonoCamera()
    cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    cam_right.setResolution(
        dai.MonoCameraProperties.SensorResolution.THE_400_P)

    # RGB CAMERA
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(
        dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setColorOrder(
        dai.ColorCameraProperties.ColorOrder.RGB)
    cam_rgb.setInterleaved(False)
    cam_rgb.setPreviewSize(640, 400)

    # STEREO
    stereo = pipeline.createStereoDepth()
    stereo.setConfidenceThreshold(150)
    stereo.setMedianFilter(
        dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
    stereo.setLeftRightCheck(False)
    stereo.setExtendedDisparity(False)  # True = better for short-range
    stereo.setSubpixel(False)  # True = better for long-range
    stereo.setRectifyMirrorFrame(False)

    return {
        'left': cam_left,
        'right': cam_right,
        'rgb': cam_rgb,
        'stereo': stereo
    }


# In[4]:


def create_output_links(pipeline: dai.Pipeline) -> Dict[str, dai.XLinkOut]:
    xout_links = {
        'left': pipeline.createXLinkOut(),
        'right': pipeline.createXLinkOut(),
        'rgb': pipeline.createXLinkOut(),
        'disparity': pipeline.createXLinkOut()
    }
    for name, xout_link in xout_links.items():
        xout_link.setStreamName(name)
        xout_link.input.setBlocking(False)
    return xout_links


# In[5]:


def link_nodes_and_outputs(nodes: Dict[str, dai.Node],
                           outputs: Dict[str, dai.XLinkOut]) -> None:
    # Raw RGB image --> Image manipulation module
    nodes['rgb'].preview.link(outputs['rgb'].input)
    # Disparity --> link the raw left & right images as input to stereo
    nodes['left'].out.link(nodes['stereo'].left)
    nodes['right'].out.link(nodes['stereo'].right)
    nodes['stereo'].disparity.link(outputs['disparity'].input)
    nodes['stereo'].rectifiedLeft.link(outputs['left'].input)
    nodes['stereo'].rectifiedRight.link(outputs['right'].input)


# In[6]:


def create_device(pipeline: dai.Pipeline) -> dai.Device:
    device = dai.Device(pipeline, usb2Mode=False)
    device.setLogLevel(dai.LogLevel.DEBUG)
    return device


# In[7]:


def create_output_queues(device: dai.Device,
                         outputs: Dict[str, dai.XLinkOut]) -> Dict[str, dai.DataOutputQueue]:
    queues = dict()
    for name, xout_link in outputs.items():
        queues[name] = device.getOutputQueue(name=name, maxSize=1, blocking=False)
    return queues


# In[8]:


def read_data(requested_data: List[str],
              output_queues: Dict[str, dai.DataOutputQueue]) -> Dict[str, Any]:
    data = {name: None for name in requested_data}
    print("Read image")
    for name in requested_data:
        if name not in output_queues.keys():
            continue
        if name in ('left', 'right', 'rgb'):
            data[name] = output_queues[name].get().getCvFrame()
        elif name == 'disparity':
            data[name] = output_queues[name].get().getFrame()
    return data

