#!/usr/bin/env python
# coding: utf-8

# In[23]:


import os
import sys
import time
import yaml
import numpy as np
import cv2
from typing import List, Dict, Callable, Optional, Any

import depthai as dai


# In[24]:


def create_pipeline() -> dai.Pipeline:
    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_3)
    return pipeline


# In[25]:


def create_nodes(pipeline: dai.Pipeline, model_path: str) -> Dict[str, dai.Node]:
    # RGB CAMERA
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(
        dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setColorOrder(
        dai.ColorCameraProperties.ColorOrder.RGB)
    cam_rgb.setInterleaved(False)
    cam_rgb.setPreviewSize(640, 400)

    # SEMANTIC SEGMENTATION NEURAL NETWORK
    blobfile = model_path
    if not os.path.exists(blobfile) or not os.path.isfile(blobfile):
        print('Model not found! Blob file: %s' % blobfile)
        assert False
    segmentation_nn = pipeline.createNeuralNetwork()
    segmentation_nn.setBlobPath(blobfile)
    segmentation_nn.input.setBlocking(False)
    segmentation_nn.setNumInferenceThreads(2)  # 3 = max for OAK-D

    # IMAGE MANIPULATION
    manip_rgb = pipeline.createImageManip()
    manip_rgb.initialConfig.setResize(640, 480)
    manip_rgb.setWaitForConfigInput(False)
    manip_rgb.setKeepAspectRatio(False)

    return {
        'rgb': cam_rgb,
        'manip': manip_rgb,
        'seg': segmentation_nn
    }


# In[26]:


def create_output_links(pipeline: dai.Pipeline) -> Dict[str, dai.XLinkOut]:
    xout_links = {
        'rgb': pipeline.createXLinkOut(),
        'seg': pipeline.createXLinkOut()
    }
    for name, xout_link in xout_links.items():
        xout_link.setStreamName(name)
        xout_link.input.setBlocking(False)
    return xout_links


# In[27]:


def link_nodes_and_outputs(nodes: Dict[str, dai.Node],
                           outputs: Dict[str, dai.XLinkOut]) -> None:
    # Raw RGB image --> Output to host
    nodes['rgb'].preview.link(outputs['rgb'].input)
    # Raw RGB image --> Image manipulation module
    nodes['rgb'].preview.link(nodes['manip'].inputImage)
    # Manipulated RGB image --> Neural network
    nodes['manip'].out.link(nodes['seg'].input)
    # Segmentation image --> Output to host
    nodes['seg'].out.link(outputs['seg'].input)


# In[28]:


def create_device(pipeline: dai.Pipeline) -> dai.Device:
    device = dai.Device(pipeline, usb2Mode=False)
    device.setLogLevel(dai.LogLevel.DEBUG)
    return device


# In[29]:


def create_output_queues(device: dai.Device,
                         outputs: Dict[str, dai.XLinkOut]) -> Dict[str, dai.DataOutputQueue]:
    queues = dict()
    for name, xout_link in outputs.items():
        queues[name] = device.getOutputQueue(name=name, maxSize=1, blocking=False)
    return queues


# In[30]:


def read_data(requested_data: List[str],
              output_queues: Dict[str, dai.DataOutputQueue]) -> Dict[str, Any]:
    data = {name: None for name in requested_data}
    for name in requested_data:
        if name not in output_queues.keys():
            continue
        if name == 'rgb':
            data[name] = output_queues[name].get().getCvFrame()
        elif name == 'seg':
            # Read raw bytes from the final layer (output) of the neural net
            segmentation = output_queues[name].get().getLayerFp16('output')
            # Convert to numpy array
            segmentation = np.asarray(segmentation, dtype=np.float16)
            # Reshape to (C, H, W)
            segmentation = segmentation.reshape((-1, 480, 640))
            # Compress 1-hot encoding of classes to a single dimension
            segmentation = np.argmax(segmentation, axis=0).astype(np.uint8)
            # Resize to the same image size as the original RGB
            segmentation = cv2.resize(segmentation, (640, 400), cv2.INTER_NEAREST)
            # Save to dictionary
            data[name] = segmentation
    return data

