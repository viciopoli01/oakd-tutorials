#!/usr/bin/env python3
import os
import time
import copy
from typing import Optional
import rospy
import yaml
from threading import Thread
import numpy as np
import cv2

from sensor_msgs.msg import CompressedImage

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType

import semantic_segmentation


def rgb8_to_compressed_imgmsg(im: np.ndarray) -> CompressedImage:
    msg = CompressedImage()
    msg.header.stamp = rospy.Time.now()
    msg.format = 'jpeg'
    msg.data = cv2.imencode('.jpeg', im)[1].tobytes()
    # ---
    return msg


def mono8_to_compressed_imgmsg(im: np.ndarray) -> CompressedImage:
    msg = CompressedImage()
    msg.header.stamp = rospy.Time.now()
    msg.format = 'jpeg'
    msg.data = cv2.imencode('.jpeg', im)[1].tobytes()
    # ---
    return msg


class SegmentationCameraNode(DTROS):
    """Handles the camera output of the OAK-D device from DepthAI.

    Note that only one instance of this class should be used at a time.
    If another node tries to start an instance while this node is running,
    it will likely fail with an `Out of resource` exception.

    Publisher:
        ~image/compressed (:obj:`CompressedImage`): The acquired camera images

    """

    def __init__(self):
        # Initialize the DTROS parent class
        super(OAKDCameraNode, self).__init__(
            node_name='camera',
            node_type=NodeType.DRIVER,
            help="Camera driver for reading and publishing OAK-D images"
        )

        # Frame IDs
        veh_name = rospy.get_namespace().rstrip('/')
        self.frame_ids = {
            'rgb': veh_name + '/camera_rgb_frame',
            'seg': veh_name + '/camera_rgb_frame'  # segmentation is in the RGB frame
        }

        # OAK-D interface
        self._pipeline = None
        self._nodes = None
        self._outputs = None
        self._device = None
        self._output_queues = None

        # Setup publishers
        self._is_stopped = False
        self._worker = None

        self.pub_images = {
            'rgb': rospy.Publisher(
                '~image/compressed',
                CompressedImage,
                queue_size=1,
                dt_topic_type=TopicType.DRIVER,
                dt_help='The stream of JPEG compressed images from OAK-D RGB camera'),
            'seg': rospy.Publisher(
                '~image_segmentation/compressed',
                CompressedImage,
                queue_size=1,
                dt_topic_type=TopicType.DRIVER,
                dt_help='Semantic segmentation per-pixel classes')
        }

        # ---
        self.log('[OAKDCameraNode]: Initialized.')

    @property
    def is_stopped(self):
        return self._is_stopped

    def publish(self, image_msgs):
        # add time to messages
        stamp = rospy.Time.now()

        for camera_name, image_msg in image_msgs.items():
            if type(image_msg) is not CompressedImage:
                continue

            image_msg.header.stamp = stamp
            # update camera frame
            image_msg.header.frame_id = self.frame_ids[camera_name]
            # publish image
            self.pub_images[camera_name].publish(image_msg)
            # save modified image message
            image_msgs[camera_name] = image_msg

    def start(self):
        """
        Begins the camera capturing.
        """
        self.log('Start capturing.')
        # ---
        try:
            self.setup()
            # run camera thread
            self._worker = Thread(target=self.run)
            self._worker.start()
        except StopIteration:
            self.log('Exception thrown.')

    def stop(self):
        self.loginfo('Stopping OAK-D...')
        self._is_stopped = True
        # wait for the camera thread to finish
        if self._worker is not None:
            self._worker.join()
            time.sleep(2)
        self._worker = None
        # release resources
        self.release()
        time.sleep(2)
        self._is_stopped = False
        self.loginfo('OAK-D stopped.')

    def setup(self):
        if self._device is not None:
            # Close device if it was previously initialized
            self._device.close()
            time.sleep(2)

        ###### OAK-D Configuration ######

        self._pipeline = semantic_segmentation.create_pipeline()
        self._nodes = semantic_segmentation.create_nodes(self._pipeline)
        self._outputs = semantic_segmentation.create_output_links(self._pipeline)
        semantic_segmentation.link_nodes_and_outputs(self._nodes, self._outputs)
        self._device = semantic_segmentation.create_device(self._pipeline)
        self._output_queues = semantic_segmentation.create_output_queues(self._device, self._outputs)

    def release(self):
        if self._device is not None:
            self.loginfo('Releasing depthai device...')
            try:
                self._device.close()
            except Exception:
                pass
            self.loginfo('depthai device released.')
        self._device = None

    def cv2jpeg_msg(self, cv_image, is_rgb):
        if is_rgb:
            return rgb8_to_compressed_imgmsg(cv_image)
        else:
            return mono8_to_compressed_imgmsg(cv_image)

    def run(self):
        """ Image capture procedure.
            Captures images from OAK-D and publishes them
        """
        if (self._device is None) or (self._pipeline is None):
            self.logerr('Device was not initialized!')
            return

        # keep reading
        while (not self.is_stopped) and (not self.is_shutdown):
            msgs = {'rgb': None, 'seg': None}
            data = semantic_segmentation.read_data(list(msgs.keys()), self._output_queues)
            for name, im_data in data.items():
                if name == 'seg':
                    msgs[name] = self.cv2jpeg_msg(im_data, is_rgb=False)
                elif name == 'rgb':
                    msgs[name] = self.cv2jpeg_msg(im_data, is_rgb=True)

            got = list(sorted([k for k, m in msgs.items() if m is not None]))
            if len(got) > 0:
              self.log('Got {} from OAK-D'.format(', '.join(got)))
              self.publish(msgs)
            else:
              self.log('No images were read from OAK-D!')

        self.loginfo('Camera worker stopped.')

    def on_shutdown(self):
        self.stop()

if __name__ == "__main__":
    # Initialize the node
    oakd_node = SegmentationCameraNode(node_name="oakd_segmentation_node")
    # Keep it spinning
    rospy.spin()
