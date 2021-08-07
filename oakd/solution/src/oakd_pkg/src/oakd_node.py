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

import basics


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


class OAKDCameraNode(DTROS):
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
            'left': veh_name + '/camera_left_frame',
            'right': veh_name + '/camera_right_frame',
            'rgb': veh_name + '/camera_rgb_frame',
            'disparity': veh_name + '/camera_left_frame'  # disparity/depth are in left camera frame
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
            'left': rospy.Publisher(
                '~image_left/compressed',
                CompressedImage,
                queue_size=1,
                dt_topic_type=TopicType.DRIVER,
                dt_help='The stream of JPEG compressed images from OAK-D left camera'),
            'right': rospy.Publisher(
                '~image_right/compressed',
                CompressedImage,
                queue_size=1,
                dt_topic_type=TopicType.DRIVER,
                dt_help='The stream of JPEG compressed images from OAK-D right camera'),
            'rgb': rospy.Publisher(
                '~image/compressed',
                CompressedImage,
                queue_size=1,
                dt_topic_type=TopicType.DRIVER,
                dt_help='The stream of JPEG compressed images from OAK-D RGB camera'),
            'disparity': rospy.Publisher(
                '~image_disparity/compressed',
                CompressedImage,
                queue_size=1,
                dt_topic_type=TopicType.DRIVER,
                dt_help='Disparity image from OAK-D stereo')
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

        self._pipeline = basics.create_pipeline()
        self._nodes = basics.create_nodes(self._pipeline)
        self._outputs = basics.create_output_links(self._pipeline)
        basics.link_nodes_and_outputs(self._nodes, self._outputs)
        self._device = basics.create_device(self._pipeline)
        self._output_queues = basics.create_output_queues(self._device, self._outputs)

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
            msgs = {'left': None, 'right': None, 'rgb': None, 'disparity': None}
            data = basics.read_data(list(msgs.keys()), self._output_queues)
            for name, im_data in data.items():
                if name in ('left', 'right', 'disparity'):
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
    oakd_node = OAKDCameraNode(node_name="oakd_node")
    # Keep it spinning
    rospy.spin()
