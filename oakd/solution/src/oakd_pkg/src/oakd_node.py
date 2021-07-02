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

from sensor_msgs.msg import CompressedImage, CameraInfo
from sensor_msgs.srv import SetCameraInfo, SetCameraInfoResponse

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
    """Handles the imagery.

    The node handles the image stream, initializing it, publishing frames
    according to the required frequency and stops it at shutdown.

    Note that only one instance of this class should be used at a time.
    If another node tries to start an instance while this node is running,
    it will likely fail with an `Out of resource` exception.

    The configuration parameters can be changed dynamically while the node is running via
    `rosparam set` commands.

    Configuration:
        ~framerate (:obj:`float`): The camera image acquisition framerate, default is 20.0 fps
        ~res_w (:obj:`int`): The desired width of the acquired image, default is 640px
        ~res_h (:obj:`int`): The desired height of the acquired image, default is 480px

    Publisher:
        ~image/compressed (:obj:`CompressedImage`): The acquired camera images
        ~camera_info (:obj:`CameraInfo`): The camera parameters

    Service:
        ~set_camera_info:
            Saves a provided camera info
            to `/data/config/calibrations/camera_intrinsic/HOSTNAME.yaml`.

            input:
                camera_info (`CameraInfo`): The camera information to save

            outputs:
                success (`bool`): `True` if the call succeeded
                status_message (`str`): Used to give details about success

    """

    def __init__(self):
        # Initialize the DTROS parent class
        super(OAKDCameraNode, self).__init__(
            node_name='camera',
            node_type=NodeType.DRIVER,
            help="Camera driver for reading and publishing OAK-D images"
        )
        # Add the node parameters to the parameters dictionary and load their default values
        self._res_w = DTParam(
            '~res_w',
            param_type=ParamType.INT,
            default=640,
            help="Horizontal resolution (width) of the produced image frames."
        )
        self._res_h = DTParam(
            '~res_h',
            param_type=ParamType.INT,
            default=400,
            help="Vertical resolution (height) of the produced image frames."
        )
        self._framerate = DTParam(
            '~framerate',
            param_type=ParamType.INT,
            default=20,
            help="Framerate at which images frames are produced"
        )

        # define parameters
        self._framerate.register_update_callback(self.parameters_updated)
        self._res_w.register_update_callback(self.parameters_updated)
        self._res_h.register_update_callback(self.parameters_updated)

        # intrinsic calibration
        veh_name = rospy.get_namespace().rstrip('/')
        self.frame_ids = {
            'left': veh_name + '/camera_left_frame',
            'right': veh_name + '/camera_right_frame',
            'rgb': veh_name + '/camera_rgb_frame',
            'disparity': veh_name + '/camera_left_frame'  # disparity/depth are in left camera frame
        }
        self.cali_file_folder = '/data/config/calibrations/camera_intrinsic/oakd/'

        cali_files = {
            'left': self.cali_file_folder + veh_name + '_left.yaml',
            'right': self.cali_file_folder + veh_name + '_right.yaml',
            'rgb': self.cali_file_folder + veh_name + '_rgb.yaml'
        }

        for cam, cali_file in cali_files.items():
            # locate calibration yaml file or use the default otherwise
            if not os.path.isfile(cali_file):
                self.logwarn('Calibration not found: %s.\n Using default instead.' % cali_file)
                cali_files[cam] = None

        # load the calibration file
        self.original_camera_infos = dict()
        for cam, cali_file in cali_files.items():
            if cali_file is None:
                self.original_camera_infos[cam] = self.default_intrinsics(cam)
                cali_file = '(default)'
            else:
                self.original_camera_infos[cam] = self.load_camera_info(cali_file)
            self.log('For camera %s, using calibration file: %s' % (cam, cali_file))
            self.original_camera_infos[cam].header.frame_id = self.frame_ids[cam]

        self.current_camera_infos = copy.deepcopy(self.original_camera_infos)
        for cam in cali_files.keys():
            self.update_camera_params(cam)

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
                '~image_rgb/compressed',
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

        self.pub_camera_infos = {
            'left': rospy.Publisher(
                '~left_camera_info',
                CameraInfo,
                queue_size=1,
                dt_topic_type=TopicType.DRIVER,
                dt_help='Left camera calibration information, the message content is fixed'),
            'right': rospy.Publisher(
                '~right_camera_info',
                CameraInfo,
                queue_size=1,
                dt_topic_type=TopicType.DRIVER,
                dt_help='Right camera calibration information, the message content is fixed'),
            'rgb': rospy.Publisher(
                '~rgb_camera_info',
                CameraInfo,
                queue_size=1,
                dt_topic_type=TopicType.DRIVER,
                dt_help='RGB camera calibration information, the message content is fixed')
        }

        # Setup service (for camera_calibration)
        self.srv_set_camera_infos = {
            'left': rospy.Service(
                '~set_left_camera_info',
                SetCameraInfo,
                self.srv_set_left_camera_info_cb),
            'right': rospy.Service(
                '~set_right_camera_info',
                SetCameraInfo,
                self.srv_set_right_camera_info_cb),
            'rgb': rospy.Service(
                '~set_rgb_camera_info',
                SetCameraInfo,
                self.srv_set_rgb_camera_info_cb)
        }
        # ---
        self.log('[OAKDCameraNode]: Initialized.')

    @property
    def is_stopped(self):
        return self._is_stopped

    def parameters_updated(self):
        self.stop()
        for cam in self.original_camera_infos.keys():
            self.update_camera_params(cam)
        self.start()

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
            # publish camera info
            if camera_name in ('left', 'right', 'rgb'):
                self.current_camera_infos[camera_name].header.stamp = stamp
                self.pub_camera_infos[camera_name].publish(
                    self.current_camera_infos[camera_name])

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

    def srv_set_left_camera_info_cb(self, req):
        self.log('[srv_set_left_camera_info_cb] Callback!')
        return self.srv_set_camera_info_cb(req, 'left')

    def srv_set_right_camera_info_cb(self, req):
        self.log('[srv_set_right_camera_info_cb] Callback!')
        return self.srv_set_camera_info_cb(req, 'right')

    def srv_set_rgb_camera_info_cb(self, req):
        self.log('[srv_set_rgb_camera_info_cb] Callback!')
        return self.srv_set_camera_info_cb(req, 'rgb')

    def srv_set_camera_info_cb(self, req, cam):
        filename = self.cali_file_folder + rospy.get_namespace().strip('/') + ('_%s.yaml' % cam)
        response = SetCameraInfoResponse()
        response.success = self.save_camera_info(req.camera_info, filename, cam)
        response.status_message = 'Wrote to %s' % filename
        return response

    def save_camera_info(self, camera_info_msg, filename, camera_name):
        """Saves intrinsic calibration to file.

            Args:
                camera_info_msg (:obj:`CameraInfo`): Camera Info containing calibration
                filename (:obj:`str`): filename where to save calibration
                camera_name (:obj:`str`): OAK-D camera name (left, right, rgb)
        """
        # Convert camera_info_msg and save to a yaml file
        self.log('[save_camera_info] filename: %s' % filename)

        # Converted from camera_info_manager.py
        calib = {
            'image_width': camera_info_msg.width,
            'image_height': camera_info_msg.height,
            'camera_name': rospy.get_name().lstrip('/').split('/')[0] + ('_%s' % camera_name),
            'distortion_model': camera_info_msg.distortion_model,
            'distortion_coefficients': {
                'data': camera_info_msg.D,
                'rows': 1,
                'cols': 5
            },
            'camera_matrix': {
                'data': camera_info_msg.K,
                'rows': 3,
                'cols': 3
            },
            'rectification_matrix': {
                'data': camera_info_msg.R,
                'rows': 3,
                'cols': 3
            },
            'projection_matrix': {
                'data': camera_info_msg.P,
                'rows': 3,
                'cols': 4
            }
        }
        self.log('[save_camera_info] calib %s' % calib)
        try:
            f = open(filename, 'w')
            yaml.safe_dump(calib, f)
            return True
        except IOError:
            return False

    def update_camera_params(self, camera_name):
        """ Update the camera parameters based on the current resolution.

        The camera matrix, rectification matrix, and projection matrix depend on
        the resolution of the image.
        As the calibration has been done at a specific resolution, these matrices need
        to be adjusted if a different resolution is being used.
        """
        original_info = self.original_camera_infos[camera_name]
        scale_width = float(self._res_w.value) / original_info.width
        scale_height = float(self._res_h.value) / original_info.height

        scale_matrix = np.ones(9)
        scale_matrix[0] *= scale_width
        scale_matrix[2] *= scale_width
        scale_matrix[4] *= scale_height
        scale_matrix[5] *= scale_height

        # adjust the camera matrix resolution
        self.current_camera_infos[camera_name].height = self._res_h.value
        self.current_camera_infos[camera_name].width = self._res_w.value

        # adjust the K matrix
        self.current_camera_infos[camera_name].K = np.array(original_info.K) * scale_matrix

        # adjust the P matrix
        scale_matrix = np.ones(12)
        scale_matrix[0] *= scale_width
        scale_matrix[2] *= scale_width
        scale_matrix[5] *= scale_height
        scale_matrix[6] *= scale_height
        self.current_camera_infos[camera_name].P = np.array(original_info.P) * scale_matrix

    @staticmethod
    def load_camera_info(filename):
        """Loads the camera calibration files.

        Loads the intrinsic and extrinsic camera matrices.

        Args:
            filename (:obj:`str`): filename of calibration files.

        Returns:
            :obj:`CameraInfo`: a CameraInfo message object

        """
        with open(filename, 'r') as stream:
            calib_data = yaml.safe_load(stream)
        cam_info = CameraInfo()
        cam_info.width = calib_data['image_width']
        cam_info.height = calib_data['image_height']
        cam_info.K = calib_data['camera_matrix']['data']
        cam_info.D = calib_data['distortion_coefficients']['data']
        cam_info.R = calib_data['rectification_matrix']['data']
        cam_info.P = calib_data['projection_matrix']['data']
        cam_info.distortion_model = calib_data['distortion_model']
        return cam_info

    @staticmethod
    def default_intrinsics(camera_name):
        calib = CameraInfo()
        calib.width = 640
        calib.height = 400
        calib.K = [320.0, 0.0, 320.0,
                   0.0, 320.0, 200.0,
                   0.0, 0.0, 1.0]
        calib.D = [0.0, 0.0, 0.0, 0.0, 0.0]
        calib.R = [1.0, 0.0, 0.0,
                   0.0, 1.0, 0.0,
                   0.0, 0.0, 1.0]
        calib.P = [200.0, 0.0, 320.0, 0.0,
                   0.0, 200.0, 320.0, 0.0,
                   0.0, 0.0, 1.0, 0.0]
        return calib

if __name__ == "__main__":
    # Initialize the node
    oakd_node = OAKDCameraNode(node_name="oakd_node")
    # Keep it spinning
    rospy.spin()
