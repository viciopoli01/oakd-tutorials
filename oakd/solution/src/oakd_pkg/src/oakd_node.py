#!/usr/bin/env python3
import os
import time
from typing import Optional

import numpy as np
import rospy
import yaml
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import Twist2DStamped, WheelEncoderStamped, EpisodeStart

import basics

if __name__ == "__main__":
    # Initialize the node
    oakd_node = basics.OAKDCameraNode(node_name="oakd_node")
    # Keep it spinning
    rospy.spin()
