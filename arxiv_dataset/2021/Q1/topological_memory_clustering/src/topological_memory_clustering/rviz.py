from __future__ import print_function

import rospy
from visualization_msgs.msg import Marker,MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from copy import deepcopy
import numpy as np

# Init - place this into your script!
'''
rospy.init_node('vis')
pub = rospy.Publisher('/quadrotor_paths', MarkerArray, queue_size=1)
'''

class_colors_rviz = [ColorRGBA(1,165/255.,0,1),
                     ColorRGBA(0,0,1,1),
                     ColorRGBA(1,0,0,1),
                     ColorRGBA(0,1,0,1),
                     ColorRGBA(1,0,1,1),
                     ColorRGBA(160/255.,32/255.,240/255.,1),
                     ColorRGBA(0,1,1,1)]
# RGBA(255, 165, 0, 1)
# 0,0,1,1
# 1,0,0,1
# 0,1,0,1
# 1,0,1,1
# (160, 32, 240, 1)


def get_delete_all_marker_array():
    # Delete
    ma = MarkerArray()
    m = Marker()
    m.action = m.DELETEALL
    ma.markers.append(m)
    return ma


def get_quadrotor_paths_as_marker_array(X, idx=None, debug=False):
    # New
    ma = MarkerArray()

    for i in range(X.shape[0]):
        m = Marker()
        m.action = m.ADD
        m.type = m.LINE_STRIP
        m.header.frame_id = 'exotica/world_frame'
        m.pose.orientation.w = 1.0
        m.scale.x = 0.03
        if idx is not None:
            m.color = deepcopy(class_colors_rviz[idx[i]])
            m.color.a = 0.05
        else:
            m.color.r = np.random.uniform(0., 1.)
            m.color.g = np.random.uniform(0., 1.)
            m.color.b = np.random.uniform(0., 1.)
            m.color.a = 1.0
        m.id = i
        for t in range(X.shape[1]):
            p = Point(X[i,t,0],X[i,t,1],X[i,t,2])
            m.points.append(p)
        ma.markers.append(m)

    debug and print(len(ma.markers))
    return ma
