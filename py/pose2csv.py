import rosbag
import numpy as np
from geometry_msgs.msg import PoseStamped

bagfile = "/home/chao/Workspace/ws/llol_ws/data/2021-09-09-22-46-07.bag"

poses = []

with rosbag.Bag(bagfile, "r") as bag:
    for topic, msg, t in bag.read_messages():
        p = msg.pose.position
        q = msg.pose.orientation
        t = msg.header.stamp.to_sec()
        pose = np.array([t, p.x, p.y, p.z, q.x, q.y, q.z, q.w])
        poses.append(pose)

poses = np.array(poses)
np.savetxt("/home/chao/Workspace/ws/llol_ws/data/nc-05.txt", poses, fmt="%10.10f")