import rosbag
import numpy as np
from geometry_msgs.msg import PoseStamped

bagfile = "/home/chao/Workspace/ws/llol_ws/bags/nc-01-rigid-pr256-cc16-2021-09-07-16-12-41.bag"

poses = []

with rosbag.Bag(bagfile, "r") as bag:
    for topic, msg, t in bag.read_messages():
        p = msg.pose.position
        q = msg.pose.orientation
        t = msg.header.stamp.to_sec()
        pose = np.array([t, p.x, p.y, p.z, q.x, q.y, q.z, q.w])
        poses.append(pose)

poses = np.array(poses)
np.savetxt("/home/chao/Workspace/ws/llol_ws/bags/nc-01-rigid-pr256-cc16.txt", poses, fmt="%10.10f")