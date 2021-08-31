import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import math
import rosbag
from sensor_msgs.msg import Imu


class MeanVar:
    def __init__(self, n):
        self.n = 0
        self.mean = np.zeros((n))
        self.var_sum = np.zeros((n))

    def __call__(self, x):
        x = np.asarray(x)
        self.n += 1
        dx = x - self.mean
        dx_n = dx / self.n
        self.mean += dx_n
        self.var_sum += dx * dx_n * (self.n - 1)

    @property
    def var(self):
        if self.n <= 1:
            return np.zeros_like(self.var_sum)
        return self.var_sum / (self.n - 1)


bagfile = "/home/chao/Workspace/dataset/bags/mrsl-static-2021-08-31-12-29-39.bag"
imu_topic = "/os_node/imu"

mv = MeanVar(3)

with rosbag.Bag(bagfile, "r") as bag:
    for topic, msg, t in bag.read_messages([imu_topic]):
        if topic == imu_topic:
            w = msg.angular_velocity
            mv([w.x, w.y, w.z])

print(mv.mean)
print(mv.var)