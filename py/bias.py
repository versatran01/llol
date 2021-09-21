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


bagfile = "/home/chao/Workspace/dataset/bags/static-2021-09-03-21-12-12.bag"
imu_topic = "/os_node/imu"

mva = MeanVar(3)
mvw = MeanVar(3)

with rosbag.Bag(bagfile, "r") as bag:
    for topic, msg, t in bag.read_messages([imu_topic]):
        if topic == imu_topic:
            w = msg.angular_velocity
            a = msg.linear_acceleration
            mvw([w.x, w.y, w.z])
            mva([a.x, a.y, a.z])

print("gyr")
print(mvw.mean)
print(np.sqrt(mvw.var))

print("acc")
print(mva.mean)
print(np.sqrt(mva.var))