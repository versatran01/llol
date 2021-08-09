import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

import rosbag
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError


# %%
def imshownn(ax, image, *args, **kwargs):
    return ax.imshow(image, *args, **kwargs, interpolation="nearest")


@njit
def destagger(image: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    out = image.copy()
    cols = image.shape[1]
    for i in range(image.shape[0]):
        o = offsets[i]
        out[i, o:] = image[i, :cols - o]
    return out


@njit
def calc_grid_curve(rimg: np.ndarray, cell_shape) -> np.ndarray:
    grid_shape = (rimg.shape[0] // cell_shape[0],
                  rimg.shape[1] // cell_shape[1])
    grid = np.zeros(grid_shape, np.float32)

    for gr in range(grid.shape[0]):
        for gc in range(grid.shape[1]):
            sr = gr * cell_shape[0]
            sc = gc * cell_shape[1]
            cell = rimg[sr, sc:sc + cell_shape[1]]
            mid = cell[cell_shape[1] // 2]
            grid[gr, gc] = np.abs(np.sum(cell) / cell_shape[1] / mid - 1)

    return grid


@njit
def calc_grid_std(rimg: np.ndarray, cell_shape) -> np.ndarray:
    grid_shape = (rimg.shape[0] // cell_shape[0],
                  rimg.shape[1] // cell_shape[1])
    grid = np.zeros(grid_shape, np.float32)

    min_pts = int(cell_shape[1] / 4 * 3)

    for gr in range(grid.shape[0]):
        for gc in range(grid.shape[1]):
            sr = gr * cell_shape[0]
            sc = gc * cell_shape[1]
            cell = rimg[sr, sc:sc + cell_shape[1]]
            mean = np.mean(cell)
            grid[gr, gc] = np.std(cell) / mean
    return grid


# %%
bagfile = "/home/chao/Workspace/dataset/bags/2021-08-07-21-34-10.bag"
image_topic = "/os_node/image"
cinfo_topic = "/os_node/camera_info"

image_msg = None
cinfo_msg = None

with rosbag.Bag(bagfile, "r") as bag:
    for topic, msg, t in bag.read_messages([image_topic, cinfo_topic]):
        if topic == image_topic:
            image_msg = msg
        elif topic == cinfo_topic:
            cinfo_msg = msg

        if image_msg is not None and cinfo_msg is not None:
            break

# %%
bridge = CvBridge()
sweep = bridge.imgmsg_to_cv2(image_msg, "32FC4")
offsets = np.array(cinfo_msg.D, int)
rimg = sweep[..., -1]
rimg2 = destagger(rimg, offsets)

f, ax = plt.subplots(2, 1, sharex="all", sharey="all")
imshownn(ax[0], rimg, cmap="pink")
imshownn(ax[1], rimg2, cmap="pink")

# %%
# compute variance per cell
cell_shape = (2, 16)
grid_std = calc_grid_std(rimg, cell_shape)  # 0.04
grid_curve = calc_grid_curve(rimg, cell_shape)  # 0.04ms
print("num std: ", np.sum(grid_std >= 0))
print("num curve: ", np.sum(grid_curve >= 0))

f, ax = plt.subplots(2, 2)
imshownn(ax[0, 0], grid_std)
imshownn(ax[0, 1], grid_curve, vmax=0.1)
ax[1, 0].hist(grid_std.ravel(), bins=100)
ax[1, 1].hist(grid_curve.ravel(), bins=100)
