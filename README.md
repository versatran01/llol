# :laughing: LLOL: Low-Latency Odometry for Spinning Lidars

Also see https://github.com/versatran01/rofl-beta.

## Reference

LLOL: Low-Latency Odometry for Spinning Lidars

Chao Qu, Shreyas S. Shivakumar, Wenxin Liu, Camillo J. Taylor

https://arxiv.org/abs/2110.01725

https://youtu.be/MmiTMFt9YdU

## Usage

Sample data at
https://www.dropbox.com/s/v4cth3z7hrqjsvf/raw-perch-loop-2021-09-06-17-22-06.bag?dl=0

Clone 
https://github.com/KumarRobotics/ouster_decoder


Open rviz using the config in `launch/llol.rviz`

First run ouster driver
```
roslaunch ouster_decoder driver.launch
```

Then run ouster decoder
```
roslaunch ouster_decoder decoder.launch
```

Then run odom
```
roslaunch llol llol.launch
```

Run bag.

See CMakeLists.txt for dependencies.
You may also check our [Github Action build
file](https://github.com/versatran01/llol/blob/main/.github/workflows/build.yaml) for instructions on how to build LLOL in Ubuntu 20.04 with ROS Noetic.

To run multithread and show timing every 5s do
```
roslaunch llol llol.launch tbb:=1 log:=5
```

This is the open-source version, some advanced features may be missing.

