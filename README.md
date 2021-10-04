# Low-Latency Odometry for Spinning Lidars

Sample data at
https://www.dropbox.com/s/v4cth3z7hrqjsvf/raw-perch-loop-2021-09-06-17-22-06.bag?dl=0

Clone 
https://github.com/versatran01/ouster_decoder


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

