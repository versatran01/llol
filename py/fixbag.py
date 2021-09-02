import rosbag
from tqdm import tqdm
from pathlib import Path

in_bag_file = "/home/chao/Workspace/ws/ouster_ws/2021-09-02-00-13-32.bag"
out_bag_file = "/home/chao/Workspace/ws/ouster_ws/morgtown-scan1.bag"


tf_msg = None
tf_saved = False

with rosbag.Bag(out_bag_file, "w") as out_bag:
    with rosbag.Bag(in_bag_file, "r") as in_bag:
        for topic, msg, t in tqdm(in_bag.read_messages()):
            if topic == "/tf_static":
                tf_msg = msg
            else:
                out_bag.write(topic, msg, msg.header.stamp)
                if not tf_saved:
                    print("write /tf_static")
                    out_bag.write("/tf_static", tf_msg, msg.header.stamp)
                    tf_saved = True
