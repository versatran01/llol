import rosbag
from tqdm import tqdm
from pathlib import Path

in_bag_file = ""
out_bag_file = ""

with rosbag.Bag(out_bag_file, "w") as out_bag:
    with rosbag.Bag(in_bag_file, "r") as in_bag:
        for topic, msg, t in tqdm(in_bag.read_messages()):
            out_bag.write(topic, msg, msg.header.stamp)
