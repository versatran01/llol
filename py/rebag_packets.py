import rosbag
from tqdm import tqdm
from pathlib import Path

out_bag_file = f"/home/chao/Documents/morgown.bag"
bag_dir = f"/home/chao/Downloads"
bag_dir = Path(bag_dir)

in_bag_files = []
for p in bag_dir.iterdir():
    if p.suffix == ".bag":
        in_bag_files.append(p)
in_bag_files = list(sorted(in_bag_files))
print(in_bag_files)

imu_topic = "/os1_node/imu_packets"
lidar_topic = "/os1_node/lidar_packets"
new_imu_topic = "/os_node/imu_packets"
new_lidar_topic = "/os_node/lidar_packets"


def rewrite(in_bag_file, out_bag):
    with rosbag.Bag(in_bag_file, "r") as in_bag:
        for topic, msg, t in tqdm(in_bag.read_messages([imu_topic, lidar_topic])):
            if topic == imu_topic:
                out_bag.write(new_imu_topic, msg)
            elif topic == lidar_topic:
                out_bag.write(new_lidar_topic, msg)


with rosbag.Bag(out_bag_file, "w") as out_bag:
    for in_bag_file in in_bag_files:
        print(in_bag_file)
        rewrite(in_bag_file, out_bag)
