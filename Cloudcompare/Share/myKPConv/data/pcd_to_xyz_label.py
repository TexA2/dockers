import numpy as np
import os
from tqdm import tqdm
from pypcd4 import PointCloud


script_dir = os.path.dirname(os.path.abspath(__file__))

cloud_path = os.path.join(script_dir, "cloud/cloud.pcd")

#cloud_path = "./cloud.pcd"


cloud = PointCloud.from_path(cloud_path)
print(cloud.fields)
pcnp = cloud.numpy(("x","y","z"))

file_name = os.path.join(script_dir, "testing/cloud.xyz_label_conf")
#file_name = "cloud" + ".xyz_label_conf"
with open(file_name, 'w') as f:
    for point in tqdm(pcnp):
        x, y, z = point
        f.write(f"{x: .2f} {y: .2f} {z: .2f} 1000\n")
