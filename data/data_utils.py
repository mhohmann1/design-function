import numpy as np
from args import args
from scipy.spatial.transform import Rotation as R

def random_pc(data, num_points=args.points):
    idx = np.random.choice(data.shape[0], num_points, replace=False)
    data = data[idx, :]
    return data

def random_translation(pointcloud, translation_range=(-0.2, 0.2), xyz2=None):
    if xyz2 is None:
        xyz2 = np.random.uniform(low=translation_range[0], high=translation_range[1], size=[3])
    translated_pointcloud = np.add(pointcloud, xyz2)
    return translated_pointcloud

def random_rotation(pointcloud, angle=None):
    if angle is None:
        angle = np.random.choice([0, 90, 180, 270])
    rotation = R.from_euler("z", angle, degrees=True)
    return rotation.apply(pointcloud)

def random_mirror(pointcloud, plane=None):
    if plane is None:
        plane = np.random.choice(["xz", "yz"])

    mirrored_pointcloud = np.copy(pointcloud)

    if plane == "xz":
        mirrored_pointcloud[:, 1] = -mirrored_pointcloud[:, 1]
    elif plane == "yz":
        mirrored_pointcloud[:, 0] = -mirrored_pointcloud[:, 0]
    return mirrored_pointcloud