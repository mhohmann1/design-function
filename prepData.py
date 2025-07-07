import glob
import os
import trimesh
from args import args
from trimesh.interfaces.gmsh import load_gmsh
import numpy as np
import pandas as pd
from data.data_utils import random_pc
from tqdm import tqdm

UNPROCESSED_DATA_PATH = "data/dataset"

file_paths_STP = glob.glob(os.path.join(UNPROCESSED_DATA_PATH, "**", "*.step"), recursive=True)

for file in tqdm(file_paths_STP):
    gmsh = load_gmsh(file)
    mesh = trimesh.Trimesh(**gmsh)
    points, idx = trimesh.sample.sample_surface(mesh, count=10_000, seed=args.seed)
    save_path = file.replace("step", "npz")
    np.savez_compressed(save_path, points=points, idx=idx)
    print(f"Saved under {save_path}")

file_paths_XLSX = glob.glob(os.path.join(UNPROCESSED_DATA_PATH, "**", "*.xlsx"), recursive=True)

for file in tqdm(file_paths_XLSX):
    df = pd.read_excel(file)
    points = df[['x (mm)', 'y (mm)', 'z (mm)']].to_numpy()
    points = random_pc(points, 10_000)

    save_path = file.replace("Data/", "Data/Bauteil/")

    save_path = save_path.replace("xlsx", "npz")
    save_path = save_path.replace(" ", "_")

    save_dir = os.path.dirname(save_path)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    np.savez_compressed(save_path, points=points)
    print(f"Saved under {save_path}")