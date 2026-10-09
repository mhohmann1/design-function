import os
from torch.utils.data import Dataset
import torch
import glob
import numpy as np
from args import args
from data.data_utils import random_pc, random_rotation, random_translation
from tqdm import tqdm
import random
import re
import json

class Data(Dataset):
    def __init__(self, data_dir, augmentation, num_points, get_scale=False):
        self.data_dir = data_dir
        self.paths = []
        self.num_points = num_points
        self.augmentation = augmentation

        self.paths = glob.glob(os.path.join(data_dir, "**"))

        if get_scale:
            self.global_min = np.full(3, np.inf)
            self.global_max = np.full(3, -np.inf)
            get_files = glob.glob(os.path.join(data_dir, "**", "*.npz"), recursive=True)
            for file in tqdm(get_files):
                current_file = np.load(file)["points"]
                file_max = np.max(current_file, axis=0)
                file_min = np.min(current_file, axis=0)

                self.global_min = np.minimum(self.global_min, file_min)
                self.global_max = np.maximum(self.global_max, file_max)

            with open(config_path=os.path.join(os.path.dirname(__file__), "scaling_values.json"), "r") as f:
                scaling = json.load(f)

            scaling["global_min"] = self.global_min.tolist()
            scaling["global_max"] = self.global_max.tolist()

            with open(config_path, "w") as f:
                json.dump(scaling, f, indent=4)

            print(self.global_min, self.global_max)

        else:
            config_path = os.path.join(
                os.path.dirname(__file__),
                "scaling_values.json"
            )

            with open(config_path, "r") as f:
                scaling = json.load(f)

            self.global_min = np.array(scaling["global_min"])
            self.global_max = np.array(scaling["global_max"])

            self.die_min = np.array(scaling["die_min"])
            self.die_max = np.array(scaling["die_max"])

            self.punch_min = np.array(scaling["punch_min"])
            self.punch_max = np.array(scaling["punch_max"])

            self.part_min = np.array(scaling["part_min"])
            self.part_max = np.array(scaling["part_max"])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]

        m_pc = random_pc(np.load(path + "/Data/01_Matrize_Viertel.npz")["points"], args.points)
        s_pc = random_pc(np.load(path + "/Data/03_Stempel_Viertel.npz")["points"], args.points)

        bt_file = random.choice(glob.glob(os.path.join(path, "**", "Bauteil", "*.npz"), recursive=True))

        bt_pc = random_pc(np.load(bt_file)["points"], args.points)

        if self.augmentation:
            m_pc = m_pc - self.die_min
            s_pc = s_pc - self.punch_min
            bt_pc = bt_pc - self.part_min

            m_pc = (m_pc - self.global_min) / (self.global_max - self.global_min) #  * 2 - 1
            s_pc = (s_pc - self.global_min) / (self.global_max - self.global_min) #  * 2 - 1
            bt_pc = (bt_pc - self.global_min) / (self.global_max - self.global_min) #  * 2 - 1

        filename = os.path.basename(bt_file)
        match = re.search(r'\d+', filename)
        if match:
            bh_f = (int(match.group()) - 15000) / 25000
        else:
            bh_f = -1

        return torch.tensor(m_pc, dtype=torch.float32), torch.tensor(s_pc, dtype=torch.float32), torch.tensor(bt_pc, dtype=torch.float32), torch.tensor(bh_f, dtype=torch.float32)