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

            print(self.global_min, self.global_max)

        else:
            self.global_min = np.array([0., 0., 0])
            self.global_max = np.array([279.99963885 - 3.27888232e-04, 229.99968253 - 3.47547511e-04, 50.5 - 5.00000000e-01])

            self.die_min = np.array([3.27888232e-04, 3.47547511e-04, 5.00000000e-01])
            self.die_max = np.array([279.99963885, 229.99968253, 50.5])

            self.punch_min = np.array([1.22097060e-04, 2.58775917e-04, -5.04998589e+01])
            self.punch_max = np.array([174.06161467, 99.06198772, -0.47961212])

            self.part_min = np.array([0., 0., -54.3315239])
            self.part_max = np.array([2.44548386e+02, 1.89805313e+02, 1.04953878e-01])

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