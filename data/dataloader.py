import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PARAMETER_COLUMNS = ["radii1", "radii2", "cr", "delta", "height"]
TOOLS = ["die", "holder", "punch"]


class PointCloudDataset(Dataset):
    """Load one part and its tools per geometry and holder force."""

    def __init__(self, data_dir, num_points=None, split_file=None):
        if num_points is not None and (not isinstance(num_points, int) or num_points <= 0):
            raise ValueError("num_points must be a positive integer or None")
        self.data_dir = Path(data_dir)
        self.num_points = num_points
        self.parameters = pd.read_csv(
            self.data_dir / "parameters.csv", dtype={"folder": str, "holder_force": int},
        ).set_index(["folder", "holder_force"], verify_integrity=True)
        index = self.parameters.index
        if split_file is not None:
            entries = [line.strip() for line in Path(split_file).read_text().splitlines() if line.strip()]
            if len(entries) != len(set(entries)):
                raise ValueError(f"Duplicate part paths in {split_file}")
            part_index = {f"{name}/parts/part_{force}.npz": (name, force) for name, force in index}
            unknown = set(entries) - part_index.keys()
            if unknown:
                raise ValueError(f"Parts in {split_file} are missing from parameters.csv: {sorted(unknown)}")
            index = [part_index[entry] for entry in entries]
        self.part_files = []
        skipped = 0
        for name, force in index:
            folder = self.data_dir / name
            part_file = folder / "parts" / f"part_{force}.npz"
            if (not part_file.is_file()
                    or not all((folder / "tools" / f"{tool}.npz").is_file() for tool in TOOLS)):
                skipped += 1
                continue
            self.part_files.append(part_file)
        if skipped:
            warnings.warn(f"Skipping {skipped} samples with missing part or tools", stacklevel=2)
        if not self.part_files:
            raise ValueError(f"No parts with complete tools found in {data_dir}")

    def __len__(self):
        return len(self.part_files)

    def __getitem__(self, index):
        part_file = self.part_files[index]
        folder = part_file.parent.parent
        force = int(part_file.stem.removeprefix("part_"))
        row = self.parameters.loc[(folder.name, force)]
        sample = {
            "folder": folder.name,
            "holder_force": torch.tensor(force, dtype=torch.float32),
            "min_thickness": torch.tensor(row["min_thickness"], dtype=torch.float32),
            "parameters": torch.tensor(
                row[PARAMETER_COLUMNS].to_numpy(dtype=np.float32),
            ),
        }
        paths = {tool: folder / "tools" / f"{tool}.npz" for tool in TOOLS}
        paths["part"] = part_file
        for name, path in paths.items():
            with np.load(path) as data:
                points = data["points"]
                if self.num_points is not None:
                    if self.num_points > len(points):
                        raise ValueError(f"Requested {self.num_points} points, but {path} has {len(points)}")
                    if self.num_points < len(points):
                        indices = np.random.choice(len(points), self.num_points, replace=False)
                        points = points[indices]
                sample[name] = torch.tensor(points, dtype=torch.float32)
        return sample


class Data(PointCloudDataset):
    """Apply the existing CVAE normalization and return (die, punch, part, bhf)."""

    def __init__(self, data_dir, augmentation, num_points, get_scale=False, split_file=None):
        super().__init__(data_dir, num_points=num_points, split_file=split_file)
        self.augmentation = augmentation
        config_path = Path(__file__).with_name("scaling_values.json")
        with config_path.open() as f:
            scaling = json.load(f)

        if get_scale:
            global_min = np.full(3, np.inf)
            global_max = np.full(3, -np.inf)
            for file in self.data_dir.rglob("*.npz"):
                with np.load(file) as data:
                    points = data["points"]
                    global_min = np.minimum(global_min, points.min(axis=0))
                    global_max = np.maximum(global_max, points.max(axis=0))
            scaling["global_min"] = global_min.tolist()
            scaling["global_max"] = global_max.tolist()
            with config_path.open("w") as f:
                json.dump(scaling, f, indent=4)

        self.global_min = torch.tensor(scaling["global_min"], dtype=torch.float32)
        self.global_max = torch.tensor(scaling["global_max"], dtype=torch.float32)
        self.die_min = torch.tensor(scaling["die_min"], dtype=torch.float32)
        self.punch_min = torch.tensor(scaling["punch_min"], dtype=torch.float32)
        self.part_min = torch.tensor(scaling["part_min"], dtype=torch.float32)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        die, punch, part = sample["die"], sample["punch"], sample["part"]
        if self.augmentation:
            scale = self.global_max - self.global_min
            die = (die - self.die_min - self.global_min) / scale
            punch = (punch - self.punch_min - self.global_min) / scale
            part = (part - self.part_min - self.global_min) / scale
        bhf = (sample["holder_force"] - 15000) / 25000
        return die, punch, part, bhf
