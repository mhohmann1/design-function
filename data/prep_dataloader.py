from torch.utils.data import DataLoader, Subset, random_split
from args import args
import torch

torch.manual_seed(args.seed)

def get_dataloaders(Data, augmentation=True):
    dataset = Data(data_dir=args.path, augmentation=augmentation, num_points=args.points)
    folders = sorted({path.parent.parent.name for path in dataset.part_files})
    train_size = int(args.train_size * len(folders))
    valid_size = int(args.val_size * len(folders))
    test_size = len(folders) - train_size - valid_size

    folder_splits = random_split(folders, [train_size, valid_size, test_size], generator=torch.Generator().manual_seed(args.seed))
    subsets = []
    for split in folder_splits:
        split_folders = set(split)
        indices = [i for i, path in enumerate(dataset.part_files) if path.parent.parent.name in split_folders]
        subsets.append(Subset(dataset, indices))
    train_data, valid_data, test_data = subsets

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False, pin_memory=True)

    return train_loader, valid_loader, test_loader
