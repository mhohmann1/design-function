from torch.utils.data import DataLoader, random_split
from args import args
import torch

torch.manual_seed(args.seed)

def get_dataloaders(Data, augmentation=True):
    dataset = Data(data_dir=args.path, augmentation=augmentation, num_points=args.points)
    train_size = int(args.train_size * len(dataset))
    valid_size = int(args.val_size * len(dataset))
    test_size = len(dataset) - train_size - valid_size

    train_data, valid_data, test_data = random_split(dataset, [train_size, valid_size, test_size], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False, pin_memory=True)

    return train_loader, valid_loader, test_loader