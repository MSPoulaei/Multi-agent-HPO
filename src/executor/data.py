import torch
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as T
import torchvision.datasets as dsets

def get_transforms(augment="basic"):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    if augment == "basic":
        train_tf = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
    else:
        train_tf = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.1,0.1,0.1,0.05),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    return train_tf, test_tf

def get_dataloaders(data_root, batch_size, num_workers=4, augment="basic", val_split=5000, seed=1337):
    train_tf, test_tf = get_transforms(augment)
    # Dataset names not exposed to agents
    full_train = dsets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
    test_set = dsets.CIFAR10(root=data_root, train=False, download=True, transform=test_tf)

    train_len = len(full_train) - val_split
    val_len = val_split
    gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [train_len, val_len], generator=gen)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader