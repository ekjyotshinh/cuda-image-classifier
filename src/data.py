# src/data.py
import torch
from torchvision import datasets, transforms

def make_loaders(batch_size=128, num_workers=2):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),
                             (0.2470,0.2435,0.2616))
    ])
    train = datasets.CIFAR10(root="data", train=True, download=True, transform=tfm)
    test  = datasets.CIFAR10(root="data", train=False, download=True, transform=tfm)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader  = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader, 10
