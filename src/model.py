# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.num_classes = num_classes

        # Placeholder for fc; will initialize dynamically
        self.fc = None

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))

        # Dynamically compute flattened size
        if self.fc is None:
            n_features = x.numel() // x.shape[0]  # total features per sample
            self.fc = nn.Linear(n_features, self.num_classes).to(x.device)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def replace_with_custom(model, custom_fn):
    # a new forward function
    def new_fwd(x):
        x = F.relu(model.conv1(x))
        x = model.pool(x)
        # custom conv instead of model.conv2
        x = F.relu(custom_fn(x, model.conv2.weight, model.conv2.bias))
        x = model.pool(x)
        x = x.view(x.size(0), -1)
        return model.fc(x)

    # replace forward method of the model
    model.forward = new_fwd
    return model
