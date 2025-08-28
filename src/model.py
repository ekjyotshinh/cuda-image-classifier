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
    # A new forward function that includes the fc layer initialization
    def new_fwd(x):
        x = F.relu(model.conv1(x))
        # Use the custom conv function for the second convolution
        x = model.pool(F.relu(custom_fn(x, model.conv2.weight, model.conv2.bias)))

        # Dynamically compute flattened size (this was the missing piece)
        if model.fc is None:
            n_features = x.numel() // x.shape[0]
            model.fc = nn.Linear(n_features, model.num_classes).to(x.device)

        x = x.view(x.size(0), -1)
        x = model.fc(x)
        return x

    # Replace the model's forward method with our new one
    model.forward = new_fwd
    return model
