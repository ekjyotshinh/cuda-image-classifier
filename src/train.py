# src/train.py
import torch, time
from torch import optim, nn
from src.data import make_loaders
from src.model import SimpleCNN

def train(device=None, epochs=2, lr=1e-3, batch=128):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, num_classes = make_loaders(batch)
    model = SimpleCNN(num_classes=num_classes).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    for e in range(1, epochs+1):
        t0 = time.time()
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
        dt = time.time()-t0
        print(f"Epoch {e}: {dt:.2f}s")

    torch.save(model.state_dict(), "weights.pt")
    print("Training done.")
