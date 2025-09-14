import torch
from torch.utils.data import DataLoader
from load_data import VideoClipDataset
from model_3dcnn import Simple3DCNN
import torch.nn as nn
import torch.optim as optim
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
train_dataset = VideoClipDataset("processed_dataset/train", clip_len=16)
val_dataset = VideoClipDataset("processed_dataset/val", clip_len=16)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Model
model = Simple3DCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):
    model.train()
    train_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Accuracy = {correct / total:.2f}")
