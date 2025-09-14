from torch.utils.data import DataLoader
import torch.optim as optim
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Load dataset
train_dataset = FightDataset("processed_dataset/train")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Model, Loss, Optimizer
model = Simple3DCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    for clips, labels in train_loader:
        clips, labels = clips.to(device), labels.to(device)

        outputs = model(clips)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    acc = correct / len(train_dataset)
    print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")

# Save model
torch.save(model.state_dict(), "fight_detection_3dcnn.pth")
