import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Devine le chiffre écrit à la main dans MNIST avec un réseau de neurones simple
# Ce script crée un petit réseau de neurones pour classer les chiffres manuscrits du dataset MNIST.

# 1. Préparation des données
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# 2. Réseau simple (images 28x28 → 784 entrées)
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)  # 10 classes (0 à 9)

    def forward(self, x):
        x = x.view(-1, 28*28)  # aplatir l'image
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = MNISTModel()

# 3. Perte et optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Entraînement
for epoch in range(5):  # 5 époques pour aller vite
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Époque {epoch+1}, Perte: {total_loss:.4f}")