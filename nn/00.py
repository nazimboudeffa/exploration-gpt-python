import torch
import torch.nn as nn
import torch.optim as optim

# Exemple simple de réseau de neurones pour une tâche de classification binaire
# Ce script crée un petit réseau de neurones pour classer des données fictives en deux classes.

# Exemple : données d'entrée fictives (100 exemples, 10 features)
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))  # 100 étiquettes binaires (0 ou 1)

# Définir le modèle
class PetitReseau(nn.Module):
    def __init__(self):
        super(PetitReseau, self).__init__()
        self.fc1 = nn.Linear(10, 32)     # couche d'entrée -> couche cachée
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 2)      # couche cachée -> sortie (2 classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialisation
model = PetitReseau()

# Fonction de perte et optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraînement simple
for epoch in range(20):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/20, Perte: {loss.item():.4f}")