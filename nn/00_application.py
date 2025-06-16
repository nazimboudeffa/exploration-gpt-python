import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
import torch.optim as optim

# Exemple : Dataset personnalisé
class MonDataset(Dataset):
    def __init__(self, chemin_csv):
        data = pd.read_csv(chemin_csv)
        self.X = torch.tensor(data.drop(columns=['label']).values, dtype=torch.float32)
        self.y = torch.tensor(data['label'].values, dtype=torch.long)  # ou float pour régression

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Charger les données
dataset = MonDataset("chemin/vers/tes_donnees.csv")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Réseau simple
class Reseau(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 2)  # à adapter selon nb classes ou sortie régression

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Instancier modèle
input_dim = dataset.X.shape[1]
model = Reseau(input_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Boucle d'entraînement
for epoch in range(20):
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")