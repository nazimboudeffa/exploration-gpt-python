import torch
import torch.nn as nn
import torch.optim as optim

# Somme de deux nombres avec un réseau de neurones simple
# Ce script crée un petit réseau de neurones pour apprendre à prédire la somme de deux nombres.

# 1. Données fictives
X = torch.randn(100, 2)  # 100 lignes avec 2 colonnes (x1, x2)
y = X.sum(dim=1, keepdim=True)  # y = x1 + x2

# 2. Définir un petit réseau de neurones
model = nn.Sequential(
    nn.Linear(2, 4),  # couche 1 : 2 entrées -> 4 neurones
    nn.ReLU(),
    nn.Linear(4, 1)   # couche 2 : 4 neurones -> 1 sortie (la somme)
)

# 3. Perte et optimiseur
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 4. Entraînement
for epoch in range(100):
    y_pred = model(X)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Perte: {loss.item():.4f}")

# 5. Test simple
test = torch.tensor([[3.0, 5.0]])
pred = model(test)
print(f"Prédiction pour [3.0, 5.0] : {pred.item():.2f} (attendu : 8.0)")