import torch
import torch.nn as nn
import torch.optim as optim
import random

# Pierre-papier-ciseaux avec un réseau de neurones simple
# Ce script crée un petit réseau de neurones pour prédire le coup suivant d'un joueur humain dans une partie de pierre-papier-ciseaux.

# 1. Préparer les données (historique de coups)
# Ex: [ton dernier coup, ton avant-dernier coup] => prédire ton prochain coup
# Valeurs: 0 = pierre, 1 = papier, 2 = ciseaux

# Simuler un historique de joueur humain
def generate_data(n=100):
    X, y = [], []
    history = [random.randint(0, 2) for _ in range(2)]
    for _ in range(n):
        next_move = random.choices([0, 1, 2], weights=[0.5, 0.25, 0.25])[0]  # humain un peu prévisible
        X.append(history[-2:])
        y.append(next_move)
        history.append(next_move)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y)

X, y = generate_data()

# 2. Réseau très simple
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 3)  # 3 coups possibles
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 3. Entraînement
for epoch in range(100):
    outputs = model(X)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Époque {epoch}, perte : {loss.item():.4f}")

# 4. Test interactif
def jouer():
    mapping = {0: "pierre", 1: "papier", 2: "ciseaux"}
    historique = [0, 1]  # Exemples initiaux
    while True:
        try:
            toi = int(input("Ton coup (0=pierre, 1=papier, 2=ciseaux, -1=stop) : "))
            if toi == -1:
                break
            entree = torch.tensor([historique[-2:]], dtype=torch.float32)
            prediction = model(entree)
            coup_ia = torch.argmax(prediction).item()
            print(f"L'IA joue : {mapping[coup_ia]}")

            # résultat
            if coup_ia == toi:
                print("Égalité !")
            elif (coup_ia - toi) % 3 == 1:
                print("L'IA gagne 😎")
            else:
                print("Tu gagnes 🎉")
            historique.append(toi)
        except:
            print("Entrée invalide")