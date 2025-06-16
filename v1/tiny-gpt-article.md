✨ Générer du texte avec un mini GPT en PyTorch

📚 Introduction

Les modèles de langage de type GPT (Generative Pretrained Transformers) sont capables de générer du texte de manière fluide à partir d’un simple prompt. Ici, nous allons construire une version très simplifiée d’un tel modèle en PyTorch, entraînée caractère par caractère à partir d’un fichier texte.

Ce projet couvre les étapes suivantes :

1-Lecture et encodage d’un texte
2-Préparation des batchs avec DataLoader
3-Définition d’un mini-modèle GPT
4-Entraînement

5-Génération de texte avec sampling contrôlé (temperature, top-k, top-p)

📝 Prétraitement du texte
On commence par charger un fichier texte (exemple.txt), nettoyer le contenu, et encoder chaque caractère en entier. On crée un vocabulaire simple basé sur les caractères uniques.

```
with open('exemple.txt', 'r', encoding='utf-8') as f:
    text = f.read().strip().lower()

chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
```

📦 Génération de batchs avec Dataset et DataLoader

Plutôt que de coder manuellement la génération des batchs, on utilise les outils standard de PyTorch :

```

class CharDataset(torch.utils.data.Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(data) - block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+block_size]
        y = self.data[idx+1:idx+1+block_size]
        return x, y

dataset = CharDataset(data, block_size=64)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

```

🧠 Mini GPT : un modèle simple

Notre modèle se compose uniquement d'une couche d'embedding suivie d’une couche linéaire qui prédit le caractère suivant :

```
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, n_embed=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx):
        x = self.embed(idx)         # [batch, time, n_embed]
        logits = self.lm_head(x)    # [batch, time, vocab_size]
        return logits
```

⚙️ Entraînement

Le modèle est entraîné à prédire chaque caractère suivant dans les séquences fournies par le DataLoader.

```
model = TinyGPT(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

data_iter = iter(dataloader)
for step in range(500):
    try:
        x, y = next(data_iter)
    except StopIteration:
        data_iter = iter(dataloader)
        x, y = next(data_iter)

    logits = model(x)
    loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Étape {step} – Perte : {loss.item():.4f}")
```

🪄 Génération de texte avec sampling contrôlé

La génération de texte se fait un caractère à la fois, à partir d’un prompt. On peut contrôler la créativité avec :

- temperature : ajuste la "netteté" des probabilités (baisse = plus conservateur)
- top_k : ne considère que les k tokens les plus probables
- top_p : garde les tokens les plus probables jusqu’à atteindre p de probabilité cumulée

```
@torch.no_grad()
def generate(model, prompt, max_new_tokens=100, temperature=1.0, top_k=None, top_p=None):
    model.eval()
    idx = torch.tensor([stoi[c] for c in prompt], dtype=torch.long).unsqueeze(0)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)[:, -1, :] / temperature

        if top_k is not None:
            vals, inds = torch.topk(logits, top_k)
            mask = torch.full_like(logits, float('-inf'))
            mask.scatter_(1, inds, vals)
            logits = mask

        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(probs, dim=-1)
            mask = cumprobs > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = 0
            logits[0, sorted_indices[0][mask[0]]] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

    return ''.join([itos[i.item()] for i in idx[0]])
```

🧪 Exemple d’utilisation

```
prompt = "un matin,"
text = generate(model, prompt, max_new_tokens=200, temperature=0.8, top_k=30, top_p=0.9)
print(text)
```

✅ Résultat

Avec un entraînement de quelques centaines d’itérations, le modèle peut déjà générer un texte "cohérent" dans le style du corpus utilisé.

🔚 Conclusion

Ce mini-GPT montre qu’il est possible de générer du texte avec un modèle très simple, sans transformer complet ni multi-têtes. Il est parfait pour l'apprentissage des fondamentaux du machine learning sur du texte. On peut ensuite enrichir :

-En ajoutant des blocs de transformer
-En entraînant sur des mots ou des tokens (niveau BPE)
-En sauvegardant les checkpoints
-En déployant une interface Web (Streamlit, Gradio...)