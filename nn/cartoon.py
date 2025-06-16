import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# 1. Charger le modèle CartoonGAN (Hayao)
model = torch.hub.load("bryandlee/animegan2-pytorch", "generator", pretrained="hayao").eval()

# 2. Transformer l’image
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

def cartoonize(img_path):
    image = Image.open(img_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)[0]

    output_image = transforms.ToPILImage()(output.clamp(0, 1))
    return output_image

# 3. Utiliser une image
img_path = "ton_image.jpg"  # Remplace par le chemin de ton image
output = cartoonize(img_path)

# 4. Afficher
plt.imshow(output)
plt.axis("off")
plt.title("Image cartoonisée")
plt.show()