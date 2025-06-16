import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Charger le modèle CartoonGAN style "hayao"
model = torch.hub.load('bryandlee/animegan2-pytorch', 'generator', pretrained='hayao').eval()

# Préparer la transformation d'image
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# Fonction de cartoonisation
def cartoonize(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output_tensor = model(input_tensor)[0]

    output_image = transforms.ToPILImage()(output_tensor.clamp(0, 1))
    return output_image

# Exemple : appliquer à une image
image_path = "ton_image.jpg"  # Remplace par ton chemin
cartoon_image = cartoonize(image_path)

# Affichage
plt.imshow(cartoon_image)
plt.axis("off")
plt.title("Style Hayao (CartoonGAN)")
plt.show()