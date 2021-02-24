from torchvision import models, transforms, datasets
import torch
from PIL import Image

from ActivationMapsExtractor import save_all_activation_maps


def image_loader(image_name, transform):
    image = Image.open(image_name)
    image = transform(image)
    image = image.to(device)
    return image


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)

# Preprocess for Resnet18 input
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
transform = transforms.Compose(
    [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
)
# TODO: Add normalization to input image as required in Resnet18

image = image_loader("imageNet/sample.JPEG", transform)

# From ActivationMapExtractor.py
save_all_activation_maps(model, image)
# Save the images in ./imagenet-activation-maps/
