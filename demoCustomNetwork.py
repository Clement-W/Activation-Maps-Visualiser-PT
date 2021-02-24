from ActivationMapsExtractor import (
    show_activation_maps,
    load_model_from_config_file,
    save_all_activation_maps,
)
from Network import Network

import yaml
from torchvision import datasets, transforms
import torch

# Download the dataset
def get_data(train_bool=True):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
    )

    dataset = datasets.MNIST(
        root="data/", download=True, train=train_bool, transform=transform
    )
    return dataset


# Make the dataset loader
def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = yaml.safe_load(open("./NNConfigFile.yaml"))

    test_loader = make_loader(get_data(False), batch_size=config["batch_size"])
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    image = images[0]

    model = load_model_from_config_file(config)  # Load trained model

    # layer_name = 'conv1'
    # show_activation_maps(model,layer_name,image)

    save_all_activation_maps(model, image)