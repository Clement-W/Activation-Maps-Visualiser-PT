import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import image
import pathlib
import tqdm
import sys

import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, transforms

sys.path.append("./CustomNetwork/")
from Network import Network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model from the config file defined in demoCustomNetwork.py
def load_model_from_config_file(config):
    model = Network(config["kernels"], config["classes"]).to(device)
    model.load_state_dict(torch.load(config["model_pt_path"]))
    model.eval()
    return model


# Load model with the path to the state dict
def load_model_from_model_path(path_to_model):
    model = Network().to(device)
    model.load_state_dict(torch.load(path_to_model))
    model.eval()
    return model


# Hook the activation of the layer
def get_activation(name, activation_list):
    def hook(model, input, output):
        activation_list[name] = output.detach()

    return hook


# Return the prediction of the model for the input image
def get_prediction(model, image):
    with torch.no_grad():
        output = model(image[None, ...]).float().to(device)
        _, prediction = torch.max(output.detach(), 1)
    return prediction.item()


# Return the activation maps for every layers in a list
def get_activation_maps_list(model, image):
    activation_list = {}
    for name, layer in model.named_modules():
        layer.register_forward_hook(get_activation(name, activation_list))

    with torch.no_grad():
        output = model(image[None, ...]).float().to(device)
    
    return activation_list

# Return the activation maps for a specific layer
def get_activation_maps_by_layer(activation_list, layer_name):
    activ_maps = activation_list[layer_name].detach()
    activ_maps = torchvision.utils.make_grid(activ_maps)
    activ_maps = activ_maps.cpu().numpy()
    return activ_maps


# Return the matplotlib figure displaying the activation maps
def create_figure(activ_maps, layer_name, image, prediction):

    # Display the activation maps with matplotlib
    fig = plt.figure(figsize=(10, 10), dpi=80)
    fig.suptitle(
        "Activation maps of the {} {} filters ({}*{}) for the given image ".format(
            activ_maps.shape[0], layer_name, activ_maps.shape[1], activ_maps.shape[1]
        )
    )
    plot_size = (int)(
        np.sqrt(activ_maps.shape[0])
    ) + 1  # Square root of the number of conv filters to get a square plot
    #  +1 To include the input image

    # Make plot with size plot_size * plot_size
    fig.add_subplot(plot_size, plot_size, 1)
    plt.axis("off")
    plt.gca().set_title("input image : {} predicted".format(prediction))
    if image.shape[0] == 1:  # Black and White image
        plt.imshow(
            image.detach().cpu().reshape((image.shape[1], image.shape[2])), cmap="gray"
        )
    elif image.shape[0] == 3:  # RGB image
        plt.imshow(image.detach().cpu().permute(1, 2, 0))

    for i in range(1, activ_maps.shape[0]):
        fig.add_subplot(plot_size, plot_size, i + 1)
        plt.axis("off")
        plt.imshow(activ_maps[i])

    return fig


# Show the activation maps of a specified layer
def show_activation_maps(model, layer_name, image):

    activ_maps_list = get_activation_maps_list(model, image)
    activ_maps_layer = get_activation_maps_by_layer(activ_maps_list,layer_name)

    prediction = get_prediction(model, image)

    fig = create_figure(activ_maps_layer, layer_name, image, prediction)
    modelName = model.__class__.__name__
    save_activation_map(fig, layer_name, modelName)
    plt.show()


# Save the activation maps in the folder './activation-maps/'
def save_activation_map(figure, layer_name, modelName, index=-1):
    pathlib.Path("./{}-activation-maps/".format(modelName)).mkdir(exist_ok=True)
    if index == -1:
        plt.savefig(
            "./{}-activation-maps/activ_map_{}.png".format(modelName, layer_name)
        )
    else:
        plt.savefig(
            "./{}-activation-maps/{}activ_map_{}.png".format(
                modelName, index, layer_name
            )
        )


# Save the activation maps of every layer of the network
def save_all_activation_maps(model, image):

    total = [i for i, (_, _) in enumerate(model.named_modules())][-1]
    print("\n[+] Saving {} images to {}".format(total, pathlib.Path.cwd()))

    modelName = model.__class__.__name__

    progressB = tqdm.tqdm(enumerate(model.named_modules()), total=total)

    activ_maps_list = get_activation_maps_list(model, image)

    for idx, (layer_name, _) in progressB:

        activ_maps_layer = get_activation_maps_by_layer(activ_maps_list,layer_name)

        prediction = get_prediction(model, image)
        fig = create_figure(activ_maps_layer, layer_name, image, prediction)
        save_activation_map(fig, layer_name, modelName, idx)
        plt.close("all")
