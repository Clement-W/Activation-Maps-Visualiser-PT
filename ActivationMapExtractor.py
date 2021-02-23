import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import image
import pathlib
import tqdm

import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, transforms

from Network import Network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model from the config file defined in demo.py
def load_model_from_config_file(config):
    model = Network(config['kernels'],config['classes']).to(device)
    model.load_state_dict(torch.load(config['model_pt_path']))
    model.eval()
    return model

# Load model with the path to the state dict
def load_model_from_model_path(path_to_model):
    model = Network().to(device)
    model.load_state_dict(torch.load(path_to_model))
    model.eval()
    return model

# Hook the activation of the layer
def get_activation(name,activation_list):
    def hook(model, input, output):
        activation_list[name] = output.detach()
    return hook

# Return the prediction of the model for the input image
def get_prediction(model,image):
    with torch.no_grad():
        output = model(image[None,...]).float().to(device)
        _,prediction = torch.max(output.detach(),1)
    return prediction.item()

# Return the activation maps the layer
def get_activation_maps(model,layer_name,image):

    activation_list={}
    for name, layer in model.named_modules():
        layer.register_forward_hook(get_activation(name,activation_list))
    # TODO: Improve this : We needd to get the activation layer for the specified layer_name and not every layers

    with torch.no_grad():
        output = model(image[None,...]).float().to(device)
    activ_maps = activation_list[layer_name].detach()
    activ_maps = torchvision.utils.make_grid(activ_maps)
    activ_maps = activ_maps.cpu().numpy()

    return activ_maps

# Return the matplotlib figure displaying the activation maps
def create_figure(activ_maps,layer_name,image,prediction):

    # Display the activation maps with matplotlib
    fig = plt.figure(figsize=(7,7),dpi=100)
    fig.suptitle('Activation maps of the {} {} filters ({}*{}) for the given image '.format(activ_maps.shape[0],layer_name,activ_maps.shape[1],activ_maps.shape[1])  )
    plot_size = (int)(np.sqrt(activ_maps.shape[0]))+1 # Square root of the number of conv filters to get a square plot
    # +1 To include the input image

    # Make plot with size plot_size * plot_size
    fig.add_subplot(plot_size,plot_size,1)
    plt.axis('off')
    plt.gca().set_title('input image : {} predicted'.format(prediction))
    plt.imshow(image.detach().cpu().reshape((28,28)),cmap='gray')

    for i in range(1,activ_maps.shape[0]):
        fig.add_subplot(plot_size,plot_size,i+1)
        plt.axis('off')
        plt.imshow(activ_maps[i])

    return fig

# Show the activation maps of a specified layer
def show_activation_maps(model,layer_name,image):

    activ_maps = get_activation_maps(model,layer_name,image)

    prediction = get_prediction(model,image)

    fig = create_figure(activ_maps,layer_name,image,prediction)
    save_activation_map(fig,layer_name)
    plt.show()


# Save the activation maps in the folder './activation-maps/'
def save_activation_map(figure,layer_name):
    pathlib.Path('./activation-maps/').mkdir(exist_ok=True)
    plt.savefig('./activation-maps/activ_map_{}.png'.format(layer_name))


# Save the activation maps of every layer of the network
def save_all_activation_maps(model,image): 

    total = [i for i,(_,_) in enumerate(model.named_modules())][-1]

    progressB = tqdm.tqdm(enumerate(model.named_modules()),total=total)
    
    for idx, (layer_name, _) in progressB:

        activ_maps = get_activation_maps(model,layer_name,image)

        prediction = get_prediction(model,image)

        fig = create_figure(activ_maps,layer_name,image,prediction)
        save_activation_map(fig,layer_name)
        plt.close('all')
        progressB.set_description("Saving {} images...".format(total))