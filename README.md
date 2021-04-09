
<!-- PROJECT LOGO -->
<br />

<h1 align="left">Activation maps visualizer for PyTorch</h1>

<p align="center">
    <img src="example.png" alt="example" >
</p>



<!-- ABOUT THE PROJECT -->
## About The Project

I've created a little PyTorch script to see the activation maps of a specific, or all the CNN's layers. 
Since I'm trying to switch from Tensorflow/Keras to PyTorch, I thought it would be interesting to do this little project to get used to handling torch objects and tensors.

### Built With

* [Python 3.8](https://www.python.org/)
* [PyTorch](https://pytorch.org/)
* [Matplotlib](https://matplotlib.org)


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.

### Prerequisites

* Python  ⩾ 3.8
  ```sh
  sudo apt install python3 python3-pip
  ```

### Installation


1. Clone the repo
   ```sh
   git clone https://github.com/Clement-W/Activation-Maps-Visualiser-PT.git
   cd Activation-Maps-Visualiser-PT/
   ```
3. Create and activate a virtual environment
   ```sh
   pip3 install virtualenv --upgrade
   virtualenv venv
   source venv/bin/activate
   ```
4. Install the requirements
   ```sh
   pip3 install -r requirements.txt
   ```


<!-- USAGE EXAMPLES -->
## Usage


### Use ActivationMapExtractor.py

There is two main way to use this python script :

* Call the function **show_activation_maps(model, layer_name, image)**. This function show the activation maps for a specific layer in the model. 
* Call the function **save_all_activation_maps(model, image,path_to_directory)**. This function save the activation maps of every layers of the model in the specified directory. 

Check the next section to see an example.




### Demo with RestNet-18

Import modules :
```py
from torchvision import models, transforms, datasets
import torch
from PIL import Image
from ActivationMapsExtractor import save_all_activation_maps, show_activation_maps
```
Set torch device :
```py
device = torch.device("cuda:0"  if torch.cuda.is_available() else  "cpu")
```
Load pretrained RestNet-18 model from PyTorch :
```py
model = models.resnet18(pretrained=True)
```
Load an image to feed the ResNet-18 model :
```py
image = Image.open("imageNet/sample.JPEG")
transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
image = transform(image).to(device)
```

If you want to see the activation maps of a specific layer by it's name for that input image :
```py
show_activation_maps(model,"conv1",image)
```

If you want to save the activation maps of every layers for that input image :
```py
save_all_activation_maps(model, image,"./ResNet-activation-maps")
```


<!-- CONTRIBUTING -->
## Contributing

I'm still learning PyTorch, so feel free to use Issues or PR to report errors and/or propose additions or corrections to my code. Any contributions you make are **greatly appreciated**.


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<!-- TODO LIST -->
## Todo list

- [ ] Add a Jupyter notebook demo
