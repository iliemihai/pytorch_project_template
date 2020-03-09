# PyTorch Project Template

Implement your PyTorch projects the smart way.

A **Scalable template** for 
PyTorch projects, with examples in Image Segmentation, Object classification, GANs and Reinforcement Learning.

Given the nature of deep learning projects, we do not get the chance to think much about the project structure or the code modularity. After working with different deep learning projects and facing problems with files organization and code repetition, we came up with a **modular project structure** to accommodate any PyTorch project.
We also wanted to provide **a base for different PyTorch models for the community** to build upon.

This is a joint work between [Hager Rady](https://github.com/hagerrady13/) and [Mo'men AbdelRazek](https://github.com/moemen95)

### Table of Contents: 
<!-- Table of contents generated generated by http://tableofcontent.eu -->
- [PyTorch Project Template](#pytorch-project-template)
    - [Why this template?](#why-this-template)
    - [Tutorials](#tutorials)
    - [Contribution](#contribution)
    - [Template Class Diagram](#template-class-diagram)
    - [Referenced Repos](#referenced-repos)
    - [Repos Migration Summary](#repos-migration-summary)
    - [Template Repo Structure](#repo-structure)
    - [Requirements](#requirements)
    - [Future Work](#future-work)
    - [License](#license)
    
### Why this template?

We are proposing **a baseline** for any PyTorch project to give you a quick start, where you will get the time to focus on your model's implementation and we will handle the rest. The novelty of this approach lies in:
- Providing **a scalable project structure**, with a template file for each.
- Introducing the usage of a config file that handle all the hyper-parameters related to a certain problem.
- Embedding examples from various problems inside the template, where you can run any of them independently with a **single change** in the config file name.
- **Tutorials** to get you started.

### Tutorials:
We are providing a series of tutorials to get your started

* [Getting Started Tutorial](https://github.com/moemen95/PyTorch-Project-Template/blob/master/tutorials/getStarted_tutorial.md): where we provide a guide on the main steps to get started on your project.
* [Mnist Tutorial](https://github.com/moemen95/PyTorch-Project-Template/blob/master/tutorials/mnist_tutorial.md): Here we take an already implemented NN model on Mnist and adapt it to our template structure.

### Contribution:
* We want this template to be **a central place** for many variations of the well-known PyTorch Deep learning models.
* We are welcoming any contribution from the community that will make this come true so we urge people to add their PyTorch models into the template.
* We are also welcoming any proposed changes or discussions about the class design pattern used in this project.

### Template Class Diagram:
![alt text](utils/assets/class_diagram.png "Template Class diagram")

This is to ensure that our proposed project structure is compatible with different problems and can handle all the variations related to any of them.

### Repo Structure:
After adding all our examples, the repo has the following structure:
```
├── agents
|  └── dcgan.py
|  └── condensenet.py
|  └── mnist.py
|  └── dqn.py
|  └── example.py
|  └── base.py
|  └── erfnet.py
|
├── configs
|  └── dcgan_exp_0.py
|  └── condensenet_exp_0.py
|  └── mnist_exp_0.py
|  └── dqn_exp_0.py
|  └── example_exp_0.py
|  └── erfnet_exp_0.py
|
├── data
|
├── datasets
|  └── cifar10.py
|  └── celebA.py
|  └── mnist.py
|  └── example.py
|  └── voc2012.py
|
├── experiments
|
├── graphs
|  └── models
|  |  └── custome_layers
|  |  |  └── denseblock.py
|  |  |  └── layers.py
|  |  |
|  |  └── dcgan_discriminator.py
|  |  └── dcgan_generator.py
|  |  └── erfnet.py
|  |  └── erfnet_imagenet.py
|  |  └── condensenet.py
|  |  └── mnist.py
|  |  └── dqn.py
|  |  └── example.py
|  |
|  └── losses
|  |  └── loss.py
|
├── pretrained_weights
|
├── tutorials
|
├── utils
|  └── assets
|
├── main.py
└── run.sh
```

### To install environment
```
cd pytorch_project_template
sudo pip3 install virtualenv      # This may already be installed
virtualenv -p python3.6 .env       # Create a virtual environment (python3)
# Note: you can also use "virtualenv .env" to use your default python (please note we support 3.6)
source .env/bin/activate         # Activate the virtual environment
pip3 install -r requirements.txt  # Install dependencies
# Work on the assignment for a while ...
deactivate                       # Exit the virtual environment

```

### Requirements:
```
numpy
scipy
Pillow
matplotlib
scikit-learn
skorch
tqdm
easydict
spacy
imageio
torch==1.4.0
torchtext
torchvision
tensorboardX
transformers
tokenizers
```

### Future Work:

We are planning to add more examples into our template to include various categories of problems. Next we are going to include the following:

* [MobilenetV2](https://github.com/MG2033/MobileNet-V2)
* [visual-interaction-networks-pytorch](https://github.com/MrGemy95/visual-interaction-networks-pytorch)
* [variational-Autoencoder-pytorch](https://github.com/pytorch/examples/tree/master/vae)

### License:
This project is licensed under MIT License - see the LICENSE file for details
