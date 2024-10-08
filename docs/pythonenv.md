# Python Environment

All code is assumed to be run on a Linux environment with python `3.10` and cuda 11.8 installed.

## Create Environment

First create the python environment using the virtualenv command where LOCATION is the directory.

`python3 -m virtualenv /LOCATION/flowMatching`

Then activate the environment with

`source LOCATION/flowMatching/bin/activate`

## Install pyTorch

the latest version of pytorch with cuda 11.8 can be installed using this command

`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

## Install requirements.txt

All requirements for this project can be installed using

`pip install -r requirements.txt`

# Installing Body Model

A body model will be needed to display the generated poses. 
SMPL is the one that was chosen for this project, but it is very easy to modify the genreation to work for SMPL+X and SMPLify-X models

To download SMPL model go to the [project page](https://smpl.is.tue.mpg.de/) and add it to the dataset file. 