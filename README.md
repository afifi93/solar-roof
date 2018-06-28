# Solar Roof

Is your roof green enough?!

### Abstract

Detect and segment objects of interest given an aerial/satellite images.

Traditionally, when a customer wants to install solar panels on their house, the company would send out a team to measure the roof dimension and cutomized/find the solar panels that will fit. This back-and-forth takes a lot of time and increases the risk of workers' injury. This project implemented a variation of convolutional neural network, U-net, to segment individual's roof using aerial/satellite images.

Due to the limited labeled roof images, I implemented a U-net model on segment building with public aerial images (https://project.inria.fr/aerialimagelabeling/). Roof boundary is very close to the building boundray from the top-view. Few modifications will be needed to improve the accuray of roof boundary. 

Googld slides can be found [here] (https://goo.gl/xcTBAW)

# Installation

The file, requirement.txt, contains the python package dependencies for this project. Installation can be performed via 


pip install -r requirement.txt

# Input images

Please find a small part of the public data set for training this model in datasets folder. 

# Pre-processing

Input data were RGB images and in PNG format. 
The public data set is high resolution (5000x5000) but too big to start with to train the MVP model. Therefore, I sliced the originial image evenly into 625 smaller images, resulting 625 (200 * 200) images. The label set was preprocessed the same as training set. 

# Example

train.py in the example folder can be served as an template on how to incoporate all functions from sun_roof folder, train the U-net and make inference.

# Model
![Unet](https://github.com/julia78118/SolarRoof/blob/master/Unet.jpg)

The U-net model code base was inspired by https://github.com/zhixuhao/unet but heavily modified for this project.

# Inference

Pre-trained weights can be downloaded from DropBox (https://www.dropbox.com/s/qkpamyh618n0kj9/unet_sgd70.hdf5?dl=0)
Test images can also be found in the datasets folder. 
Perform via command line

# Post-processing

The boundary will be shown on the original input image

# Packaging

Install as single package using pip install setup.py
