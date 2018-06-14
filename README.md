# SolarRoof

Is your roof green enough?!

### Abstract

Traditionally, when a customer wants to install solar panels on their house, the company would send out a team to measure the roof dimension and cutomized/find the solar panels that will fit. This back-and-forth takes a lot of time and increases the risk of workers' injury. This project implemented a variation of convolutional neural network, called U-net, to segment individual's roof using aerial/satelite images.

Due to the limited labeled roof images, I implemented a U-net model on segment building with public aerial images. Roof boundary is very close to the building boundray from the top-view. Few modifications will be needed to improve the accuray of roof boundary. 

# Installation

The file, requirement.txt, contains the python package dependencies for this project. Installation can be performed via 


pip install -r requirement.txt

# Input images

Please find the public data set for training this model in dataset folder

# Preprocessing

Input data were converted to 8bit (pixel range from 0-255), .png format and gray-scale. 
The public data set is high resolution (5000x5000) but too big for my local computer to train the model. Therefore, I sliced the originial image evenly into 100 smaller images, resulting 100 (500 * 500) images. The label set was preprocessed the same as training set. 


# Model
![Unet](http://url/to/unet.png)
The U-net model code base was inspired by https://github.com/zhixuhao/unet but heavily modified for this project.

# Inference

Test images can also be found in the dataset folder. 
Perform via command line

# Postprocessing

