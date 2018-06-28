# Solar Roof

Is your roof green enough?!

### Abstract

Detect and segment objects of interest given an aerial/satellite images.

Today, solar panel installation requires a crew of workers to measure your roof dimensions to create a custom engineered solution. This process is costly and can lead to worker injury. My project automates this process by leveraging satellite images and deep learning segmentation to create accurate traces of roof lines.

Due to the limited labeled roof images, I implemented a U-net model on segmenting building with [public aerial images](https://project.inria.fr/aerialimagelabeling/). Roof boundary is very close to the building boundray from the top-view. Few modifications will be needed to improve the accuray of roof boundary. 

Googld slides can be found [here](https://goo.gl/xcTBAW)

# Installation

The file, `requirement.txt`, contains the python package dependencies for this project. Installation can be performed via 
```
pip install -r requirement.txt
```

# Input images

Please find a small set of the public data for training this model in `datasets` folder. 

# Pre-processing

Input data were RGB images and in PNG format. 
The public data set is high resolution (5000x5000) but too inefficient to train the MVP model. Therefore, I sliced the originial image evenly into 625 smaller images, resulting 625 (200 * 200) images. The label set was preprocessed the same as training set. Final model was trained with 20,000 images for 70 epochs.

# Example

`train.py` in the `example` folder can be served as an template on how to incoporate all functions from `sun_roof` folder, to train the U-net and to make inference.

# Model
![Unet](https://github.com/julia78118/SolarRoof/blob/master/Unet.jpg)

The U-net model code base was inspired by https://github.com/zhixuhao/unet but heavily modified for this project.

# Inference

Pre-trained weights can be downloaded from [DropBox](https://www.dropbox.com/s/qkpamyh618n0kj9/unet_sgd70.hdf5?dl=0).
Test set images can also be found in the `datasets` folder. 


# Post-processing

The traced roof lines would be shown on the original input image

# Packaging

Install as a single package  
```
pip install setup.py
```
