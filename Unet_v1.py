
# coding: utf-8

# In[1]:


import os
import numpy as np
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


# In[2]:


def unet(pretrained_weights = None, input_size = (4800, 4800, 1)):
        
        inputs = Input(input_size)
        
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        print ("conv1 shape:", conv1.shape)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        print ("conv1 shape:", conv1.shape)
        pool1 = MaxPooling2D(pool_size = (2,2))(conv1)
        print ("pool1 shape:", pool1.shape)
        
        conv2 = Conv2D(128, 3, activation  = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        print ("conv2 shape:", conv2.shape)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        print ("conv2 shape:", conv2.shape)
        pool2 = MaxPooling2D(pool_size = (2,2))(conv2)
        print ("pool2 shape:", pool2.shape)
        
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        print ("conv3 shape:", conv3.shape)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        print ("conv3 shape:", conv3.shape)
        pool3 = MaxPooling2D(pool_size = (2,2))(conv3)
        print ("pool3 shape:", pool3.shape)
        
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation  = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        print ("conv4 shape:", conv4.shape)
        drop4 = Dropout(0.5)(conv4)
        print ("drop4 shape:", drop4.shape)
        pool4 = MaxPooling2D(pool_size = (2,2))(drop4)
        print ("pool4 shape:", pool4.shape)
        
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        print ("conv5 shape:", conv5.shape)
        drop5 = Dropout(0.5)(conv5)
        print ("drop5 shape:", drop5.shape)
        
        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        #up6 = concatenate([UpSampling2D(size = (2,2))(drop5), drop4], axis = 3)
        merge6 = concatenate([drop4, up6], axis = 3)
        #merge6 = merge([drop4, up6], mode = 'concat', concat_axis = 3)
        conv6 = Conv2D(512, 3, activation =  'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
        
        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        #up7 = concatenate([UpSampling2D(size = (2,2))(conv6), conv3], axis = 3)
        merge7 = concatenate([conv3, up7], axis = 3)
        #merge7 = merge([conv3, up7], mode = 'concat', concat_axis = 3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
        
        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        #merge8 = merge([conv2, up8], mode = 'concat', concat_axis = 3)
        merge8 = concatenate([conv2, up8], axis = 3)
        #up8 = concatenate([UpSampling2D(size = (2,2))(conv7), conv2], axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
        
        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        #merge9 = merge([conv1, up9], mode = 'concat', concat_axis = 3)
        merge9 = concatenate([conv1, up9], axis = 3)
        #up9 = concatenate([UpSampling2D(size = (2,2))(conv8), conv1], axis = 3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        print ("conv9 shape:", conv9.shape)
        
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
        print ("conv10 shape:", conv10.shape)
        
        model = Model(input = inputs, output = conv10)
        
        model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        #model.summary() 
        
        if (pretrained_weights):
            model.load_weights(pretrained_weights)
        
        return model

