"""
    Script to build U-net
    """
from keras.models import Model
from keras.layers import (Input, Conv2D, concatenate,
                          MaxPooling2D, UpSampling2D, Dropout)


def unet(pretrained_weights=None, input_size=(256, 256, 3)):

    inputs = Input(input_size)

    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    print("pool1 shape:", pool1.get_shape())

    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    print("pool2 shape:", pool2.get_shape())

    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    print("pool3 shape:", pool3.get_shape())

    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    print("pool4 shape:", pool4.get_shape())

    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)
    print("conv5 shape:", conv5.get_shape())

    merge6 = concatenate([conv4, (UpSampling2D(size=(2, 2))(conv5))], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv6)
    print("conv6 shape:", conv6.get_shape())

    merge7 = concatenate([conv3, (UpSampling2D(size=(2, 2))(conv6))], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv7)
    print("conv7 shape:", conv7.get_shape())

    merge8 = concatenate([conv2, (UpSampling2D(size=(2, 2))(conv7))], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv8)
    print("conv8 shape:", conv8.get_shape())

    merge9 = concatenate([conv1, (UpSampling2D(size=(2, 2))(conv8))], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    print("conv9 shape:", conv9.get_shape())

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    print("conv10 shape:", conv10.get_shape())

    model = Model(inputs=inputs, outputs=conv10)

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
