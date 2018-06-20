# Functions for pre-processing steps
import numpy as np
import math

def data_is_ok(data, raise_exception=False):
    """Perform a check to ensure the image data is in the correct range
        Args
        ----
        data (np.array) : the image data
        use (str) : ['obj', 'patch', 'None'] the type (or use) of image passed,
        this is for checking the image shape, if None, don't check the shape
        raise_exception (bool) : raise exception if data is not ok
        Returns
        -------
        (bool) : True if data is OK, otherwise False
        """
    try:
        assert data.dtype == np.uint8
        assert data.max() <= 255
        assert data.min() >= 0

        # make sure data wasn't normalized to [0,1]
        assert data.max() > 1.0

    except AssertionError as e:
        if raise_exception:
            raise e
        else:
            _data_is_ok = False
    else:
        _data_is_ok = True
        print (_data_is_ok)
    return _data_is_ok

def image_save_preprocessor(img, report=True):
    """Normalize the image
        Procedure
        ---------
        - Convert higher bit images (16, 10, etc) to 8 bit
        - Set color channel to the last channel
        - Drop Alpha layer and conver b+w -> RGB
        TODO
        ----
        Correctly handle images with values [0,1)
        Args
        ----
        img (np array) : raw image data
        report (bool) : output a short log on the imported data
        Returns
        -------
        numpy array of cleaned image data with values [0, 255]
        """

    data = np.asarray(img)

    if data.ndim == 3:
        # set the color channel to last if in channel_first format
        if data.shape[0] <= 4:
            data = np.rollaxis(data, 0, 3)

        # remove alpha channel
        if data.shape[-1] == 4:
            data = data[...,:3]

    # if > 8 bit, shift to a 255 pixel max
    bitspersample = int(math.ceil(math.log(data.max(), 2)))
    if bitspersample > 8:
        data >>= bitspersample - 8


    # if data [0, 1), then set range to [0,255]
    if bitspersample <= 0:
        data *= 255

    data = data.astype(np.uint8)

    if report:
        print("Cleaned To:")
        print("\tShape: ", data.shape)
        print("\tdtype: ", data.dtype)


    # Make sure the data is actually in the correct format
    data_is_ok(data, raise_exception=True)

    return data

def preprocess_image_for_model(data, raise_exception=False):
    """Process data in the manner expected by retinanet
        preprocessor That takes a clean image and performs final adjustments
        before it's feed into a model
        Convert RGB -> BGR
        normalize in the VGG16 way
        Notes
        -----
        - handles batch or single image
        - do NOT use this with Retina net built in pre processor, the pre-processor,
        will repeat these commands.
        References
        ----------
        (*) keras_retinanet : https://github.com/fizyr/keras-retinanet
        Args
        ----
        data (np.array) : of shape ( _, 400, 400, 3) for obj identification
        or (_, 200, 200, 3) for patch identification
        Returns
        -------
        normalized data of the same shape
        """

    try:
        data_is_ok(data, raise_exception=True)
    except AssertionError as e:
        data = image_save_preprocessor(data, report=False)
        # Doing this for debug purposes
        if raise_exception:
            raise e

    # cast as float
    data = data.astype(np.float32)

    # Normalize to the mean of the color channels
#    data[...,0] -= 103.939
#    data[...,1] -= 116.779
#    data[...,2] -= 123.78

    return data
    

