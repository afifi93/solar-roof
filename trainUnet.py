
# coding: utf-8

#get_ipython().run_line_magic('run', 'Unet_v1.ipynb')
#get_ipython().run_line_magic('run', 'Preprocess.ipynb')
from Unet_v2 import *
from Preprocess import *
import matplotlib.pyplot as plt
from Post import *
import numpy as np

## generate input image/label as a set
data_gen_args = dict()
myGene = trainGenerator(1, 'notebook', 'image', 'label', data_gen_args, save_to_dir = None)

#check(myGene)

model = unet()
model_checkpoint = ModelCheckpoint('unet_.hdf5', monitor = 'loss', verbose = 1, save_best_only = True)
history = model.fit_generator(myGene, steps_per_epoch = 10, epochs = 4, callbacks = [model_checkpoint, model_history])

hist_array = np.array(history.history['loss'])
np.savetxt('loss_history.txt', hist_array, delimiter=',')

# plot history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# generate test image
testGene = testGenerator('notebook/test')

model = unet()
model.load_weights('unet_.hdf5')
results = model.predict_generator(testGene, 1, verbose = 1)
saveResult('notebook', results)

import matplotlib.image as pimg
#import cv2 as cv

img = pimg.imread('0_predictgray32.png')
thre = 0.49

binary = to_binary(img, 0.49)
plt.imshow(binary, 'gray')
plt.show()
#io.imsave(os.path.join('notebook/binary', 'predictgray32.png'), binary)
