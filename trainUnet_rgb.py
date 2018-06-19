# This is the main script
from Unet_v1 import *
from newProcess import *
#import matplotlib.pyplot as plt
from Post import *
import numpy as np
#import cv2
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, History, TensorBoard
#from time import time

## generate input image/label as a set
# trainGenerator(batch_size, train_path, image_folder, mask_folder)
data_gen_args = ()
myGene = trainGenerator(10, '', 'images', 'label', data_gen_args, save_to_dir = None)

model = unet()
model.compile(optimizer = Adam(lr = 5e-5), loss = IoU_loss, metrics = ['binary_crossentropy', IoU_coef_int])
model_checkpoint = ModelCheckpoint('unet_10.hdf5', monitor = 'loss', verbose = 1, save_best_only = True)
#tb = TensorBoard(log_dir='logs/{}'.format(time()))
#file_writer = tf.summary.FileWriter('logs', sess.graph)
history = model.fit_generator(myGene, steps_per_epoch = 1, epochs = 1000, callbacks = [model_checkpoint, TensorBoard('logs')])

# Save loss/accuracy history as txt file
hist_array = np.array(history.history['loss'])
np.savetxt('loss_10.txt', hist_array, delimiter=',')

# plot history
# summarize accuracy history for training and validation
#plt.plot(history.history['acc'])
##plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.yscale('log')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()

# summarize loss history for training and validation
#plt.plot(history.history['loss'])
##plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()

# generate test set; test_path as input
testGene = testGenerator('test')

model = unet()
model.compile(optimizer = Adam(lr = 5e-5), loss = IoU_loss, metrics = ['binary_crossentropy', IoU_coef_int])
# Load pre-trained weights
model.load_weights('unet_10.hdf5')
# Perform inference
results = model.predict_generator(testGene, 2, verbose = 1)

# Save prediction image in the provided path
saveResult('', results)

# Post-processing prediction images; input as (pred_file, test_path, original_size)
#draw_bound(results, 'test', (200, 200))
