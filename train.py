# This is the main script
from Unet_v1 import *
from check_data import *
from preprocess_train_val import *
from post import *
import numpy as np
import glob
import os
import shutil
from keras import backend as K
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, History, TensorBoard
from keras.optimizers import Adam, SGD

## split input to training/validation sets
base_dir = ''
sourceN = base_dir + 'train/images'
destN = base_dir + 'val/images'
sourceP = base_dir + 'train/label'
destP = base_dir + 'val/label'

filesN = os.listdir(sourceN)
filesP = os.listdir(sourceP)
##
#for f in filesN:
#    if np.random.rand(1) < 0.01:
#        shutil.move(sourceN + '/'+ f, destN + '/'+ f)
#        shutil.move(sourceP + '/'+ f, destP + '/'+ f)
print(len(os.listdir(sourceN)))
print(len(os.listdir(sourceP)))
print(len(os.listdir(destN)))
print(len(os.listdir(destP)))
tra_num = len(os.listdir(sourceN))
val_num = len(os.listdir(destN))
#
## generate input image/label as a set
## trainGenerator(batch_size, train_path, image_folder, mask_folder)
data_gen_args = ()
mytrain = trainGenerator(32, 'train', 'images', 'label', data_gen_args, save_to_dir = None)

# generate validation sets
myval = valGenerator(32, 'val', 'images', 'label', data_gen_args, save_to_dir = None)
model = unet()
optimizer = SGD(lr=1e-4, momentum=0.9, nesterov=True)
model.compile(optimizer = optimizer, loss = IoU_loss, metrics = ['accuracy', 'binary_crossentropy', IoU_coef_int])

# training model
model_checkpoint = ModelCheckpoint('unet_sgd35.hdf5', monitor = 'loss', verbose = 1, save_best_only = True)
tb = TensorBoard(log_dir='./logs', write_graph=True, write_images=True)
history = model.fit_generator(mytrain, steps_per_epoch = (tra_num/32), epochs = 35, callbacks = [model_checkpoint, tb], validation_data = myval, validation_steps=(val_num/32))

# Save loss/accuracy history as txt file
#print(history.history.keys())
train_array = np.array(history.history['loss'])
val_array = np.array(history.history['val_loss'])
np.savetxt('loss_history35.txt', train_array, delimiter=',')
np.savetxt('val_history35.txt', val_array, delimiter=',')


# plot history
# summarize loss history for training and validation
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')
#plt.show()
#
# generate test set; test_path as input
testGene = testGenerator('test')

model = unet()
model.compile(optimizer = optimizer, loss = IoU_loss, metrics = ['accuracy', 'binary_crossentropy', IoU_coef_int])
# Load pre-trained weights
model.load_weights('unet_sgd35.hdf5')
# Perform inference
results = model.predict_generator(testGene, 7, verbose = 1)

# Save prediction image in the provided path
saveResult('', results)
#
## Post-processing prediction images; input as (pred_file, test_path, original_size)
##draw_bound(results, 'test', (200, 200))
