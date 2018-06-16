# This is the main script
from Unet_rgb import *
from Preprocess_rgb import *
import matplotlib.pyplot as plt
#from Post import *
import numpy as np
import cv2
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, History

# generate input image/label as a set
# trainGenerator(batch_size, train_path, image_folder, mask_folder)
data_gen_args = ()
myGene = trainGenerator(50, '', 'images', 'label', data_gen_args, save_to_dir = None)

model = unet()
model.compile(optimizer = Adam(lr = 1e-4), loss = IoU_loss, metrics = ['accuracy', IoU_coef, IoU_coef_int])
model_checkpoint = ModelCheckpoint('unet_rgb.hdf5', monitor = 'loss', verbose = 1, save_best_only = True)
history = model.fit_generator(myGene, steps_per_epoch = 50, epochs = 300, callbacks = [model_checkpoint])

# Save loss history as txt file
hist_array = np.array(history.history['loss'])
np.savetxt('loss_history.txt', hist_array, delimiter=',')

# plot history
# summarize accuracy history for training and validation
plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize loss history for training and validation
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# generate test set; test_path as input
testGene = testGenerator('test')

model = unet()
model.compile(optimizer = Adam(lr = 1e-4), loss = IoU_loss, metrics = ['accuracy', IoU_coef, IoU_coef_int])
# Load pre-trained weights
model.load_weights('unet_rgb.hdf5')
# Perform inference
results = model.predict_generator(testGene, steps=1, verbose = 1)
# Save prediction image in the provided path
saveResult('', results)

# Post-processing prediction images
#img = pimg.imread('0_predict.png')
#thre = 0.49
#
#binary = to_binary(img, 0.49)
#io.imsave(os.path.join('predict', 'predbinary.png'), binary)
