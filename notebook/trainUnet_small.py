from Unet_small import *
from Preprocess_small import *
import matplotlib.pyplot as plt
from Post import *
import numpy as np
import cv2

# generate input image/label as a set
data_gen_args = ()
myGene = trainGenerator(50, '', 'imagegray', 'labelgray', data_gen_args, save_to_dir = None)

model = unet()
model_checkpoint = ModelCheckpoint('unet_small600.hdf5', monitor = 'loss', verbose = 1, save_best_only = True)
history = model.fit_generator(myGene, steps_per_epoch = 4, epochs = 600, callbacks = [model_checkpoint])

hist_array = np.array(history.history['loss'])
np.savetxt('loss_history600.txt', hist_array, delimiter=',')

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
testGene = testGenerator('testgray')

model = unet()
model.load_weights('unet_small600.hdf5')
results = model.predict_generator(testGene, steps=1, verbose = 1)
saveResult('predict', results)


#img = pimg.imread('0_predict.png')
#thre = 0.49
#
#binary = to_binary(img, 0.49)
#io.imsave(os.path.join('predict', 'predbinary.png'), binary)
