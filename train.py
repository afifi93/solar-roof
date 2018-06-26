"""This python file is main running script
    """
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import SGD
from preprocess import split_train_val, train_generator, valid_generator, test_generator
from post import jaccard_coef, jaccard_loss, save_history, save_result, draw_boundary
from Unet import unet

# split training and validation sets
tra_num, val_num = split_train_val(
    '', 'train/images', 'val/images', 'train/label', 'val/label')

# train_generator(batch_size, train_path, image_folder, mask_folder)
data_gen_args = ()
mytrain = train_generator(32, 'train', 'images', 'label',
                          data_gen_args, save_to_dir=None)

# generate validation sets
myval = valid_generator(32, 'val', 'images', 'label',
                        data_gen_args, save_to_dir=None)
model = unet()
optimizer = SGD(lr=1e-3, momentum=0.9, nesterov=True)
model.compile(optimizer=optimizer, loss=jaccard_loss, metrics=[
              'accuracy', 'binary_crossentropy', jaccard_coef])

# training model
model_checkpoint = ModelCheckpoint(
    'unet_sgd70.hdf5', monitor='loss', verbose=1, save_best_only=True)
tb = TensorBoard(log_dir='./logs', write_graph=True, write_images=True)
history = model.fit_generator(mytrain, steps_per_epoch=(tra_num / 32), epochs=70, callbacks=[
                              model_checkpoint, tb], validation_data=myval, validation_steps=(val_num / 32))

save_history(history)

# generate test set; test_path as input
testGene = test_generator('test')

model = unet()
model.compile(optimizer=optimizer, loss=jaccard_loss, metrics=[
              'accuracy', 'binary_crossentropy', jaccard_coef])

# Load pre-trained weights
model.load_weights('unet_sgd70.hdf5')

# Perform inference
results = model.predict_generator(testGene, 54, verbose=1)

# Save prediction image in the provided path
save_result('', results)

# Post-processing prediction images
# Input as (pred_file, test_path, pred_img_size)
draw_boundary(results, 'test', (256, 256))
