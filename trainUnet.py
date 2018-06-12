
# coding: utf-8

# In[ ]:


#get_ipython().run_line_magic('run', 'Unet_v1.ipynb')
#get_ipython().run_line_magic('run', 'Preprocess.ipynb')
from Unet_v1 import *
from Preprocess import *

# In[ ]:


data_gen_args = dict()
myGene = trainGenerator(5, 'Data/aerialsample/Train/images', 'output', 'output', data_gen_args, save_to_dir = None)
model = unet()
model_checkpoint = ModelCheckpoint('unet_aerial.hdf5', monitor = 'loss', verbose = 1, save_best_only = True)
model.fit_generator(myGene, steps_per_epoch = 3, epochs = 3, callbacks = [model_checkpoint])


def read_layer(model, x, layer_name):
    """Return the activation values for the specifid layer"""
    # Create Keras function to read the output of a specific layer
    get_layer_output = keras.function([model.layers[0].input], [model.get_layer(layer_name).output])
    outputs = get_layer_output([x])[0]
    tensor_summary(outputs)
    return outputs[0]

def view_layer(model, x, layer_name, cols=5):
    outputs = read_layer(model, x, layer_name)
    display_images([outputs[:,:,i] for i in range(10)], cols=cols)

view_layer(model, myGene, "conv2D_2")
view_layer(model, myGene, "conv2D_22")
# In[ ]:


testGene = testGenerator('Data/aerialsample/test/image')
model = unet()
model.load_weights('unet_aerial.hdf5')
results = model.predict_generator(testGene, 4, verbose = 1)
saveResult('Data/aerialsample/test', results)

