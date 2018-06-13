
# coding: utf-8

# In[ ]:


#get_ipython().run_line_magic('run', 'Unet_v1.ipynb')
#get_ipython().run_line_magic('run', 'Preprocess.ipynb')
from Unet_v1 import *
from Preprocess import *


# In[ ]:

# In[ ]:
data_gen_args = dict()
myGene = trainGenerator(3, 'notebook/Data/aerialsample/Train', 'images', 'label', data_gen_args, save_to_dir = None)
model = unet()
model_checkpoint = ModelCheckpoint('unet_aerial.hdf5', monitor = 'loss', verbose = 1, save_best_only = True)
model.fit_generator(myGene, steps_per_epoch = 3, epochs = 3, callbacks = [model_checkpoint])



testGene = io.imread('/Users/JuliaChen/Google Drive/InsightAI/notebook/Data/aerialsample/test/image/austin0.tif')
#display_images([testGene], cols =2)

model = unet()
model.load_weights('unet_aerial.hdf5')

#view_layer(model, testGene, "conv2D_2")
#view_layer(model, testGene, "conv2D_22")


results = model.predict_generator(testGene, 4, verbose = 1)
saveResult('Data/aerialsample/test', results)

