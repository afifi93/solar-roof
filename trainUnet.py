
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


# In[ ]:


testGene = testGenerator('Data/aerialsample/test/image')
model = unet()
model.load_weights('unet_aerial.hdf5')
results = model.predict_generator(testGene, 4, verbose = 1)
saveResult('Data/aerialsample/test', results)

