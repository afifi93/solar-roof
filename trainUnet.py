
# coding: utf-8

# In[ ]:


#get_ipython().run_line_magic('run', 'Unet_v1.ipynb')
#get_ipython().run_line_magic('run', 'Preprocess.ipynb')
from Unet_v2 import *
from Preprocess import *
import matplotlib.pyplot as plt
# In[ ]:


data_gen_args = dict()
myGene = trainGenerator(1, 'notebook', 'image', 'label', data_gen_args, save_to_dir = 'notebook/ex')
model = unet()
model_checkpoint = ModelCheckpoint('unet_gray.hdf5', monitor = 'loss', verbose = 1, save_best_only = True)
model.fit_generator(myGene, steps_per_epoch = 2, epochs = 2, callbacks = [model_checkpoint])

# In[ ]:


testGene = testGenerator('notebook/test')

model = unet()
model.load_weights('unet_gray.hdf5')
results = model.predict_generator(testGene, 1, verbose = 1)
saveResult('notebook', results)

