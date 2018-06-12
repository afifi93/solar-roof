
# coding: utf-8

# In[1]:


import Augmentor
p = Augmentor.Pipeline('Data/aerialsample/Train/images')
p.ground_truth('Data/aerialsample/Train/gt')

p.rotate90(probability=0.5)
p.rotate270(probability=0.5)
p.flip_left_right(probability=0.8)
p.flip_top_bottom(probability=0.3)
p.sample(10)

