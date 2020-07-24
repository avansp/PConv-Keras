#!/usr/bin/env python
# coding: utf-8

# # CelebA Partial Convolution
# ----
# Modified from https://github.com/MathiasGruber/PConv-Keras/blob/master/notebooks/Step4%20-%20Imagenet%20Training.ipynb
# 
# This uses smaller size of images, so it should be quicker.

# In[ ]:


import os
import pandas as pd
import datetime
from pconv_keras.util import MaskGenerator
from pconv_keras.generator import AugmentingDataGenerator
from pconv_keras.pconv_model import PConvUnet

from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from IPython.display import clear_output

BATCH_SIZE=16


# ## Create Test & Train Generator

# In[ ]:


# Create training generator
train_datagen = AugmentingDataGenerator(  
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1./255,
    horizontal_flip=True
)
train_generator = train_datagen.flow_from_dataframe(
    pd.read_csv("/mnt/data/data/celeba/train.csv"),
    MaskGenerator(256, 256, 3),
    folder='/mnt/data/data/celeba/',
    target_size=(256, 256), 
    batch_size=BATCH_SIZE
)


# In[ ]:


# Create validation generator
val_datagen = AugmentingDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_dataframe(
    pd.read_csv("/mnt/data/data/celeba/val.csv"), 
    MaskGenerator(256, 256, 3), 
    folder='/mnt/data/data/celeba/',
    target_size=(256,256), 
    batch_size=BATCH_SIZE, 
    seed=42
)


# In[ ]:


# Create testing generator
test_datagen = AugmentingDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    pd.read_csv("/mnt/data/data/celeba/test.csv"), 
    MaskGenerator(256, 256, 3), 
    folder='/mnt/data/data/celeba/',
    target_size=(256, 256), 
    batch_size=BATCH_SIZE, 
    seed=42
)


# # Training on ImageNet

# In[ ]:


# Pick out an example
test_data = next(test_generator)
(masked, mask), ori = test_data


# In[ ]:


def plot_callback(model, folder):
    """Called at the end of each epoch, displaying our previous test images,
    as well as their masked predictions and saving them to disk"""
    
    # Get samples & Display them        
    pred_img = model.predict([masked, mask])
    pred_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # Clear current output and display test images
    for i in range(len(ori)):
        _, axes = plt.subplots(1, 3, figsize=(20, 5))
        axes[0].imshow(masked[i,:,:,:])
        axes[1].imshow(pred_img[i,:,:,:] * 1.)
        axes[2].imshow(ori[i,:,:,:])
        axes[0].set_title('Masked Image')
        axes[1].set_title('Predicted Image')
        axes[2].set_title('Original Image')
                
        plt.savefig(os.path.join(folder, f"00img_{i}_{pred_time}.png"))
        plt.close()


# ### Phase 1 - with batch normalization

# In[ ]:


# Instantiate the model
model = PConvUnet(img_rows=256, img_cols=256,
                  vgg_weights=r"/mnt/data/train_camp/pconv_keras_imagenet/pytorch_to_keras_vgg16.h5")
FOLDER = r'/mnt/data/train_camp/pconv_keras_celeba/imagenet_phase1_paperMasks'
TEST_SAMPLE_FOLDER = os.path.join(FOLDER, 'test_samples')
if not os.path.isdir(TEST_SAMPLE_FOLDER):
    os.makedirs(TEST_SAMPLE_FOLDER)


# In[ ]:


# Run training for certain amount of epochs
model.fit_generator(
    train_generator, 
    steps_per_epoch=len(train_datagen.generator),
    validation_data=val_generator,
    validation_steps=len(val_datagen.generator),
    epochs=10,  
#     verbose=0,
    callbacks=[
        TensorBoard(
            log_dir=FOLDER,
            write_graph=False
        ),
        ModelCheckpoint(
            os.path.join(FOLDER, 'weights.{epoch:02d}-{loss:.2f}.h5'),
            monitor='val_loss', 
            save_best_only=True, 
            save_weights_only=True
        ),
        LambdaCallback(
            on_epoch_end=lambda epoch, logs: plot_callback(model, TEST_SAMPLE_FOLDER)
        )
    ]
)


# In[ ]:


print("FINISHED!!!!!")


# In[ ]:




