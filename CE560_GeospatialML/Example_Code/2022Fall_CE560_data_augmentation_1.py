# -*- coding: utf-8 -*-
"""
Last updated Feb/13/2022

@ author: Sreenivas Bhattiprolu, ZEISS
@ modified by: Jaehoon Jung, OSU

Keras data augmentation tutorial part 1 
"""


from keras.preprocessing.image import ImageDataGenerator 
from skimage import io

datagen = ImageDataGenerator( 
        rotation_range = 45,      #-- Random rotation between 0 and 45
        width_shift_range=[-20,20],  #-- min and max shift in pixels
        height_shift_range=0.2,  #-- Can also define as % shift (min/max or %)
        shear_range = 0.2, # Shear angle in counter-clockwise direction in degrees
        zoom_range = 0.2, # Range for random zoom
        horizontal_flip = True, 
        brightness_range = (0.5, 1.5), 
        fill_mode='reflect') #-- use 'constant' to make black background (zero pixels)

x = io.imread('dog.jpg')  
x = x.reshape((1, ) + x.shape)  # add a new dimension (number of images)
i = 0
for batch in datagen.flow(x, batch_size = 2, #-- transform images N times 
                          save_to_dir ='augmented/',  #-- directory to save the augmented images
                          save_prefix ='aug', save_format ='jpeg'): 
    i += 1
    if i > 6: 
        break
