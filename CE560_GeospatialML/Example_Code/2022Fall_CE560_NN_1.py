# -*- coding: utf-8 -*-
"""
Last updated Oct/26/2022

@author: Jaehoon Jung, PhD, OSU
"""
#-- Ctrl + 1 to comment out multiple lines
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # uncomment this and restart kernel to not use GPU  

from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()
#-- view images 
for i in range(9): #-- return 0 - 8 
	plt.subplot(330 + 1 + i) 
	plt.imshow(train_images[i])
plt.show()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images/255.0 #-- scale data between 0 and 1
test_images = test_images/255.0

# %% model
#-- to define a fully connected network, you can simply stack your dense layers one after another using a sequential model  
#-- to create a deep NN, all you need to do is stack your dense layers over and over and create more hierarchical model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28), input_dim = 1),
    keras.layers.Dense(64,activation="sigmoid"), # relu
    keras.layers.Dense(64,activation="sigmoid"), # relu
    keras.layers.Dense(10,activation="softmax")
    ])

# %% compile and train the model 
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]) # if your labels are integers, use 'sparse_categorical_crossentropy'
model.fit(train_images,train_labels, epochs=10)
# model.save('NN_tutorial1.hdf5'); 

# %% evaluation
test_loss, test_acc = model.evaluate(test_images,test_labels)
print("tested acc:", test_acc)

# %% prediction 
prediction = model.predict(test_images)
for i in range(5):
    plt.figure()
    plt.grid(False)
    plt.imshow(test_images[i])
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()

#-- By default, Spyder plots will be shown in the IPython console, 
#-- but this can be annoying when saving and interacting with the plots we make. 
#-- You can change how plots are displayed in Spyder to have them show up in a separate window 
#-- 1. Tools -> Preferences -> IPython console -> Graphics -> Graphics backend -> Automatic 
#-- 2. Restart Spyder

# %% use pre-trained model
if 0:
    import keras
    from keras.models import load_model
    model = load_model("NN_tutorial1.hdf5", compile=False)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    data = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = data.load_data()
    test_loss, test_acc = model.evaluate(test_images,test_labels)
    print("tested acc:", test_acc)