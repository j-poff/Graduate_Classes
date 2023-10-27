# -*- coding: utf-8 -*-
"""
Last updated Oct/26/2022

@author: Jaehoon Jung, PhD, OSU
"""
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # uncomment this and restart kernel to not use GPU  

import time
import tensorflow as tf
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
for i in range(9):
	plt.subplot(330 + 1 + i)
	plt.imshow(train_images[i])
train_images = train_images/255.0 ## scale data between 0 and 1
test_images = test_images/255.0
train_images = tf.expand_dims(train_images, axis=-1)
test_images = tf.expand_dims(test_images, axis=-1)

# %% model 
model = keras.Sequential([ #-- he_uniform: Draws samples from a uniform distribution within [-limit, limit]
    keras.layers.Conv2D(28, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(28, 28, 1)), 
    keras.layers.BatchNormalization(), #-- help address vanishing gradient problem 
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.2), ##--help address overfitting problem 
    keras.layers.Conv2D(56, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'), ## he_uniform is ideal for Relu
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(28,activation="relu"), # sigmoid
    keras.layers.Dense(28,activation="relu"), # sigmoid
    keras.layers.Dense(10,activation="softmax")
    ])

# %% compile 
opt1 = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9) #-- momentum helps converge to global miminum
opt2 = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt1, loss='sparse_categorical_crossentropy', metrics=['accuracy']) #-- the slope of Cross Entropy for a bad prediction is larger than sqaured sum residuals
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# %% training 
start_time = time.time()
history = model.fit(train_images, train_labels, verbose=1, epochs=10, batch_size=64) #-- batch size between 32 - 64
print("\n--- %.3f seconds ---\n" % (time.time() - start_time))

# %% plot the training accuracy and loss at each epoch
loss = history.history['loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.clf()
plt.plot(epochs, loss, 'y', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
acc = history.history['accuracy']
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# %% evaluation  
test_loss, test_acc = model.evaluate(test_images,test_labels)
print("tested acc:", test_acc)

# %% prediction 
prediction = model.predict(test_images)
plt.figure()
for i in range(5):
    plt.figure()
    plt.grid(False)
    plt.imshow(test_images[i])
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()

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