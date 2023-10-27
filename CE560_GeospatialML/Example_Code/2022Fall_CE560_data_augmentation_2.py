# -*- coding: utf-8 -*-
"""
Last updated Oct/26/2022

@author: Jaehoon Jung, PhD, OSU

Keras data augmentation tutorial part 2
"""
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # uncomment this and restart kernel to not use GPU  

import time
import tensorflow as tf
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img 

data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data() # split into training and testing
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
for i in range(9):
	plt.subplot(330 + 1 + i)
	plt.imshow(train_images[i])
train_images = train_images/255.0 #-- normalize between 0 and 1
test_images = test_images/255.0
train_images = tf.expand_dims(train_images, axis=-1) #-- reshpae training data into the format for model.fit (# data + dimension)
test_images = tf.expand_dims(test_images, axis=-1)

# %% model 
model = keras.Sequential([
    keras.layers.Conv2D(28, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(28, 28, 1)),
    keras.layers.BatchNormalization(), #-- help address vanishing gradient problem 
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.2), #-- help address overfitting problem 
    keras.layers.Conv2D(56, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'), #-- he_uniform is ideal for Relu
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
model.compile(optimizer=opt1, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# %% training
#-- data augmentation
train_datagen = ImageDataGenerator(rotation_range=5,  #-- Too much rotation may hurt accuracy
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range = 0.1,
    vertical_flip=False,
    horizontal_flip = True,
    fill_mode="reflect")
train_generator = train_datagen.flow(
    train_images,
    train_labels,
    batch_size = 32)  #-- images to generate in a batch
start_time = time.time()
history = model.fit(train_generator, verbose=1, epochs=50, batch_size=1) #-- batch size between 32 - 64
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