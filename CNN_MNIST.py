# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 22:17:05 2020

@author: gilli
"""

import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt

'Importing the MNIST number dataset'

from tensorflow.keras.datasets import mnist

'seperating the dataset into the training and test sets'
(x_train, y_train), (x_test, y_test) = mnist.load_data()

'Finding out the shape of the data'
print(x_train.shape)

single_image = x_train[0]

#print(single_image)

print(single_image.shape)

plt.imshow(single_image)


'Importing the to_categorical utility in order to prevent the model from seeing this as a regression problem'
from tensorflow.keras.utils import to_categorical


'Finding out how many different categories we are dealing with, In this case -> 10'
y_example = to_categorical(y_train)
print(y_example.shape)

y_cat_test = to_categorical(y_test,10)
y_cat_train = to_categorical(y_train,10)

'''Normalizing the data. First we find what the max value is of each datapoint(image pixel) 
and divide every point by this max value'''

print(single_image.max())

x_train = x_train/255
x_test = x_test/255

'Reshaping the data to be read as number of instances, size and number of channels'
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000,28,28,1)



'MODEL CREATION'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten


model = Sequential()

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(28, 28, 1), activation='relu',))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

# FLATTEN IMAGES FROM 28 by 28 to 764 BEFORE FINAL LAYER
model.add(Flatten())

# 128 NEURONS IN DENSE HIDDEN LAYER (YOU CAN CHANGE THIS NUMBER OF NEURONS)
model.add(Dense(128, activation='relu'))

# LAST LAYER IS THE CLASSIFIER, THUS 10 POSSIBLE CLASSES
model.add(Dense(10, activation='softmax'))

# https://keras.io/metrics/
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']) # we can add in additional metrics https://keras.io/metrics/

'Incorporating a simple early stopping function into the training of our model'
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss',patience=2)
model.fit(x_train,y_cat_train,epochs=10,validation_data=(x_test,y_cat_test),callbacks=[early_stop])

print(model.metrics_names)
losses = pd.DataFrame(model.history.history)

print(losses.head())

losses[['accuracy','val_accuracy']].plot()
losses[['loss','val_loss']].plot()

print(model.metrics_names)
print(model.evaluate(x_test,y_cat_test,verbose=0))

from sklearn.metrics import classification_report,confusion_matrix
predictions = model.predict_classes(x_test)

print(classification_report(y_test,predictions))
confusion_matrix(y_test,predictions)