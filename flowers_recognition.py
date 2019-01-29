#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 22:35:59 2018

@author: hakunamatata
"""

import os
print(os.listdir('flowers'))

# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
 
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
%matplotlib inline  
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

#preprocess.
from keras.preprocessing.image import ImageDataGenerator

#dl libraraies
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical

# specifically for cnn
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
 
import tensorflow as tf
import random as rn

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2                  
import numpy as np  
from tqdm import tqdm
import os                   
from random import shuffle  
from zipfile import ZipFile
from PIL import Image

X = []
Z = []
img_size = 155
sunflower_dir = 'flowers/sunflower'
daisy_dir = 'flowers/daisy'
tulip_dir = 'flowers/tulip'
rose_dir = 'flowers/rose'
dandi_dir = 'flowers/dandelion'

def assign_label(img,flower_type):
    return flower_type

def make_train_data(flower_type, DIR):
    for img in tqdm(os.listdir(DIR)):
        label = assign_label(img, flower_type)
        path = os.path.join(DIR, img)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img_size, img_size))
        
        X.append(np.array(img))
        Z.append(str(label))
        
make_train_data('Daisy', daisy_dir)
print(len(X))

make_train_data('Rose', rose_dir)
make_train_data('Tulip', tulip_dir)
make_train_data('Sunflower', sunflower_dir)
make_train_data('Dandelion', dandi_dir)
print(len(X))

fig, ax = plt.subplots(5,2)
fig.set_size_inches(15, 15)
for i in range(5):
    for j in range(2):
        l = rn.randint(0, len(Z))
        ax[i, j].imshow(X[l])
        ax[i, j].set_title('Flower: ' + Z[l])
        
plt.tight_layout()

le = LabelEncoder()

Y = le.fit_transform(Z)
Y = to_categorical(Y, 5)
X = np.array(X)
X = X/255
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25)

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5, 5),
                 padding = 'Same', activation = 'relu',
                 input_shape = (155, 155, 3)))
model.add(MaxPooling2D())
model.add(Dropout(rate = 0.1))
model.add(Conv2D(filters = 64, kernel_size = (3, 3),
                 padding = 'Same', activation = 'relu',))
model.add(Dropout(rate = 0.1))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2, 2)))

model.add(Conv2D(filters = 96, kernel_size = (3, 3),
                 padding = 'Same', activation = 'relu'))
model.add(Dropout(rate = 0.1))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2, 2)))

model.add(Conv2D(filters = 96, kernel_size = (3, 3),
                 padding = 'Same', activation = 'relu'))
model.add(Dropout(rate = 0.1))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2, 2)))
model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(rate = 0.1))
model.add(Activation('relu'))
model.add(Dense(5, activation = 'softmax'))

batch_size = 128
epochs = 50

datagen = ImageDataGenerator(rotation_range = 0,
                             zoom_range = 0.1,
                             width_shift_range = 0.2,
                             height_shift_range = 0.2,
                             horizontal_flip = True,
                             vertical_flip = False)

datagen.fit(x_train)

model.compile(optimizer = Adam(), loss = 'categorical_crossentropy',
              metrics = ['accuracy'])
model.summary()

History = model.fit_generator(datagen.flow(x_train, y_train, batch_size = batch_size,),
                              epochs = epochs, validation_data = (x_test, y_test),
                              verbose = 1,
                              steps_per_epoch = x_train.shape[0]//batch_size)

model.save_weights('checkpoint')
model.save('flower_recognition.h5')
