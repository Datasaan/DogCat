# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 08:12:09 2016

@author: pc
"""

import os 
import gc
import pandas  as pd
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils
# Image details
ROWS=64
COLUMNS=64
CHANNEL=3


# Get image paths
train_dir='/home/sanjeet/Downloads/DogCat/train/'
test_dir= '/home/sanjeet/Downloads/DogCat/test/'

train=pd.DataFrame()
test=pd.DataFrame()
train['paths']=[train_dir+i for i in os.listdir(train_dir)]  # 25k images in train
test['paths']=[test_dir+i for i in os.listdir(test_dir)] 
# tagging the True for dogs , false for cats
train['dogs']= train['paths'].str.contains('dog')

# Lets train on 1000 dogs and 1000 cats 
def slice_dataset(train,n):
    a=((train[train['dogs']]).iloc[0:n])
    b=(train[train['dogs']==False]).iloc[0:n]
    return (pd.concat([a,b]).reset_index().drop('index',axis=1)).iloc[np.random.permutation(2*n)]

# read an image
def read_image(path):
    img=cv2.imread(path,cv2.IMREAD_COLOR)
    return cv2.resize(img,(ROWS,COLUMNS),interpolation=cv2.INTER_CUBIC)

def prep_data(images):
    count=len(images)
    data= np.ndarray((count,CHANNEL,ROWS,COLUMNS),dtype=np.uint8)
    for i,image_file in enumerate(images):
        image=read_image(image_file)
        data[i]=image.T
        print i
    return data
#sliced=slice_dataset(train,3000) # sliced['dogs'] for labels
#labels=sliced['dogs']*1
train_data=prep_data(train['paths'])
labels=train['dogs']*1



optimizer = RMSprop(lr=1e-4)
objective = 'binary_crossentropy'


def catdog():

    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, ROWS, COLUMNS), activation='relu'))
    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    #model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    #model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    #model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])

    return model


model = catdog()

nb_epoch = 10
batch_size = 16

## Callback for loss logging per epoch
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

def run_catdog():
    history = LossHistory()
    model.fit(train_data, labels, batch_size=batch_size, nb_epoch=nb_epoch,validation_split=0.25, verbose=2, shuffle=True, callbacks=[history, early_stopping])
    return

run_catdog()
del train_data
gc.collect()
test_data=prep_data(test['paths'])
p=model.predict(test_data)
sub=pd.DataFrame()
sub['label']=p[:,0]
sub['id']=test['paths']
sub['id']=sub['id'].str.replace('/home/sanjeet/Downloads/DogCat/test/','')
sub['id']=sub['id'].str.replace('.jpg','')
sub.to_csv('sub1.csv',index=False)