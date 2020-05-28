# -*- coding: utf-8 -*-
"""
Created on Fri May 15 14:12:02 2020

@author: Fredrik Möller
"""

#%% build model You need to fetch data to be able to run this script 

import keras as ks 
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D


import datetime
# import pandas as pd 


max_features = 5000
maxlen = len(X_train_pad[0])
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250



model = Sequential()
# embedding layer that maps the vocad indexes into embedding_dim dimensions, 
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
model.add(Dropout(0.2))

# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1
                 ))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add vanilla fully connected layer:
model.add(Dense(hidden_dims, activation = 'relu'))
model.add(Dropout(0.2))

# classification layer, single neuron with sigmoid activation function, nr_classes = 2 
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss='mean_squared_error',
              optimizer='Adam',
              metrics=['accuracy'])


#%%
#% Train model, the data need to be defined from main before it is run 

early_stopping = ks.callbacks.EarlyStopping(monitor = 'val_loss', patience = 2, verbose = 1, restore_best_weights= True)

epochs = 50
batch_size = 256


model.fit(X_train_pad, y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks = [early_stopping],
          validation_data=(X_test_pad, y_test))


#%
#% save model in the folder models 
datetime_object = datetime.datetime.now()
date = str(datetime_object.date())
time = str(datetime_object.time())
time = time.split(':')
time = time[0:2]
path = 'C:/Users/Fredrik Möller/Documents/MPSYS/Master_thesis/Code/GIT/Thesis-backup/models'
model.save(path + '/' + date + '-' + time[0] + '-' + time[1])
print(date+'-'+time[0]+'-'+time[1])
print('Training acc', model.history.history['acc'][-3])
