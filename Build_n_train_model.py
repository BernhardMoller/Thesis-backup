# -*- coding: utf-8 -*-
"""
Created on Fri May 15 14:12:02 2020

@author: Fredrik Möller
"""

#%% build model You need to fetch data to be able to run this script 

import keras as ks 
from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D , Embedding , Dense, Dropout 
import numpy as np 

import datetime
# import pandas as pd 
eval = True
save_mod = False
res_all = np.empty([5,12])
for tst in range(1):
    print("starting build of model " + str(tst+1))
    
    max_features = max_features
    maxlen = len(X_train_pad[0])
    embedding_dims = 50
    filters = 250
    kernel_size = 3
    
    # kernel_size = 50
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
    # model.add(Flatten())
    model.add(Dense(1, activation = 'sigmoid'))

    
    model.compile(loss='mean_squared_error',
                  optimizer='Adam',
                  metrics=['accuracy'])
    
    
    #%
    #% Train model, the data need to be defined from main before it is run 
    
    early_stopping = ks.callbacks.EarlyStopping(monitor = 'val_loss', patience = 2, verbose = 1, restore_best_weights= True)
    
    epochs = 50
    # batch_size = 256
    batch_size = 32

    print("starting training of model " + str(tst+1))

    
    model.fit(X_train_pad, y_train,
              batch_size=batch_size,
              epochs=epochs,
              callbacks = [early_stopping],
              validation_data=(X_test_pad, y_test))


    if save_mod:
       datetime_object = datetime.datetime.now()
       date = str(datetime_object.date())
       time = str(datetime_object.time())
       time = time.split(':')
       time = time[0:2]
       path = 'C:/Users/Fredrik Möller/Documents/MPSYS/Master_thesis/Code/GIT/Thesis-backup/models/ver'
       model.save(path + '/' + date + '-' + time[0] + '-' + time[1])
       print(date+'-'+time[0]+'-'+time[1])
       print('Training acc', model.history.history['acc'][-3])
    

    if eval:
        #% for verification testing can remove after 
        print("starting evaulation of model " + str(tst+1))
    
        indexes_tp = []
        indexes_tn = []
        indexes_fp = []
        indexes_fn = []
        res = []
        
        
        score_predict = model.predict_classes(X_test_pad)
        confusion = score_predict - np.expand_dims(y_test,1)
        for i in range(len(y_test)):
            if confusion[i] == 1: # false positive 
                indexes_fp.append(i)
            elif confusion[i] == -1: # false negative 
                indexes_fn.append(i)
            else: # sort rest if their target is true or false 
                if y_test.iloc[i]== 1:
                    indexes_tp.append(i)
                else:
                    indexes_tn.append(i)
        
        res.append(model.history.history['acc'][-3])
        res.append(model.history.history['val_acc'][-3])
        res.append(len(y_train))
        res.append(y_train.sum()/len(y_train))
        res.append(len(y_test))
        res.append(y_test.sum()/len(y_test))
        res.append(len(indexes_tp))
        res.append(len(indexes_tn))
        res.append(len(indexes_fp))
        res.append(len(indexes_fn))
        res.append(len(indexes_tp) / (len(indexes_tp) + len(indexes_fn)))
        res.append(len(indexes_tp) / (len(indexes_tp) + len(indexes_fp)))
        
        
        
        res = np.array(res)
        res = res.reshape(1,len(res))

        res_all[tst,:] = res[0]