# -*- coding: utf-8 -*-
"""
Created on Thu May 28 19:25:49 2020

@author: Fredrik Möller
"""

# from keras.layers import Dense, Dropout
# from keras.layers import Embedding
# from keras.layers import Conv1D, GlobalMaxPooling1D
# from keras.datasets import imdb
from sklearn.model_selection import train_test_split
# from keras.regularizers import l2

import fun_c as f 
import numpy as np

# import seaborn as sns


reduce_padding = True

data = f.Get_data(63536)
# data = f.Get_mosque_data()

X = data.iloc[:,1]
y = data.iloc[:,0]



if reduce_padding:
    ##preprocess to try and reduce the padding of the sentences by removing the longest sentences 
    N_sigma_len = 3
    
    X_tmp = f.Split_sentences(X)
    # create an array which contains the lengths of all sentences in X
    X_len = np.array([len(sent) for sent in X_tmp])
    # create a T-F array of which sentences that are shorter than +N sigmas 
    X_ok_len = X_len <= X_len.mean() + N_sigma_len * X_len.std()
    # get the indexes of the sentences that are sorter than N sigmas 
    index_X_ok = [i for i, x in enumerate(X_ok_len) if x]
    # write over the old data 
    X = data.iloc[index_X_ok,1]
    y = data.iloc[index_X_ok,0]
    max_len = np.ceil(X_len.mean() + N_sigma_len * X_len.std())

##
# split into training and test data 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

if reduce_padding:
    data_train_reduce_pad = X_train.to_frame()
    data_test_reduce_pad = X_test.to_frame()
    
    
    data_train_reduce_pad['target'] = y_train.values
    data_test_reduce_pad['target'] = y_test.values
    
    data_train_reduce_pad.to_csv('data_train_reduce_pad')
    data_test_reduce_pad.to_csv('data_test_reduce_pad')
else:
    data_train = X_train.to_frame()
    data_test = X_test.to_frame()
    
    
    data_train['target'] = y_train.values
    data_test['target'] = y_test.values
    
    data_train.to_csv('data_train')
    data_test.to_csv('data_test')