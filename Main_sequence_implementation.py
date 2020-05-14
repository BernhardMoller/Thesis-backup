# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:55:39 2020

@author: Fredrik Möller
"""

import keras as ks 
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
# from keras.datasets import imdb
from sklearn.model_selection import train_test_split
# from keras.regularizers import l2

import innvestigate as inn
import fun_c as f 
import numpy as np


#%%
# set parameters:
max_features = 5000
maxlen = 400
batch_size = 64
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 5

# print('Loading data...')
# # (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
# print(len(x_train), 'train sequences')
# print(len(x_test), 'test sequences')

# total number of data points available is 63536
data = f.Get_data(63536)

# split into training and test data 
X = data.iloc[:,1]
y = data.iloc[:,0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train_split = f.Split_sentences(X_train)
X_test_split = f.Split_sentences(X_test)

t = ks.preprocessing.text.Tokenizer(num_words = max_features)
t.fit_on_texts(X_train_split)

X_train_seq = t.texts_to_sequences(X_train_split)
X_test_seq= t.texts_to_sequences(X_test_split)

# pad sequences to the max length 
X_train_pad = sequence.pad_sequences(X_train_seq, maxlen=maxlen, padding = 'post')
X_test_pad = sequence.pad_sequences(X_test_seq, maxlen=maxlen, padding = 'post')

# X_train_pad = np.expand_dims(X_train_pad,1)
# X_test_pad = np.expand_dims(X_test_pad,1)

# y_train =  np.expand_dims(y_train,1)
# y_test =  np.expand_dims(y_test,1)

# input_shape = np.shape(X_train_pad[0,:,:])
#%% build the model 

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
early_stopping = ks.callbacks.EarlyStopping(monitor = 'val_loss', patience = 3, verbose = 1, restore_best_weights= True)



model.fit(X_train_pad, y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks = [early_stopping],
          validation_data=(X_test_pad, y_test))

#%% need to split the trained model into two. Embedder and everything else. Due to the embedding layer not beeing supported in the innvestigate package

embedding_layer = model.layers[0]

emb_mod = Sequential()
emb_mod.add(embedding_layer)

emb_mod.compile('Adam', loss = 'mean_squared_error')

model_rest = model.layers[1:]

new_mod = Sequential()
for lay in model_rest:
    new_mod.add(lay)

new_mod.compile('Adam', loss = 'mean_squared_error')



#%% test so that the outputs of the split models are consistent and compatible with eachother

for target , i in y_test:
    print(i)


#%%
# test to run one input sentence first through the separate embedding layer and then the rest of the model. 
indexes = [21 , 100 , 1400]
for index in indexes:
    sent = X_test_pad[index]
    one_sent = np.expand_dims(sent,0)
    
    test_target = y_test.iloc[index]
    
    emb_mat = emb_mod.predict(one_sent) # OK 
    # use output of embedding layer to predict in the new model 
    y_hat = new_mod.predict_classes(emb_mat) # OK
    # everything runs as expected , also with using multiple sentences as input (the whole test_input)
    result = y_hat - test_target
    accuracy = 1 -(np.sum(np.abs(result))/len(result))
    
    #% create and test analyser 
    # analyzer = inn.create_analyzer('gradient',new_mod)
    analyzer = inn.create_analyzer('lrp.epsilon',new_mod, epsilon = 0.00001)
    # analyzer = inn.create_analyzer('lrp.alpha_2_beta_1',new_mod)
    
    
    lrp_result = analyzer.analyze(emb_mat)
    lrp_result_seq = np.sum(lrp_result, axis = 2)
    
    text = t.sequences_to_texts(one_sent)
    text_split = f.Split_sentences(text)
    text_plot = text_split[0]
    
    # isolate results from words, disregard from empty spaces
    score_words = lrp_result_seq[0][0:np.shape(text_split)[1]]
    
    # plot heatmap and prediction results
    f.plot_text_heatmap(text_plot, score_words)
    print('target =', test_target , 'predicted =' ,y_hat[0][0])
