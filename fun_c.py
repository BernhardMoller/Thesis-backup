# -*- coding: utf-8 -*-
"""
Created on Wed May  6 13:57:55 2020

@author: Fredrik Möller
"""
import pandas as pd 
import keras as ks 
from sklearn.feature_extraction import stop_words
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from matplotlib import cm, transforms


#Function that returns the raw data containing the word mosque and its labels 
def Get_mosque_data():
    data_all = pd.read_csv ('C:/Users/Fredrik Möller/Documents/MPSYS/Master_thesis/Code/Data/data_terrorincident_eng_26-03-2020_60k - data_terrorincident_eng_26-03-2020xxx.csv.csv', delimiter=',')
    mosque_data = data_all[['score','mosque_highlighted']]
    mosque_data = mosque_data.dropna(how='any')
    return mosque_data


def Get_data(nr_data):
    data_all = pd.read_csv ('C:/Users/Fredrik Möller/Documents/MPSYS/Master_thesis/Code/Data/data_terrorincident_eng_26-03-2020_60k - data_terrorincident_eng_26-03-2020xxx.csv.csv', delimiter=',')
    data = data_all[['score','fragment']]
    rand_prem = np.random.permutation(len(data))
    data = data.iloc[rand_prem[:nr_data]]
    return data
    
# unused ##################################
def Train_tokenizer(data, nb_words,use_stop_words):
    # no maximum vocal size is used, English stop words from sklearn can be used 
    data_with_words_separated = []
    for sent in data: 
        # split the sentences into words 
        words = sent.split()
        # save split sencencs in a list 
        data_with_words_separated.append(words)
        
    # setting up ad training the  on the training data. 
    if use_stop_words == True:
        t = ks.preprocessing.text.Tokenizer(num_words = nb_words ,filters=stop_words.ENGLISH_STOP_WORDS)
    else:
        t = ks.preprocessing.text.Tokenizer(num_words = nb_words)
    
    t.fit_on_texts(data_with_words_separated)
    return t
###########################################

def Split_sentences(data):
    data_with_words_separated = []
    for sent in data: 
        # split the sentences into words 
        words = sent.split()
        # save split sencencs in a list 
        data_with_words_separated.append(words)
        
    return data_with_words_separated
        

def Matrix_from_split_data(data, max_length, lib_size, t):
    # create empty array to store matrixes in, will be removed
    X_train_bin_mat = np.zeros((max_length , lib_size))
    X_train_bin_mat = np.expand_dims(X_train_bin_mat, axis = (0,3))
    
    # for each sentence in training data 
    for sent in data:
        # create a binare matrix rep
        tmp_mat = t.texts_to_matrix(sent)
        # transpose it due to direction of padding cannot be changed
        tmp_mat_tp = np.transpose(tmp_mat)
        # pad the sentence to max length of traing data 
        tmp_mat_tp_pad = pad_sequences(tmp_mat_tp , maxlen=max_length, padding='post')
        # transpose back to original
        tmp_mat_pad = np.transpose(tmp_mat_tp_pad)
        tmp_mat_pad = np.expand_dims(tmp_mat_pad, axis = (0,3))
        # stack the matrixes in 3d to get the shape of standard image analysis format 
        X_train_bin_mat = np.concatenate((X_train_bin_mat, tmp_mat_pad), axis = 0)

            
    return X_train_bin_mat[1:,:,:,:] # return all elements except the first empty one 

def Shorten_sentences(data_X , data_y , max_length): 
    # removing any sentences longer than max sentences in traing 
    tmp_X = []
    tmp_y = []
    indexes = []
    for i in range(len(data_X)):
        if max_length >= len(data_X[i]):
            tmp_X.append(data_X[i])
            tmp_y.append(data_y[i])
        else:
            indexes.append(i)
            
    X_short = tmp_X
    y_short = pd.Series(tmp_y) # convert to series due to keras complaining otherwise 
    return X_short , y_short , indexes


# shamelessly borrowed function for plotting a heatmap with the relevancy scores, from iNNvestigate github repo
def plot_text_heatmap(words, scores, title="", width=10, height=0.2, verbose=0, max_word_per_line=20, savefig = 0):
    fig = plt.figure(figsize=(width, height))
    
    ax = plt.gca()

    ax.set_title(title, loc='left')
    tokens = words
    if verbose > 0:
        print('len words : %d | len scores : %d' % (len(words), len(scores)))

    cmap = plt.cm.ScalarMappable(cmap=cm.bwr)
    cmap.set_clim(0, 1)
    
    canvas = ax.figure.canvas
    t = ax.transData

    # normalize scores to the followings:
    # - negative scores in [0, 0.5]
    # - positive scores in (0.5, 1]
    normalized_scores = 0.5 * scores / np.max(np.abs(scores)) + 0.5
    
    if verbose > 1:
        print('Raw score')
        print(scores)
        print('Normalized score')
        print(normalized_scores)

    # make sure the heatmap doesn't overlap with the title
    loc_y = -0.2

    for i, token in enumerate(tokens):
        *rgb, _ = cmap.to_rgba(normalized_scores[i], bytes=True)
        color = '#%02x%02x%02x' % tuple(rgb)
        
        text = ax.text(0.0, loc_y, token,
                       bbox={
                           'facecolor': color,
                           'pad': 5.0,
                           'linewidth': 1,
                           'boxstyle': 'round,pad=0.5'
                       }, transform=t)

        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()
        
        # create a new line if the line exceeds the length
        if (i+1) % max_word_per_line == 0:
            loc_y = loc_y -  2.5
            t = ax.transData
        else:
            t = transforms.offset_copy(text._transform, x=ex.width+15, units='dots')

    if verbose == 0:
        ax.axis('off')
        
    return fig 




