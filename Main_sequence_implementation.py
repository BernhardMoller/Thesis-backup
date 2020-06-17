# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:55:39 2020

@author: Fredrik Möller
"""
import string
import keras as ks 
from keras.models import load_model
from keras.preprocessing import sequence
from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.layers import Embedding
# from keras.layers import Conv1D, GlobalMaxPooling1D
# from keras.datasets import imdb
from sklearn.model_selection import train_test_split
# from keras.regularizers import l2

import innvestigate as inn
import fun_c as f 
import numpy as np
import pandas as pd 

# import seaborn as sns
import matplotlib.pyplot as plt

#%%
# set parameters:
max_features = 5000
maxlen = 75 # cap to 75 

use_word_filter_remove = False
use_word_filter_drop_word = False
use_word_filter_drop_sent = True

if 1 < use_word_filter_remove + use_word_filter_drop_word + use_word_filter_drop_sent:    
   raise Exception('More than one filter is set to be applied to the data, check settings and try again')

dr = 0.1

stop_words_from = 'freq' # 'freq' , 'tfidf' , 'lrp' ,'stop_words


if use_word_filter_remove or use_word_filter_drop_word or use_word_filter_drop_sent: 
    filter_words = []
    name = 'filter_words_' + stop_words_from +'.txt'
    with open(name, "r") as file:
        for line in file:
            filter_words.append(line.strip())

reduce_padding = False


replace_not_remove = False
norm_token = 'Nom-Token'
# total number of data points available is 63536
# data = f.Get_mosque_data()

# get stopwords 

stop_words = f.get_stop_words()


data_train , data_test = f.train_test_data(reduce_pad = reduce_padding)

X_train = data_train['fragment']
X_test = data_test['fragment']

y_train = data_train['target']
y_test = data_test['target']

X_train_indexes = X_train.index
X_test_indexes = X_test.index

X_train = f.covert_to_lowercase(X_train)
X_test = f.covert_to_lowercase(X_test)

X_train_split = f.Split_sentences(X_train)
X_test_split = f.Split_sentences(X_test)

# remove punctuations and commas, also convert to lowercase 
X_train_split = f.remove_punct(X_train_split)
X_test_split = f.remove_punct(X_test_split)


if use_word_filter_remove:
    X_train_split , y_train = f.apply_word_filter_remove(X_train_split, y_train , filter_words)
    X_test_split , y_test = f.apply_word_filter_remove(X_test_split , y_test , filter_words)


if use_word_filter_drop_word:
    X_train_split , y_train = f.apply_word_filter_drop_word(X_train_split, y_train , filter_words , dr)
    X_test_split , y_test = f.apply_word_filter_drop_word(X_test_split , y_test , filter_words, dr)
    

if use_word_filter_drop_sent: 
    X_train_split , y_train , removed_index = f.apply_word_filter_drop_sent(X_train_split , y_train , filter_words , dr)
    X_test_split , y_test , removed_index = f.apply_word_filter_drop_sent(X_test_split , y_test , filter_words, dr)
    

t = ks.preprocessing.text.Tokenizer(num_words = max_features)

t.fit_on_texts(X_train_split)

X_train_seq = t.texts_to_sequences(X_train_split)
X_test_seq= t.texts_to_sequences(X_test_split)

# pad sequences to the max length 
X_train_pad = sequence.pad_sequences(X_train_seq, maxlen=maxlen, padding = 'post')
X_test_pad = sequence.pad_sequences(X_test_seq, maxlen=maxlen, padding = 'post')


#%%
# read stop words 
read_tmp = []
with open("stop_words.txt", "r") as file:
  for line in file:
    read_tmp.append(line.strip())
stop_words = np.array(read_tmp)

# words_wrong_classification = list(set(skewed_words_fp + skewed_words_fn))
words_not_in_stop_words = [w for w in filter_words if w not in stop_words]

with open("not_in_stop_words.txt", "w") as file:
    for s in words_not_in_stop_words:
        file.write(str(s) +"\n")

#%% load a pretrained model from the models folder 

model_name = '2020-05-29-13-28' # baseline model 
path = 'C:/Users/Fredrik Möller/Documents/MPSYS/Master_thesis/Code/GIT/Thesis-backup/models'


# model_name = '2020-06-16-08-58'
# path = 'C:/Users/Fredrik Möller/Documents\MPSYS/Master_thesis/Code/GIT/Thesis-backup/models/ver'

model = load_model(path + '/' + model_name)

#%% start of LRP exploration 
#
#first we need need to split the trained model into two. Embedding layer and everything else. Due to the embedding layer not beeing supported in the iNNvestigate package

## Seperate model for embedding layer 
embedding_layer = model.layers[0]

emb_mod = Sequential()
emb_mod.add(embedding_layer)

emb_mod.compile(optimizer = 'Adam', loss = 'mean_squared_error')
####

## seperate model for everything else in the model
model_rest = model.layers[1:]

new_mod = Sequential()
for lay in model_rest:
    new_mod.add(lay)

new_mod.compile(optimizer = 'Adam', loss = 'mean_squared_error')
####

epsilon = 0.01
# epsilon = 1
### create LRP analyzer 
#% create and test analyser 
# analyzer = inn.create_analyzer('gradient',new_mod)
analyzer = inn.create_analyzer('lrp.epsilon',new_mod, epsilon = epsilon)
# analyzer = inn.create_analyzer('lrp.alpha_2_beta_1',new_mod)
####

#%% check which words have the highest lrp score in each class for X_test_pad
lrp_fp = np.zeros((1, max_features))
lrp_fn = np.zeros((1, max_features))
lrp_tp = np.zeros((1, max_features))
lrp_tn = np.zeros((1, max_features))


score_predict = model.predict_classes(X_test_pad)
confusion = score_predict - np.expand_dims(y_test,1)

# text_seq_pad = X_test_pad

# perform LRP on the selected sentences and save the relevance per toekn in a matrix for further evaluation 
lrp_tmp = np.zeros((1, max_features))
lrp_per_token = np.zeros((len(X_test_pad), max_features))



for idx , sent in enumerate(X_test_pad): 
    if np.mod(idx,1000) == 0: print(idx)   
    
    sent = np.expand_dims(sent , 0 )
    emb_mat = emb_mod.predict(sent) 
    y_hat = new_mod.predict_classes(emb_mat) 
    
    lrp_result = analyzer.analyze(emb_mat)
    lrp_result_seq = np.sum(lrp_result, axis = 2)
    
    lrp_tmp = np.zeros((1, max_features))
    for i in range(np.shape(lrp_result)[1]):
        index = sent[0,i]
        tmp_lrp_result = lrp_result_seq[0,i]
        lrp_tmp[0,index] = lrp_tmp[0,index] + tmp_lrp_result
        
    lrp_per_token[idx, : ] = lrp_tmp

    if confusion[idx] == 1: # false positive # add the lrp to the correct bin dependent on the classification score 
        lrp_fp = lrp_fp + lrp_tmp
    elif confusion[idx] == -1: # false negative 
        lrp_fn = lrp_fn + lrp_tmp
    else: # sort rest if their target is 1 or 0
        if y_test.iloc[idx]== 1:
            lrp_tp = lrp_tp + lrp_tmp
        else:
            lrp_tn = lrp_tn + lrp_tmp

# # get mean included the padding token 
lrp_fp_mean = lrp_fp.mean()
lrp_fn_mean = lrp_fn.mean()
lrp_tp_mean = lrp_tp.mean()
lrp_tn_mean = lrp_tn.mean()
#get deviation 
lrp_fp_std = lrp_fp.std()
lrp_fn_std = lrp_fn.std() 
lrp_tp_std = lrp_tp.std()
lrp_tn_std = lrp_tn.std()


#%% get the words that are stat outliers 
lrp_N_sigma = 3

# FP case 
# only loking at + sigma since pos lrp score support the wrong classification
tmp_bool = lrp_fp >= lrp_fp_mean + lrp_N_sigma * lrp_fp_std 
more_than_sigma_fp = [i for i, x in enumerate(tmp_bool[0]) if x ]
if 0 in more_than_sigma_fp: more_than_sigma_fp.remove(0) # remove 0 if in list since index 0 is the padding index and does not exist in the tokenizer 
skewed_words_fp = [ t.index_word[index] for index in more_than_sigma_fp ]

# FN case 
# only loking at + sigma since pos lrp score support the wrong classification
tmp_bool = lrp_fn >= lrp_fn_mean + lrp_N_sigma * lrp_fn_std 
more_than_sigma_fn = [i for i, x in enumerate(tmp_bool[0]) if x ]
if 0 in more_than_sigma_fn: more_than_sigma_fn.remove(0)
skewed_words_fn = [ t.index_word[index] for index in more_than_sigma_fn]

# TP case 
# only loking at - sigma since neg lrp score oppose the correct classification
tmp_bool = lrp_tp <= lrp_tp_mean - lrp_N_sigma * lrp_tp_std 
less_than_sigma_tp = [i for i, x in enumerate(tmp_bool[0]) if x ]
if 0 in less_than_sigma_tp: less_than_sigma_tp.remove(0)
skewed_words_tp = [ t.index_word[index] for index in less_than_sigma_tp]

# TN case 
# only loking at - sigma since neg lrp score oppose the correct classification
tmp_bool = lrp_tn <= lrp_tn_mean - lrp_N_sigma * lrp_tn_std 
less_than_sigma_tn = [i for i, x in enumerate(tmp_bool[0]) if x ]
if 0 in less_than_sigma_tn: less_than_sigma_tn.remove(0)
skewed_words_tn = [ t.index_word[index] for index in less_than_sigma_tn]

# take all skewed words supporting an erronous prediction 
skewed_words_lrp = list(set(skewed_words_fp + skewed_words_fn))
# remove stopwords from filter 
skewed_words_lrp = f.remove_stopwords(skewed_words_lrp)


#%%
# write filter words 
skewed_words_tmp = skewed_words_lrp
skewed_words_tmp.sort()


with open("filter_words_lrp.txt", "w") as file:
    for s in skewed_words_tmp:
        file.write(str(s) +"\n")


# words_wrong_classification = list(set(skewed_words_fp + skewed_words_fn))
words_not_in_stop_words = [w for w in skewed_words_lrp if w not in stop_words]

#%% get lrp score per class, sorted by target
lrp_N_sigma = 2

lrp_pred_pos = lrp_tp + lrp_fp
lrp_pred_neg = lrp_tn + lrp_fn

# with the padding token 
lrp_pos_mean = lrp_pred_pos.mean()
lrp_neg_mean = lrp_pred_neg.mean()

lrp_pos_std = lrp_pred_pos.std()
lrp_neg_std = lrp_pred_neg.std()

# pos class 
tmp_bool = lrp_pred_pos >= lrp_pos_mean + lrp_N_sigma * lrp_pos_std 
more_than_sigma_pos = [i for i, x in enumerate(tmp_bool[0]) if x ]
if 0 in more_than_sigma_pos: more_than_sigma_pos.remove(0)
skewed_words_pos = [ t.index_word[index] for index in more_than_sigma_pos]
#neg class 
tmp_bool = lrp_pred_neg >= lrp_neg_mean + lrp_N_sigma * lrp_neg_std 
more_than_sigma_neg = [i for i, x in enumerate(tmp_bool[0]) if x ]
if 0 in more_than_sigma_neg: more_than_sigma_neg.remove(0)
skewed_words_neg = [ t.index_word[index] for index in more_than_sigma_neg]


skewed_words_pos = f.remove_stopwords(skewed_words_pos)
skewed_words_neg = f.remove_stopwords(skewed_words_neg)


#%% try to get stemming on the tokenizer indexes so that LRP can be summerized for each baseword in stemmed words 

from nltk.stem import PorterStemmer
# from nltk.tokenize import sent_tokenize, word_tokenize

lrp_N_sigma = 1

ps = PorterStemmer()

# stem all the words in the tokenizer 
stemmed_t= []
for i in range(1,max_features+1):
    # tmp.append(t.index_word[i])
    stemmed_word = ps.stem(t.index_word[i])
    if stemmed_word not in stemmed_t:
        stemmed_t.append(stemmed_word)


lrp_data = lrp_fp
lrp_stemmed_token = np.zeros([1,len(stemmed_t)])
for i , score in enumerate(lrp_data[0]):
    if i == 0:
        continue
    
    tmp_stemmed = ps.stem(t.index_word[i])
    for index , word in enumerate(stemmed_t):
        if tmp_stemmed == word:
            lrp_stemmed_token[0,index] = lrp_stemmed_token[0,index] + score

lrp_stemmed_token_mean = lrp_stemmed_token.mean()
lrp_stemmed_token_std = lrp_stemmed_token.std()

tmp_bool = lrp_stemmed_token >= lrp_stemmed_token_mean + lrp_N_sigma * lrp_stemmed_token_std 
more_than_sigma_pos = [i for i, x in enumerate(tmp_bool[0]) if x ]
skewed_stemmed_words_support = [ stemmed_t[index] for index in more_than_sigma_pos]


tmp_bool = lrp_stemmed_token <= lrp_stemmed_token_mean - lrp_N_sigma * lrp_stemmed_token_std 
less_than_sigma_pos = [i for i, x in enumerate(tmp_bool[0]) if x ]
skewed_stemmed_words_against = [ stemmed_t[index] for index in less_than_sigma_pos]

#%% plotting time 

fig_width = 15
fig_hight = 4 
marker_size1 = 2 
marker_size2 = 2
colour = 'red'
leg = ['LRP < mean + 2 sigma', 'LRP >= mean + 2 sigma']
title1 = 'LRP score per index'
title2 = 'LRP score per index w/o padding index'
x_label = 'Tokenizer index'
ylabel = 'LRP score'

type_conf = [' (fp)' , ' (fn)' , ' (tp)' , ' (tn)', ' (predicted pos)' , (' (prediced neg)')]

data_plot = [lrp_fp , lrp_fn , lrp_tp , lrp_tn, lrp_pred_pos , lrp_pred_neg]
indexes_plot = [more_than_sigma_fp , more_than_sigma_fn , less_than_sigma_tp , less_than_sigma_tn, more_than_sigma_pos, more_than_sigma_neg]

for idx , dat in enumerate(data_plot):
    
    plt.figure()
    fig, ax = plt.subplots(1, 2, figsize=(fig_width,fig_hight))
    # plot with the padding index 
    ax[0].scatter(y=dat[0] , x = range(len(dat[0])) , s = marker_size1)
    ax[0].scatter(y=dat[0,indexes_plot[idx]] , x = indexes_plot[idx] , s = marker_size2 , c= colour)
    # plot without the padding index 
    ax[1].scatter(y=dat[0,1:] , x = range(len(dat[0])-1) , s = marker_size1)
    ax[1].scatter(y=dat[0,indexes_plot[idx]] , x = indexes_plot[idx] , s = marker_size2 , c= colour)
    
    ax[0].legend(leg , loc = 0)
    ax[1].legend(leg , loc = 4)
    
    
    ax[0].set_title(title1 + type_conf[idx])
    ax[1].set_title(title2 + type_conf[idx])
    
    ax[0].set_xlabel(x_label)
    ax[1].set_xlabel(x_label)
    
    ax[0].set_ylabel(ylabel)
    ax[1].set_ylabel(ylabel)



#%% Get data from local indexes, convert to gobal idenxes so that they allways can be found 
# local to global indexes 
# local_indexes = indexes_fp
# global_indexes = X_test.index[local_indexes]
# read_global = False 
# # global_indexes = [6612, ]

# # read from global indexes instead of X_test 
# if read_global:
# # from global indexes perfrom preproccesseing so that the sentenes can be evaluated 
#     score_eval = []
#     text = []
#     for index in global_indexes: 
#         sample = data.loc[index]
#         score_eval.append(sample[0])
#         tmp_text = sample[1].split()
#         text.append(tmp_text)
        
    
    
#     text_seq = t.texts_to_sequences(text)
#     data_for_lrp = sequence.pad_sequences(text_seq, maxlen= maxlen, padding = 'post' ) # regular padding 

# if not read_global: 
#     data_for_lrp = X_test_pad

# # perform LRP on the selected sentences and save the relevance per toekn in a matrix for further evaluation 
# lrp_tmp = np.zeros((1, max_features))
# lrp_per_token = np.zeros((1, max_features))

# for sent in data_for_lrp: 
#     # tmp = sent    
#     sent = np.expand_dims(sent , 0 )
#     emb_mat = emb_mod.predict(sent) 
#     y_hat = new_mod.predict_classes(emb_mat) 
    
#     lrp_result = analyzer.analyze(emb_mat)
#     lrp_result_seq = np.sum(lrp_result, axis = 2)
    
#     lrp_tmp = np.zeros((1, max_features))
#     for i in range(np.shape(lrp_result)[1]):
#         index = sent[0,i]
#         tmp_lrp_result = lrp_result_seq[0,i]
#         lrp_tmp[0,index] = lrp_tmp[0,index] + tmp_lrp_result
        
#     lrp_per_token = np.append(lrp_per_token,lrp_tmp,0)



# # sum all relevancy scores per token and get the corresponding words

# lrp_sum = np.sum(lrp_per_token,0)
# non_zero_index = np.nonzero(lrp_sum)

# lrp_sum_non_zero = lrp_sum[non_zero_index]

# lrp_tokens = t.sequences_to_texts(non_zero_index)
# lrp_tokens =  f.Split_sentences(lrp_tokens)
# lrp_tokens = lrp_tokens[0]
# lrp_tokens.insert(0,'padding_token')


# # get top N highest relevance contributers 
# N = 20
# top_N_index = lrp_sum_non_zero.argsort()[-N:]
# top_N_lrp = lrp_sum_non_zero[top_N_index]
# if 'top_N_words' not in locals():
#     top_N_words = np.array(lrp_tokens)[top_N_index]
# else:
#     np.append(top_N_words,top_N_words)
    
# ## train new tokenizer with the highest relevancies removed 

# # tt = ks.preprocessing.text.Tokenizer(num_words = max_features, filters = top_N_words)

# # X_test_seq_new = tt.texts_to_sequences(X_test_split)
# # X_test_pad_new = sequence.pad_sequences(X_test_seq_new, maxlen=maxlen, padding = 'post')

    
# #%% check to see if there is some relation between the total relevancy + and - and the distrubution of classes in the training data set 
# names = ['TP' , 'TN' , 'FP' , 'FN']
# tmp = [indexes_tp , indexes_tn , indexes_fp, indexes_fn]
# for i in range(len(tmp)):
#     local_indexes = tmp[i]
#     global_indexes = X_test.index[local_indexes]
    
#     # global_indexes = [6612, ]
    
#     # from global indexes perfrom preproccesseing so that the sentenes can be evaluated 
#     score_eval = []
#     text = []
#     lrp_pos = 0
#     lrp_neg = 0
#     for index in global_indexes: 
#         sample = data.loc[index]
#         score_eval.append(sample[0])
#         tmp_text = sample[1].split()
#         text.append(tmp_text)
        
    
    
#     text_seq = t.texts_to_sequences(text)
#     text_seq_pad = sequence.pad_sequences(text_seq, maxlen= maxlen, padding = 'post' ) # regular padding 
    
   
#     for sent in text_seq_pad: 
#         # tmp = sent    
#         sent = np.expand_dims(sent , 0 )
#         emb_mat = emb_mod.predict(sent) 
#         y_hat = new_mod.predict_classes(emb_mat) 
        
#         lrp_result = analyzer.analyze(emb_mat)
#         lrp_result_seq = np.sum(lrp_result, axis = 2)
        
#         for lrp in lrp_result_seq[0]:
#             if lrp <= 0:
#                 lrp_min = lrp_neg + lrp
#             else:
#                 lrp_pos = lrp_pos + lrp
                
#     lrp_sum = lrp_pos + np.abs(lrp_neg)
#     print(names[i])
#     # print('lrp positive =' , lrp_pos)
#     # print('lrp negative =' , lrp_min)
#     print('% devision positive= ', lrp_pos/lrp_sum )
#     print('% devision negative= ', lrp_neg/lrp_sum )
    
#%% PLOTT HEATMAP
# test to run one input sentence first through the separate embedding layer and then the rest of the model. 
# intresting_data = [22]
# indexes = intresting_data


# examples of when relevance score sum is < 0 
indexes = [402 , 374 , 375, 12, 213, 261, 295, 553]

for index in indexes:
    sent = X_test_pad[index]
    
    if all(v == 0 for v in sent): # found some indexes which the sentence is all zero (no words are in the dictionary).
        continue                  # We just skip these examples for now   
    one_sent = np.expand_dims(sent,0)
    
    test_target = y_test.iloc[index]
    
    emb_mat = emb_mod.predict(one_sent) # OK 
    # use output of embedding layer to predict in the new model 
    y_hat = new_mod.predict_classes(emb_mat) # OK
    # everything runs as expected , also with using multiple sentences as input (the whole test_input)
    result = y_hat - test_target
    accuracy = 1 -(np.sum(np.abs(result))/len(result))
    
    lrp_result = analyzer.analyze(emb_mat)
    lrp_result_seq = np.sum(lrp_result, axis = 2)
    
    text = t.sequences_to_texts(one_sent)
    text_split = f.Split_sentences(text)
    text_plot = text_split[0]
    
    # isolate results from words, disregard from empty spaces
    score_words = lrp_result_seq[0][0:np.shape(text_split)[1]]
    
    # plot heatmap and prediction results
    f.plot_text_heatmap(text_plot, score_words ,title = ['GI: ', X_test_indexes[index]])
    print('target =', test_target , 'predicted =' ,y_hat[0][0], 'index' , index)