# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:55:39 2020

@author: Fredrik Möller
"""

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

# import seaborn as sns
import matplotlib.pyplot as plt


#%%
# set parameters:
max_features = 5000
maxlen = 75 # cap to 75 

use_stopwords = True
reduce_padding = False

# total number of data points available is 63536
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

X_train_indexes = X_train.index
X_test_indexes = X_test.index


X_train_split = f.Split_sentences(X_train)
X_test_split = f.Split_sentences(X_test)

# chose to use stopwords or not 
if use_stopwords:
    filter_words = []
    with open("filter_words.txt", "r") as file:
      for line in file:
        filter_words.append(line.strip())
    filter_words = np.array(filter_words)
    t = ks.preprocessing.text.Tokenizer(num_words = max_features, filters = filter_words)
else:
    t = ks.preprocessing.text.Tokenizer(num_words = max_features)


t = ks.preprocessing.text.Tokenizer(num_words = max_features)
t.fit_on_texts(X_train_split)

X_train_seq = t.texts_to_sequences(X_train_split)
X_test_seq= t.texts_to_sequences(X_test_split)

# pad sequences to the max length 
X_train_pad = sequence.pad_sequences(X_train_seq, maxlen=maxlen, padding = 'post')
X_test_pad = sequence.pad_sequences(X_test_seq, maxlen=maxlen, padding = 'post')

#%% load a pretrained model from the models folder 

model_name = '2020-05-27-09-32'
path = 'C:/Users/Fredrik Möller/Documents/MPSYS/Master_thesis/Code/GIT/Thesis-backup/models'
model = load_model(path + '/' + model_name)


#%%
#% # to continue a model need to either be defined, compiled, and trained from the "Build_n_train_model" script
# or loaded from the top of this script 
#
# Make prediction on all the test data, sort the result into a "confusion" vector, fp-fn-tp-tn. 
#only local indexes relating to the sentences position in X_test_pad is saved

indexes_tp = []
indexes_tn = []
indexes_fp = []
indexes_fn = []

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
            
# print('Accuracy:' , 1-(len(indexes_fp)+len(indexes_fn))/len(y))

#% print data from experiment of using stopwords and reducing padding 

# keys=model.history.history.keys()
res = []

print(model_name)
if reduce_padding:
    print('reduce padding' ,1 )
    # res.append(1)
else:
    print('reduce padding' ,0 )
    # res.append(0)

if use_stopwords:
    print('stop words' ,1 )
    # res.append(1)
else:
    print('stop words' ,0 )
    # res.append(0)

# for key in keys:
#     value = model.history.history[key][-3] # -3 due to early stopping used during training 
#     print(key + ':', value)
#     res.append(value)
print('Accuracy:' , 1-(len(indexes_fp)+len(indexes_fn))/len(y_test))
print('length test data: ' ,len(y_test))
print('TP :', len(indexes_tp))
print('TN :', len(indexes_tn))
print('FP :', len(indexes_fp))
print('FP :', len(indexes_fn))
res.append(1-(len(indexes_fp)+len(indexes_fn))/len(y_test))
res.append(len(y_test))
res.append(len(indexes_tp))
res.append(len(indexes_tn))
res.append(len(indexes_fp))
res.append(len(indexes_fn))

res = np.array(res)
res = res.reshape(1,len(res))
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

### create LRP analyzer 
#% create and test analyser 
# analyzer = inn.create_analyzer('gradient',new_mod)
analyzer = inn.create_analyzer('lrp.epsilon',new_mod, epsilon = 0.00001)
# analyzer = inn.create_analyzer('lrp.alpha_2_beta_1',new_mod)
####

#%% check which words have the highest lrp score in each class for X_test_pad
lrp_fp = np.zeros((1, max_features))
lrp_fn = np.zeros((1, max_features))
lrp_tp = np.zeros((1, max_features))
lrp_tn = np.zeros((1, max_features))


score_predict = model.predict_classes(X_test_pad)
confusion = score_predict - np.expand_dims(y_test,1)

lrp_per_token = np.zeros((1, max_features))

for idx , sent in enumerate(X_test_pad): 
    # tmp = sent    
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
        
    lrp_per_token = np.append(lrp_per_token,lrp_tmp,0)
    
    if confusion[idx] == 1: # false positive # add the lrp to the correct bin dependent on the classification score 
        lrp_fp = lrp_fp + lrp_tmp
    elif confusion[idx] == -1: # false negative 
        lrp_fn = lrp_fn + lrp_tmp
    else: # sort rest if their target is 1 or 0
        if y_test.iloc[idx]== 1:
             lrp_tp = lrp_tp + lrp_tmp
        else:
             lrp_tn = lrp_tn + lrp_tmp

# get mean 
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
lrp_N_sigma = 2 

# FP case 
# only loking at + sigma since pos lrp score support the wrong classification
tmp_bool = lrp_fp >= lrp_fp_mean + lrp_N_sigma * lrp_fp_std 
more_than_sigma= [i for i, x in enumerate(tmp_bool[0]) if x ]
if 0 in more_than_sigma: more_than_sigma.remove(0) # remove 0 if in list since index 0 is the padding index and does not exist in the tokenizer 
skewed_words_fp = [ t.index_word[index] for index in more_than_sigma ]

# FN case 
# only loking at + sigma since pos lrp score support the wrong classification
tmp_bool = lrp_fn >= lrp_fn_mean + lrp_N_sigma * lrp_fn_std 
more_than_sigma= [i for i, x in enumerate(tmp_bool[0]) if x ]
if 0 in more_than_sigma: more_than_sigma.remove(0)
skewed_words_fn = [ t.index_word[index] for index in more_than_sigma]

# TP case 
# only loking at + sigma since pos lrp score support the wrong classification
tmp_bool = lrp_tp >= lrp_tp_mean + lrp_N_sigma * lrp_tp_std 
more_than_sigma= [i for i, x in enumerate(tmp_bool[0]) if x ]
if 0 in more_than_sigma: more_than_sigma.remove(0)
skewed_words_tp = [ t.index_word[index] for index in more_than_sigma]

# TN case 
# only loking at + sigma since pos lrp score support the wrong classification
tmp_bool = lrp_tn >= lrp_tn_mean + lrp_N_sigma * lrp_tn_std 
more_than_sigma= [i for i, x in enumerate(tmp_bool[0]) if x ]
if 0 in more_than_sigma: more_than_sigma.remove(0)
skewed_words_tn = [ t.index_word[index] for index in more_than_sigma]

# read stop words 
read_tmp = []
with open("stop_words.txt", "r") as file:
  for line in file:
    read_tmp.append(line.strip())
stop_words = np.array(read_tmp)

words_wrong_classification = list(set(skewed_words_fp + skewed_words_fn))
words_not_in_stop_words = [w for w in words_wrong_classification if w not in stop_words]



#%% Get data from local indexes, convert to gobal idenxes so that they allways can be found 
# local to global indexes 
local_indexes = indexes_fp
global_indexes = X_test.index[local_indexes]
read_global = False 
# global_indexes = [6612, ]

# read from global indexes instead of X_test 
if read_global:
# from global indexes perfrom preproccesseing so that the sentenes can be evaluated 
    score_eval = []
    text = []
    for index in global_indexes: 
        sample = data.loc[index]
        score_eval.append(sample[0])
        tmp_text = sample[1].split()
        text.append(tmp_text)
        
    
    
    text_seq = t.texts_to_sequences(text)
    data_for_lrp = sequence.pad_sequences(text_seq, maxlen= maxlen, padding = 'post' ) # regular padding 

if not read_global: 
    data_for_lrp = X_test_pad

# perform LRP on the selected sentences and save the relevance per toekn in a matrix for further evaluation 
lrp_tmp = np.zeros((1, max_features))
lrp_per_token = np.zeros((1, max_features))

for sent in data_for_lrp: 
    # tmp = sent    
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
        
    lrp_per_token = np.append(lrp_per_token,lrp_tmp,0)



# sum all relevancy scores per token and get the corresponding words

lrp_sum = np.sum(lrp_per_token,0)
non_zero_index = np.nonzero(lrp_sum)

lrp_sum_non_zero = lrp_sum[non_zero_index]

lrp_tokens = t.sequences_to_texts(non_zero_index)
lrp_tokens =  f.Split_sentences(lrp_tokens)
lrp_tokens = lrp_tokens[0]
lrp_tokens.insert(0,'padding_token')


# get top N highest relevance contributers 
N = 20
top_N_index = lrp_sum_non_zero.argsort()[-N:]
top_N_lrp = lrp_sum_non_zero[top_N_index]
if 'top_N_words' not in locals():
    top_N_words = np.array(lrp_tokens)[top_N_index]
else:
    np.append(top_N_words,top_N_words)
    
## train new tokenizer with the highest relevancies removed 

# tt = ks.preprocessing.text.Tokenizer(num_words = max_features, filters = top_N_words)

# X_test_seq_new = tt.texts_to_sequences(X_test_split)
# X_test_pad_new = sequence.pad_sequences(X_test_seq_new, maxlen=maxlen, padding = 'post')

    
#%% check to see if there is some relation between the total relevancy + and - and the distrubution of classes in the training data set 
names = ['TP' , 'TN' , 'FP' , 'FN']
tmp = [indexes_tp , indexes_tn , indexes_fp, indexes_fn]
for i in range(len(tmp)):
    local_indexes = tmp[i]
    global_indexes = X_test.index[local_indexes]
    
    # global_indexes = [6612, ]
    
    # from global indexes perfrom preproccesseing so that the sentenes can be evaluated 
    score_eval = []
    text = []
    lrp_pos = 0
    lrp_neg = 0
    for index in global_indexes: 
        sample = data.loc[index]
        score_eval.append(sample[0])
        tmp_text = sample[1].split()
        text.append(tmp_text)
        
    
    
    text_seq = t.texts_to_sequences(text)
    text_seq_pad = sequence.pad_sequences(text_seq, maxlen= maxlen, padding = 'post' ) # regular padding 
    
   
    for sent in text_seq_pad: 
        # tmp = sent    
        sent = np.expand_dims(sent , 0 )
        emb_mat = emb_mod.predict(sent) 
        y_hat = new_mod.predict_classes(emb_mat) 
        
        lrp_result = analyzer.analyze(emb_mat)
        lrp_result_seq = np.sum(lrp_result, axis = 2)
        
        for lrp in lrp_result_seq[0]:
            if lrp <= 0:
                lrp_min = lrp_neg + lrp
            else:
                lrp_pos = lrp_pos + lrp
                
    lrp_sum = lrp_pos + np.abs(lrp_neg)
    print(names[i])
    # print('lrp positive =' , lrp_pos)
    # print('lrp negative =' , lrp_min)
    print('% devision positive= ', lrp_pos/lrp_sum )
    print('% devision negative= ', lrp_neg/lrp_sum )
    
#%% PLOTT HEATMAP
# test to run one input sentence first through the separate embedding layer and then the rest of the model. 
# intresting_data = [22]
# indexes = intresting_data


nr_plots = 10
indexes = indexes_fn[:10]

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
    print('target =', test_target , 'predicted =' ,y_hat[0][0])