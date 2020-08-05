# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:55:39 2020

@author: Fredrik Möller
"""

from keras.models import load_model

import innvestigate as inn
import fun_c as f 
import numpy as np
# import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt

# logging package 
# import logging



#%% load a pretrained model from the models folder 

model_name = '2020-05-29-13-28' # baseline model 
path = 'C:/Users/Fredrik Möller/Documents/MPSYS/Master_thesis/Code/GIT/Thesis-backup/models'


# model_name = '2020-06-16-08-58'
# path = 'C:/Users/Fredrik Möller/Documents\MPSYS/Master_thesis/Code/GIT/Thesis-backup/models/ver'

model = load_model(path + '/' + model_name)

#%%
# set parameters:
max_features = 5000
nr_features = 5000
maxlen = 75 # cap to 75 

dr = 0.3

use_word_filter_remove = False
use_word_filter_drop_word = False
use_word_filter_drop_sent = False
mosque_setup = False
reduce_padding = False
return_text = False

stop_words_from = 'tfidf' # 'freq' , 'tfidf' , 'lrp' ,'stop_words


if 1 < use_word_filter_remove + use_word_filter_drop_word + use_word_filter_drop_sent:    
   raise Exception('More than one filter is set to be applied to the data, check settings and try again')

"reading word filter"
filter_words = []
if use_word_filter_remove or use_word_filter_drop_word or use_word_filter_drop_sent: 
    name = 'filter_words_' + stop_words_from +'.txt'
    with open(name, "r") as file:
        for line in file:
            filter_words.append(line.strip())

# use_word_filter_remove , use_word_filter_drop_word , use_word_filter_drop_sent , filter_words , dr ,mosque_setup , flip_mosque_target , reduce_padding

input_vars = [use_word_filter_remove ,
              use_word_filter_drop_word ,
              use_word_filter_drop_sent ,
              filter_words ,
              dr,
              mosque_setup ,
              reduce_padding , 
              max_features , 
              maxlen, 
              return_text]

X_train_pad , y_train , X_test_pad , y_test , t = f.do_preprocess(input_vars)
use_word_filter_drop_sent = False


return_text = True
input_vars = [use_word_filter_remove ,
              use_word_filter_drop_word ,
              use_word_filter_drop_sent ,
              filter_words ,
              dr,
              mosque_setup ,
              reduce_padding , 
              max_features , 
              maxlen, 
              return_text]
split_texts = f.do_preprocess(input_vars)
return_text = False
X_train_split = split_texts[0]
X_test_split = split_texts[2]


mosque_setup = True
input_vars = [use_word_filter_remove ,
              use_word_filter_drop_word ,
              use_word_filter_drop_sent ,
              filter_words ,
              dr,
              mosque_setup ,
              reduce_padding , 
              max_features , 
              maxlen, 
              return_text]
mosque_data = f.do_preprocess(input_vars)

# set all mosque targets to 1 in the training data 
# check if the index is in y_train since data can be removed
if mosque_setup:
    for index in mosque_data[5]:
        if index in y_train.index:
           y_train[index] = 1 
    
    X_test_pad_mosque = X_test_pad[mosque_data[6]]
    y_test_mosque = y_test[mosque_data[6]]
    mosque_setup = False


# logging commands 
# log = logging.getLogger(__name__)
# logging.setLevel(logging.INFO)
# log.info(f'Apa bepa cepa')

#% small fix to shorten the base training data set based on prob 
# import random

# dr = 0.22

# filtered_data = []
# filtered_target = []

# for i , sent in enumerate(X_train_pad):
#     if dr > random.random():
#         continue
#     filtered_data.append(sent)
#     filtered_target.append(y_train[i])

# tmp = np.empty([len(filtered_target),75])
# for i , sent in enumerate(filtered_data):
#     tmp[i,:] = sent



# X_train_pad =  tmp
# y_train = pd.Series(filtered_target)
  

#%% Check which framgent have bee n worngfullt classified. 


evaluate_mosque = True
print('##########################################')
print('load pretrained models')
print('##########################################')
path = 'C:/Users/Fredrik Möller/Documents/MPSYS/Master_thesis/Code/GIT/Thesis-backup/models/ver'

if not evaluate_mosque:
    if 'Baseline_0' not in locals():
        Baseline_0 = load_model(path + '/' + 'Baseline_0')
        Baseline_022 = load_model(path + '/' + 'Baseline_022')
        NCOF_02 = load_model(path + '/' + 'NCOF_02')
        TFIDF_03 = load_model(path + '/' + 'TFIDF_03')
        LRP_045 = load_model(path + '/' + 'LRP_045')
    model_names = ['Baseline_0' , 'Baseline_022' , 'NCOF_02' , 'LRP_045' , 'TFIDF_03']
    models = [Baseline_0 , Baseline_022 , NCOF_02 , LRP_045 , TFIDF_03]
else:   
    print('loading mosque trained models')
    if 'Baseline_0_mosque' not in locals():
        Baseline_0_mosque = load_model(path + '/' + 'Baseline_0_mosque')
        Baseline_022_mosque = load_model(path + '/' + 'Baseline_022_mosque')
        NCOF_02_mosque = load_model(path + '/' + 'NCOF_02_mosque')
        TFIDF_03_mosque = load_model(path + '/' + 'TFIDF_03_mosque')
        LRP_025_mosque = load_model(path + '/' + 'LRP_025_mosque')
    model_names = ['Baseline_0_mosque' , 'Baseline_022_mosque' , 'NCOF_02_mosque' , 'LRP_025_mosque', 'TFIDF_03_mosque']
    models = [Baseline_0_mosque , Baseline_022_mosque , NCOF_02_mosque , LRP_025_mosque , TFIDF_03_mosque]

#%%
# # use for mosque tests 
# X_test_pad = mosque_test
# y_test = mosque_test_target
# ###########################
eval_data = X_test_pad_mosque
eval_targets = y_test_mosque
# ###########################
# eval_data = X_test_pad
# eval_targets = y_test

print('##########################################')
print('Calculates confusion matrix for loaded models and data')
print('##########################################')
res_pred = np.empty([len(eval_targets), len(models)])
for i , mod in enumerate(models):
    print('Starting evaluation of model ' , str(mod), '(', str(i + 1) , '/', str(len(models)),')' )
    y_hat = mod.predict_classes(eval_data)
    res_pred[:,i] = y_hat.squeeze()
    
# -1 = FN , 1 = FP , 0 = correct predicted
confusion_all = res_pred - np.expand_dims(eval_targets,1)

print('##########################################')
print('Get indexes where models prediction differ')
print('##########################################')
index_in_all = []
index_in_filter = []
for  i , line in enumerate(confusion_all): # for each row in confusion
    if np.all(line[0] != line[1:]): # if model prediction not equal baseline
        index_in_all.append(i)
        
    elif np.all(line[0] != line[2:]):
        index_in_filter.append(i)

# get the summerized lrp score for mosque data
print('##########################################')
print('get the summerized lrp score for mosque data')
print('##########################################')
epsilon = 0.01
lrp_tp_all = []
lrp_tn_all = []
lrp_fp_all = []
lrp_fn_all = []
lrp_all = []
for model in models: 
    emb_layer , other_layers = f.split_model(model)
    lrp_tp , lrp_tn , lrp_fp , lrp_fn , lrp_per_token = f.Get_LRP_per_token(model = model, emb_layer = emb_layer , other_layers = other_layers, epsilon = epsilon, data = eval_data , targets = eval_targets, nr_features = max_features)
    lrp_tp_all.append(lrp_tp)
    lrp_tn_all.append(lrp_tn)
    lrp_fp_all.append(lrp_fp)
    lrp_fn_all.append(lrp_fn)
    lrp_all.append(lrp_per_token)

#%%
print('##########################################')
print('get outliers for the selected lrp data')
print('##########################################')
words_all = []
index_all = []
sigma = 3
for elm in lrp_fp_all:
    elm_data = elm
    outliers_word , outliers_index = f.Get_lrp_outliers(lrp_data = elm_data , sigma = sigma, pm = 1, tokenizer = t)
    words_all.append(outliers_word)
    index_all.append(outliers_index)


#%% Get a seperate summation for each token for relevance in relation to a pos or neg classification for the entirery of the training data 

lrp_pos_all = lrp_tp[0] + lrp_fp[0]
lrp_neg_all = (lrp_tn[0] + lrp_fn[0])*-1

tmp_pos = np.zeros_like(lrp_pos_all)
tmp_neg = np.zeros_like(lrp_neg_all)

for i , elm in enumerate(lrp_pos_all):
    if elm < 0:
        tmp_neg[i] = tmp_neg[i] + elm
    else:
        tmp_pos[i] = tmp_pos[i] + elm

for i , elm in enumerate(lrp_neg_all):
    if elm < 0:
        tmp_neg[i] = tmp_neg[i] + elm
    else:
        tmp_pos[i] = tmp_pos[i] + elm
        
    
    
def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in values]

tmp_pos = zero_to_nan(tmp_pos)
tmp_neg = zero_to_nan(tmp_neg)

nr_elements = 500
plt.figure()

plt.scatter(range(1,500) , tmp_pos[1:nr_elements] , s =2)
plt.scatter(range(1,500) , tmp_neg[1:nr_elements] , s =2)


plt.figure()
plt.bar(range(1,nr_elements),tmp_pos[1:nr_elements], edgecolor = 'b')
plt.bar(range(1,nr_elements),tmp_neg[1:nr_elements], edgecolor = 'r')

# plt.scatter(range(nr_points), tmp_neg[:nr_points], s = 2)


#%% Produce LRP analysis score for the diffrent model and make plots of them

epsilon = 0.01
element = 121 # which element in mosque test data do you want to investigate?
# element = 145 # which element in mosque test data do you want to investigate?

index = mosque_data[6][element]
sent = X_test_pad_mosque[element]
target = y_test_mosque[index]
# index = index_in_filter[6] 
words = []
for elm in sent:
    if elm != 0:
        words.append(t.index_word[elm])



lrp_mod = []
lrp_mod_all = np.empty((len(models), len(words)))

for i , model in enumerate(models):
    print('evaluating model ', str(i+1), '/' , str(len(models)))
    emb_mod , new_mod = f.split_model(model)
    ann_input =emb_mod.predict(np.expand_dims(sent,0))
    new_mod.predict_classes(ann_input)

    analyzer = inn.create_analyzer('lrp.epsilon', new_mod, epsilon = epsilon)

    lrp = analyzer.analyze(ann_input)
    lrp_result_seq = np.sum(lrp, axis = 2)
    lrp_mod = []
    for k , elm in enumerate(words):
        lrp_mod_all[i,k] = lrp_result_seq[0,k]

words = np.expand_dims(words,0)
presentable_results = np.concatenate((words,lrp_mod_all),0)

for j , elm in enumerate(lrp_mod_all):
    f.plot_text_heatmap(words[0], elm , title = model_names[j])
    
print('target: ', str(target))
print('predicted: ', str(res_pred[element]))
print('confusion: ', str(confusion_all[element]))

#%%
wrong_target = []
for i , line in enumerate(confusion_all):
    if line[0]==1 and line[4] == 0:
        wrong_target.append(i)

#%% Produces LRP score for the test data based on the model in var explorer

"seperate the embedding layer from the rest of the layers since the embedding layer does comply to back propagation regulations"
emb_mod , new_mod = f.split_model(model)

"calculates the lrp score per confusion quadrant for the given data"
"If bog standard LRP application is needed set epsilon to zero"
lrp_tp , lrp_tn , lrp_fp , lrp_fn , lrp_per_token = f.Get_LRP_per_token(model=model, emb_layer = emb_mod, other_layers = new_mod, epsilon = 0.01, data = X_train_pad, targets = y_train, nr_features = max_features)

"calculate the mean of the lrp score for each conf quadrant "
# lrp_fp_mean = lrp_fp.mean()
# lrp_fn_mean = lrp_fn.mean()
# lrp_tp_mean = lrp_tp.mean()
# lrp_tn_mean = lrp_tn.mean()
"calculate the std of the lrp score for each conf quadrant "
# lrp_fp_std = lrp_fp.std()
# lrp_fn_std = lrp_fn.std() 
# lrp_tp_std = lrp_tp.std()
# lrp_tn_std = lrp_tn.std()

#%%
" get the N sigma outliers for each conf quadrant"
sigma = 3
# tp
tp_word , tp_index = f.Get_lrp_outliers(lrp_data =  lrp_tp, sigma = sigma , pm = 1, tokenizer = t)
# tn
tn_word , tn_index = f.Get_lrp_outliers(lrp_data =  lrp_tn, sigma = sigma , pm = 1, tokenizer = t)
# fp 
fp_word , fp_index = f.Get_lrp_outliers(lrp_data =  lrp_fp, sigma = sigma , pm = 1, tokenizer = t)
# fn
fn_word , fn_index = f.Get_lrp_outliers(lrp_data =  lrp_fn, sigma = sigma , pm = 1, tokenizer = t)

"summerize the outliers from FP and FN to the LRP filter"
"Presentes the result as aplhabetically sorted"
filter_lrp = set(fp_word + fn_word)
filter_lrp = f.remove_stopwords(filter_lrp)
filter_lrp.sort()

# %
with open("filter_words_lrp.txt", "w") as file:
    for s in filter_lrp:
        file.write(str(s) +"\n")

#%% Lemmatize the LRP result for each confusion classification 

"Lemmatize the N sigma LRP score outliers per token for each confusion classification "
sigma = 3
# TP
tp_lemma_words , tp_lemma_index = f.Get_lemma_lrp(lrp_data = lrp_tp , sigma = sigma , pm=-1, tokenizer = t )

# TN
tn_lemma_words , tn_lemma_index = f.Get_lemma_lrp(lrp_data = lrp_tn , sigma = sigma , pm=-1, tokenizer = t )

# FP 
fp_lemma_words , fp_lemma_index = f.Get_lemma_lrp(lrp_data = lrp_fp , sigma = sigma , pm=-1, tokenizer = t )

# FN 
fn_lemma_words , fn_lemma_index = f.Get_lemma_lrp(lrp_data = lrp_fn , sigma = sigma , pm=-1, tokenizer = t )


#%% plotting time 

fig_width = 15
fig_hight = 4 
marker_size1 = 2 
marker_size2 = 2
colour = 'red'
leg = ['LRP < mean + 3 sigma', 'LRP >= mean + 3 sigma']
title1 = 'LRP score per index'
title2 = 'LRP score per index w/o padding index'
x_label = 'Tokenizer index'
ylabel = 'LRP score'

type_conf = [' (fp)' , ' (fn)' , ' (tp)' , ' (tn)', ' (predicted pos)' , ' (prediced neg)']

data_plot = [lrp_fp , lrp_fn , lrp_tp , lrp_tn]
indexes_plot = [fp_index , fn_index , tp_index , tn_index ]

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
    f.plot_text_heatmap(text_plot, score_words ,title = ['Placeholder text'])
    print('target =', test_target , 'predicted =' ,y_hat[0][0], 'index' , index)
    
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
#%% plot distrubution of lengths 

tmp_length = [len(sent) for sent in X_train_split]
sns.set_style("darkgrid")

plt.figure()
fig, ax = plt.subplots()

sns.distplot(tmp_length, bins = 50, label = "Distrubution of sentence length")
plt.axvline(75, 0 ,max(tmp_length), ls = '--' , c = 'red', label= "Sentence length = 75")

ax.set_ylabel('% chance of sentence with length')
ax.set_xlabel('Sentence length [words]')
ax.set_title('Distrubution of sentence length in the training data')
ax.legend() 

