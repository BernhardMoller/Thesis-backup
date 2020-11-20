# -*- coding: utf-8 -*-
"""
Created on Wed May  6 13:57:55 2020

@author: Fredrik Möller
"""
import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
plt.style.use('seaborn')
from matplotlib import cm, transforms

import string
import random

import innvestigate as inn

import nltk 
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer 

from keras.models import Sequential
from keras.preprocessing import sequence
import keras as ks 


#Function that returns the raw data containing the word mosque and its labels 
def Get_mosque_data():
    data_all = pd.read_csv ('C:/Users/Fredrik Möller/Documents/MPSYS/Master_thesis/Code/Data/data_terrorincident_eng_26-03-2020_60k - data_terrorincident_eng_26-03-2020xxx.csv.csv', delimiter=',')
    mosque_data = data_all[['score','mosque_highlighted']]
    mosque_data = mosque_data.dropna(how='any')
    return mosque_data

def train_test_data(reduce_pad):
    if reduce_pad:
        data_train = pd.read_csv ('C:/Users/Fredrik Möller/Documents/MPSYS/Master_thesis/Code/GIT/Thesis-backup/train_test_data/data_train_reduce_pad', delimiter=',')
        data_test = pd.read_csv ('C:/Users/Fredrik Möller/Documents/MPSYS/Master_thesis/Code/GIT/Thesis-backup/train_test_data/data_test_reduce_pad', delimiter=',')
    else:    
        data_train = pd.read_csv ('C:/Users/Fredrik Möller/Documents/MPSYS/Master_thesis/Code/GIT/Thesis-backup/train_test_data/data_train', delimiter=',')
        data_test = pd.read_csv ('C:/Users/Fredrik Möller/Documents/MPSYS/Master_thesis/Code/GIT/Thesis-backup/train_test_data/data_test', delimiter=',')
        
    return  data_train , data_test 

def Get_data(nr_data):
    data_all = pd.read_csv ('C:/Users/Fredrik Möller/Documents/MPSYS/Master_thesis/Code/Data/data_terrorincident_eng_26-03-2020_60k - data_terrorincident_eng_26-03-2020xxx.csv.csv', delimiter=',')
    data = data_all[['score','fragment']]
    rand_prem = np.random.permutation(len(data))
    data = data.iloc[rand_prem[:nr_data]]
    return data

def covert_to_lowercase(data):
    # data need to be formated as a series object to be able to e converted
    data = data.str.lower()
    return data 

def remove_punct(data_split):
    data_split_tmp= []
    table = str.maketrans('', '', string.punctuation)
    for sent in data_split:
        data_split_tmp.append( [w.translate(table) for w in sent])
    return data_split_tmp

##############################################################
"does all preprocessessing of the static training and test data"
"depending on the active bool variables"

def do_preprocess(input_vars):
    
    # use_word_filter_remove , use_word_filter_drop_word , use_word_filter_drop_sent , filter_words , dr ,mosque_setup  , reduce_padding
    use_word_filter_remove = input_vars[0]
    use_word_filter_drop_word = input_vars[1]
    use_word_filter_drop_sent = input_vars[2]
    filter_words = input_vars[3]
    dr = input_vars[4]
    mosque_setup = input_vars[5]
    reduce_padding = input_vars[6]
    max_features = input_vars[7]
    maxlen = input_vars[8]
    return_text = input_vars[9]
    
    data_train , data_test = train_test_data(reduce_pad = reduce_padding)

    X_train = data_train['fragment']
    X_test = data_test['fragment']
    
    y_train = data_train['target']
    y_test = data_test['target']
    
    # X_train_indexes = X_train.index
    # X_test_indexes = X_test.index
    
    "converting all letters to lowercase"
    X_train = covert_to_lowercase(X_train)
    X_test = covert_to_lowercase(X_test)
    
    "splitting the sentences into seperate words"
    X_train_split = Split_sentences(X_train)
    X_test_split = Split_sentences(X_test)
    
    # remove punctuations and commas, also convert to lowercase 
    X_train_split = remove_punct(X_train_split)
    X_test_split = remove_punct(X_test_split)
    
    if use_word_filter_remove:
        X_train_split , y_train = apply_word_filter_remove(X_train_split, y_train , filter_words)
    
    
    if use_word_filter_drop_word:
        X_train_split , y_train = apply_word_filter_drop_word(X_train_split, y_train , filter_words , dr)
            
        
    if use_word_filter_drop_sent: 
        X_train_split , y_train , removed_index = apply_word_filter_drop_sent(X_train_split , y_train , filter_words , dr)
                        
        
    if return_text:
        return [X_train_split , y_train , X_test_split , y_test]
    
    "creating tokenizer"
    t = ks.preprocessing.text.Tokenizer(num_words = max_features)
    "Fitting the tokenizer on training data"
    t.fit_on_texts(X_train_split)
    "converting the training and test data to integer sequences"
    X_train_seq = t.texts_to_sequences(X_train_split)
    X_test_seq= t.texts_to_sequences(X_test_split)
    
    "padding the sequences to a consistent length"
    # pad sequences to the max length 
    X_train_pad = sequence.pad_sequences(X_train_seq, maxlen=maxlen, padding = 'post')
    X_test_pad = sequence.pad_sequences(X_test_seq, maxlen=maxlen, padding = 'post')
    
    
    if mosque_setup:
    
        mosque_data = Get_mosque_data()
        
        # y = mosque_data['score']
        X = mosque_data['mosque_highlighted']
        X = covert_to_lowercase(X)
        X = Split_sentences(X)
        X = remove_punct(X)
        X = t.texts_to_sequences(X)
        
        
        
        print("Getting mosque indexes in the test data set")
        mosque_in_test = []
        for  i , sent in enumerate(X):
            for j , text in enumerate(X_test_seq):
                if sent == text:
                    mosque_in_test.append(j)
        
        print("Getting mosque indexes in the train data set")
        mosque_in_train = []
        for  i , sent in enumerate(X):
            for j , text in enumerate(X_train_seq):
                if sent == text:
                    mosque_in_train.append(j)
        
        mosque_train = X_train_pad[mosque_in_train]
        mosque_train_target = y_train[mosque_in_train]
        
        mosque_test = X_test_pad[mosque_in_test]
        mosque_test_target = y_train[mosque_in_test]
        
        return [mosque_train , mosque_train_target , mosque_test , mosque_test_target , t , mosque_in_train , mosque_in_test]
        
        
    return [X_train_pad , y_train , X_test_pad , y_test , t]
##############################################################
##############################################################
" calculates NCOF score for a given data set, can also produce plots of the results"
def Get_NCOF(data, targets, nr_features , sigma , tokenizer, plot_fig ):
    "data needs to be a sequence tokenized sentence, targets the sentences targets"
    "nrfeatures is how many tokens is in your tokenizers dictionary, i.e. the size of the tokenizers dictionary"
    "Sigma is at how many sigmas you want the method to scelect words from"
    "Tokenizer is the tokenizer you have used to sequence your sentences"
    
    belong_neg = np.zeros((1,nr_features))
    belong_pos = np.zeros((1,nr_features))
   
    for i in range(len(data)):
        sent = data[i]
        target = targets[i]
    
        if target == 0: # if sentence belongs to negative, calc nr of occurences 
            for index in sent:
                belong_neg[0,index] = belong_neg[0,index] + 1
        else: # if sentence belongs to positive class, calc occurence 
            for index in sent:
                belong_pos[0,index] = belong_pos[0,index] + 1
     
    # calc prob of each word occuring as an element in a sentence for each class 
    p_word_in_pos = belong_pos / np.sum(belong_pos)
    p_word_in_neg = belong_neg / np.sum(belong_neg)
    
    # calc NCOF, mean and std 
    occurency = p_word_in_pos - p_word_in_neg
    word_deviation = occurency.std()
    word_mean = occurency.mean()

    # get the words which are NCOF +- outliers based on input sigma 
    tmp_bool = occurency <= word_mean - sigma * word_deviation 
    less_than_sigma = [i for i, x in enumerate(tmp_bool[0]) if x]
    tmp_bool = occurency >= word_mean + sigma * word_deviation 
    more_than_sigma = [i for i, x in enumerate(tmp_bool[0]) if x]
    
    # go from tokens to text words 
    words_pos = [ tokenizer.index_word[index] for index in more_than_sigma ]
    words_neg = [ tokenizer.index_word[index] for index in less_than_sigma ]
    
    if '' in words_neg:
        words_neg.remove('')
    # reassign variable 
    pos_index = more_than_sigma
    neg_index = less_than_sigma
    
    # join outlier words to a single list 
    words_both = words_pos + words_neg
    # remove stopwords from the joined list 
    skewed_words = remove_stopwords(words_both) # uses functon from fun_c
    # get indexes of the remaining outlier words 
    skewed_indexes = [tokenizer.word_index[word] for word in skewed_words]
    
    # choose to plot figues based on input param
    if plot_fig:
        dot = 5
        # plot of NCOF without stopwords in outliers 
        plt.figure()
        fig, ax = plt.subplots()
        ax.scatter(y=occurency , x = range(len(occurency[0])) , s = dot, label = 'NCOF inliers & stop words' )
        ax.scatter(y = occurency[0,skewed_indexes] , x = skewed_indexes, s = dot , c= 'red', label =  'NCOF 3 sigma outliers')
        ax.set_ylabel('NCOF score')
        ax.set_xlabel('Integer representation index')
        ax.set_title('NCOF score between + & - class')
        ax.legend()
    
        skewed_indexes_w_stop_words= [tokenizer.word_index[word] for word in words_both]
        # plot of NCOF with stopwords in outliers 
        plt.figure()
        fig, ax = plt.subplots()
        ax.scatter(y=occurency , x = range(len(occurency[0])) , s = dot, label = 'NCOF inliers' )
        ax.scatter(y = occurency[0,skewed_indexes_w_stop_words] , x = skewed_indexes_w_stop_words, s = dot , c= 'red', label =  'NCOF 3 sigma outliers & stop words')
        ax.set_ylabel('NCOF score')
        ax.set_xlabel('Integer representation index')
        ax.set_title('NCOF score between + & - class')
        ax.legend()
        
        # zoomed version of the no stopwords plot
        points = range(300)
        plt.figure()
        fig, ax = plt.subplots()
        ax.scatter(y=occurency[0][points] , x = points , s = dot*2, label = 'NCOF inliers & stop words' )
        ax.scatter(y = occurency[0,skewed_indexes] , x = skewed_indexes, s = dot*2 , c= 'red', label =  'NCOF 3 sigma outliers')
        ax.set_ylabel('NCOF score')
        ax.set_xlabel('Integer representation index')
        ax.set_title('NCOF score between + & - class, cropped')
        ax.legend()
        
        plt.figure()
        fig, ax = plt.subplots()
        ax.scatter(y=p_word_in_pos , x = range(len(p_word_in_pos[0])) , s = dot, label = 'Token' )
        ax.set_ylabel('NCOF score')
        ax.set_xlabel('Integer representation index')
        ax.set_title('NCOF score for tokens, from positive sentences')
        ax.legend()
        
        plt.figure()
        fig, ax = plt.subplots()
        ax.scatter(y=p_word_in_neg , x = range(len(p_word_in_neg[0])) , s = dot, label = 'Token' )
        ax.set_ylabel('NCOF score')
        ax.set_xlabel('TInteger representation index')
        ax.set_title('NCOF score for tokens, from negative sentences')
        ax.legend()
    
    return words_pos, words_neg , pos_index, neg_index , skewed_words ,skewed_indexes

##############################################################
##############################################################
" function used in the Tf-idf calculation"
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
##############################################################

##############################################################
" calculates Tf-idf score for a given data set given two classes, can also produce plots of the results"
def Get_tfidf(data, targets , nr_features,  nr_sigmas , plot_fig):
    "Data does not need to be split into seperate words"
    "nr_sigmas is at what sigma you want to feth our outliers from"
    "plot_fig is a tuple that indicates if plots should be produced "
    # create tfidf vectorizer 
    vectorizer = TfidfVectorizer(max_features = nr_features)
    print('here0')
    vectorizer.fit(data)
    print('here')
    # fit vectorizer to corpus 
    X_tfidf = vectorizer.transform(data)
    # extract all feature names in alphabetical order 
    feature_names = vectorizer.get_feature_names()    
    # create vars for storing result
    print('here1')
    tfidf_score_pos = np.zeros([1,len(feature_names)])
    tfidf_score_neg = np.zeros([1,len(feature_names)])

    # calculate tfidf score based of target of a sentence and store sum 
    for i , elm in enumerate(X_tfidf):
        sent = X_tfidf[i]
        sorted_items=sort_coo(sent.tocoo())
        if targets[i] == 0:
            for tup in sorted_items:
                index , score = tup
                tfidf_score_neg[0,index] = tfidf_score_neg[0,index] + score
        else: 
            for tup in sorted_items:
                index , score = tup
                tfidf_score_pos[0,index] = tfidf_score_pos[0,index] + score
    print('here2')                
    tfidf_std_neg = tfidf_score_neg.std()
    tfidf_mean_neg = tfidf_score_neg.mean()
    
    tfidf_std_pos = tfidf_score_pos.std()
    tfidf_mean_pos = tfidf_score_pos.mean()

    # get tf-idf scores that are +- N sigmas 
    skewed_indexes_tfidf_neg = []
    skewed_words_tfidf_neg = []
    
    skewed_indexes_tfidf_pos = []
    skewed_words_tfidf_pos = []
    # calculate the outlier words from the pos and neg class 
    for i in range(len(tfidf_score_neg[0])):
        if tfidf_score_neg[0,i] >= tfidf_mean_neg + nr_sigmas * tfidf_std_neg:
            skewed_indexes_tfidf_neg.append(i)
            skewed_words_tfidf_neg.append(feature_names[i])
    
    for i in range(len(tfidf_score_pos[0])):
        if tfidf_score_pos[0,i] >= tfidf_mean_pos + nr_sigmas * tfidf_std_pos:
            skewed_indexes_tfidf_pos.append(i)
            skewed_words_tfidf_pos.append(feature_names[i])
                
    tmp_neg_ind=set(skewed_indexes_tfidf_neg)
    tmp_pos_ind=set(skewed_indexes_tfidf_pos)

    set_difference_ind = tmp_neg_ind ^ tmp_pos_ind
    
    # take the set difference the get a not in b 
    pos_ind_not_in_neg =list( tmp_pos_ind - tmp_neg_ind)
    
    # set difference b not in a 
    neg_ind_not_in_pos = list(tmp_neg_ind - tmp_pos_ind)

    # get the word that are in both pos and negative outliers 
    tmp_intersection = list(tmp_neg_ind.intersection(tmp_pos_ind))
    
    
    # get unique words for neg and pos & union 
    words_unique_pos = [feature_names[i] for i in pos_ind_not_in_neg]
    words_unique_neg = [feature_names[i] for i in neg_ind_not_in_pos]
    words_intersection = [feature_names[i] for i in tmp_intersection]  
    words_set_difference = [feature_names[i] for i in set_difference_ind]   
    
    words_tmp = words_set_difference
    words_tmp.sort()
    
    words_tmp = remove_stopwords(words_tmp)
    
    with open("filter_words_tfidf.txt", "w") as file:
        for s in words_tmp:
            file.write(str(s) +"\n")
        # get indexes of the remaining outlier words     
    
    # print(neg_ind_not_in_pos)
    words_unique_neg.append('new')
    tmp_words_neg = remove_stopwords(words_unique_neg)
    tmp_vec= []
    
    for i,elm in enumerate(feature_names):
       if elm in tmp_words_neg:
           tmp_vec.append(i)
        
    print(tmp_vec)
    
    if plot_fig:
        
        dot = 8
        # neg class 
        plt.figure()
        fig, ax = plt.subplots()
        
        
        ax.scatter(y=tfidf_score_neg , x = range(tfidf_score_neg.shape[1]) , s = dot , label = 'Tf-idf inliers & stop words')
        ax.scatter(y = tfidf_score_neg[0,tmp_vec] , x = tmp_vec, s = dot , c= 'red', label = 'Tf-idf 3 sigma unique neg')
        ax.scatter(y = tfidf_score_neg[0,tmp_intersection] , x = tmp_intersection, s = dot , c= 'black', label = 'Tf-idf 3 sigma outliers, in both')
        
        ax.set_ylabel('Tf-idf score')
        ax.set_xlabel('Tf-idf Integer representation index')
        ax.set_title('Tf-idf score of negative class, with 3 sigma outliers highlighted')
        ax.legend() 
        # pos class 
        plt.figure()
        fig, ax = plt.subplots()
        ax.scatter(y=tfidf_score_pos , x = range(tfidf_score_pos.shape[1]) , s = dot , label = 'Tf-idf inliers & stop words')
        ax.scatter(y = tfidf_score_pos[0,pos_ind_not_in_neg] , x = pos_ind_not_in_neg, s = dot , c= 'red', label = 'Tf-idf 3 sigma outliers, unique pos')
        ax.scatter(y = tfidf_score_pos[0,tmp_intersection] , x = tmp_intersection, s = dot , c= 'black', label = 'Tf-idf 3 sigma outliers, in both')
        
        ax.set_ylabel('Tf-idf score')
        ax.set_xlabel('Tf-idf Integer representation index')
        ax.set_title('Tf-idf score of positive class, with 3 sigma outliers highlighted')
        ax.legend()
    return words_unique_pos , words_unique_neg , words_intersection , words_set_difference
##############################################################

##############################################################
"Functions calculate the LRP score for a model and presents the results in terms of tokens available in the data and also based on classification"
"This function assumed that you have an embedding layer in you model"
def Get_LRP_per_token(model, emb_layer , other_layers, epsilon, data , targets, nr_features):
    "model: base keras model used to classify the data. the model that you have trained"
    "emb_layer: the models keras embedding layer seperated from the rest of your model, need to be compiled as a seperate model. Needed due to rlp does not work on embedding layer due to no rules for back prop"
    "other_layers: all the layers in model - the embedding layer, compiled into a seperate model"
    "epsilon: epsilon used in the LRP calculation (epsilon rule)"
    "data: sentenced in the same format as the data used to train the model"
    "targets: the targets of 'data' "
    "nr_features: number of features in your tokenizer dictionary"
    
    "the model fore some reason needs to make one prediction before it can be translated for lrp prop"
    sent = np.expand_dims(data[0] , 0 )
    emb_mat = emb_layer.predict(sent) 
    y_hat = other_layers.predict_classes(emb_mat) 

    
    analyzer = inn.create_analyzer('lrp.epsilon', other_layers, epsilon = epsilon)
    print("Completed the LRP analyser of the input model")
    
    lrp_fp = np.zeros((1, nr_features))
    lrp_fn = np.zeros((1, nr_features))
    lrp_tp = np.zeros((1, nr_features))
    lrp_tn = np.zeros((1, nr_features))
    
    
    score_predict = model.predict_classes(data)
    confusion = score_predict - np.expand_dims(targets,1)
    print("Completed prediction of the data and made a confusion array")
    # text_seq_pad = X_test_pad
    
    # perform LRP on the selected sentences and save the relevance per toekn in a matrix for further evaluation 
    lrp_tmp = np.zeros((1, nr_features))
    lrp_per_token = np.zeros((len(data), nr_features))
    
    
    
    for idx , sent in enumerate(data): 
        if np.mod(idx,1000) == 0: print("Completed lrp-calculations for " , idx , " / " , len(targets))   
        
        sent = np.expand_dims(sent , 0 )
        emb_mat = emb_layer.predict(sent) 
            
        if np.mod(idx,1000) == 0: print("Managed to make predictions on embedding layer")
        
        lrp_result = analyzer.analyze(emb_mat)
        lrp_result_seq = np.sum(lrp_result, axis = 2)
        if np.mod(idx,1000) == 0: print("Completed LRP propagation over model and summerized to the tokens in the sentece")
        
        
        if np.mod(idx,1000) == 0: print("Sorts score into a culmative score per token ")
        lrp_tmp = np.zeros((1, nr_features))
        for i in range(np.shape(lrp_result)[1]):
            index = sent[0,i]
            tmp_lrp_result = lrp_result_seq[0,i]
            lrp_tmp[0,index] = lrp_tmp[0,index] + tmp_lrp_result

        if np.mod(idx,1000) == 0: print("Sorts lrp result based on confusion")            
        lrp_per_token[idx, : ] = lrp_tmp
        if confusion[idx] == 1: # false positive # add the lrp to the correct bin dependent on the classification score 
            lrp_fp = lrp_fp + lrp_tmp
        elif confusion[idx] == -1: # false negative 
            lrp_fn = lrp_fn + lrp_tmp
        else: # sort rest if their target is 1 or 0
            if targets.iloc[idx]== 1:
                lrp_tp = lrp_tp + lrp_tmp
            else:
                lrp_tn = lrp_tn + lrp_tmp
        
    return lrp_tp , lrp_tn , lrp_fp , lrp_fn , lrp_per_token

##############################################################

##############################################################
def Get_lrp_outliers(lrp_data , sigma , pm, tokenizer):
    " gets the sigma outerliers from an lrp array where each index is the lrp score for the token with the same index in the tokenizer"
    "lrp_data: array containiing lrp score per token"
    "sigma: how many sigmas you wan to pick outliers from"
    "pm: do you want to look at + or - outliers? 1=> + , -1 => - "
    "tokenizer: the tokenizer used to tokenizer the sentences the LRP data comes from"
    
    mean = lrp_data.mean()
    std = lrp_data.std()
    
    if pm == 1:
        tmp_bool = lrp_data >= mean + (sigma * std)
        
    elif pm== -1:
        tmp_bool = lrp_data <= mean - (sigma * std)
    else:
        print("Invalid value of pm variable, pm is ether 1 or -1")
        print("Breaking function early")
        return
        
    # chech the size of tmp_bool and handle the array accordingly
    if len(tmp_bool) < 1:
        outliers_index = [i for i, x in enumerate(tmp_bool) if x ]
        
    else:
        outliers_index = [i for i, x in enumerate(tmp_bool[0]) if x ]
        
    if 0 in outliers_index: outliers_index.remove(0) # remove 0 if in list since index 0 is the padding index and does not exist in the tokenizer 
    outliers_word = [ tokenizer.index_word[index] for index in outliers_index ]
    
    return outliers_word , outliers_index
##############################################################

##############################################################
def Get_lemma_lrp(lrp_data , sigma, pm ,tokenizer ):
    lm = WordNetLemmatizer()
    lemma_t= []
    for i in range(1,tokenizer.num_words+1):
        # tmp.append(t.index_word[i])
        lemma_word = lm.lemmatize(tokenizer.index_word[i])
        if lemma_word not in lemma_t:
            lemma_t.append(lemma_word)
    
    lrp_lemmad_token = np.zeros([1,len(lemma_t)])
    for i , score in enumerate(lrp_data[0]):
        if i == 0:
            continue
        
        tmp_lemm = lm.lemmatize(tokenizer.index_word[i])
        for index , word in enumerate(lemma_t):
            if tmp_lemm == word:
                lrp_lemmad_token[0,index] = lrp_lemmad_token[0,index] + score

    lrp_lemmad_token_mean = lrp_lemmad_token.mean()
    lrp_lemmad_token_std = lrp_lemmad_token.std()

    if pm == 1:
        tmp_bool = lrp_lemmad_token >= lrp_lemmad_token_mean + sigma * lrp_lemmad_token_std 
        index_words = [i for i, x in enumerate(tmp_bool[0]) if x ]
        lemmad_words = [ lemma_t[index] for index in index_words]
    
    elif pm == -1: 
        tmp_bool = lrp_lemmad_token <= lrp_lemmad_token_mean - sigma * lrp_lemmad_token_std 
        index_words = [i for i, x in enumerate(tmp_bool[0]) if x ]
        lemmad_words = [ lemma_t[index] for index in index_words]


    return lemmad_words , index_words

##############################################################

##############################################################
" function takes in a keras model where the first layer is an embedding layer and split the model into two."
"The first returned model is the embedding layer and the second is the rest of the available layers"
"The function compiles the models such that they can be used."
def split_model(model):
    embedding_layer = model.layers[0]

    emb_mod = Sequential()
    emb_mod.add(embedding_layer)
    "the optimizer and the loss is irrelevant if the layer is not updated. But needed for the model to compile"
    emb_mod.compile(optimizer = 'Adam', loss = 'mean_squared_error')


    ## seperate model for everything else in the model
    model_rest = model.layers[1:]

    new_mod = Sequential()
    
    for lay in model_rest:
        new_mod.add(lay)
    
    "the optimizer and the loss is irrelevant if the layer is not updated. But needed for the model to compile"
    new_mod.compile(optimizer = 'Adam', loss = 'mean_squared_error')
    
    
    return emb_mod , new_mod
##############################################################
def Split_sentences(data):
    data_with_words_separated = []
    for sent in data: 
        # split the sentences into words 
        words = sent.split()
        # save split sencencs in a list 
        data_with_words_separated.append(words)
        
    return data_with_words_separated
        
def apply_word_filter_remove(data, target , filter):
    filtered_data = []
    filtered_target = []
    for i , sent in enumerate(data):
        filtered_sent = []

        for word in sent:
            if word in filter:
                continue
            else:
                filtered_sent.append(word)
        filtered_data.append(filtered_sent)
        filtered_target.append(target[i])
    return filtered_data , pd.Series(filtered_target)

def apply_word_filter_drop_word(data, target , filter, dr):
    filtered_data = []
    filtered_target = []
    for i , sent in enumerate(data):
        filtered_sent = []

        for word in sent:
            if word in filter and dr > random.random(): # drop word randomly  if seed is lower than dr 
                continue
            else:
                filtered_sent.append(word)
        filtered_data.append(filtered_sent)
        filtered_target.append(target[i])

    return filtered_data , pd.Series(filtered_target)

def apply_word_filter_drop_sent(data, target , filter, dr):
    filtered_data = []
    filtered_target = []
    removed_index= []
    drop_sent = False
    for i , sent in enumerate(data):
        for word in sent:
            if word in filter and dr > random.random(): # drop word randomly  if seed is lower than dr 
                drop_sent = True
                removed_index.append(i)
                break
            
        if  not drop_sent:
            filtered_data.append(sent)
            filtered_target.append(target[i])

        else:    
            drop_sent = False
    return filtered_data ,  pd.Series(filtered_target) , removed_index

###################################################
"fetches stopwords from txt file from the same folder which this file is in "
def get_stop_words():
    read_tmp = []
    with open("stop_words.txt", "r") as file:
      for line in file:
        read_tmp.append(line.strip())
    stop_words = np.array(read_tmp)
    return stop_words
###################################################

###################################################
"uses get_stop_words to remove those words from a list of words"
def remove_stopwords(data):
    stop_words = get_stop_words()
    data = [w for w in data if w not in stop_words]
    return data
###################################################

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


# borrowed function for plotting a heatmap with the relevancy scores, from iNNvestigate github repo
def plot_text_heatmap(words, scores, title="", width=25.5, height=0.9, verbose=0, max_word_per_line=20, savefig = 0):
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
    # loc_y = -0.2
    loc_y = 0.3
    loc_x = 0.01
    for i, token in enumerate(tokens):
        *rgb, _ = cmap.to_rgba(normalized_scores[i], bytes=True)
        color = '#%02x%02x%02x' % tuple(rgb)
        
        text = ax.text(loc_x, loc_y, token,
                       fontsize=30,
                       bbox={
                           'facecolor': color,
                           'pad': 5.0,
                           'linewidth': 1,
                           'boxstyle': 'round,pad=0.4'
                       }, transform=t)

        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()
        
        # create a new line if the line exceeds the length
        if (i+1) % max_word_per_line == 0:
            loc_y = loc_y + 5
            t = ax.transData
        else:
            t = transforms.offset_copy(text._transform, x=ex.width+60 , units='dots')

    if verbose == 0:
        ax.axis('off')
        
    if savefig:
        print('saving:' , title)
        path = 'C:/Users/Fredrik Möller/Documents/MPSYS/Master_thesis/raport figs/'
        plt.savefig(path+title+'.jpg', format='jpg')
    return fig 




