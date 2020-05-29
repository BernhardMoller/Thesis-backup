# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:00:52 2020

@author: Fredrik Möller
"""

import fun_c as f 
import numpy as np
import matplotlib.pyplot as plt
import keras as ks
# import seaborn as sns


data_train , data_test = f.train_test_data(reduce_pad = reduce_padding)

X_train = data_train['fragment']
X_test = data_test['fragment']

y_train = data_train['target']
y_test = data_test['target']


# what to conduct all tests on 
X = X_train
y = y_train

X_split = f.Split_sentences(X)
X_split = f.remove_punct(X_split)



max_features = 5000

t = ks.preprocessing.text.Tokenizer(num_words = max_features)
t.fit_on_texts(X_split)


X_sequence = t.texts_to_sequences(X_split)


X_matrix = t.texts_to_matrix(X_split)


sum_word_occurence = np.sum(X_matrix,0)

nr_of_words_to_plot = 1000
plt.figure()
plt.plot(sum_word_occurence[1:nr_of_words_to_plot]/len(X))
plt.xlabel("Index in tokenzer library")
plt.ylabel("% Occurence frequency in all classes")
x_max= nr_of_words_to_plot
plt.axis([-100*0.1, x_max*1.1, -0.05, 1])
plt.show()

#%%

# check class belonging of word indexes. for each occurence of the index. Can be multiple times per sentence
belong_neg = np.zeros((1,max_features))
belong_pos = np.zeros((1,max_features))

for i in range(len(X_sequence)):
    sent = X_sequence[i]
    target = y[i]

    if target == 0: # if sentence belongs to negative, calc nr of occurences 
        for index in sent:
            belong_neg[0,index] = belong_neg[0,index] + 1
    else: # if sentence belongs to positive class, calc occurence 
        for index in sent:
            belong_pos[0,index] = belong_pos[0,index] + 1
    
# values over 1 indicate that the word occure more than once per sentence 
belong_neg_normalized = belong_neg/(len(y)-y.sum())
belong_pos_normalized = belong_pos / y.sum() 

# sorted indexes from smallest to largest
belong_neg_index = belong_neg.argsort()
belong_pos_index = belong_pos.argsort()


plt.figure()
tmp_plot = belong_pos_normalized[0,belong_pos_index]
plt.scatter(y=tmp_plot[0] , x = belong_pos_index[0] , s = 2)
plt.ylabel("% Occurence frequency in positive class")
plt.xlabel("Tokenizer index")
tmp_plot = belong_neg_normalized[0,belong_neg_index]
plt.scatter(y=tmp_plot[0] , x = belong_neg_index[0] , s = 1)
plt.ylabel("% Occurence frequency in negative class")
plt.xlabel("Tokenizer index")
plt.close()

# take the two occurency vectors and negate one from the other. 
# negative values indicate that the word is more frequent in the negative classification class and positive values the opposite
occurency = belong_pos - belong_neg
occurency_index = occurency.argsort()


word_deviation = occurency.std()
word_mean = occurency.mean()

# sort out the words that have a skewed distrubution of +- 2 sigma 
indexes_skewed = []
skewed_words = []
nr_sigmas = 2 


tmp_bool = occurency <= word_mean - nr_sigmas * word_deviation 
less_than_sigma = [i for i, x in enumerate(tmp_bool[0]) if x]
tmp_bool = occurency >= word_mean + nr_sigmas * word_deviation 
more_than_sigma = [i for i, x in enumerate(tmp_bool[0]) if x]
          
skewed_indexes = less_than_sigma + more_than_sigma

pos_words = [ t.index_word[index] for index in more_than_sigma ]
neg_words = [ t.index_word[index] for index in less_than_sigma ]


skewed_words = [ t.index_word[index] for index in skewed_indexes ]
###

# pack word and comp score neatly 
freq_result = []
for i in range(len(skewed_indexes)):
    tmp_result = (skewed_words[i] , occurency[0,skewed_indexes[i]])
    freq_result.append(tmp_result)
#####

plt.figure()
fig, ax = plt.subplots()

ax.scatter(y=occurency , x = range(len(occurency[0])) , s = 1, label = 'f-diff inliers' )
ax.scatter(y = occurency[0,skewed_indexes] , x = skewed_indexes, s = 1 , c= 'red', label =  'f-diff 2 sigma outliers')

ax.set_ylabel('(f + class)- (f - class)')
ax.set_xlabel('Tokenizer index')
ax.set_title('Comparative occurence frequency between + & - class')
ax.legend()


tmp_skewed = skewed_words
tmp_skewed.sort()
with open("filter_words_freq.txt", "w") as file:
    for s in tmp_skewed:
        file.write(str(s) +"\n")



# plt.figure()
# sns.distplot(occurency)
# plt.ylabel("Dist")
# plt.xlabel("Comparative class frequency")


#%% test TF-IDF
# https://kavita-ganesan.com/extracting-keywords-from-text-tfidf/#.Xst37WgzaUk
from sklearn.feature_extraction.text import TfidfVectorizer

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results



# create tfidf vectorizer 
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
# fit vectorizer to corpus 
X_tfidf = vectorizer.transform(X)
# extract all feature names in alphabetical order 
feature_names = vectorizer.get_feature_names()

# create empty vectors to store the sum of tfidf scores for pos and neg class 
# tfidf_sum_pos = np.zeros([1,len(feature_names)])
# tfidf_sum_neg = np.zeros([1,len(feature_names)])

# # sum the tf-idf scores for each token in positive and negative class 
# for i in range(X_tfidf.shape[0]):
#     sent = X_tfidf[i]
#     target = y[i]
#     sorted_items=sort_coo(sent.tocoo())

#     if target == 0:
#         for tup in sorted_items:
#             index , score = tup
#             tfidf_sum_neg[0,index] = tfidf_sum_neg[0,index] + score
#     else:
#         for tup in sorted_items:
#             index , score = tup
#             tfidf_sum_pos[0,index] = tfidf_sum_pos[0,index] + score
tfidf_score_pos = np.zeros([1,len(feature_names)])
tfidf_score_neg = np.zeros([1,len(feature_names)])

for i in range(X_tfidf.shape[0]):
    sent = X_tfidf[i]
    sorted_items=sort_coo(sent.tocoo())
    if y[i] == 0:
        for tup in sorted_items:
            index , score = tup
            tfidf_score_neg[0,index] = tfidf_score_neg[0,index] + score
    else: 
        for tup in sorted_items:
            index , score = tup
            tfidf_score_pos[0,index] = tfidf_score_pos[0,index] + score
            
# get comparative tf-idf scores for 
# comp_tfidf = tfidf_sum_pos - tfidf_sum_neg
tfidf_std_neg = tfidf_score_neg.std()
tfidf_mean_neg = tfidf_score_neg.mean()

tfidf_std_pos = tfidf_score_pos.std()
tfidf_mean_pos = tfidf_score_pos.mean()

# get tf-idf scores that are +- N sigmas 
skewed_indexes_tfidf_neg = []
skewed_words_tfidf_neg = []

skewed_indexes_tfidf_pos = []
skewed_words_tfidf_pos = []

nr_sigmas = 2 

for i in range(len(tfidf_score_neg[0])):
    if tfidf_score_neg[0,i] >= tfidf_mean_neg + nr_sigmas * tfidf_std_neg:
        skewed_indexes_tfidf_neg.append(i)
        skewed_words_tfidf_neg.append(feature_names[i])

for i in range(len(tfidf_score_pos[0])):
    if tfidf_score_pos[0,i] >= tfidf_mean_pos + nr_sigmas * tfidf_std_pos:
        skewed_indexes_tfidf_pos.append(i)
        skewed_words_tfidf_pos.append(feature_names[i])

# pack word and comp score neatly 
# freq_result_tfidf = []
# for i in range(len(skewed_indexes_tfidf)):
#     tmp_result = [skewed_words_tfidf[i] , tfidf_score[0,skewed_indexes_tfidf[i]]]
#     freq_result_tfidf.append(tmp_result)

# plots 
# plt.figure()
# plt.scatter(y=tfidf_score , x = range(tfidf_score.shape[1]) , s = 1)
# plt.scatter(y = tfidf_score[0,skewed_indexes_tfidf] , x = skewed_indexes_tfidf, s = 1 , c= 'red')
# plt.ylabel("Culm Tf-idf score")
# plt.xlabel("Tf-idf tokenizer word index")
# plt.title('Culmative Tf-idf score for evaluated data set')
 
# neg class 
plt.figure()
fig, ax = plt.subplots()

ax.scatter(y=tfidf_score_neg , x = range(tfidf_score_neg.shape[1]) , s = 1 , label = 'Tf-idf inliers')
ax.scatter(y = tfidf_score_neg[0,skewed_indexes_tfidf_neg] , x = skewed_indexes_tfidf_neg, s = 1 , c= 'red', label = 'Tf-idf 2 sigma outliers')

ax.set_ylabel('Tf-idf score')
ax.set_xlabel('Tf-idf Tokenizer index')
ax.set_title('Tf-idf score of negative class')
ax.legend()

# pos class 
plt.figure()
fig, ax = plt.subplots()

ax.scatter(y=tfidf_score_pos , x = range(tfidf_score_pos.shape[1]) , s = 1 , label = 'Tf-idf inliers')
ax.scatter(y = tfidf_score_pos[0,skewed_indexes_tfidf_pos] , x = skewed_indexes_tfidf_pos, s = 1 , c= 'red', label = 'Tf-idf 2 sigma outliers')

ax.set_ylabel('Tf-idf score')
ax.set_xlabel('Tf-idf Tokenizer index')
ax.set_title('Tf-idf score of positive class')
ax.legend()


tmp_skewed = list(set(skewed_words_tfidf_neg + skewed_words_tfidf_pos))
tmp_skewed.sort()
with open("filter_words_tfidf.txt", "w") as file:
    for s in tmp_skewed:
        file.write(str(s) +"\n")

# plt.figure()
# sns.distplot(tfidf_score[0])
# plt.ylabel("Dist")
# plt.xlabel("Comparative Tf-idf score")

# keywords=extract_topn_from_vector(feature_names,sorted_items,10)

#%% check if any word are both in the regular frequency and tfidf score 

in_both=[]
for i in range(len(skewed_words_tfidf)):
    word = skewed_words_tfidf[i]
    if word in skewed_words:
        in_both.append(i)

print('words that are in both')
print([(skewed_words_tfidf[index]) for index in in_both])
print(len(in_both), 'of', len(skewed_words) , 'word from the regular frequency analysis occur in both +- 2 sigma comparisons')
print(len(in_both), 'of', len(skewed_words_tfidf) , 'word from the tfidf score analysis occur in both +- 2 sigma comparisons')



# #%% Save most frequent words as stopwords/ filters 
# filter_words = skewed_words # for pure freq words 


# # stop_words = skewed_words_tfidf # for pure tf-idf words 
# # write stop words 
# with open("filter_words.txt", "w") as file:
#     for s in filter_words:
#         file.write(str(s) +"\n")
        
# # read stop words 
# read_tmp = []
# with open("stop_words.txt", "r") as file:
#   for line in file:
#     read_tmp.append(line.strip())
# read_tmp = np.array(read_tmp)


    