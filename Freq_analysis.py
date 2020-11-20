# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:00:52 2020

@author: Fredrik Möller
"""

import fun_c as f 
import numpy as np
import matplotlib.pyplot as plt
import keras as ks
import seaborn as sns
max_features = 5000



read_tmp = []
with open("stop_words.txt", "r") as file:
  for line in file:
    read_tmp.append(line.strip())
stop_words = np.array(read_tmp)

data_train , data_test = f.train_test_data(reduce_pad = reduce_padding)

X_train = data_train['fragment']
# X_test = data_test['fragment']

# y_train = data_train['target']
# y_test = data_test['target']


# what to conduct all tests on 
X = X_train

# if only manually annotated data is wanted 
# X = X[index_train]
# # X.reset_index()
# X = X.reset_index(drop=True)
# # y_train.reset_index()
# y_train = y_train.reset_index(drop=True)


#%% calcualte NCOF 
X_sequence = t.texts_to_sequences(X_train_split)


words_pos , words_neg , pos_index , neg_index , skewed_words ,skewed_indexes = f.Get_NCOF(data = X_sequence, targets = y_train, nr_features = max_features, sigma = 3, tokenizer = t , plot_fig=True)



tmp_skewed = skewed_words
tmp_skewed.sort()

nr_pos = len(f.remove_stopwords(words_pos))
nr_neg = 0
labels= 'Positive' , 'Negative'
sizes = [nr_pos , nr_neg]
explode = [ 0 , 0]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.set_title('Distrubution of word class origin, NCOF filter')

plt.show()


with open("filter_words_freq.txt", "w") as file:
    for s in tmp_skewed:
        file.write(str(s) +"\n")

#%% test TF-IDF
# https://kavita-ganesan.com/extracting-keywords-from-text-tfidf/#.Xst37WgzaUk

words_unique_pos , words_unique_neg , words_intersection , words_difference = f.Get_tfidf(data = X, targets = y_train , nr_features = max_features , nr_sigmas = 3, plot_fig = True)
 #%%
nr_pos = len(f.remove_stopwords(words_unique_pos))
nr_neg = len(f.remove_stopwords(words_unique_neg))
nr_intersection = len(f.remove_stopwords(words_intersection))

labels= 'Positive' , 'Negative' , 'Common to both classes'
sizes = [nr_pos , nr_neg , nr_intersection]
explode = [ 0 , 0 , 0]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.set_title('Distrubution of outlier origin TF-IDF')
plt.show()


nr_pos_1 = len(f.remove_stopwords(words_unique_pos))
nr_neg_1 = len(f.remove_stopwords(words_unique_neg))

labels= 'Positive' , 'Negative'
sizes = [nr_pos , nr_neg]
explode = [ 0 , 0]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.set_title('Distrubution of word class origin, TF-IDF filter')
plt.show()



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





    