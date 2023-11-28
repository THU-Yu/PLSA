import pandas as pd
import re
from copy import deepcopy
import json as js
import numpy as np
data = pd.read_csv(r'Task-Corpus.csv')
# print(type(data['ABSTRACT'].to_list()))
document_abstract = data['ABSTRACT'].to_list()
N = len(document_abstract)
with open("stopwords.txt", "r", encoding='utf-8') as f:
    stopword = f.read()
stopword_list = stopword.split('\n')
word_list = []
word_index_dictionary = {}
current_index = 0
formula = stopword_list[891:]
formula.append('\"')
document_abstract_after_word_segmentation = []
for i in range(N):
    # Replace special characters and mathematical formulas in the abstract
    abstract = re.sub('[0-9]|[\u00e0-\u9fa5]|-|\.|\(|\)|,|\'', ' ', document_abstract[i])
    abstract = abstract.split()
    # if i == 448:
    #     print(abstract)
    new_abstract = []
    for word in abstract:
        # delete word length less than 3
        if len(word) < 3:
            continue
        # delete word in stopwords list
        if word in stopword_list:
            continue
        # delete word include math symbol
        delete_word = False
        for symbol in formula:
            if symbol in word:
                delete_word = True
                break
        if delete_word:
            continue
        new_abstract.append(word)
        if word not in word_list:
            word_list.append(word)
            word_index_dictionary.update({word: current_index})
            current_index += 1
    # if i == 0:
        # print(new_abstract)
        # print(word_index_dictionary)
    document_abstract_after_word_segmentation.append(deepcopy(new_abstract))

print(len(word_list))
with open('dictionary.json', 'w') as f:
    f.write(js.dumps({'word_list': word_list}))
document_word_num_list = []
for i in range(N):
    freq = np.zeros(len(word_list))
    abstract = document_abstract_after_word_segmentation[i]
    for word in abstract:
        freq[word_index_dictionary[word]] += 1
    document_word_num_list.append(deepcopy(freq))
document_word_num_list = np.array(document_word_num_list)
print(document_word_num_list.shape)
np.save('document_word_num_list.npy', document_word_num_list)