import json as js
import numpy as np
from EM import EM

with open('dictionary.json', 'r') as f:
    string = f.read()
    data = js.loads(string)
word_list = data['word_list']
Z = 4
W = len(word_list)
max_step = 10
epsilon = 1e-2
try_times = 5
document_word_num_list = np.load('document_word_num_list.npy')
Theta, Lambda = EM(document_word_num_list, W, Z, max_step, epsilon, try_times)
np.save('Theta.npy', Theta)
np.save('Lambda.npy', Lambda)