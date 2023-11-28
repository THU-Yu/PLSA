import numpy as np
import json as js
import pandas as pd
with open('dictionary.json', 'r') as f:
    string = f.read()
    data = js.loads(string)
word_list = data['word_list']
Theta = np.load('Theta.npy')
Lambda = np.load('Lambda.npy')
Z, W = Theta.shape
data = None
for i in range(Z):
    theta = Theta[i].tolist()
    word_with_theta = [list(t) for t in zip(word_list, theta)]
    word_with_theta_sorted = sorted(word_with_theta, key=lambda x: x[1], reverse=True)
    if i == 0:
        data = pd.DataFrame(np.array(word_with_theta_sorted), columns=['Topic%d_Word'%(i+1), 'Topic%d_Probability'%(i+1)])
    else:
        data = pd.concat([data, pd.DataFrame(np.array(word_with_theta_sorted), columns=['Topic%d_Word'%(i+1), 'Topic%d_Probability'%(i+1)])], axis=1, sort=False)
data.to_csv("Theta_sorted.csv",index=False,sep=',')
lambda_data = pd.DataFrame(Lambda.reshape((1,4)), columns=['Topic1_Probability', 'Topic2_Probability', 'Topic3_Probability', 'Topic4_Probability'])
lambda_data.to_csv("Lambda.csv",index=False,sep=',')