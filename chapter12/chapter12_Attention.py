# -*- coding: utf-8 -*-
# 1.导库
from random import randint
import numpy as np
from numpy import array,argmax,array_equal
from tensorflow.keras.layers import LSTM,Dense
from custom_recurrents import AttentionDecoder
from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed

# https://github.com/datalogue/tensorflow.keras-attention/tree/master/models  custom_reccurrents.py tdd.py

# 2.创建输入数据
def generate_sequence(length,n_unique):
    return [randint(0,n_unique-1) for _ in range(length)]

def one_hot_encode(sequence,n_unique):
    encoding = list()
    for value in sequence:
        vector = [0 for _ in range(n_unique)]
        vector[value] = 1
        encoding.append(vector)
    return array(encoding)

def get_pair(n_in,n_out,vocab_size):
    sequence_in = generate_sequence(n_in,vocab_size)
    sequence_out = sequence_in[:n_out] + [0 for _ in range(n_in - n_out)]
    sequence_out = [element // 2 for element in sequence_out]
    X = one_hot_encode(sequence_in,vocab_size)
    y = one_hot_encode(sequence_out,vocab_size)
    X = X.reshape((1,X.shape[0],X.shape[1]))
    y = y.reshape((1,y.shape[0],y.shape[1]))
    return X,y

def one_hot_decode(encoded_seq):
    return [argmax(vector) for vector in encoded_seq]
    
# 3.设置超参
n_features = 50 
n_timesteps_in = 6
n_timesteps_out = 3

# 4.定义模型
model = Sequential()
model.add(LSTM(150,input_shape=(n_timesteps_in,n_features),return_sequences=True))
model.add(AttentionDecoder(150,n_features))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# 5.训练模型
for epoch in range(5000):
    X,y = get_pair(n_timesteps_in,n_timesteps_out,n_features)
    model.fit(X,y,epoch=1,verbose=2)


# 6.模型预测与性能评估
total,correct = 100,0
for _ in range(total):
    X,y = get_pair(n_timesteps_in,n_timesteps_out,n_features)
    yhat = model.predict(X,verbose=0)
    if array_equal(one_hot_decode(y[0]),one_hot_decode(yhat[0])):
        correct += 1
print('accuracy: %.2f%%'%(float(correct)/float(total)*100.0))

'''
###### 
for _ in range(10):
    X,y = get_pair(n_timesteps_in,n_timesteps_out,n_features)
    yhat = model.predict(X,verbose=0)
    print('input is ',one_hot_decode(X[0]),'Expected: ',one_hot_decode(y[0]),'predicted: ',one_hot_decode(yhat[0]))
    
    
model = Sequential()
model.add(LSTM(150,input_shape=(n_timesteps_in,n_features)))
model.add(RepeatVector(n_timesteps_in))
model.add(LSTM(150,return_sequences=True))
model.add(TimeDistributed(Dense(n_features,activation='softmax')))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
'''    




































