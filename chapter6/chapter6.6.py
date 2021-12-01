# -*- coding: utf-8 -*-
# 1.导入库
from keras.layers import GRU,LSTM,Dense
from keras.layers.embeddings import Embedding

from keras.models import Sequential
import numpy as np
from keras.datasets import  imdb
from keras.preprocessing import sequence

'''
model =Sequential()

model.add(GRU(units=128,input_shape=(input.shape[1],input.shape[2]),activation='relu'))

model.add(LSTM(...,return_sequences=True,input_shape=()))
model.add(LSTM(...,return_sequences=True))
model.add(LSTM(...,return_sequences=True))
model.add(LSTM(...,return_sequences=True))
model.add(Dense())
model.add(Embedding(input_dim=1000,output_dim=64,input_length=10))

'''

# 2.载入数据
vocab_size = 5000
(X_train,y_train),(X_test,y_test) = imdb.load_data(num_words=vocab_size)

# 3.限制序列的长度
review_length_max = 500
X_train = sequence.pad_sequences(X_train,maxlen=review_length_max)
X_test= sequence.pad_sequences(X_test,maxlen=review_length_max)
embedding_length = 32

# 4.网络搭建
model = Sequential()
model.add(Embedding(vocab_size,embedding_length,input_length=review_length_max))
model.add(LSTM(128))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)