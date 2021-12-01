# -*- coding: utf-8 -*-

#autoencoder.compile(optimizer='adam',loss='mse')
#autoencoder.compile(optimizer='adam',loss='binary_crossentroy')

##### Auto Encoder 
# 1.导库
import keras
from keras.layers import Dense,Activation,Input
from keras.models import  Sequential
import matplotlib.pyplot as plt
from keras import regularizers

# 2.载入数据
def load_dataset():
    (X_train,_),(X_test,_) = keras.datasets.mnist.load_data()
    X_train = X_train/255
    X_test = X_test/255
    return X_train,X_test
X_train,X_test = load_dataset()

# 3.数据前处理
X_train_flatten = X_train.reshape((X_train.shape[0],-1))
X_test_flatten = X_test.reshape((X_test.shape[0],-1))
'''
# 4.搭建网络
input_size = 784
hidden_size = 128
code_size = 32
model = Sequential()
model.add(Dense(hidden_size,input_shape=(input_size,),activation='relu'))
model.add(Dense((code_size),activation='relu'))
model.add(Dense(hidden_size,activation='relu'))
model.add(Dense(input_size,activation='relu'))
model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train_flatten,X_train_flatten,epochs=5,validation_data=(X_test_flatten,X_test_flatten))

# 5.结果预测
reconstructed = model.predict(X_test_flatten)
plt.figure
plt.subplot(211)
plt.imshow(X_test[10],cmap='Greys')
plt.subplot(212)
plt.imshow(reconstructed[10].reshape(28,28),cmap='Greys')
plt.show()
'''
### 7.4
code_size = 64
input_size = 784
model = Sequential()
model.add(Dense(code_size,input_shape=(input_size,),activation='relu',activity_regularizer=regularizers.l1(10e-05)))
model.add(Dense(input_size,activation='sigmoid'))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train_flatten,X_train_flatten,epochs=5,validation_data=(X_test_flatten,X_test_flatten))
reconstructed = model.predict(X_test_flatten)
plt.figure
plt.subplot(211)
plt.imshow(X_test[10],cmap='Greys')
plt.subplot(212)
plt.imshow(reconstructed[10].reshape(28,28),cmap='Greys')
plt.show()














































