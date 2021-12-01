# -*- coding: utf-8 -*-
### 去噪AE 
# 1.导库
import  keras
from keras.layers import Dense,Activation,Input
from keras.models import  Sequential
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model,save_model
from keras.utils import plot_model

# 2.载入数据
def load_datadset():
    (X_train,_),(X_test,_) = keras.datasets.mnist.load_data()
    X_train = X_train/255
    X_test = X_test/255
    return X_train,X_test
X_train,X_test = load_datadset()

# 3.数据前处理
X_train_flatten = X_train.reshape((X_train.shape[0],-1))
X_test_flatten = X_test.reshape((X_test.shape[0],-1))
noise_factor = 0.5
X_train_noise = X_train_flatten + noise_factor*np.random.normal(loc=1.0,scale=1.0,size=X_train_flatten.shape)
x_test_nosie = X_test_flatten + noise_factor*np.random.normal(loc=1.0,scale=1.0,size=X_test_flatten.shape)
X_train_noise = np.clip(X_train_noise,0.,1.)
X_test_noise = np.clip(x_test_nosie,0.,1.)

n = 10
plt.figure(figsize=(20,2))
for i in range(1,n):
    ax = plt.subplot(1,n,i)
    plt.imshow(X_test_noise[i].reshape(28,28),cmap='Greys')
plt.show()

# 4.搭建网络
input_size = 784
hidden_size = 128
code_size = 32

model = Sequential()
model.add(Dense(hidden_size,input_shape=(input_size,),activation='relu'))
model.add(Dense(hidden_size,activation='relu'))
model.add(Dense(hidden_size,activation='relu'))
model.add(Dense(input_size,activation='sigmoid'))
#在原始的新模型搭建的时候增加的模型重新训练对比实验结果 
model.add(Dense(input_size,activation='sigmoid'))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X_train_noise,X_train_flatten,epochs=10,validation_data=(X_test_noise,X_test_flatten))
#model.save(r'Testsave\test.h5')

#model = load_model(r'Testsave\test2.h5')
#model.add(Dense(input_size,activation='sigmoid'))
#model.fit(X_train_noise,X_train_flatten,epochs=10,validation_data=(X_test_noise,X_test_flatten))
plot_model(model,to_file='Saved_mode2.png',show_shapes=True)

# 5.预测
reconstructed = model.predict(X_test_noise)
n = 10
plt.figure(figsize=(20,2))
for i in range(1,n):
    ax = plt.subplot(2,n,i)
    plt.imshow(X_test_noise[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(2,n,i+n)
    plt.imshow(reconstructed[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()



























