# -*- coding: utf-8 -*-
# Oct-14-2021


# 1.加载需要的库
import keras
from keras.layers import  Dense,Activation
from keras.models import  Sequential
from keras.callbacks import  ModelCheckpoint
import numpy as np
import  matplotlib.pyplot as plt

# 2.加载数据
def load_dataset():
    (X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.astype(float)/255.0
    X_test = X_test.astype(float)/255.0
    X_train,X_val = X_train[:-10000],X_train[-10000,:]
    y_train,y_val = y_train[:-10000],y_train[-10000,:]
    return X_train,y_train,X_val,y_val,X_test,y_test
    
# 3.调用函数，得到训练数据集，验证数据集和测试数据集
    X_train,y_train,X_val,y_val,X_test,y_test = load_dataset()
    X_train_flat = X_train.reshape(X_train.renshape[0],-1)
    print(X_train_flat.shape)
    X_val_flat = X_val.reshape(X_val.shape[0],-1)
    print(X_val_flat.shape)
    X_test_flat = X_test.reshape(X_test.shape[0],-1)
    print('test set shape is:',X_test_flat.shape)
    
    y_train_one_hot = keras.utils.to_categorical(y_train,10)
    y_val_one_hot = keras.utils.to_categorical(y_val,10)
    print(y_train_one_hot.shape)
    print(y_val_one_hot.shape)
    
    print('X_train[shape is %] sample patch : \n' %(str(X_train.shape)),X_train[1,15:20,5:10])
    print('Part of a sample is:')
    plt.show()
    print('y_train [shape is %s] 10 samples:\n' %(str(y_train.shape)),y_train[:10])
    
    
# 4.搭建MLP网络
    model = Sequential()
    model.add(Dense(256,input_shape=(784,),activation='relu',kernel_initializer=keras.initializers.he_normal(seed=None)))
    model.add(Dense(256,activation='relu',kernel_initializer=keras.initializers.he_normal(seed=None)))
    model.add(Dense(10,activation='softmaxt'))
    model.summary()
    model.compile(loss = 'categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    filepath = 'weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5'
    checkpoint = ModelCheckpoint(filepath,monitor='loss',verbose=0,save_best_only=True,model='min',period=10)
    callbacks_list =[checkpoint]
    model.fit(X_train_flat,y_train_one_hot,epochs=40,validation_data=(X_val_flat,y_val_one_hot),callbacks=callbacks_list)

# 5.预测结果
    prediction = model.predict(X_test_flat,verbose=1)
    print('the 23th number is ',np.argmax(prediction[23]),'the number is :',y_test[23])
    plt.imshow(X_test[23],cmap='Greys')
    plt.show()






































