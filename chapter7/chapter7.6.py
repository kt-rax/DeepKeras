# -*- coding: utf-8 -*-
### 上色器
# 1.导库
from keras.layers import Dense,Input
from keras.layers import Conv2D,Flatten,Reshape,Conv2DTranspose
from keras.models import  Model
from keras.utils import  plot_model
from keras.callbacks import  ReduceLROnPlateau,ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import os
from  keras import  backend as K
from keras.datasets import cifar10

# 2.载数据
(X_train,_),(X_test,_) = cifar10.load_data()
img_rows = X_train.shape[1]
img_cols = X_train.shape[2]
channels = X_train.shape[3]

# 3.数据前处理
def rgb2gray(rgb):
    return np.dot(rgb[...,:3],[0.2126,0.7152,0.0722])

# 4.预览彩色图像与灰度图像 
color_images = X_test[:100]
color_images = color_images.reshape((10,10,X_test.shape[1],X_test.shape[2],X_test.shape[3]))
color_images = np.vstack([np.hstack(i) for i in color_images])
plt.figure()
plt.axis('off')
plt.title('Color images')
plt.imshow(color_images)
plt.show()

gray_images = rgb2gray(X_test[:100])
gray_images = gray_images.reshape((10,10,X_test.shape[1],X_test.shape[2]))
gray_images = np.vstack([np.hstack(i) for i in gray_images])
plt.figure()
plt.axis('off')
plt.title('gray images')
plt.imshow(gray_images,cmap='Greys')
plt.show()

# 5.数据前处理
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255
X_train_gray = rgb2gray(X_train)
X_test_gray = rgb2gray(X_test)
X_train_gray = X_train_gray.astype('float32')/255
X_test_gray = X_test_gray.astype('float32')/255
X_train_gray = X_train_gray.reshape((X_train_gray.shape[0],X_train.shape[1],X_train.shape[2],1))
X_test_gray = X_test_gray.reshape((X_test_gray.shape[0],X_test.shape[1],X_test.shape[2],1))
X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2],3))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],3))


# 6.参数设定
input_shape = (X_train.shape[1],X_train.shape[2],1)
batch_size = 32
kernel_size = 3
latent_dim = 256

# 7.搭建模型
inputs = Input(shape=input_shape,name='encoder_input')
x = inputs
layer_filters = [64,128,256]
for filters in layer_filters:
    x = Conv2D(filters = filters,kernel_size=kernel_size,strides=2,activation='relu',padding='same')(x)
shape = K.int_shape(x)
x = Flatten()(x)
latent = Dense(latent_dim,name='latent_vector')(x)
#ValueError: Graph disconnected: cannot obtain value for tensor Tensor("encoder_input_9:0", shape=(?, 32, 32, 1), dtype=float32) at layer "encoder_input". The following previous layers were accessed without issue: []
encoder = Model(inputs,latent,name='encoder')
encoder.summary()
latent_inputs = Input(shape=(latent_dim,),name='decoder_input')

x = Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
x = Reshape((shape[1],shape[2],shape[3]))(x)
for filters in layer_filters[::-1]:
    x = Conv2DTranspose(filters=3,kernel_size=kernel_size,strides=2,activation='relu',padding='same')(x)
outputs = Conv2DTranspose(filters=channels,kernel_size=kernel_size,activation='sigmoid',padding='same',name='decoder_output')(x)

decoder = Model(latent_inputs,outputs,name='decoder')
decoder.summary()
autoencoder = Model(inputs,decoder(encoder(inputs)),name='autoencoder')
autoencoder.summary()

plot_model(decoder,to_file='decoder.png',show_shapes=True)
plot_model(encoder,to_file='encoder.png',show_shapes=True)
save_dir = os.path.join(os.getcwd(),'saved_models')
model_name = 'colorized_ae_model'


if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
filepath = os.path.join(save_dir,model_name)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),cooldown=0,patience=5,verbose=1,min_lr=0.5e-6)
checkpoint = ModelCheckpoint(filepath=filepath,monitor='val_loss',verbose=1,save_best_only=True)
autoencoder.compile(loss='mse',optimizer='adam')
callbacks = [lr_reducer,checkpoint]
autoencoder.fit(X_train_gray,X_train,validation_data=(X_test_gray,X_test),epochs=1,batch_size=batch_size,callbacks=callbacks)


# 使用保存的模型继续训练




# 8.显示彩色图像
'''
X_decoded = autoencoder.predict(X_test_gray)
imgs = X_decoded[:100]
imgs = imgs.reshape((10,10),imags.shape[0],imgs.shape[1],channels))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Colorized test images(Predicted)')
plt.imshow(imgs,interpoloation='none')
plt.show()

'''





































