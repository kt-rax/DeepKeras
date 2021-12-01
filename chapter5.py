# -*- coding: utf-8 -*-
'''
from keras.datasets import  cifar10
(X_train,y_train),(X_test,y_test) = cifar10.load_data()


from keras.layers import  Conv2D
from keras.models import  Sequential

model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu',input_shape=(rows,columns,3)))
model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='valid',activation'relu'))


##

from keras.layers import MaxPooling2D
MaxPooling2D(pool_size=((2,2),strides=None,padding='valid')


################    黑白手写数字识别           
# 1.载入需要的库 
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras import  backend as K 
import matplotlib.pyplot as plt
from keras.utils import plot_model
 
# 2.设置参数
batch_size= 4 
num_classes = 10
epochs = 10
rows,cols = 28,28  

# 3.载入数据
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0],rows,cols,1)
x_test = x_test.reshape(x_test.shape[0],rows,cols,1) 
input_shape = (rows,cols,1)

# 4.数据前处理
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
# 把标签变成one-hot向量 
y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)

# 5.网络搭建
model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(2,3),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))
plot_model(model,to_file='./mnist.jpg',show_shapes=True)

# 6.模型编译与拟合
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
xxx =model.fit(x_train,y_train,batch_size= batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test))

plt.figure
plt.plot(xxx.history['acc'])
plt.plot(xxx.history['val_acc'])
plt.xlabel('epochs')
plt.ylabel('acc')
plt.title('train & test acc')
plt.legend(['train acc','val_acc'],loc='lower right')
plt.savefig('Mniacc.jpg')
plt.close()

plt.figure
plt.plot(xxx.history['loss'])
plt.plot(xxx.history['val_loss'])
plt.title('train & test loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train loss','test loss'],loc='upper right')
plt.savefig('Mniloss.jpg')
plt.close()


'''
###############   彩色图像分类
# 1.载入需要的库
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D,Dense,Dropout,Activation,Flatten,MaxPooling2D
import matplotlib.pylab as plt
import numpy as np
from keras.utils import plot_model

# 2.设置参数 
batch_size = 16
num_classes = 20
epochs = 10

# 3.载入数据
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
print('x_trian is shape:',x_train.shape)

# 4.数据前处理
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)

# 5.网络搭建
model = Sequential()
model.add(Conv2D(32,(3,3),padding='same',input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
plot_model(model,to_file='./Cifar10.png',show_shapes=True)

# 6.模型编译与拟合
model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
xxx=model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,validation_data=(x_test,y_test),verbose=1,shuffle=True)

# 7.预测实验
cifar10_labels = np.array(['airplen','automobile','bird','cat','deer','dag','frog','horse','ship','truck'])

x_input = np.array(x_test[3]).reshape(1,32,32,3)
prediction = model.predict(x_input)

print(cifar10_labels[np.argmax(prediction)])

plt.figure
plt.plot(xxx.history['acc'])
plt.plot(xxx.history['val_acc'])
plt.xlabel('epochs')
plt.ylabel('acc')
plt.title('train & test acc')
plt.legend(['train acc','val_acc'],loc='lower right')
plt.savefig('Cifar10acc.jpg')
plt.close()

plt.figure
plt.plot(xxx.history['loss'])
plt.plot(xxx.history['val_loss'])
plt.title('train & test loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train loss','test loss'],loc='upper right')
plt.savefig('Cifar10loss.jpg')
plt.close()
































