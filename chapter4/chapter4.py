# -*- coding: utf-8 -*-
# Oct-14-2021
from keras.models import Sequential

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

model = Sequential()
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

model.fit(X_train_flat,y_train_one_hot,epochs=40,validation_data=(X_val_flat,y_val_one_hot),callbacks=callbacks_list)
# validation_data,表示比较准确率 1

# keras中正则化的使用 
from keras import regularizers
model.add(Dense(128,input_shape=(784,),kernel_regularizer=regularizers.l2(0.1))
          
from keras.layers import Dropout
model.add(Dropout(0.2))

from keras.layers import BatchNormalization
from keras.layers import Dense
model.add(Dense(128,activation='relu'))
model.add(Dense(128,use_bias=False))
model.add(BatchNormalization())
model.add(activation('relu'))

from keras import optimizers
optimizer = optimizers.Adagrad(learning_rate=0.01)

from keras import  optimizers
optimizer = optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,asmgrad=False)

from keras import  optimizers
optimizer = optimizers.rmsprop(learning_rate=0.001,rho=0.9)

from keras import  optimizers
optimizer = optimizers.SGD(learning_rate=0.01,momentum=0.9,nesterov=True)