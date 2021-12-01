# -*- coding: utf-8 -*-
# 1.导库
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,Activation,Add,Dense,AveragePooling2D,Flatten,BatchNormalization
from tensorflow.keras.utils import  plot_model
from tensorflow.keras.models import  Model
from kt_package.Personal_module import insert_layers_nonseq 

'''
# 2.搭建Identity Block
def identity_block(filters,X):
    X_shortcut = X
    X = Conv2D(filters=filters,kernel_size=(3,3),padding='same')(X)
    X = BatchNormalization(axis=-1)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=filters,kernel_size=(3,3),padding='same')(X)
    X = BatchNormalization(axis=-1)(X)
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    return X

# 3.搭建Convolutional Block
def Convolutional_block(filters,X):
    X_shortcut = X
    X = Conv2D(filters=filters,kernel_size=(3,3))(X)
    X = BatchNormalization(axis=-1)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=filters,kernel_size=(3,3))(X)
    X = BatchNormalization(axis=-1)(X)
    X_shortcut = Conv2D(filters=filters,kernel_size=(1,1),strides=(1,1))(X_shortcut)
    X = BatchNormalization(axis=-1)(X_shortcut)
    X = Activation('relu')(X)
    return X

# 4.搭建34层的RestNet
def RestNet(input_shape,num_classes):
    X_input = Input(input_shape)
    X = Conv2D(filters=64,kernel_size=(7,7))(X_input)
    X = BatchNormalization(axis=-1)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(3,3),strides=(2,2))(X)
    X = Convolutional_block(64,X)
    X = identity_block(64,X)
    X = identity_block(64,X)
    X = Convolutional_block(128,X)
    X = identity_block(128,X)
    X = identity_block(128,X)
    X = identity_block(128,X)
    X = Convolutional_block(256,X)
    X = identity_block(256,X)
    X = identity_block(256,X)
    X = identity_block(256,X)
    X = identity_block(256,X)
    X = identity_block(256,X)
    X = Convolutional_block(512,X)
    X = identity_block(512,X)
    X = identity_block(512,X)
    X = AveragePooling2D(pool_size=(2,2),padding='same')(X)
    X = Flatten()(X)
    X = Dense(units = num_classes,activation='softmax')(X)
    model = Model(inputs = X_input,outputs=X)
    model.summary()
    plot_model(model,to_file='Restnet.png',show_shapes=True,show_layer_names=True)
    return model

# 5.模型编译
input_shape = (300,300,3)
num_classes = 100
model = RestNet(input_shape=input_shape,num_classes=num_classes)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
'''
# -*- coding: utf-8 -*-

import re
from tensorflow.keras.models import Model
    
### 用Keras快速搭建ResNet 
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import  image
from tensorflow.keras.applications.resnet50 import  preprocess_input,decode_predictions
import numpy as np
from tensorflow.keras.layers import Dropout

model = ResNet50(weights='imagenet')
model.summary()
plot_model(model,to_file='Restnet_BeforeInsert.png',show_shapes=True,show_layer_names=True)
#model_t = model

#model = model

def dropout_layer_factory():
    return Dropout(rate=0.2,name='Dropout')

model_t = insert_layers_nonseq(model,'.*activation.*',dropout_layer_factory)

model_t.summary()
plot_model(model_t,to_file='Restnet_AterInsert.png',show_shapes=True,show_layer_names=True)


img_path = './people.jpg'
img = image.load_img(img_path,target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)
x = preprocess_input(x)
y_pred = model.predict(x)
print(decode_predictions(y_pred,top=3)[0])

y_pred_ = model_t.predict(x)
print(decode_predictions(y_pred_,top=3)[0])
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    