# -*- coding: utf-8 -*-
# 1.导库
from keras.layers import Conv2D,MaxPooling2D,Input
from keras.utils.vis_utils import plot_model
from keras.models import Model
import keras
from kt_package.Personal_module import print_time

# 2.搭建Inception模块:不能使用Sequential()
input_layer = Input(shape=(28,28,192))
branch1 = Conv2D(filters=64,kernel_size=(1,1),padding='same',activation='relu')(input_layer)
branch2 = Conv2D(filters=16,kernel_size=(1,1),padding='same',activation='relu')(input_layer)
branch2 = Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu')(branch2)
branch3 = Conv2D(filters=16,kernel_size=(1,1),padding='same',activation='relu')(input_layer)
branch3 = Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu')(branch3)
branch4 = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same')(input_layer)
branch4 = Conv2D(filters=16,kernel_size=(1,1),padding='same',activation='relu')(branch4)
Inception_output = keras.layers.concatenate([branch1,branch2,branch3,branch4],axis=3)

model = Model(inputs=input_layer,outputs=Inception_output,name='incepition_model')
model.summary()
plot_model(model,to_file='inception.jpg',show_shapes=True,show_layer_names=True)

### 使用keras快速搭建Inception V3模型 
from keras.applications.inception_v3 import  InceptionV3
from keras.preprocessing import  image
from keras.applications.inception_v3 import  preprocess_input,decode_predictions
import numpy as np

model = InceptionV3(weights='imagenet')
### 如果需要重新训练该模型模型
#inception_v3_model = InceptionV3(input_shape=input_shape,include_top=False,weight=None)# model = Model(inputs=incepition_v3_model.input,output = y)
print_time()
img_path = 'test.jpg'
img = image.load_img(img_path,target_size=(299,299))
X = image.img_to_array(img)
X = np.expand_dims(X,axis=0)
X = preprocess_input(X)
y_pred = model.predict(X)
print(decode_predictions(y_pred,top=3)[0])
print_time()
print_time()
