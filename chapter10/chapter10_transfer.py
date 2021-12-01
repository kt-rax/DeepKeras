# -*- coding: utf-8 -*-
from keras.applications.vgg16 import  VGG16
from keras.models import  Model
from keras.layers import  Dense,Flatten

model = VGG16()

model.summary()

from keras.utils import plot_model
plot_model(model)

model.layers.pop()
output_layer = Dense(3,activation='softmax')(model.outputs)


model = VGG16(include_top = False,input_shape=input_shape)

flat_1 = Flatten()(model.outputs)
fc1 = Dense(1024,activation='relu')(flat_1)
fc2 = Dense(3,activation='softmax')(fc1)
model = Model(input=model.inputs,outputs=fc2)

model = VGG(include_top=False,input_shape=input_shape)
for layer in model.layers:
    layer.trainable = False
    

model.get_layer('Block5_conv3').trainable = False