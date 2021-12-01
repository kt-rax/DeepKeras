# -*- coding: utf-8 -*-
# 1.导库
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from keras.models import Sequential,save_model,load_model
#from keras.utils import  plot_model
from keras.utils.vis_utils import  plot_model
import matplotlib.pyplot as plt


import re
from tensorflow.keras.models import Model

import re
from tensorflow.keras.models import Model

def insert_layers_nonseq(model,layer_regex,insert_layer_factory,insert_layer_name=None,position='after'):
    
    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of':{},'new_output_tensor_of':{}}
    
    # set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update({layer_name:[layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)
    
    # Set hte output tensor of the input layer
    network_dict['new_output_tensor_of'].update({model.layers[0].name:model.input})
    
    # Iterate over all layers after the input
    model_outputs = []
    
    for layer in model.layers[1:]:
        
        # Determin input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux] for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]
        # Insert layer if name matches the regular expression
        if re.match(layer_regex,layer.name):
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before,after,replace')
                
            new_layer = insert_layer_factory()
            
            if insert_layer_name:
                new_layer.name = insert_layer_name
            else:
                new_layer.name = '{}_{}'.format(layer.name,new_layer.name)
            
            x = new_layer(x)
            print('new layer:{} old layer:{} Type:{}'.format(new_layer.name,layer.name,position))
    
            if position == 'before':
                x = layer(x)
        else:
            x = layer(layer_input)
            
        # set new output tensor(the original one,or the one of the inserted layer)
        network_dict['new_output_tensor_of'].update({layer.name:x})
        
        # Save tensor in output list if it is output in intial model
        if layer_name in model.output_names:
            model_outputs.append(x)
    
    # debug AttributeError: 'tuple' object has no attribute 'layer'
    #model = Model(inputs=model.inputs,outputs=model.outputs)
    plot_model(model,to_file='Restnet_AterInsert.png',show_shapes=True,show_layer_names=True)
    
    return Model(inputs=model.inputs,outputs=model.outputs)

# 2.添加卷积层
model = Sequential()
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu'))
model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256,kernel_size=(3,3),padding='same',activation='relu'))
model.add(Conv2D(filters=256,kernel_size=(3,3),padding='same',activation='relu'))
model.add(Conv2D(filters=256,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu'))
model.add(Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu'))
model.add(Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu'))
model.add(Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu'))
model.add(Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

# 3.添加全连接层
model.add(Flatten())
model.add(Dense(units=4096,activation='relu'))
model.add(Dense(units=4096,activation='relu'))
model.add(Dense(units=4096,activation='relu'))

# 4.查看模型
model.summary()
plot_model(model,to_file='vgg_16.png',show_shapes=True,show_layer_names=True)

#### 使用keras快速搭建VGG16
from keras.applications.vgg16 import  VGG16
model_2 = VGG16()
print(model_2.summary())
plot_model(model_2,to_file='keras_vgg_16.png',show_shapes=True,show_layer_names=True)

#### 使用现在的VGG模型进行预测
from keras.preprocessing import image
from keras.models import Model
from keras.applications.vgg16 import preprocess_input,decode_predictions,VGG16
import numpy as np

''' 
model_3 = VGG16(weights='imagenet')
# 模型裁剪迁移学习,替换最后一层  --- 未调通
model_3.layers.pop()
output_layer = Dense(3,activation='softmax',name='Transfered')(model_3.outputs)
#plot_model(model_3,to_file='keras_vgg_16_T.png',show_shapes=True,show_layer_names=True)
'''

# 模型裁剪修改 a.末尾修改
model_3 = VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3))
flat_1 = Flatten()(model_3.outputs)
fc1 = Dense(1024,activation='relu')(flat_1)
fc2 = Dense(3,activation='softmax')(fc1)
model_3 = Model(inputs=model_3.inputs,outputs=fc2)
plot_model(model_3,to_file='keras_vgg_16_T.png',show_shapes=True,show_layer_names=True)
# b.中间修改 查看plot的图在block5_conv1与block5_conv2之间加3层


def Conv2D_layer_factory():
    return Conv2D(name='block3_conv3')
model = insert_layer_nonseq(model, '.*activation.*', dropout_layer_factory)




# Fix possible problems with new model
model.save('temp.h5')
model = load_model('temp.h5')

model.summary()









'''
img_path = 'test.jpg'
img = image.load_img(img_path,target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)
y_pred = model_3.predict(x)
#print(decode_predictions(y_pred,top=5)[0])
print(y_pred)

'''











