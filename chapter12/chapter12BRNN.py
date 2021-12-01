# -*- coding: utf-8 -*-

# 1.导库
import numpy as np
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers.core import Activation,Dropout,Dense
from tensorflow.keras.layers import LSTM,Input,Bidirectional
from kt_package import print_time

#model.add(Bid)

# 2.模拟X数据
X = np.array([x**2 for x in range(60)])

# 3.改变X数据的格式
X = X.reshape(12,5,1)

# 4.模拟Y数据
Y = []
for x in X:
    Y.append(x.sum)
Y = np.array(Y)

# 5.构建BRNN
model = Sequential()
model.add(Bidirectional(LSTM(64,activation='relu',input_shape=( 5,1))))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()
model.fit(X,Y,epochs=1000,verbose=1)


# 6.验证BRNN
test_input = np.array([3600,3721,3844,3969,4096])
test_input = test_input.reshape((1,5,1))
test_output = model.predict(test_input,verbose=0)
print(test_output)

