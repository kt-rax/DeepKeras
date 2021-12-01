# -*- coding: utf-8 -*-

'''
#================s数学原理实现线性回归
import matplotlib.pyplot as plt

# 1.载入需要的库与数据 
X_train = [1,1.2,1.4,1.5,1.8,2,2.5,3,4,5,6.2]
Y_train = [2,5,8,9,12,16.5,23.5,43.5,68.5,89.3,120]

# 2.训练参数初始化 
W = 0
b = 0

# 3.超参数设定
learning_rate = 0.01
num_epochs = 50
n = len(X_train)

# 4.建立数学模型
def pred(W,b,X):
    return W*X+b 


# 5利用梯度下降法进行线性回归
cost = []
epochs = [x for x in range(num_epochs)]
for i in range(num_epochs):
    loss = 0.0
    grad_W = 0.0
    grad_b =0.0
    for j in range(n):
        X = X_train[j]
        y = Y_train[j]
        loss += (1/n)*(y-pred(W,b,X))**2
        grad_W += -(2/n)*(y-pred(W,b,X))*X
        grad_b += -(2/n)*(y-pred(W,b,X))
    W = W - learning_rate*grad_W
    b = b - learning_rate*grad_b
    
    cost.append(loss)

# 6.绘图
plt.plot(epochs,cost)
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.title('loss function')
plt.show()

# 利用keras实现线性回归
from keras.layers import Dense
from keras.models import  Sequential
import  numpy as np
import  matplotlib.pylab as plt

# 1.数据
X_train = [1,1.2,1.4,1.5,1.8,2,2.5,3,4,5,6.2]
y_train = [3,5,8,9,12,16.5,23.5,43.5,68.5,89.3,120]

# 2.数据预处理
X_train = np.array(X_train)
y_train = np.array(y_train)
X_train = np.reshape(X_train,(11,1))
y_train = np.reshape(y_train,(11,1))

# 3.建立模型
model = Sequential()
model.add(Dense(1,input_shape=(1,),activation=None))
model.compile(optimizer='sgd', loss='mse')
model.summary()

# 4.模型拟合
model.fit(x=X_train,y=y_train,epochs=50,verbose=2)

# 5.查看参数
for layer in model.layers:
    weights = layer.get_weights()
    print(weights)

# 6.结果预测
test = np.array([1.8])
test = np.reshape(test,(1,1))
test_result = model.predict(test)

print(test_result)
'''
# 数学实现多元线性回归
import matplotlib.pyplot as plt
import  numpy as np
from keras.layers import  Dense
from keras.models import  Sequential

# 1 数据
X_train = [[1,3.89],[1.2,4.0],[1.4,4.1],[1.5,4.18],[1.8,4.27],[2,4.38],[2.5,4.49],[3,4.62],[4,4.78],
           [5,4.89],[6.2,5.1]]
y_train = [3,5,8,9,12,16.5,23.5,43.7,68.5,89.3,120]

# 2.参数初始化
W = [0.0,0.0]
b = 0

# 3.超参
learning_rate = 0.001
num_epochs = 10

# 4.定义模型
def pred(W,b,X):
    return np.dot(W,X)+b

# 5.梯度下降线性回归
cost = []
n = len(X_train)
epochs = [x for x in range(num_epochs)]
for i in range(num_epochs):
    loss = 0.0
    grad_W = [0.0,0.0]
    grad_b = 0.0
    for j in range(n):
        X = X_train[j]
        y = y_train[j]
        loss += (1/n)*(y - pred(W,b,X))**2
        grad_W += -(2/n)*np.dot(y-pred(W,b,X),X)
        grad_b += -(2/n)*(y-pred(W,b,X))
    W = W - learning_rate * grad_W
    b = b - learning_rate * grad_b
    cost.append(loss)
    
    
plt.plot(epochs,cost)
plt.xlabel('Epochs') 
plt.ylabel('loss')
plt.title('loss function')
plt.show()


# keras实现多元线性回归
X_train = [np.array(element) for element in X_train]
X_train = np.array(X_train)
y_train = np.array(y_train)
X_train = np.reshape(X_train,(11,1,2))
y_train = np.reshape(y_train,(11,1,1))

model = Sequential()
model.add(Dense(1,input_shape=(1,2),activation=None))
model.compile(optimizer='sgd', loss='mse')
model.summary()


# 4.模型拟合
model.fit(x=X_train,y=y_train,epochs=50,verbose=2)

# 5.查看参数
for layer in model.layers:
    weights = layer.get_weights()
    print(weights)

# 6.结果预测
test = np.array([1.8,4]).reshape(1,1,2)
test_result = model.predict(test)

print(test_result)

















