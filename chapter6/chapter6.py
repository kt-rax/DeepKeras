# -*- coding: utf-8 -*-
from keras.layers import SimpleRNN
from keras.models import Sequential
import  matplotlib.pyplot as plt
import  numpy as np
from math import exp
import pandas as pd
from keras.layers import Dense,SimpleRNN
from keras import initializers
from keras import optimizers


initializer = initializers.Orthogonal(gain=1.0,seed=None)
#model = Sequential()
#model.add(SimpleRNN(units=128,input_shape=(input.shape[1],input.shape[2]),activation='relu'))

# 生成假数据 
def modified_sigmoid(t):
    x = 10000/(1+exp(-0.02*(t-500)))+np.random.randint(-400,400)
    return x

N = 1000
t = np.arange(0,N)
x = map(modified_sigmoid,t)
x = list(x)
plt.figure
plt.xlabel('days')
plt.ylabel('population')
plt.title('plague model')
plt.plot(t,x)
plt.savefig('W.png')
plt.show()
plt.close()

# 数据前处理
x = np.array(x)
num_time_steps = 4
train,test = x[0:800],x[800:1000]
test = np.append(test,np.repeat(test[-1],num_time_steps))
train = np.append(train,np.repeat(train[-1],num_time_steps))

def Sequence_generator(input_data,num_time_steps):
    X,Y = [],[]
    for i in range(len(input_data)-num_time_steps):
        X.append(input_data[i:i+num_time_steps])
        Y.append(input_data[i+num_time_steps])
    return np.array(X),np.array(Y)

x_train,y_train = Sequence_generator(train,num_time_steps=num_time_steps)
x_test ,y_test = Sequence_generator(test,num_time_steps=num_time_steps)
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

# RNN搭建
model = Sequential()
model.add(SimpleRNN(units=64,input_shape=(4,1),activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1))

model.compile(loss='mse',optimizer=optimizers.SGD(lr=0.01,clipvalue=0.5))

#model.compile(loss='mse',optimizer='RMSprop')
model.fit(x_train,y_train,epochs=50,batch_size=16,verbose=2)


# 结果预测
trainPredict = model.predict(x_train)
testPredict = model.predict(x_test)
predicted = np.concatenate((trainPredict,testPredict),axis=0)
df = pd.DataFrame(x)
index = df.index.values
plt.figure
plt.plot(index,df[0],'b')
plt.plot(index,predicted,'g')
plt.legend(['real data','predicted'])
plt.title('real data VS predicted data')
plt.xlabel('days')
plt.ylabel('population')
plt.axvline(df.index[800],c='r')
plt.savefig('A.png')
plt.show()
plt.close()

# 模型性能测试
draw_x = x[800:1000].reshape(200,1)
diff = (testPredict - draw_x)/testPredict
dataframe = pd.DataFrame(diff)
ax = dataframe.plot()
vals = ax.get_yticks()
ax.set_xlabel('days')
ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])
ax.legend('percentage')
ax.set_title('percentage diff')
plt.savefig('B.png')
plt.show()
plt.close()



















