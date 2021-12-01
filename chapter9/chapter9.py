# -*- coding: utf-8 -*-
# 1.导库
from numpy import zeros,ones
from numpy.random import randn
from numpy.random import randint
from keras.datasets.cifar10 import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import  Dense,Reshape,Flatten,Conv2D,Conv2DTranspose,LeakyReLU,Dropout
from matplotlib import pyplot

# 2.构建生成器 
def define_generator(latent_dim):
    model = Sequential()
    n_nodes = 256*4*4
    model.add(Dense(n_nodes,input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((4,4,256)))
    model.add(Conv2DTranspose(128,kernel_size=(4,4),strides=(2,2),padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128,kernel_size=(4,4),strides=(2,2),padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128,kernel_size=(4,4),strides=(2,2),padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(3,(3,3),activation='tanh',padding='same'))
    
    optimizer = Adam(lr=0.0002,beta_1=0.5)
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    return model

# 3构建鉴别器
def define_discriminator(in_shape=(32,32,3)):
    model = Sequential()
    model.add(Conv2D(64,kernel_size=(3,3),padding='same',input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128,kernel_size=(3,3),strides=(2,2),padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128,kernel_size=(3,3),strides=(2,2),padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(256,kernel_size=(3,3),strides=(2,2),padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1,activation='sigmoid'))
    optimizer = Adam(lr=0.0002,beta_1=0.5)
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    return model

# 4.定义GAN
def define_gan(gen_model,dis_mode):
    dis_mode.trainable = False
    model = Sequential()
    model.add(gen_model)
    model.add(dis_mode)
    opt = Adam(lr=0.0002,beta_1=0.5)
    model.compile(loss='binary_crossentropy',optimizer=opt)
    return model
    
# 5.从CIFAR-10数据集载入数据
def load_real_samples():
    (trainX,_),(_,_) = load_data()
    X = trainX.astype('float32')
    X = (X-127.5)/127.5
    return X

# 6.从真实数据中随机采样
def generate_real_samples(dataset,n_samples):
    ix = randint(0,dataset.shape[0],n_samples)
    X = dataset[ix]
    y = ones((n_samples,1))
    return X,y

# 7.生成用于触发器工作的随机向量
def generate_latent_points(latent_dim,n_samples):
    x_input = randn(latent_dim*n_samples)
    x_input = x_input.reshape(n_samples,latent_dim)
    return x_input

# 8.用生成器生成假图像
def generate_fake_samples(gen_model,latent_dim,n_samples):
    x_input = generate_latent_points(latent_dim,n_samples)
    X = gen_model.predict(x_input)
    y = zeros((n_samples,1))
    return X,y

# 9.绘制与保存图像
def save_plot(examples,epoch,n=7):
    examples = (examples+1)/2.0
    for i in range(n*n):
        pyplot.subplot(n,n,n+1)
        pyplot.axis('off')
        pyplot.imshow(examples[i])

    filename = 'generated_plot_e%03d.png'
    pyplot.savefig(filename)
    pyplot.close()

# 10.评估模型性能
def summarize_performance(epoch,gen_model,dis_mode,dataset,latent_dim,n_samples=150):
    X_real,y_real = generate_real_samples(dataset, n_samples)
    _,acc_real = dis_mode.evaluate(X_real,y_real,verbose=0)
    x_fake,y_fake = generate_fake_samples(gen_model,latent_dim,n_samples)
    _,acc_fake = dis_mode.evaluate(x_fake,y_fake,verbose=0)
    print('>Accuracy real: %.0f%%, fake: %.0f%%' %(acc_real*100,acc_fake*100))
    save_plot(x_fake,epoch)
    filename = 'generator_model_%03d.h5'%(epoch+1)
    gen_model.save(filename) 

# 11.训练部署
def train(gen_model,dis_mode,gan_mode,dataset,latent_dim,n_epochs=2,batch_size=128):
    bat_per_epo = int(dataset.shape[0]/batch_size)
    half_batch = int(batch_size / 2)
    for i in range (n_epochs):
        for j in range(bat_per_epo):
            X_real,y_real = generate_real_samples(dataset,half_batch)
            d_loss1,_ = dis_mode.train_on_batch(X_real,y_real)
            X_fake,y_fake = generate_fake_samples(gen_model, latent_dim, half_batch)
            d_loss2,_ = dis_mode.train_on_batch(X_fake,y_fake)
            X_gan = generate_latent_points(latent_dim, batch_size)
            y_gan = ones((batch_size,1))
            g_loss = gen_model.train_on_batch(X_gan,y_gan)
            print('>%d, %d/%d, d1 = %.3f, d2 = %.3f g=%.3f'%(i+1,j+1,bat_per_epo,d_loss1,d_loss2,g_loss))
            summarize_performance(i,gen_model,dis_mode,dataset,latent_dim)
            
# 12.模型训练
latent_dim = 100
dis_mode = define_discriminator()
gen_model = define_generator(latent_dim)
gan_mode = define_gan(gen_model,dis_mode)
dataset = load_real_samples()
train(gen_model, dis_mode, gan_mode, dataset, latent_dim)
 
# 13.生成器的作用
def create_plot(examples,n):
    for i in range(n*n):
        pyplot.subplot(n,n,1+i)
        pyplot.axis('off')
        pyplot.imshow(examples[i,:,:])
    pyplot.show()
    
    
    
    
    
    
    
    
    
    
    
    
    