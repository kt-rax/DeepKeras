# -*- coding: utf-8 -*-
# 1.导库
from numpy import zeros,ones
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.datasets.cifar10 import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense,Reshape,Flatten,Conv2D,Conv2DTranspose,LeakyReLU,Dropout
from matplotlib import pyplot
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
import os
import numpy as np

# 2.构建生成器 
def define_generator(latent_dim):
    model = Sequential(name='generator')
    n_nodes = 256*4*4
    model.add(Dense(n_nodes,input_dim=latent_dim,name='generatorr'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((4,4,256)))
    model.add(Conv2DTranspose(128,kernel_size=(4,4),strides=(2,2),padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128,kernel_size=(4,4),strides=(2,2),padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128,kernel_size=(4,4),strides=(2,2),padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(3,(3,3),activation='tanh',padding='same'))
    plot_model(model,to_file='genearator.png',show_shapes=True)
    return model

# 3构建鉴别器
def define_discriminator(in_shape=(32,32,3)):
    model = Sequential(name='discriminator')
    model.add(Conv2D(64,kernel_size=(3,3),padding='same',input_shape=in_shape,name='discriminator'))
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
    plot_model(model,to_file='discriminator.png',show_shapes=True)
    return model

# 4.定义GAN
def define_gan(gen_model,dis_mode):
    dis_mode.trainable = False
    model = Sequential(name='gan')
    model.add(gen_model)
    model.add(dis_mode)
    opt = Adam(lr=0.0002,beta_1=0.5)
    model.compile(loss='binary_crossentropy',optimizer=opt)
    plot_model(model,to_file='gan.png',show_shapes=True)
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
def save_plot(examples,Experiment,epoch,n=7):
    examples = (examples+1)/2.0
    for i in range(n*n):
        #pyplot.subplot(n,n,n+1)
        pyplot.subplot(n,n,1+i)
        pyplot.axis('off')
        pyplot.imshow(examples[i])

    filename = './Experiment%d/Experiment%d_generated_plot_e%03d.png'%(Experiment+1,Experiment+1,epoch+1)
    pyplot.suptitle(filename)
    pyplot.savefig(filename)
    pyplot.close()

def Oringnal_plot(examples,n=7):
    
    examples = (examples+1)/2.0
      
    for i in range(n*n):
        #pyplot.subplot(n,n,n+1)
        pyplot.subplot(n,n,1+i)
        pyplot.axis('off')
      
        pyplot.imshow(examples[i])    
    filename = 'Orignal_plot.png'
    pyplot.suptitle(filename)
    pyplot.savefig(filename)
    pyplot.close()

# 10.评估模型性能
def summarize_performance(epoch,gen_model,dis_model,gan_model,dataset,latent_dim,Experiment,log_file,n_samples=150):
    # real image and labels
    X_real,y_real = generate_real_samples(dataset, n_samples)
    _,acc_real = dis_model.evaluate(X_real,y_real,verbose=0)
    # fake image and lalels
    x_fake,y_fake = generate_fake_samples(gen_model,latent_dim,n_samples)
    _,acc_fake = dis_model.evaluate(x_fake,y_fake,verbose=0)

    print('>Accuracy real: %.0f%%, fake: %.0f%%' %(acc_real*100,acc_fake*100),end='   ')
    log_file.write('>Accuracy real: %.0f%%, fake: %.0f%%\n' %(acc_real*100,acc_fake*100))
    save_plot(x_fake,Experiment,epoch)
    filename_gen = './Experiment%d/Experiment%d_generator_model_%03d.h5'%(Experiment+1,Experiment+1,epoch+1)
    gen_model.save(filename_gen) 
    filename_dis = './Experiment%d/Experiment%d_discriminator_model_%03d.h5'%(Experiment+1,Experiment+1,epoch+1)
    dis_model.save(filename_dis) 
    filename_gan = './Experiment%d/Experiment%d_gan_model_%03d.h5'%(Experiment+1,Experiment+1,epoch+1)
    gan_model.save(filename_gan)
    
    #log_file.close()
    
    return log_file,acc_real,acc_fake
    
# 11.训练部署
def train(gen_model,dis_model,gan_model,dataset,latent_dim,Experiment,n_epochs=30,batch_size=64):
    bat_per_epo = int(dataset.shape[0]/batch_size)
    half_batch = int(batch_size / 2)
    
    #mk log dir and mk log file
    os.mkdir(os.path.join('%s/Experiment%d/'%(os.getcwd(),Experiment+1)))
    log_file = open(os.path.join('%s/Experiment%d/'%(os.getcwd(),Experiment+1))+'Experiment%d_log.txt'%(Experiment+1),'w+')
    #log_file_name = 'Experiment%d_log.txt'%(Experiment+1)
    #log_file = open(log_file_name, 'w+')
    acc_real_history = []
    acc_fake_history = []
    d_loss1_history = []
    d_loss2_history = []
    g_loss_history = []
    for i in range (n_epochs):
        for j in range(bat_per_epo):
            X_real,y_real = generate_real_samples(dataset,half_batch)
            d_loss1,_ = dis_model.train_on_batch(X_real,y_real)
            X_fake,y_fake = generate_fake_samples(gen_model, latent_dim, half_batch)
            d_loss2,_ = dis_model.train_on_batch(X_fake,y_fake)
            X_gan = generate_latent_points(latent_dim, batch_size)
            y_gan = ones((batch_size,1))
            g_loss = gan_model.train_on_batch(X_gan,y_gan)
            if(j+1) % 100 == 0:
                print('>Experiment:%d  epoch:%3d, %4d/%4d, d1 = %.4f, d2 = %.4f g=%.4f'%(Experiment+1,i+1,j+1,bat_per_epo,d_loss1,d_loss2,g_loss))
                log_file.write('>Experiment:%d  epoch:%3d, %4d/%4d, d1 = %.4f, d2 = %.4f g=%.4f\n'%(Experiment+1,i+1,j+1,bat_per_epo,d_loss1,d_loss2,g_loss))
        if(i+1) % 10 == 0:
            log_file,acc_real,acc_fake = summarize_performance(i,gen_model,dis_model,gan_model,dataset,latent_dim,Experiment,log_file)
            acc_real_history = np.append(acc_real_history,acc_real)
            acc_fake_history = np.append(acc_fake_history,acc_fake)
            d_loss1_history = np.append(d_loss1_history, d_loss1)
            d_loss2_history = np.append(d_loss2_history, d_loss2)
            g_loss_history = np.append(g_loss_history, g_loss)

            #loss_history = loss_history.update('d_loss1':d_loss1,'d_loss2':d_loss2,'g_loss':g_loss)
            print('nepochs: %d  bat_per_epc: %d'%(i,j))
            log_file.write('nepochs: %d  bat_per_epc: %d\n'%(i,j))
            
    log_file.close()
    
    pyplot.figure
    pyplot.plot(d_loss1_history)
    pyplot.plot(d_loss2_history)
    pyplot.plot(g_loss_history)
    pyplot.legend(['d_loss1','d_loss2','g_loss'])
    pyplot.xlabel('epochs')
    pyplot.ylabel('loss')
    pyplot.savefig('./Experiment%d/Experiment%d_loss.jpg'%(Experiment+1,Experiment+1))
    pyplot.close()
    pyplot.figure
    pyplot.plot(acc_real_history)
    pyplot.plot(acc_fake_history)
    pyplot.legend(['acc_real','acc_fake'])
    pyplot.xlabel('epochs')
    pyplot.ylabel('accuracy')
    pyplot.title('Real & Fake Accuray by discriminator')
    pyplot.savefig('./Experiment%d/Experiment%d_Accuracy.jpg'%(Experiment+1,Experiment+1))
    pyplot.close()    
          
# 12.模型训练
for Experiment in range(9,10):   
    latent_dim = 100
    dis_model = define_discriminator()
    gen_model = define_generator(latent_dim)
    #gen_model = load_model(r'./generator_model_100.h5')
    gan_model = define_gan(gen_model,dis_model)
    dataset = load_real_samples()
    #
    Orignal_examples,_= generate_real_samples(dataset,n_samples=100)
    Oringnal_plot(Orignal_examples,n=7)
    
    train(gen_model, dis_model, gan_model, dataset, latent_dim,Experiment)
     
    # 13.生成器的使用
    def create_plot(examples,n):
        for i in range(n*n):
            pyplot.subplot(n,n,1+i)
            pyplot.axis('off')
            pyplot.imshow(examples[i,:,:])
        pyplot.show()
    Experiment += 1
    
    
#model = load_model(os.path.join(".//","generator_model_100.h5"))
    
    
for Experiment in range(1,3):
    print(Experiment)
    
    
    
    
    