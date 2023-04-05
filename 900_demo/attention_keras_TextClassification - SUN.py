# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 05:29:50 2020
机器学习与临床决策——注意力机制
数据是IMDB，评论文本分类
@author: David
"""
import sys
file_dir = 'C:\\Python Scripts\\0-attention\\attention-master\\'
sys.path.append(file_dir)
sys.path.append("..")
#%%

from tensorflow import keras
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from matplotlib import pyplot as plt
import pandas as pd

from keras import backend as K
#from tensorflow.keras.engine.topology import Layer
from tensorflow.keras.layers import Layer #ZERO
import attention_keras#add byZERO





'''############################################################################'''

class Self_Attention(Layer):

    def __init__(self, output_dim, **kwargs):
        
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        #inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3,input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

        super(Self_Attention, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])

        print("WQ.shape",WQ.shape)

        print("K.permute_dimensions(WK, [0, 2, 1]).shape",K.permute_dimensions(WK, [0, 2, 1]).shape)#矩阵转置 #ZERO

        QK = K.batch_dot(WQ,K.permute_dimensions(WK, [0, 2, 1]))#矩阵转置 #ZERO

        QK = QK / (64**0.5) #缩放

        QK = K.softmax(QK) #归一化

        print("QK.shape",QK.shape)

        V = K.batch_dot(QK,WV) #内积

        return V

    def compute_output_shape(self, input_shape):

        return (input_shape[0],input_shape[1],self.output_dim)


'''############################################################################'''





'''----------------------------------- Data ------------------------------------'''

max_features = 20000

print('Loading data...')

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
#标签转换为独热码
y_train, y_test = pd.get_dummies(y_train),pd.get_dummies(y_test)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

#%%数据归一化处理

maxlen = 64

print('Pad sequences (samples x time)')

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

#%%


'''################################### Model ##################################'''

batch_size = 32
from tensorflow.keras.models import Model
#from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.optimizers import Adam,Nadam, SGD
from tensorflow.keras.layers import *
#from Attention_keras import Attention,Position_Embedding
from attention_keras import Attention,Position_Embedding



S_inputs = Input(shape=(64,), dtype='int32') # 64是文本的长度
embeddings = Embedding(max_features, 128)(S_inputs)# 128是embedding的维度

'''+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'''
O_seq = Self_Attention(128)(embeddings)
'''+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'''

O_seq = GlobalAveragePooling1D()(O_seq) # 拉平为vector
O_seq = Dropout(0.5)(O_seq)
outputs = Dense(2, activation='softmax')(O_seq)


model = Model(inputs=S_inputs, outputs=outputs)

print(model.summary())
# try using different optimizers and different optimizer configs
opt = Adam(lr=0.0002,decay=0.00001)
loss = 'categorical_crossentropy'
model.compile(loss=loss,
             optimizer=opt,
             metrics=['accuracy'])


from tensorflow.keras.utils import plot_model
print("layer nums:", len(model.layers))
plot_model(model, to_file='text_attention.png', show_shapes=True, show_layer_names=False)

# In[*]


print(model.summary())
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='TextAT.png',show_shapes=True)



#%%
print('Trainning-----------------------------------------------------------------')

history = model.fit(x_train, y_train,
         batch_size=batch_size,
         epochs=5,
         validation_data=(x_test, y_test))

#%%
'''------------------------------- Visualization ------------------------------'''
print('Visualing-----------------------------------------------------------------')



import matplotlib.pyplot as plt
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy TEXT')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right') 
plt.show()
# summarize history for loss 
plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.title('model loss TEXT')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right') 
plt.show()


#model.save("imdb.h5")