# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 00:52:53 2018

@author: Lxiao217
"""
import numpy as np


#np.random.shuffle(data)    #打乱数据

#假设x 是一个形状为(samples,features) 的二维矩阵
#标准化
x = np.array([[8.,2.],[20.,4.]])
x -= x.mean(axis = 0)
x /= x.std(axis = 0)
print(x)

#使用l2正则化
from keras.models import model
from keras import layers
from keras import regularizers
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
activation='relu', input_shape=(10000,)))
#l2(0.001) 的意思是该层权重矩阵的每个系数都会使网络总损失增加0.001 * weight_coefficient_value。
regularizers.l1(0.001) #L1 正则化
regularizers.l1_l2(l1=0.001, l2=0.001) #同时做L1 和L2 正则化

















