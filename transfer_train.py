# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 14:43:49 2021

@author: LiXian
"""
import sound
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

try:
    path = open('path.txt', 'r+')
except:
    raise IOError('无法打开地址文件')

path_txt = path.readlines()

if path_txt:
    a = input('数据地址：{}, Y/N? '.format(str(path_txt[-1])))
    if a == 'y' or a=='Y':
        path_txt=path_txt[-1]
    elif a == 'n' or a=='N':
        path_txt = input('请输入数据地址：')
        path.write('\n' + path_txt)
    else: 
        raise TypeError('输入格式不正确')
        
else:
    path_txt = input('请输入数据地址：')
    path.write(path_txt)
path.close()

path_train=os.path.join(str(path_txt),'train')
path_test=os.path.join(str(path_txt),'test')

classes,X_data,Y_data=sound.data_generation(path_train,method='cnn')
index=np.arange(len(Y_data))
np.random.shuffle(index)
X_data=X_data[index]
Y_data=Y_data[index]
X_data=X_data.reshape(-1,224,224,1)
X_data=tf.cast(X_data, dtype=tf.float32)

dict = {}
for key in Y_data:
    dict[key] = dict.get(key, 0) + 1
print(dict)


classes_test,X_test,Y_test=sound.data_generation(path_test,method='cnn')
X_test=X_test.reshape(-1,224,224,1)
X_test=tf.cast(X_test, dtype=tf.float32)

b_s=16


data_train_all=tf.data.Dataset.from_tensor_slices((X_data,Y_data))
data_train=data_train_all.take(2300).shuffle(500).batch(b_s).prefetch(1)
data_valid=data_train_all.skip(2300).batch(b_s).prefetch(1)

base_model=sound.cnn(5)
base_model.load_weights('cnn_clf_weight').expect_partial()

print(base_model.summary())

output=keras.layers.Dense(len(classes),activation='softmax')(base_model.get_layer('dropout').output)

cnn_clf=keras.Model(inputs=base_model.input,outputs=output)

for layer in base_model.layers:
    layer.trainable = False
    
cnn_clf.compile(loss="sparse_categorical_crossentropy",
            optimizer=keras.optimizers.Adam(lr=0.01,
                                            beta_1=0.9,
                                            beta_2=0.999),
            metrics=['accuracy'])

history=cnn_clf.fit(data_train,
                    epochs=20,
                    validation_data=data_valid,
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                             min_delta=0.001,
                                                             mode='max',
                                                             patience=3,
                                                             restore_best_weights=True)])

for layer in base_model.layers:
    layer.trainable = True
    
cnn_clf.compile(loss="sparse_categorical_crossentropy",
            optimizer=keras.optimizers.Adam(lr=0.001,
                                            beta_1=0.9,
                                            beta_2=0.999),
            metrics=['accuracy'])

history=cnn_clf.fit(data_train,
                    epochs=100,
                    validation_data=data_valid,
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                             min_delta=0.001,
                                                             mode='max',
                                                             patience=5,
                                                             restore_best_weights=True)])

cnn_clf.save_weights('cnn_clf_weight_6Classes')


