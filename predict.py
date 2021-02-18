# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 17:31:27 2021

@author: LiXian
"""
import sound
import os
import joblib
import numpy as np
import tensorflow as tf

try:
    path=open('path.txt','r+')
except:
    raise IOError('无法打开地址文件')

path_txt=path.readlines()

if path_txt:
    a=input('数据地址：{}, Y/N? '.format(str(path_txt[-1])))
    if a=='y':
        path_txt=path_txt[-1]
    elif a=='n':
        path_txt=input('请输入数据地址：')
        path.write('\n'+path_txt)
        
else:
    path_txt=input('请输入数据地址：')
    path.write(path_txt)
path.close()

train_path=os.path.join(str(path_txt),'train')
predict_path=os.path.join(str(path_txt),'predict')

method=input('输入判别方法，svm/cnn? ')

if method=='svm':
    mtd=np.load('meanstd.npz')
    svm_clf=joblib.load('svm_clf.pkl')
    u=mtd['arr_0']
    s=mtd['arr_1']
    
    classes,X_test=sound.data_predict(train_path,predict_path,u=u,s=s)
    print(classes)
    
    Y_pred=svm_clf.predict(X_test)
    result=np.array(classes)[Y_pred]
    print(result)

elif method=='cnn':
    cnn_clf=sound.cnn()
    cnn_clf.load_weights('cnn_clf_weight').expect_partial()
    
    cls_predict,X_predict=sound.data_predict(train_path, predict_path,method='cnn')
    X_predict=X_predict.reshape(-1,224,224,1)
    X_predict=tf.cast(X_predict,dtype=tf.float32)
    results_prob=cnn_clf.predict(X_predict)
    results=np.argmax(results_prob,axis=1)
    results=np.array(cls_predict)[results]
    print(results)
