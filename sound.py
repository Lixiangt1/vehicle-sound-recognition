# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 17:31:27 2021

@author: LiXian
"""

import wave
import numpy as np
import scipy.signal as signal
import cv2
import os
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import joblib
from tensorflow import keras
from model import resnet18

class MethodError(Exception):
    pass

def wavread(path):
    f = wave.open(path, "rb")
    params = f.getparams()
    nchannel,sampwith,fs,N=params[0:4]
    wavdata = f.readframes(N)
    f.close()
    wav_data = np.frombuffer(wavdata, dtype=np.int16)
    # wav_data=wav_data/(np.std(wav_data, ddof = 1))
    # wav_data=20*np.log10(wav_data/(2*10**(-5)))
    wav_data.shape = -1, nchannel
    wav_data=wav_data.T
    wav_data=(wav_data-wav_data.mean(axis=1)[:,None])/((np.max([[1e-5,x] for x in np.std(wav_data,axis=1)],axis=1))[:,None])
    return nchannel,fs, wav_data

def fea_extra(wav_data,fs,fre_low,fre_high):
    delta_f=5
    win_length=fs/delta_f
    fre, t, Zxx = signal.stft(wav_data, fs, nperseg=win_length)
    Zxx_Nor=np.abs(Zxx)
    low=int(fre_low/delta_f)
    high=int(fre_high/delta_f)+1
    fre_ch=fre[low:high]
    Zxx_Nor_ch=Zxx_Nor[low:high]

    # Zxx_Nor_ch=np.int64(Zxx_Nor_ch>np.mean(Zxx_Nor_ch)) #二值化
    # plt.pcolormesh(t, fre_ch, np.abs(Zxx_Nor_ch))
    # plt.show()
    return fre_ch,t,Zxx_Nor_ch

def listdir(path):
    list_name=[]
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        list_name.append(file_path)
    return list_name

def data_generation(path,fre_low=20,fre_high=1500,method='svm'):
    if method=='svm':
        dim=64
    elif method=='cnn':
        dim=224
    else:
        raise MethodError('Method is svm or cnn')
        
    classes = os.listdir(path)

    X_data = []
    Y_data = []
    label=0
    for name in listdir(path):
        for file in listdir(name):
            n,fs,data=wavread(file)
            for i in range(n):
                fre,t,fea=fea_extra(data[i],fs,fre_low,fre_high)
                x_data=cv2.resize(fea,(dim,dim))
                X_data.append(x_data)
                Y_data.append(label)
        label += 1
 
    X_data=np.array(X_data)
    
    if method=='svm':
        X_data=X_data.reshape(-1,dim**2)
        u=X_data.mean(axis=0)
        s=np.std(X_data,axis=0)
        X_data=(X_data-u[None,:])/(s[None,:])
        Y_data=np.array(Y_data)
        return classes,X_data,Y_data,u,s
        
    elif method=='cnn':
        maxvalue=np.max([[1e-5,x] for x in X_data.max(axis=2).max(axis=1)],axis=1)
        X_data=X_data/(maxvalue[:,None,None])
        Y_data=np.array(Y_data)
        return classes,X_data,Y_data

def data_test(test_path,u=0,s=1,fre_low=20,fre_high=1500,method='svm'):
    if method=='svm':
        dim=64
    elif method=='cnn':
        dim=224
    else:
        raise MethodError('Method is svm or cnn')
    
    classes = os.listdir(test_path)    
    X_data = []
    Y_data = []
    label=0
    for name in listdir(test_path):
        for file in listdir(name):
            n,fs,data=wavread(file)
            for i in range(n):
                fre,t,fea=fea_extra(data[i],fs,fre_low,fre_high)
                x_data=cv2.resize(fea,(dim,dim))
                X_data.append(x_data)
                Y_data.append(label)
        label += 1
        
    X_data=np.array(X_data)
    
    if method=='svm':
        X_data=X_data.reshape(-1,dim**2)
        X_data=(X_data-u[None,:])/(s[None,:])
    elif method=='cnn':
        maxvalue=np.max([[1e-5,x] for x in X_data.max(axis=2).max(axis=1)],axis=1)
        X_data=X_data/(maxvalue[:,None,None])
        
    Y_data=np.array(Y_data)    
    return classes,X_data,Y_data
    

def label_generation(classes,path):
    Y_data = []
    label=0
    for name in listdir(path):
        for file in listdir(name):
            Y_data.append(label)
        label += 1
        
    Y_data=np.array(Y_data)
    return Y_data

def data_predict(train_path,predict_path,u=0,s=1,fre_low=20,fre_high=1500,method='svm'):
    if method=='svm':
        dim=64
    elif method=='cnn':
        dim=224
    else:
        raise MethodError('Method is svm or cnn')
        
    classes=os.listdir(train_path)
    X_data = []
    for name in listdir(predict_path):
        n,fs,data=wavread(name)
        
        fre,t,fea=fea_extra(data[0],fs,fre_low,fre_high)
        x_data=cv2.resize(fea,(dim,dim))
        X_data.append(x_data)    
    X_data=np.array(X_data)
    
    if method=='svm':
        X_data=X_data.reshape(-1,dim**2)
        X_data=(X_data-u[None,:])/(s[None,:])
    elif method=='cnn':
        X_data=X_data/(np.max([[1e-5,x] for x in X_data.max(axis=2).max(axis=1)],axis=1))[:,None,None]
        
    return classes,X_data

def cnn(classes=5):
    model=resnet18(num_classes=classes)
    model.compile(loss="sparse_categorical_crossentropy",
                optimizer=keras.optimizers.Adam(lr=0.001,
                                                beta_1=0.9,
                                                beta_2=0.999),
                metrics=['accuracy'])
    return model

if __name__=='__main__':

    path=r'D:\sound\train'
    classes,X_data,Y_data,u,s=data_generation(path)
    print(classes,u,s)

    index=np.arange(len(Y_data))
    np.random.shuffle(index)
    X_data=X_data[index]
    Y_data=Y_data[index]
    print(X_data.shape, Y_data.shape)

    dict = {}
    for key in Y_data:
        dict[key] = dict.get(key, 0) + 1
    print(dict)

    

# 支持向量机超参数优化：
    svm_clf = LinearSVC()
    svm_param_grid = [{'C': [1, 10, 50, 100, 200],'loss':['hinge','squared_hinge']}]
    svm_grid_search = GridSearchCV(svm_clf, svm_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    svm_grid_search.fit(X_data, Y_data)
    c = svm_grid_search.best_params_['C']
    loss=svm_grid_search.best_params_['loss']
    print('c:{},loss:{}'.format(c,loss))
    svm_cvres = svm_grid_search.cv_results_
    for score, params in zip(svm_cvres["mean_test_score"], svm_cvres["params"]):
        print(score, params)

# 交叉验证
    svm_clf=LinearSVC(C=c,loss=loss)
    Y_pred=cross_val_predict(svm_clf,X_data,Y_data,cv=5,n_jobs=-1)
    pred_results=confusion_matrix(Y_data,Y_pred)

    for i in range(len(pred_results)):
        precision=pred_results[i][i]/np.sum(pred_results,axis=0)[i]
        print('accuracy of {} is {}'.format(classes[i],precision))

# 随机森林超参数优化：
#     rf_clf=RandomForestClassifier(min_samples_split=5,min_samples_leaf=5,n_jobs=-1,)
#     rf_param_grid=[{'n_estimators':[10,50,100,200],'max_features':[0.1,0.2,0.5,0.8],'max_depth':[6,10,15,20]}]
#     rf_grid_search=GridSearchCV(rf_clf,rf_param_grid,cv=5,scoring='accuracy',n_jobs=-1)
#     rf_grid_search.fit(X_data,Y_data)
#     n_estimators=rf_grid_search.best_params_['n_estimators']
#     max_features=rf_grid_search.best_params_['max_features']
#     max_depth=rf_grid_search.best_params_['max_depth']
#     print('n_estimators:{},max_features:{},max_depth:{}'.format(n_estimators,max_features,max_depth))
#     rf_cvres=rf_grid_search.cv_results_
#     for score, params in zip(rf_cvres["mean_test_score"], rf_cvres["params"]):
#         print(score, params)
#    rf_clf.fit(X_data,Y_data)

# 测试集测试
    svm_clf.fit(X_data, Y_data)
    path_test=r'D:\sound\test'
    classes,X_test,Y_test=data_generation(path_test)
    y_test_pred=svm_clf.predict(X_test)
    test_results=confusion_matrix(Y_test,y_test_pred)
    for i in range(len(pred_results)):
        precision=test_results[i][i]/np.sum(test_results,axis=0)[i]
        print('accuracy of {} is {}'.format(classes[i],precision))

    joblib.dump(svm_clf, 'svm_clf.pkl')