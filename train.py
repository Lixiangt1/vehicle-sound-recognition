# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 11:54:54 2021

@author: LiXian
"""

import sound
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import joblib
import tensorflow as tf
from tensorflow import keras

try:
    path = open('path.txt', 'r+')
except:
    raise IOError('无法打开地址文件')

path_txt = path.readlines()

if path_txt:
    a = input('数据地址：{}, Y/N? '.format(str(path_txt[-1])))
    if a == 'y' or a =='Y':
        path_txt=path_txt[-1]
    elif a == 'n' or a =='N':
        path_txt = input('请输入数据地址：')
        path.write('\n' + path_txt)
    else: 
        raise TypeError('输入格式不正确')
        
else:
    path_txt = input('请输入数据地址：')
    path.write(path_txt)
path.close()

path_train=os.path.join(str(path_txt),'train')
classes,X_data,Y_data,u,s=sound.data_generation(path_train)
np.savez('meanstd', u,s)
print(classes)

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
svm_grid_search.fit(X_data,Y_data)
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
path_test=os.path.join(str(path_txt),'test')
cls_test,X_test,Y_test=sound.data_test(path_test, u=u, s=s)

y_test_pred=svm_clf.predict(X_test)
test_results=confusion_matrix(Y_test,y_test_pred)
for i in range(len(pred_results)):
    precision=test_results[i][i]/np.sum(test_results,axis=0)[i]
    print('accuracy of {} is {}'.format(classes[i],precision))

joblib.dump(svm_clf, 'svm_clf.pkl')


# 卷积网络模型训练
classes,X_data,Y_data=sound.data_generation(path_train,method='cnn')
X_data=X_data[index]
Y_data=Y_data[index]
X_data=X_data.reshape(-1,224,224,1)
X_data=tf.cast(X_data, dtype=tf.float32)

classes_test,X_test,Y_test=sound.data_generation(path_test,method='cnn')
X_test=X_test.reshape(-1,224,224,1)
X_test=tf.cast(X_test, dtype=tf.float32)

b_s=16
epoches=100
 
cnn_clf=sound.cnn(5)

data_train_all=tf.data.Dataset.from_tensor_slices((X_data,Y_data))
data_train=data_train_all.take(1800).shuffle(500).batch(b_s).prefetch(1)
data_valid=data_train_all.skip(1800).batch(b_s).prefetch(1)

history=cnn_clf.fit(data_train,
                    epochs=epoches,
                    validation_data=data_valid,
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                             min_delta=0.001,
                                                             mode='max',
                                                             patience=7,
                                                             restore_best_weights=True)])

cnn_clf.evaluate(X_test,Y_test)

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

cnn_clf.save_weights('cnn_clf_weight')
