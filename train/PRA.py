import scipy as sp
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import pandas as pd


f1=pd.read_csv('/home/mt/learn/NGSIM数据处理/csv/合并数据/v1-y4.csv')
f2=pd.read_csv('/home/mt/learn/NGSIM数据处理/csv/101/8:20-8:35/new/v1-y4.csv')
train_data = f1[f1.columns[2:-1]]
train_truth = f1['label']
test_data=f2[f2.columns[3:-1]]
test_truth=f2['label']
train_data

#调用逻辑斯特回归
model = LogisticRegression(class_weight=None)
model.fit(train_data, train_truth)
print(model)       #输出模型
# make predictions
predict_reg= model.predict(test_data)       #测试样本预测
#输出结果
print(metrics.classification_report(test_truth, predict_reg))       #输出结果，精确度、召回率、f-1分数
print(metrics.confusion_matrix(test_truth, predict_reg))




from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
model1 = Sequential()
model1.add(Dense(64, activation='relu', input_dim=6))
model1.add(Dropout(0.5))
model1.add(Dense(32, activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(1, activation='sigmoid'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model1.compile(loss='mse',optimizer=sgd,metrics=['accuracy'])
model1.fit(x_train, y_train,epochs=100,batch_size=128)
print(model1)
# model1.save("/home/mt/learn/NGSIM数据处理/model/算法模型/bp.h5")
# model.load_weights('my_model_weights.h5')
pre1=model1.predict_classes(x_test)
print(metrics.classification_report(y_test, pre1))       #输出结果，精确度、召回率、f-1分数
print(metrics.confusion_matrix(y_test, pre1))