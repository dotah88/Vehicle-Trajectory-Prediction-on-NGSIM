import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.models import load_model
from sklearn import metrics
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

df = pd.read_csv("/home/mt/learn/NGSIM数据处理/csv/80/4:00-4:15/Double_train.csv")
f=pd.read_csv("/home/mt/learn/NGSIM数据处理/csv/80/4:00-4:15/Doubel_label.csv")
df2=pd.read_csv("/home/mt/learn/NGSIM数据处理/csv/80/4:00-4:15/Double_train1.csv")
f2=pd.read_csv("/home/mt/learn/NGSIM数据处理/csv/80/4:00-4:15/Double_label1.csv")
double_label=[]
double_label2=[]

label=f.iloc[:,9].values
label2=f2.iloc[:,9].values
for i in range(len(label)):
    if label[i]==0:
        double_label.append([1,0,0])
    elif label[i]==1:
        double_label.append([0,1,0])
    elif label[i]==2:
        double_label.append([0,0,1])
for i in range(len(label2)):
    if label2[i]==0:
        double_label2.append([1,0,0])
    elif label2[i]==1:
        double_label2.append([0,1,0])
    elif label2[i]==2:
        double_label2.append([0,0,1])

train_y=[]
train_y2=[]
data_num=int(len(label)/40)
data_num2=int(len(label2)/40)

for i in range(data_num):
    train_y.append(double_label[40*i])
for i in range(data_num2):
    train_y2.append(double_label2[40*i])
train_y=np.array(train_y)
train_y2=np.array(train_y2)

data=df.iloc[:,2:9].values
data2=df2.iloc[:,2:9].values
data=np.reshape(data,[data_num,40,7])
data2=np.reshape(data2,[data_num2,40,7])



train20_42=[]
train20_43=[]
train20_20=[]
val20_42=[]
val20_43=[]
val20_20=[]
# for i in range(len(data)):
#     if i==0:
#         continue
#     else:
#         if data[i][0]!=data[i-1][0] or (data[i][0]==data[i-1][0] and data[i][7]!=data[i-1][7]):
#             train20.extend(data[i-1-a] for a in range(40))
# print(len(train20))


def train_data(data,label,start_frame,end_frame):
    x_train = []
    y_train = []
    total_frame = end_frame - start_frame
    data_num=len(data)
    for i in range(data_num):
        x_train.extend(data[i][a] for a in range(start_frame,end_frame))
    for i in range(data_num):
        y_train.append(label[40 * i])
    x_train=np.reshape(x_train,[data_num,total_frame,7])
    y_train = np.array(y_train)
    return x_train,y_train

#
# for i in range(len(data)):
#     train20_42.extend(data[i][a] for a in range(20))
# train20_42=np.reshape(train20_42,[data_num,20,7])
# for i in range(len(data2)):
#     val20_42.extend(data2[i][a] for a in range(20))
# val20_42=np.reshape(val20_42,[data_num2,20,7])
#
#
#
# for i in range(len(data)):
#     train20_20.extend(data[i][20+a] for a in range(20))
# train20_20=np.reshape(train20_20,[data_num,20,7])
# for i in range(len(data2)):
#     val20_20.extend(data2[i][20+a] for a in range(20))
# val20_20=np.reshape(val20_20,[data_num2,20,7])
#
#
# for i in range(len(data)):
#     train20_43.extend(data[i][a] for a in range(10))
# train20_43=np.reshape(train20_43,[data_num,10,7])
# for i in range(len(data2)):
#     val20_43.extend(data2[i][a] for a in range(10))
# val20_43=np.reshape(val20_43,[data_num2,10,7])





def shuffle(X,Y):
  np.random.seed(10)
  randomList = np.arange(X.shape[0])
  np.random.shuffle(randomList)
  return X[randomList], Y[randomList]




def splitData(X,Y,rate):
  X_train = X[int(X.shape[0]*rate):]
  Y_train = Y[int(Y.shape[0]*rate):]
  X_val = X[:int(X.shape[0]*rate)]
  Y_val = Y[:int(Y.shape[0]*rate)]
  return X_train, Y_train, X_val, Y_val
#
# data=data[800:]
# double_label=double_label[800*40:]
# data2=data2[600:]
# double_label2=double_label2[600*40:]

x_train,y_train=train_data(data=data,label=double_label,start_frame=0,end_frame=5)
x_test,y_test=train_data(data=data2,label=double_label2,start_frame=20,end_frame=30)
X_train,Y_train=shuffle(x_train,y_train)
X_val,Y_val=shuffle(x_test,y_test)


def buildManyToOneModel(shape):
  model = Sequential()
  # model.add(LSTM(10, input_length=shape[1], input_dim=shape[2]))
  model.add(LSTM(128, input_shape=(shape[1],shape[2])))
  model.add(Dropout(0.5))
  # output shape: (1, 1)
  model.add(Dense(3,activation='softmax'))
  model.compile(loss="categorical_cro"
                     "ssentropy", optimizer="adam", metrics=['accuracy'])
  model.summary()
  return model

# model = buildManyToOneModel(X_train.shape)
# # callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
# # model.fit(X_train, Y_train, epochs=100, batch_size=128, validation_data=data_num(X_val, Y_val))
# model.fit(X_train, Y_train, epochs=100, batch_size=128,validation_data=(X_val, Y_val))
# # model.save("/home/mt/learn/NGSIM数据处理/model/model30-40.h5")
model = load_model('/home/mt/learn/NGSIM数据处理/model/train2.h5')

loss, accuracy = model.evaluate(X_val,Y_val)
print(accuracy)
y_predict=model.predict(X_val)
y_predict=np.round(y_predict)
y_val=[]
pre_val=[]
for i in range(len(y_predict)):
    if Y_val[i][0]==1:
        y_val.append([0])
    elif Y_val[i][1]==1:
        y_val.append([1])
    elif Y_val[i][2]==1:
        y_val.append([2])
for i in range(len(Y_val)):
    if Y_val[i][0]==1.0:
        pre_val.append([0])
    elif Y_val[i][1]==1.0:
        pre_val.append([1])
    elif Y_val[i][2]==1.0:
        pre_val.append([2])

# print(y_val)
# print('=================================================================')
# print(pre_val)




print(metrics.classification_report(Y_val,y_predict))       #输出结果，精确度、召回率、f-1分数
# print(metrics.confusion_matrix(Y_val, y_predict))






# y_predict=np.round(y_predict)
#
# KL_num=TL_num=TR_num=KL_pre=TL_pre=TR_pre=0
# for i in range(len(Y_val)):
#     if Y_val[i][0]==1:
#         KL_num+=1
#         if y_predict[i][0]==1.0:
#             KL_pre+=1
#     elif Y_val[i][1]==1:
#         TL_num+=1
#         if y_predict[i][1]==1.0:
#             TL_pre+=1
#     elif Y_val[i][2]==1:
#         TR_num+=1
#         if y_predict[i][2]==1.0:
#             TR_pre+=1
# KL_acc=KL_pre/KL_num
# TL_acc=TL_pre/TL_num
# TR_acc=TR_pre/TR_num
# print(accuracy)
# print(KL_acc)
# print(TL_acc)
# print(TR_acc)






