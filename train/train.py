import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.models import load_model
import random

df = pd.read_csv("/home/mt/learn/NGSIM数据处理/csv/80/5:00-5:15/训练数据新/Double_train.csv")
f=pd.read_csv("/home/mt/learn/NGSIM数据处理/csv/80/5:00-5:15/训练数据新/Double_label.csv")
df2=pd.read_csv("/home/mt/learn/NGSIM数据处理/csv/80/5:15-5:30/new/Double_train.csv")
f2=pd.read_csv("/home/mt/learn/NGSIM数据处理/csv/80/5:15-5:30/new/Double_label.csv")
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

data_num=int(len(label)/40)
data_num2=int(len(label2)/40)
data=df.iloc[:,[2,3,4,5,6,7,8,10]].values
data2=df2.iloc[:,[2,3,4,5,6,7,8,10]].values
data=np.reshape(data,[data_num,40,8])
data2=np.reshape(data2,[data_num2,40,8])

# data=data[800:]
# double_label=double_label[800*40:]
# data2=data2[600:]
# double_label2=double_label2[600*40:]
# train_data_num=len(data)
# train_data_num2=len(data2)

def pro_label(data_num,label):
    train_label=[]
    for i in range(data_num):
        train_label.append(label[i*40])
    return train_label

train_label=pro_label(data_num=data_num,label=double_label)
test_label=pro_label(data_num=data_num2,label=double_label2)



def train_data(data,label,data_num):
    x_train = []
    y_train = []
    total_frame = 10
    train_data_num=10000
    for i in range(train_data_num):
        start_frame = random.randint(0, 30)
        end_frame = start_frame + 10
        train_id = random.randint(0, data_num-1)
        x_train.append(data[train_id][start_frame:end_frame])
        y_train.append(label[train_id])
    x_train = np.reshape(x_train, [train_data_num, total_frame, 8])
    y_train=np.array(y_train)
    return x_train,y_train


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

# data=data[800:]
# double_label=double_label[800*40:]
# data2=data2[600:]
# double_label2=double_label2[600*40:]
#
# x_train,y_train=train_data(data=data,label=double_label,start_frame=0,end_frame=10)
# x_test,y_test=train_data(data=data2,label=double_label2,start_frame=0,end_frame=10)
# X_train,Y_train=shuffle(x_train,y_train)
# X_val,Y_val=shuffle(x_test,y_test)

def buildManyToOneModel(shape):
  model = Sequential()
  # model.add(LSTM(128, input_length=shape[1], input_dim=shape[2]))
  # model.add(Dropout(0.5))
  model.add(LSTM(50, input_shape=(shape[1],shape[2]),return_sequences=True))
  model.add(Dropout(0.5))
  model.add(LSTM(100,return_sequences=False))
  model.add(Dropout(0.5))
  model.add(Dense(3,activation='softmax'))
  model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
  model.summary()
  return model

x_train,y_train=train_data(data=data,label=train_label,data_num=data_num)
x_test,y_test=train_data(data=data2,label=test_label,data_num=data_num2)


model = buildManyToOneModel(x_train.shape)
model.fit(x_train, y_train, epochs=100, batch_size=128,validation_data=(x_test, y_test))
model.save('/home/mt/learn/NGSIM数据处理/model/2LSTM_new.h5')
loss, accuracy = model.evaluate(x_test,y_test)
print(loss,accuracy)