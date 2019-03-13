import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

df = pd.read_csv("/home/mt/learn/NGSIM数据处理/csv/4:00-4:15/Two_train.csv")
f=pd.read_csv("/home/mt/learn/NGSIM数据处理/csv/4:00-4:15/Two_label.csv")
df2=pd.read_csv("/home/mt/learn/NGSIM数据处理/csv/5:00-5:15/Two_train.csv")
f2=pd.read_csv("/home/mt/learn/NGSIM数据处理/csv/5:00-5:15/Two_label.csv")

label=f.iloc[:,8].values
label2=f2.iloc[:,8].values
train_y=[]
train_y2=[]
data_num=int(len(label)/40)
data_num2=int(len(label2)/40)

for i in range(data_num):
    train_y.append(label[40*i])
train_y=np.reshape(train_y,[data_num,1])
for i in range(data_num2):
    train_y2.append(label2[40*i])
train_y2=np.reshape(train_y2,[data_num2,1])

data=df.iloc[:,1:8].values
data2=df2.iloc[:,1:8].values
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

for i in range(len(data)):
    train20_42.extend(data[i][a] for a in range(20))
train20_42=np.reshape(train20_42,[data_num,20,7])
for i in range(len(data2)):
    val20_42.extend(data2[i][a] for a in range(20))
val20_42=np.reshape(val20_42,[data_num2,20,7])



for i in range(len(data)):
    train20_20.extend(data[i][20+a] for a in range(20))
train20_20=np.reshape(train20_20,[data_num,20,7])
for i in range(len(data2)):
    val20_20.extend(data2[i][20+a] for a in range(20))
val20_20=np.reshape(val20_20,[data_num2,20,7])


for i in range(len(data)):
    train20_43.extend(data[i][a] for a in range(10))
train20_43=np.reshape(train20_43,[data_num,10,7])
for i in range(len(data2)):
    val20_43.extend(data2[i][a] for a in range(10))
val20_43=np.reshape(val20_43,[data_num2,10,7])


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



X_train,Y_train=shuffle(train20_42,train_y)
X_val,Y_val=shuffle(val20_42,train_y2)


def buildManyToOneModel(shape):
  model = Sequential()
  model.add(LSTM(10, input_length=shape[1], input_dim=shape[2]))
  # output shape: (1, 1)
  model.add(Dense(1,activation='sigmoid'))
  model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
  model.summary()
  return model

x_test=X_val
y_test=Y_val
model = buildManyToOneModel(X_train.shape)
# callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
# model.fit(X_train, Y_train, epochs=100, batch_size=128, validation_data=(X_val, Y_val))
model.fit(X_train, Y_train, epochs=100, batch_size=128, verbose=1)
y_predict=model.predict(x_test)
y_test=np.reshape(y_test,-1)
y_predict=np.reshape(y_predict,-1)
combine=list(zip(y_test,y_predict))
loss, accuracy = model.evaluate(x_test, y_test)
print(loss,accuracy)
