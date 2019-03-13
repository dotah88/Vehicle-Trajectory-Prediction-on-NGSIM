import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("/home/mt/learn/NGSIM数据处理/csv/4:00-4:15/Two_train.csv")
f=pd.read_csv("/home/mt/learn/NGSIM数据处理/csv/4:00-4:15/Two_label.csv")
df2=pd.read_csv("/home/mt/learn/NGSIM数据处理/csv/5:00-5:15/Two_train.csv")
f2=pd.read_csv("/home/mt/learn/NGSIM数据处理/csv/5:00-5:15/Two_label.csv")
label=f.iloc[:,8].values
label2=f2.iloc[:,8].values
data_frame=20

train_y=[]
train_y2=[]
data_num=int(len(label)/40)
data_num2=int(len(label2)/40)

# for i in range(data_num):
#     train_y.append(label[40*i])
# for i in range(data_num2):
#     train_y2.append(label2[40*i])

data=df.iloc[:,1:8].values
data2=df2.iloc[:,1:8].values
data=np.reshape(data,[data_num,40,7])
data2=np.reshape(data2,[data_num2,40,7])

data_svm=np.reshape(data,[-1,7])
data2_svm=np.reshape(data2,[-1,7])

train1=[]
train2=[]
train20_42=[]
train20_43=[]
train20_20=[]
val20_42=[]
val20_43=[]
val20_20=[]
label_43=[]
label2_43=[]
label_1=[]
label_2=[]

a=40-data_frame
for i in range(data_num):
    if i!=0:
        label_1.append(label[40*i-a-1])
        train1.append(data_svm[40*i-a-1])
for i in range(data_num2):
    if i!=0:
        label_2.append(label[40*i-a-1])
        train2.append(data2_svm[40*i-a-1])
train1=np.array(train1)
train2=np.array(train2)
label_1=np.array(label_1)
label_2=np.array(label_2)


for i in range(data_num):
    label_43.extend(label[40*i+a] for a in range(10))
for i in range(data_num2):
    label2_43.extend(label2[40*i + a] for a in range(10))



for i in range(len(data)):
    train20_42.extend(data[i][a] for a in range(20))
train20_42=np.reshape(train20_42,[-1,7])
for i in range(len(data2)):
    val20_42.extend(data2[i][a] for a in range(20))
val20_42=np.reshape(val20_42,[-1,7])



for i in range(len(data)):
    train20_20.extend(data[i][20+a] for a in range(20))
train20_20=np.reshape(train20_20,[-1,7])
for i in range(len(data2)):
    val20_20.extend(data2[i][20+a] for a in range(20))
val20_20=np.reshape(val20_20,[-1,7])


for i in range(len(data)):
    train20_43.extend(data[i][a] for a in range(10))
train20_43=np.reshape(train20_43,[-1,7])
for i in range(len(data2)):
    val20_43.extend(data2[i][a] for a in range(10))
val20_43=np.reshape(val20_43,[-1,7])

def shuffle(X,Y):
  np.random.seed(10)
  randomList = np.arange(X.shape[0])
  np.random.shuffle(randomList)
  return X[randomList], Y[randomList]

label_43=np.reshape(label_43,-1)
label2_43=np.reshape(label2_43,-1)



x_train,y_train=shuffle(train1,label_1)
x_test,y_test=shuffle(train2,label_2)


clf = svm.SVC(C=0.8, kernel='rbf', gamma=10, decision_function_shape='ovo')
clf.fit(x_train, y_train)

print(clf.score(x_train, y_train)) # 精度
print('训练集准确率：', accuracy_score(y_train, clf.predict(x_train)))
print(clf.score(x_test, y_test))
print('测试集准确率：', accuracy_score(y_test, clf.predict(x_test)))
y_predict=clf.predict(x_test)
y_predict=y_predict.reshape(-1)
comb=zip(y_test,y_predict)
print(list(comb))

