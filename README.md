# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipment Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter Notebook

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Tarun S
RegisterNumber: 212223040226
*/
```
```c
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
## Result output:
![image](https://github.com/Tarun-2006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145584190/cba0ab37-44fc-47a6-8709-59a64f78988c)



## data.head():
![image](https://github.com/Tarun-2006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145584190/632dccc6-3b59-4d26-8cf2-bcfa1332a317)


## data.info():
![image](https://github.com/Tarun-2006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145584190/c530d131-fff6-48a8-b44b-0ce95e3985e8)


## Y_prediction value:
![image](https://github.com/Tarun-2006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145584190/55e57843-2010-454b-8cc9-e204c2ebee68)


 ## Accuracy value:
![image](https://github.com/Tarun-2006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145584190/382fa110-b93c-4a8c-8450-258c802994f1)



 



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using Python programming.
