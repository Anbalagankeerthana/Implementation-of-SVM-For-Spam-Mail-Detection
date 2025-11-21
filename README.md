# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Import the packages.

2.Analyse the data. 

3.Use modelselection and Countvectorizer to preditct the values.

4.Find the accuracy and display the result.

## Program:
Program to implement the SVM For Spam Mail Detection..
Developed by:KEERTHANA 
RegisterNumber:212224220046  
```

import chardet
file='/content/spam (1).csv'
with open(file, 'rb') as rawdata:
     print('Result output')
    result = chardet.detect(rawdata.read(10000))
result

import pandas as pd
data=pd.read_csv("/content/spam (1).csv",encoding="windows-1252")

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
print("y_pred")
y_pred


from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print("accuracy")
accuracy
```
## Output:

Data Head()


<img width="1040" height="81" alt="image" src="https://github.com/user-attachments/assets/a99bd327-a92e-4bbc-bbed-e03b784d4417" />


Data.Info


<img width="1052" height="345" alt="image" src="https://github.com/user-attachments/assets/9ed3ac2f-2d12-466f-a564-8dd76c8f44b7" />


Data.isnull().sum()


<img width="694" height="349" alt="image" src="https://github.com/user-attachments/assets/66c155b1-298e-43aa-ad22-ed63f5797876" />



y_pred()


<img width="335" height="207" alt="image" src="https://github.com/user-attachments/assets/025127aa-5600-4550-88cc-4a231884103f" />





Accuracy


<img width="407" height="100" alt="image" src="https://github.com/user-attachments/assets/f68e3755-6a81-4709-bb74-55062bfde09c" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

