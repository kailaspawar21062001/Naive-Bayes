# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 22:46:04 2023

@author: kailas
"""

PROBLEM::
BUSINESS OBJECTIVE::--1) Prepare a classification model using Naive Bayes 
for salary data.:USE BOTH 'SalaryData_Test' and 'SalaryData_Train'
    


#Import Liabrary
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pylab as plt

#Dataset
train=pd.read_csv("D:/data science assignment/Assignments/12.Naive Bayes/SalaryData_Train.csv")
test=pd.read_csv("D:/data science assignment/Assignments/12.Naive Bayes/SalaryData_Test.csv")

train
train.info()
train.describe()
train.head()
train.tail()
train.shape


test
test.info()
test.describe()
test.head()
test.tail()
test.shape

#Data Cleaning on train Datset
train[train.duplicated()]
Train = train.drop_duplicates()
Train.isna().sum()
Train.Salary.value_counts()

#Data Cleaning on test Datset
test['maritalstatus'].value_counts()
test[test.duplicated()]
Test=test.drop_duplicates()
Test.isna().sum()
Test.Salary.value_counts()


#Visulazations
#Train Dataset
sns.countplot(x='Salary',data=Train)
plt.xlabel('Salary')
plt.ylabel('count')
plt.show()
Train['Salary'].value_counts()


#Test Dataset
plt.xlabel('Salary')
sns.countplot(x='Salary',data=Test)
plt.ylabel('count')
plt.show()
Train['Salary'].value_counts()

sns.scatterplot(Train['occupation'],Train['workclass'],hue=Train['Salary'])
pd.crosstab(Train['Salary'],Train['occupation']).mean().plot(kind='bar')
pd.crosstab(Train['Salary'],Train['workclass']).mean().plot(kind='bar')
pd.crosstab(Train['Salary'],Train['relationship']).mean().plot(kind='bar')
pd.crosstab(Train['Salary'],Train['education']).mean().plot(kind='bar')


string_columns = ["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]


##Preprocessing the data.

from sklearn.preprocessing import LabelEncoder
l = LabelEncoder()
for i in string_columns:
        Train[i]= l.fit_transform(Train[i])
        Test[i]=l.fit_transform(Test[i])
        
##Capturing the column.as 'col'
col = Train.columns
col 
        
 
# storing the values in x_train,y_train,x_test & y_test for spliting the data in train and test for analysis
x_train = Train[col[0:13]].values
y_train = Train[col[13]].values
x_test = Test[col[0:13]].values
y_test = Test[col[13]].values        


##Normalmization
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)

x_train=norm_func(x_train)
x_test=norm_func(x_test)


## GaussianNB for numerical data
from sklearn.naive_bayes import GaussianNB as GB
model=GB()

#Fitting as well as Predicting the model
train_pred=model.fit(x_train,y_train).predict(x_train)
test_pred=model.fit(x_train,y_train).predict(x_test)


train_acc=np.mean(train_pred==y_train)
train_acc

test_acc=np.mean(test_pred==y_test)
test_acc

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, test_pred)
confusion_matrix

#calculating the accuracy of this model w.r.t. this dataset
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,test_pred))
