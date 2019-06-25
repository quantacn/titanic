#coding=utf-8
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier



# use pandas to manage data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]
#中文
#print (train_df.describe(include=['object']))
#显示某一个的数据，并画个图
#train_df['Age'].plot()
#plt.show()
#['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'
#查看null值的列
#print (train_df.isnull().any())
#分组分析数据
#Pclass_Survived_0 = train_df.Pclass[train_df['Survived'] == 0].value_counts()
#Pclass_Survived_1 = train_df.Pclass[train_df['Survived'] == 1].value_counts()
#Pclass_Survived = pd.DataFrame({ 0: Pclass_Survived_0, 1: Pclass_Survived_1})
#删除列的时候必须制定axis=1
train_df=train_df.drop(['Name','PassengerId','Ticket'], axis=1)
test_df = test_df.drop(['Name','Ticket'], axis=1)
age_guess = train_df['Age'].median()
train_df.loc[train_df['Age'].isnull(),'Age']=age_guess
#train_df['Age'].fillna(age_guess)
Fare_guess=train_df['Fare'].mode()
train_df['Fare'].fillna(Fare_guess)
train_df=train_df.drop('Cabin',axis=1)
train_df['Sex']=train_df['Sex'].map({'female':0,'male':1}).astype(int)
#train_df['Embarked']=train_df['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)
#SibSp Parch列合并成新列，然后处理数据，分为三类。年龄亦如此。
train_df['col_Sib_new']=train_df['SibSp']+train_df['Parch']+1
train_df=train_df.drop(['SibSp','Parch'], axis=1)
#sns.tsplot(x='Fare',y='Survived',data=train_df)
#plt.show()
def Fam_label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 7)) | (s == 1):
        return 1
    elif (s > 7):
        return 0
train_df['col_Sib_new']=train_df['col_Sib_new'].apply(Fam_label)
dummies_Embarked = pd.get_dummies(train_df['Embarked'], prefix= 'Embarked')
train_df=pd.concat([train_df,dummies_Embarked],axis=1)
train_df=train_df.drop(['Embarked'], axis=1)
print (train_df.head())

