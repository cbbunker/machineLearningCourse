# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing the Data Set
dataSet = pd.read_csv('Data.csv')
#Independent Variables
# :, :-1 take all the rows and all the columns excpet for the last one
X = dataSet.iloc[:, :-1].values
#Dependent
#Get the last column 
Y = dataSet.iloc[:, 3 ].values

#********Missing Data: Could remove an obersvation with missing data. Bad idea.
#Most common, take the mean of the column.
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
#upper bound is excluded, 1:3 is columns 1 and 2
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#********Categorical Variables: Can't keep text, need to encode them somehow
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:,0] = labelEncoder_X.fit_transform(X[:,0])
#Need to prevent ML algorithms from thinking one category is better than another.
#Need to make columns for each category
oneHotEncoder_X = OneHotEncoder(categorical_features = [0])
X = oneHotEncoder_X.fit_transform(X).toarray()
#Encoding the dependent variable field
labelEncoder_Y = LabelEncoder()
Y = labelEncoder_Y.fit_transform(Y)

#**********Test/Train Splitting
#Splitting into the training set and the test set
#Don't want to overtrain the algorithm, we need to use some entries to see if it is accurate
from sklearn.model_selection import  train_test_split
#Test size is good between 0.2-0.35
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 0)

#************Scaling
#data scaling. Variables need to be within the same range so neither variable is dominated by the other
#standardization and Normalization
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

print(X_test)

#print(dataSet)
#print(X)
#print(Y)