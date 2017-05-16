# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:30:33 2017

@author: user
"""

# Multiple linear regression

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values
                
# Label encoding to dummy variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3]) # Specify the array number
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
# Generally, regressor object automatically avoid this trap
X = X[:, 1:] 

           
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.2, random_state = 0)


