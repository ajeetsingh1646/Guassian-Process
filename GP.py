# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 22:54:25 2017

@author: ajeet
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import exp
from sklearn import gaussian_process
from scipy import linalg as l
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#Loading data
ActP1 = pd.read_table("Squash1PlayerActivity.txt")
ActP2 = pd.read_table("Squash2PlayerActivity.txt")
PosP1 = pd.read_table("squash1Position.txt")
PosP2 = pd.read_table("squash2Position.txt")

# Data manupulation and making dataframes
act1 = ActP1
x_tr1 = act1.iloc[:,:].values
x_tr1 = pd.DataFrame(x_tr1)
x_tr1 = x_tr1[[0,1,2,4,5,6,7,3]]
X1 = pd.DataFrame(x_tr1.iloc[:, :-1].values)
y1 = pd.DataFrame(x_tr1.iloc[:, 7].values)

#converting string data to the numerical data
y1 = y1.replace(["S","LL","C","CS","DL","DC","LOL","LOC","KL","KC","VLL","VC","VCS","VDL","VDC","VBN","VBO","VBR","VKL","VKC","BN","BO","BR","BS","COS"],["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25"])
X1[3] = X1[3].replace(["i","l","s","t","n"],[1,2,3,4,5])
X1[4] = X1[4].replace(["f","b"],[1,2])

#Data normalization/scalling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X1 = sc_X.fit_transform(X1)
y1 = sc_X.fit_transform(y1)

#Split data to train and test
from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size= 0.2, random_state=0)

#Function for the gaussian process 
gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
gp.fit(X_train1, y_train1)

# Pridictions 
y_pred1, sigma2_pred1 = gp.predict(X_test1, eval_MSE=True)

# plotting the predictions
plt.plot(X_train1, y_train1, 'r+')
plt.plot(X_test1, y_pred1, 'b+')
plt.xlabel("range")
plt.ylabel("Predictions")

#------------------------------------------------------------------------------------#


# Data manupulation and making dataframes
act2 = ActP2
x_tr2 = act2.iloc[:,:].values
x_tr2 = pd.DataFrame(x_tr2)
x_tr2 = x_tr2[[0,1,2,4,5,6,7,3]]
X2 = pd.DataFrame(x_tr2.iloc[:, :-1].values)
y2 = pd.DataFrame(x_tr2.iloc[:, 7].values)

#converting string data to the numerical data
y2 = y2.replace(["S","LL","C","CS","DL","DC","LOL","LOC","KL","KC","VLL","VC","VCS","VDL","VDC","VBN","VBO","VBR","VKL","VKC","BN","BO","BR","BS","COS"],["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25"])
X2[3] = X2[3].replace(["i","l","s","t","n"],[1,2,3,4,5])
X2[4] = X2[4].replace(["f","b"],[1,2])

#Data normalization/scalling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X2 = sc_X.fit_transform(X2)
y2 = sc_X.fit_transform(y2)

# Split data to train and test
from sklearn.model_selection import train_test_split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size= 0.2, random_state=0)

# Function for the gaussian process 
gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
gp.fit(X_train2, y_train2)

# Pridictions 
y_pred2, sigma2_pred2 = gp.predict(X_test2, eval_MSE=True)

# plotting the predictions
plt.plot(X_train2, y_train2, 'r+')
plt.plot(X_test2, y_pred2, 'b+')
plt.xlabel("range")
plt.ylabel("Predictions")

#-------------------------------------------------------------------------------------#

# Data manipulation
x_tr3 = PosP1
X3 = pd.DataFrame(x_tr3.iloc[:, :-1])
y3 = pd.DataFrame(x_tr3.iloc[:, 10])

# Data normalization/scalling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X3 = sc_X.fit_transform(X3)
y3 = sc_X.fit_transform(y3)

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X3[:, 0:9])
X3[:, 0:9] = imputer.transform(X3[:, 0:9])
imputer1 = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer1 = imputer1.fit(y3[:, :])
y3[:,:] = imputer1.transform(y3[:,:])

# Split data to train and test
from sklearn.model_selection import train_test_split
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size= 0.3, random_state=0)

# Function for the gaussian process 
gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
gp.fit(X_train3, y_train3)

# Pridictions 
y_pred3, sigma2_pred3 = gp.predict(X_test3, eval_MSE=True)


# plotting the predictions
plt.plot(X_train3 , y_train3, 'r+')
plt.plot(X_test3, y_pred3,"b+")
plt.xlabel("range")
plt.ylabel("Predictions")
#-------------------------------------------------------------------------------------#

# Data manipulation
x_tr4 = PosP2
X4 = pd.DataFrame(x_tr4.iloc[:, :-1])
y4 = pd.DataFrame(x_tr4.iloc[:, 10])

# Data normalization/scalling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X4 = sc_X.fit_transform(X4)
y4 = sc_X.fit_transform(y4)

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X4[:, 0:9])
X4[:, 0:9] = imputer.transform(X4[:, 0:9])
imputer1 = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer1 = imputer1.fit(y4[:, :])
y4[:,:] = imputer1.transform(y4[:,:])

# Split data to train and test
from sklearn.model_selection import train_test_split
X_train4, X_test4, y_train4, y_test4 = train_test_split(X4, y4, test_size= 0.3, random_state=0)

# Function for the gaussian process 
gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
gp.fit(X_train4, y_train4)

# Pridictions 
y_pred4, sigma2_pred4 = gp.predict(X_test4, eval_MSE=True)


# plotting the predictions
plt.plot(X_train4 , y_train4, 'r+')
plt.plot(X_test4, y_pred4, 'b+')
plt.xlabel("range")
plt.ylabel("Predictions")
