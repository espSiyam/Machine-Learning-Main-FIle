#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values #Won't work if we use 0 insted :-1
y = dataset.iloc[:,4].values

#Encoding the independent variable
#Convering CV iinto number
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
#Convering CV iinto 0 and 1
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the dummy variablr trap
X = X[:,1:]

#Splitting the dataset into training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y ,test_size = 0.2,random_state = 0)

#We dont have to use feature scaling cz library will take care of it

#Fitting Multiple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)

#Fromula : b0.x0 + b1.X1 + 2x2 + ... pxp.
#x0 = 1 so,we need to add collum full of 1 in X variable in Backward Eliminataion
#Building the optimal model using Backward Eliminataion
#We use these to find the optimal variable
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int),values = X ,axis = 1) #50 = no of line ,1 = no of collum;axis = 0 if row.1 if collum
#Adding X next to collum full of zero
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() #The lower p value , the better
regressor_OLS.summary()

X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() #The lower p value , the better
regressor_OLS.summary()

X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() #The lower p value , the better
regressor_OLS.summary()

X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() #The lower p value , the better
regressor_OLS.summary()