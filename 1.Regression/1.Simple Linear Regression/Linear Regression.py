#Simple Linear Regression

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values #Won't work if we use 0 insted :-1
y = dataset.iloc[:,1].values

#Splitting the dataset into training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y ,test_size = 1/3,random_state = 0)

#No need for Standard scalling cz library fnc take care of this

#Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
#Fitting model based on training data>We will fit that model on test data to predict result

#Predicting the test set results
y_pred = regressor.predict(X_test)

#Visualising the Training set results
plt.scatter(X_train,y_train,color = 'red')
plt.plot(X_train,regressor.predict(X_train),color = 'blue')
plt.title("Salary VS Experience(Training set)")
plt.xlabel("Years of experience") #CZ 0 collum contains exper
plt.ylabel("Salary") #CZ 1 collum contains Salary
plt.show()