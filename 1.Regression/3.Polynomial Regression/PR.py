#Simple Linear Regression

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values 
y = dataset.iloc[:,[2]].values

#No need Splitting for Standard scalling cz Dataset is small and we want accurate result

#No need for Standard scalling cz library fnc take care of this

#Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Creating polynomial feature from x
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
#First collumn add all 1 CZ we need b0 which = 1
#Fitting x poly in dataset
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

    
#Visualising the Linear Regression results
plt.scatter(X,y,color = 'red') 
plt.plot(X,lin_reg.predict(X),color = 'blue')
plt.title("True or False(Linear Regression)")
plt.xlabel("Position") #CZ 0 collum contains exper
plt.ylabel("Salary") #CZ 1 collum contains Salary
plt.show()

#Visualising the Polynomial Regression results
plt.scatter(X,y,color = 'red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color = 'blue') #Have to predict depending polynomial Fea.... and can handle new features automatically 
plt.title("True or False(Polynomial Regression)")
plt.xlabel("Position") #CZ 0 collum contains exper
plt.ylabel("Salary") #CZ 1 collum contains Salary
plt.show()

#Predicting a new result with Linear Regression
lin_reg.predict([[6.5]]) #Cz it requires 2D 

#Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))#poly_reg.fit_transform cz we need predict from a singular value