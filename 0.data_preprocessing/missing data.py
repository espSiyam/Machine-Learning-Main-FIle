#Data preprocessing

#impoting the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset  = pd.read_csv(r'C:\Users\Siyam\Desktop\Machine Learning\0.data_preprocessing\Data.csv')
X = dataset.iloc[:,[0,1,2]].values
y = dataset.iloc[:,[3]].values


#Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(X[:,1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Encoding the data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(
    transformers = [
        ('encoder',OneHotEncoder(),[0])
    ],
    remainder='passthrough'
)
X = np.array(ct.fit_transform(X))

from sklearn.preprocessing import LabelEncoder
labelencoder_y =  LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Splitting the dataset into train,test,split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) #CZ we want to learn from train data
