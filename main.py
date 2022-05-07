import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('venv/predictive_maintenance.csv')  # importing dataset

# mof=Matrix of Features
mof = df.iloc[:, :-2].values  # all the rows and columns excluding last column

# dvv=Dependant variable vector
dvv = df.iloc[:, 8:].values  # all the rows and the last column

print(mof)
print(dvv)

# Taking care of missing values
from sklearn.impute import SimpleImputer  # simpleImputer is a class

imputer = SimpleImputer(missing_values=np.nan,
                        strategy='mean')  # Imputer object finds the missing value and apply the strategy
imputer.fit(mof[:, 3:])  # missing integer value
mof[:, 3:] = imputer.transform(mof[:, 3:])  # replacing the missing value


# Encoding categorical data
from sklearn.compose import ColumnTransformer
from  sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[2])],remainder='passthrough')
ct2=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
mof = np.array(ct.fit_transform(mof))
dvv = np.array(ct2.fit_transform(dvv))
print(type(mof))  # We can't have a string and process data so we converting the string into a value with one hot encoder
print(type(dvv))
"""from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
mof=np.array((le.fit_transform(mof)))
dvv=np.array(le.fit_transform(dvv))
print(dvv)  # we Converting string to 0 and 1 to process data
print(mof)"""
# Spliting dataset into the training set and Test set
from sklearn.model_selection import train_test_split
mof_train,mof_test,dvv_train,dvv_test= train_test_split(mof,dvv,test_size=0.2,random_state=1)   # splits the datasets into 4 variable
print(len(mof_train))# ->8000 training set
print(len(mof_test))# ->2000 testing set
print(len(dvv_test))# ->2000 testing set
print(len(dvv_train))# ->8000 training set
