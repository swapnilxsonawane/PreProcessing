# Here we will be learning about the data PreProcessing. 
# In ML , it is very important to cleanse the data before its feed to the model. Without this cleaning the model
# would not work. The model will give wrong prediction. Hence we need to put an effort to make data work for the model.
# The data cleaning for each model might vary at the detail level But Basic data preprocessing is same for every model.
# Here in this section, we will be concentrating on Basic processing for each model.
# We are using two sets of data to make concepts clear.
# One data is country and average income and other is companoes and their profit data.


# This is very basic tutorial for beginners. Along the learning path we will keep on adding more adavnce features.



# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('CompaniesData.csv')


# In all of our current learning , we are assuming that there will be only one dependent variable.
# For multiple dependent variables, we will explore that area in separate tutorial. 

# Since last column is dependent variable. We will separate it from the rst of the data.
# X: will have all independent variable and Y will have only one dependent variable.


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Categorical variable  in this case is the state. This need to be transfered in to numbers.
# In ML, machine understands only numbers. Hence categorical variables are converted into Numbers.
# But at the same time , we need to take care if they are Nominal or ordinal variable.
# here in this case they are nominal.

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])

# see the output: its tranfromed into numbers 0,1 , 2
X[:, 3]

# See original MatriX X is transformed.
X

# Now depending upon the number of categ variable, encoder will add number of columns by below code.

onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
X

# We need to avoid the Dummy Variable Trap. Since  out of three columns of categorical variable if 2 are absent means it should be present in third variable. Hence one column can be deleted for categorical variable
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# 
# Feature Scaling:  It helps to normalize the data into particular range. It helps in speeding up calculations in algorithms.

# but it reduces the visibility of the data. Hence in some practices its not followed.
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)




# Importing the dataset
dataset1 = pd.read_csv('CountryIncome.csv')
X1 = dataset1.iloc[:, :-1].values
y1 = dataset1.iloc[:, 3].values

X1
y1

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X1[:, 1:3] = imputer.transform(X1[:, 1:3])

X1


