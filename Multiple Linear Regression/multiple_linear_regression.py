#import Machine learning packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import *
from sklearn.impute import *

#importing data
data_set = pd.read_csv('50_Startups.csv')
# independent feature
X = data_set.iloc[:,:-1].values
# dependent features
y = data_set.iloc[:,4].values

# encoding the categorical features
labelEncoder_X = LabelEncoder()
labelEncoder_X.fit(X[:,3])
X[:,3] = labelEncoder_X.transform(X[:,3])

#converting catgorical data to column
onehotencoder = OneHotEncoder()
X = np.append(X,onehotencoder.fit_transform(X[:,3].reshape(-1, 1)).toarray() ,axis=1) 
X= X[:,[4,5,6,0,1,2]]

# removing dummy variables
X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)
# print(X_train)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train,y_train)

print(regressor.predict(X_test))
print('')
print("")
print(y_test)

#Backward Elimination 
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
X = sm.add_constant(X.toarray())
# X = np.append(np.ones((50,1),X),axis=1)
# X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(y,X)
regressor_OLS.fit()
