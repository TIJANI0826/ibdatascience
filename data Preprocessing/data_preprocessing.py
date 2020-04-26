#import Machine learning packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import *
from sklearn.impute import *

#importing data
data_set = pd.read_csv('data.csv')
# independent feature
X = data_set.iloc[:,:-1].values
# dependent features
y = data_set.iloc[:,3].values

# # Fixing missing values
# imputer = SimpleImputer(missing_values=np.nan ,strategy="mean")
# imputer = imputer.fit(X[:,1:3])
# X[:,1:3] = imputer.transform(X[:,1:3])

# #encoding the categorical features
# labelEncoder_X = LabelEncoder()
# labelEncoder_X.fit(X[:,0])
# X[:,0] = labelEncoder_X.transform(X[:,0])
# #converting catgorical data to column
# onehotencoder = OneHotEncoder()
# X = np.append(X,onehotencoder.fit_transform(X[:,0].reshape(-1, 1)).toarray()
# ,axis=1) 

# X = np.delete(X,np.s_[0], axis = 1)

# labelencoder_y = LabelEncoder()
# y = labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

""" 
# feature scaling with StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

"""