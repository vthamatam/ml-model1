# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 16:27:29 2022

@author: vthamatam
"""

import numpy as np
import pandas as pd

training_data = pd.read_csv("storepurchasedata.csv")
training_data.describe()

X = training_data.iloc[:, :-1].values
Y = training_data.iloc[:,-1].values

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = .20,random_state=0)

#feature scale the data to make age,salary same range so ML will not get influenced  by salary which is higher range
from sklearn.preprocessing import StandardScaler

sc = StandardScaler() 
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#build clasification model using K-NN technic
from sklearn.neighbors import KNeighborsClassifier
clasifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
clasifier.fit(X_train, Y_train)
Y_pred = clasifier.predict(X_test)
Y_prob = clasifier.predict_proba(X_test)[:,1]

#confusion metrics

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

# acuracy of model

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,Y_pred))

# predict for new data 

new_predict = clasifier.predict(sc.transform(np.array([[40,20000]])))

new_predictProb = clasifier.predict_proba(sc.transform(np.array([[40,20000]])))


# pickle the model and standard scalar
import pickle

model_file = "classifier.pickle"
pickle.dump(clasifier,open(model_file,'wb'))

model_file = "sc.pickle"
pickle.dump(sc,open(model_file,'wb'))
