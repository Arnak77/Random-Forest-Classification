## Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


## Importing the dataset
dataset = pd.read_csv(r"D:\NIT\JANUARY\11-12 JAN(enseble learn)\10th,11th\5. RANDOM FOREST\Social_Network_Ads.csv")
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values


## Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


#Feature Scaling not important

## Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
clas = RandomForestClassifier(criterion= 'entropy',max_depth= None, min_samples_leaf= 4, min_samples_split= 2, n_estimators= 50)
clas.fit(X_train, y_train)


## Predicting the Test set results
y_pred = clas.predict(X_test)


## Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

## Making the accuracy_score
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)


bias = clas.score(X_train, y_train)
bias

variance = clas.score(X_test, y_test)
variance





































































