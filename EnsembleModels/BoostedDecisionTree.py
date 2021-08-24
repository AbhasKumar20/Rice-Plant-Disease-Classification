# Bagged Decision Trees for Classification

import pandas
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

dataframe = pandas.read_csv("Dataset.csv", header=None)
array = dataframe.values
X = array[:,0:11]
Y = array[:,11]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y,random_state=0)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


kfold = model_selection.KFold(n_splits=20, random_state=7)
cart = DecisionTreeClassifier()
model = BaggingClassifier(base_estimator=cart, n_estimators=100, random_state=7)
model.fit(X_train, y_train)
print 'Accuracy of Logistic regression classifier on training set: {:.2f}'.format(model.score(X_train, y_train))
print 'Accuracy of Logistic regression classifier on test set: {:.2f}'.format(model.score(X_test, y_test))
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
acc=results.mean()*100
print "Accuracy from BDT classifier(in %):",acc