
# Random Forest Classification

import pandas
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

dataframe = pandas.read_csv("Dataset.csv", header=None)
array = dataframe.values
X = array[:,0:11]
Y = array[:,11]
seed = 7
num_trees = 100
max_features = 5
kfold = model_selection.KFold(n_splits=20, random_state=seed)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
acc=results.mean()*100
print "Accuracy from RF classifier(in %):",acc