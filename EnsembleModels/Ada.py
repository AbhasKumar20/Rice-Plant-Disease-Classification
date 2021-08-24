# AdaBoost Classification
import pandas
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier

dataframe = pandas.read_csv("Dataset.csv", header=None)
array = dataframe.values
X = array[:,0:11]
Y = array[:,11]
seed = 7
num_trees = 30
kfold = model_selection.KFold(n_splits=20, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
acc=results.mean()*100
print "Accuracy from Ada classifier(in %):",acc