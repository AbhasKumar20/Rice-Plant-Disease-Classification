# Voting Ensemble for Classification
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

dataframe = pandas.read_csv("Dataset.csv", header=None)
array = dataframe.values
X = array[:,0:11]
Y = array[:,11]
seed = 7
kfold = model_selection.KFold(n_splits=20, random_state=seed)

estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))

ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)
acc=results.mean()*100
print "Accuracy from voting classifier(in %):",acc