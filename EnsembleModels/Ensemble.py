import pandas
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


dataframe = pandas.read_csv("Dataset.csv")
array= dataframe.values
X = array[:,0:11]
Y = array[:,-1]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y,random_state=0)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
seed = 7
kfold = model_selection.KFold(n_splits=25, random_state=seed)

#BDT---------------------------------------
cart = DecisionTreeClassifier()
num_trees = 200
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
model.fit(X_train, y_train)
print 'Accuracy of Bagged Decision Trees for Classification on training set: {:.3f}'.format(model.score(X_train, y_train))
print 'Accuracy of Bagged Decision Trees for Classification on test set: {:.3f}'.format(model.score(X_test, y_test))
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
acc=results.mean()*100
#print "Accuracy from BDT classifier(in %):",acc

#RF---------------------------------------------------------
max_features = 7
kfold = model_selection.KFold(n_splits=20, random_state=seed)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
model.fit(X_train, y_train)

print 'Accuracy of ET on training set: {:.3f}'.format(model.score(X_train, y_train))
print 'Accuracy of ET on test set: {:.3f}'.format(model.score(X_test, y_test))
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
acc=results.mean()*100
#print "Accuracy from RF classifier(in %):",acc

#ExtraTreesClassifier---------------------------------------------
model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print confusion_matrix(y_test, pred)
print classification_report(y_test, pred)
print 'Accuracy of RF on training set: {:.3f}'.format(model.score(X_train, y_train))
print 'Accuracy of RF on test set: {:.3f}'.format(model.score(X_test, y_test))
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
acc=results.mean()*100
#print "Accuracy from RF classifier(in %):",acc


#AdaBoostClassifier----------------------------------------------
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
model.fit(X_train, y_train)
print 'Accuracy of AdaBoostClassifier on  training set: {:.3f}'.format(model.score(X_train, y_train))
print 'Accuracy of  AdaBoostClassifier on test set: {:.3f}'.format(model.score(X_test, y_test))
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
acc=results.mean()*100
#print "Accuracy from Ada classifier(in %):",acc


#GradientBoostingClassifier------------------------------------------------------------
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
model.fit(X_train, y_train)
print 'Accuracy of GradientBoostingClassifier on training set: {:.3f}'.format(model.score(X_train, y_train))
print 'Accuracy of GradientBoostingClassifier on test set: {:.3f}'.format(model.score(X_test, y_test))
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
acc=results.mean()*100
#print "Accuracy from SGC classifier(in %):",acc

#VotingClassifier---------------------------------------------
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
ensemble = VotingClassifier(estimators)
ensemble.fit(X_train, y_train)
print 'Accuracy of VotingClassifier on training set: {:.3f}'.format(model.score(X_train, y_train))
print 'Accuracy of VotingClassifier on test set: {:.3f}'.format(model.score(X_test, y_test))
results = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)
acc=results.mean()*100
#print "Accuracy from voting classifier(in %):",acc
