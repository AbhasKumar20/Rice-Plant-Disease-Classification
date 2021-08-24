
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
from matplotlib import cm
from sklearn.model_selection import train_test_split
import numpy as np
import pandas
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from pandas.tools.plotting import scatter_matrix
from matplotlib import cm


dataframe = pd.read_csv("Dataset.csv", header=None)
array = dataframe.values
X = array[:,0:11]
Y = array[:,11]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y,random_state=0)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print 'Accuracy of Logistic regression classifier on training set: {:.3f}'.format(logreg.score(X_train, y_train))
print 'Accuracy of Logistic regression classifier on test set: {:.3f}'.format(logreg.score(X_test, y_test))


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)
print 'Accuracy of Decision Tree classifier on training set: {:.3f}'.format(clf.score(X_train, y_train))
print 'Accuracy of Decision Tree classifier on test set: {:.3f}'.format(clf.score(X_test, y_test))

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print 'Accuracy of K-NN classifier on training set: {:.3f}'.format(knn.score(X_train, y_train))
print 'Accuracy of K-NN classifier on test set: {:.3f}'.format(knn.score(X_test, y_test))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print 'Accuracy of LDA classifier on training set: {:.3f}'.format(lda.score(X_train, y_train))
print 'Accuracy of LDA classifier on test set: {:.3f}'.format(lda.score(X_test, y_test))

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print 'Accuracy of GNB classifier on training set: {:.3f}'.format(gnb.score(X_train, y_train))
print 'Accuracy of GNB classifier on test set: {:.3f}'.format(gnb.score(X_test, y_test))

from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
print 'Accuracy of SVM classifier on training set: {:.3f}'.format(svm.score(X_train, y_train))
print 'Accuracy of SVM classifier on test set: {:.3f}'.format(svm.score(X_test, y_test))


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = clf.predict(X_test)
print confusion_matrix(y_test, pred)
print classification_report(y_test, pred)


import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import matplotlib.patches as mpatches


k_range = range(1,16)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append((knn.score(X_test, y_test)*100))
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy(%)')
plt.scatter(k_range, scores)
plt.xticks((np.arange(1,16, step=1)))
plt.show()

