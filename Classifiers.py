
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
from matplotlib import cm
from sklearn.model_selection import train_test_split
import numpy as np
from pandas.tools.plotting import scatter_matrix
from matplotlib import cm


dfs = pd.read_excel('Train.xlsx', sheet_name=None)



#feature_names = ['Area', 'Peri', 'A_ratio', 'Extent','Dia','Major','Minor','Ratio','noc','angle','ff']
feature_names = ['Area','Extent','Dia','Major','Minor','Ratio','noc']
X = dfs[feature_names]
y = dfs['Label']



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print 'Accuracy of Logistic regression classifier on training set: {:.2f}'.format(logreg.score(X_train, y_train))
print 'Accuracy of Logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test))

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)
print 'Accuracy of Decision Tree classifier on training set: {:.2f}'.format(clf.score(X_train, y_train))
print 'Accuracy of Decision Tree classifier on test set: {:.2f}'.format(clf.score(X_test, y_test))

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print 'Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train))
print 'Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print 'Accuracy of LDA classifier on training set: {:.2f}'.format(lda.score(X_train, y_train))
print 'Accuracy of LDA classifier on test set: {:.2f}'.format(lda.score(X_test, y_test))

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print 'Accuracy of GNB classifier on training set: {:.2f}'.format(gnb.score(X_train, y_train))
print 'Accuracy of GNB classifier on test set: {:.2f}'.format(gnb.score(X_test, y_test))

from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
print 'Accuracy of SVM classifier on training set: {:.2f}'.format(svm.score(X_train, y_train))
print 'Accuracy of SVM classifier on test set: {:.2f}'.format(svm.score(X_test, y_test))


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = knn.predict(X_test)
print confusion_matrix(y_test, pred)
print classification_report(y_test, pred)


import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import matplotlib.patches as mpatches


k_range = range(1, 25)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks((np.arange(1, 26, step=2)))
plt.show()