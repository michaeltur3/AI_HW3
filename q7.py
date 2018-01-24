from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sfs import sfs

def getFeatureSubset(lst, V):
    res = []
    for l in lst:
        inner_res = []
        for i in range(32):
            if i in V:
                inner_res.append(l[i])
        res.append(inner_res)
    return res


def score_fun(clf, V, x, y):
    X = np.array(getFeatureSubset(x, V))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    return accuracy_score(y_test, y_predict)


flare = open('flare.csv', 'r')
x = []
y = []
for line in flare.readlines()[1:]:
    x.append(line.split(",")[:-1])
    y.append(line.split(",")[-1])

X = np.array(x)
Y = np.array(y)

classifier = KNeighborsClassifier(n_neighbors=5)
# regular KNN
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_test)
score = accuracy_score(y_test, y_predict)
print(score)

# using SFS
V = sfs(X_train, y_train, 8, classifier, score_fun)
X_train_subset = getFeatureSubset(X_train, V)
X_test_subset = getFeatureSubset(X_test, V)
classifier.fit(X_train_subset, y_train)
y_predict = classifier.predict(X_test_subset)
score = accuracy_score(y_test, y_predict)
print(score)


