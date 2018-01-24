from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
import numpy as np

flare = open('flare.csv', 'r')
x = []
y = []
for line in flare.readlines()[1:]:
    x.append(line.split(",")[:-1])
    y.append(line.split(",")[-1])

X = np.array(x)
Y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
classifier = tree.DecisionTreeClassifier(criterion="entropy")
classifier = classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_test)
score = accuracy_score(y_test, y_predict)
print(score)

classifier = tree.DecisionTreeClassifier(criterion="entropy", min_samples_leaf=20)
classifier = classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_test)
score = accuracy_score(y_test, y_predict)
print(score)

flare.close()



