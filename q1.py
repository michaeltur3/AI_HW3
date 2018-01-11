from sklearn.model_selection import StratifiedKFold
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np

flare = open('flare.csv', 'r')
x = []
y = []
for line in flare.readlines()[1:]:
    x.append(line.split(",")[:-1])
    y.append(line.split(",")[-1])

X = np.array(x)
Y = np.array(y)

kf = StratifiedKFold(n_splits=4)
classifier = tree.DecisionTreeClassifier(criterion="entropy")
score_sum = 0
confusion_mat_sum = np.zeros(shape=(2, 2))
for train, test in kf.split(X, Y):
    classifier = classifier.fit(X[train], Y[train])
    y_predict = classifier.predict(X[test])
    score_sum += accuracy_score(Y[test], y_predict)
    confusion_mat_sum += confusion_matrix(Y[test], y_predict)

score_avg = score_sum / 4
confusion_mat_avg = confusion_mat_sum / 4
print(score_avg)
print(confusion_mat_avg)

flare.close()



