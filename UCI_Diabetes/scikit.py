import numpy as np
import pandas as pd
from sklearn import linear_model, ensemble, svm, metrics
from sklearn import model_selection

# load data and split train and test
raw = np.loadtxt('diabete.data', delimiter=',')
X, y = raw[:, : 8], raw[:, -1]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)


# parameter tunings
# params = {'kernel': ['linear', 'rbf'], 'C': [0.001, 0.01, 0.1, 1], 'gamma': ['auto']}
# clf = model_selection.GridSearchCV(svm.SVC(), verbose=2, cv=5,
#                                   param_grid=params, scoring='accuracy')
# clf.fit(X_train, y_train)
# print(clf.best_estimator_)
# print(clf.cv_results_)


# # Logistics Regression
# lr = linear_model.LogisticRegression(C=20)
# # cross validation
# scores = model_selection.cross_val_score(lr, X_train, y_train, cv=5, scoring='accuracy')
# print(scores.mean(), scores.std())

# # predict on test
# lr.fit(X_train, y_train)
# y_pred = lr.predict(X_test)
# accuracy = lr.score(X_test, y_test)
# factors = pd.DataFrame(data=lr.coef_[0],
#                        index=['Number of times pregnant', 'Plasma glucose concentration', 'Diastolic blood pressure',
#                            'Triceps skin fold thickness', 'Serum insulin', 'Body mass index', 'Diabetes pedigree', 'Age'])
# print(factors)


# # Adaboost
# ada = ensemble.AdaBoostClassifier(n_estimators=50, learning_rate=0.5)
# # cross validation
# scores = model_selection.cross_val_score(ada, X_train, y_train, cv=5, scoring='accuracy')
# print(scores.mean(), scores.std())

# # predict on test
# ada.fit(X_train, y_train)
# y_pred = ada.predict(X_test)
# accuracy = ada.score(X_test, y_test)


# Support Vectors
svc = svm.SVC(kernel='linear', C=1, gamma='auto')
# # cross validation
# scores = model_selection.cross_val_score(svc, X_train, y_train, cv=5, scoring='accuracy')
# print(scores.mean(), scores.std())

# predict on test
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
accuracy = svc.score(X_test, y_test)

# get result
print('\nAccuracy: %.2f%%' % accuracy)
print('\n Confusion Matrix')
print(metrics.confusion_matrix(y_test, y_pred))
