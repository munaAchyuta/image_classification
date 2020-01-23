from numpy.random import seed

seed(1)
import os
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
import numpy as np
import pandas as pd

# Train model
from sklearn import svm, metrics, datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import SGDClassifier
import pickle

param_grid1 = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]
param_grid2 = [{'C': [0.1, 1, 10, 100, 1000]}]

train_test_filename = "../data_kfold/{}.pkl"
for k_fold in [0,1,2]:
    print("k-fold number : {}".format(k_fold))
    with open(train_test_filename.format(k_fold), 'rb') as file:
        pickle_model = pickle.load(file)
    X_train = pickle_model.get('X_tr')
    Y_train = pickle_model.get('y_tr')
    X_val = pickle_model.get('X_te')
    Y_val = pickle_model.get('y_te')
    #svc = svm.SVC()
    #clf = GridSearchCV(OneVsRestClassifier(svc), param_grid1, cv=3)
    #svc = svm.LinearSVC(penalty='l2',loss='squared_hinge')
    #clf = GridSearchCV(svc, param_grid2, cv=3, n_jobs=2)
    clf = svm.LinearSVC(penalty='l2',loss='squared_hinge',C=0.1)
    #clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
    clf.fit(X_train, Y_train)

    # View the accuracy score
    #print('Best score for training data:', clf.best_score_,"\n")

    # View the best parameters for the model found using grid search
    #print('Best C:',clf.best_estimator_.C,"\n")
    #print('Best Kernel:',clf.best_estimator_.kernel,"\n")
    #print('Best Gamma:',clf.best_estimator_.gamma,"\n")

    final_model = clf
    #final_model = clf.best_estimator_

    Y_pred = final_model.predict(X_val)

    #Y_pred = clf.predict(X_val)
    print("Classification report for - \n{}:\n{}\n".format(final_model, metrics.classification_report(Y_val, Y_pred)))

#import pickle
# Save to file in the current working directory
#pkl_filename = "../model/pickle_model.pkl"
#with open(pkl_filename, 'wb') as file:
#    pickle.dump(final_model, file)

# Load from file
#with open(pkl_filename, 'rb') as file:
#    pickle_model = pickle.load(file)