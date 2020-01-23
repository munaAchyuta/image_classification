from numpy.random import seed

seed(1)
import os
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from sklearn import svm, metrics, datasets
import pickle

base_dir = os.path.join(os.path.dirname(__file__), '../data')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# Directory with our training class1 pictures
train_blank_dir = os.path.join(train_dir, 'Blank')

# Directory with our training class2 pictures
train_cancel_dir = os.path.join(train_dir, 'Cancelled')

# Directory with our training class3 pictures
train_hand_dir = os.path.join(train_dir, 'Handwritten')

# Directory with our training class4 pictures
train_print_dir = os.path.join(train_dir, 'Printed')

# Directory with our validation class1 pictures
validation_blank_dir = os.path.join(validation_dir, 'Blank')

# Directory with our validation class2 pictures
validation_cancel_dir = os.path.join(validation_dir, 'Cancelled')

# Directory with our validation class3 pictures
validation_hand_dir = os.path.join(validation_dir, 'Handwritten')

# Directory with our validation class4 pictures
validation_print_dir = os.path.join(validation_dir, 'Printed')

# Directory with our test pictures
test_class_dir = os.path.join(test_dir, 'allclasses')

'''
# get shape of feature matrix
print('Feature matrix shape is: ', X_train.shape)

# define standard scaler
ss = StandardScaler()
# run this on our feature matrix
bees_stand = ss.fit_transform(X_train)

pca = PCA(n_components=500)
# use fit_transform to run PCA on our standardized matrix
bees_pca = ss.fit_transform(bees_stand)
# look at new shape
print('PCA matrix shape is: ', bees_pca.shape)
'''


train_test_filename = "../data_kfold/dll_{}.pkl"
for k_fold in [0,1,2]:
    print("k-fold number : {}".format(k_fold))
    with open(train_test_filename.format(k_fold), 'rb') as file:
        pickle_model = pickle.load(file)
    X_train = pickle_model.get('X_tr')
    Y_train = pickle_model.get('y_tr')
    X_val = pickle_model.get('X_te')
    Y_val = pickle_model.get('y_te')
    print((X_train.shape,Y_train.shape),(X_val.shape,Y_val.shape))
    
    for each_model_k in [0,1,2]:
        model = load_model('../model/mdl_wts_{}.hdf5'.format(each_model_k))

        pred = model.predict(X_val)
        Y_pred = np.argmax(pred, axis=1)
        Y_val_num = np.argmax(Y_val, axis=1)
        print("Classification report for - \nval_data--{} & model_k--{}:\n{}\n".format(k_fold, each_model_k, metrics.classification_report(Y_val_num, Y_pred)))