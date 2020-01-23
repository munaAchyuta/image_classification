# -*- coding: utf-8 -*-
# author - U63411
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



# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1. / 255)
#train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=40, horizontal_flip=True, fill_mode='nearest')#, rotation_range=40, zoom_range=0.2, shear_range=0.2, width_shift_range=0.2, height_shift_range=0.2
val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# training images
train_generator = train_datagen.flow_from_directory(
    train_dir,  # This is the source directory for training images
    target_size=(150, 150),  # All images will be resized to 150x150
    batch_size=1,
    class_mode='categorical',shuffle=False)#,save_to_dir='../data_repo')

# validation images
validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=1,
    class_mode='categorical',shuffle=False)

# test images
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=1,
    class_mode='categorical',
    shuffle=False)


#X_train = [i for i in train_generator]
# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
#'''
X_train = []
Y_train = []
i = 0
for batch in train_generator:#.flow(x, batch_size=1):
    i += 1
    if i > 95:
        break  # otherwise the generator would loop indefinitely
    #print(batch[0].shape,batch[1].shape)
    X_train.append(batch[0])
    Y_train.append(batch[1])
X_train = np.array(X_train)
Y_train = np.array(Y_train)
print(X_train.shape)
print(Y_train.shape)
X_train = X_train.reshape((X_train.shape[0]*X_train.shape[1]), X_train.shape[2], X_train.shape[3], X_train.shape[4])
Y_train = Y_train.reshape((Y_train.shape[0]*Y_train.shape[1]), Y_train.shape[2])
print(X_train.shape)
print(Y_train.shape)
#X_train = X_train.reshape((X_train.shape[0], -1))
Y_train_dl = np.argmax(Y_train, axis=1)
#print(X_train.shape)
#print(Y_train.shape)
#print(np.argmax(Y_train[-5:], axis=1))
#import sys;sys.exit()
#========================================
X_val = []
Y_val = []
i = 0
for batch in validation_generator:#.flow(x, batch_size=1):
    i += 1
    if i > 24:
        break  # otherwise the generator would loop indefinitely
    #print(batch[0].shape,batch[1].shape)
    X_val.append(batch[0])
    Y_val.append(batch[1])
X_val = np.array(X_val)
Y_val = np.array(Y_val)
print(X_val.shape)
print(Y_val.shape)
X_val = X_val.reshape((X_val.shape[0]*X_val.shape[1]), X_val.shape[2], X_val.shape[3], X_val.shape[4])
Y_val = Y_val.reshape((Y_val.shape[0]*Y_val.shape[1]), Y_val.shape[2])
print(X_val.shape)
print(Y_val.shape)
#X_val = X_val.reshape((X_val.shape[0], -1))
#print(X_val.shape)
Y_val_dl = np.argmax(Y_val, axis=1)
#print(Y_val.shape)
#import sys;sys.exit()
#'''

'''
# Extract features
import os, shutil
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 32

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 7, 7, 512))  # Must be equal to the output of the convolutional base
    labels = np.zeros(shape=(sample_count))
    # Preprocess data
    generator = datagen.flow_from_directory(directory,
                                            target_size=(img_width,img_height),
                                            batch_size = batch_size,
                                            class_mode='binary')
    # Pass data through convolutional base
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels
    
train_features, train_labels = extract_features(train_dir, train_size)  # Agree with our small dataset size
validation_features, validation_labels = extract_features(validation_dir, validation_size)
test_features, test_labels = extract_features(test_dir, test_size)
'''

#'''
# Concatenate training and validation sets
X_train = np.concatenate((X_train, X_val))
Y_train = np.concatenate((Y_train, Y_val))
Y_tr_te_dl = np.concatenate((Y_train_dl, Y_val_dl))
print('merged : ',X_train.shape)
print('merged : ',Y_train.shape)
print('merged : ',Y_tr_te_dl.shape)

from sklearn.model_selection import StratifiedKFold
import pickle
from numpy import save, load
#X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
#y = np.array([0, 0, 1, 1])
skf = StratifiedKFold(n_splits=3, random_state=41, shuffle=True)
#skf.get_n_splits(X, y)

train_test_filename = "../data_kfold/dl_{}.pkl"
count = 0
for train_index, test_index in skf.split(X_train, Y_tr_te_dl):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_tr, X_te = X_train[train_index], X_train[test_index]
    y_tr, y_te = Y_train[train_index], Y_train[test_index]
    tmp_dict = {'X_tr':X_tr,'X_te':X_te,'y_tr':y_tr,'y_te':y_te}
    
    with open(train_test_filename.format(count), 'wb') as file:
        pickle.dump(tmp_dict, file)
    
    count += 1