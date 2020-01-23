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
from sklearn.model_selection import StratifiedKFold
import pickle
from numpy import save, load

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

# Extract features
import os, shutil
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 1
img_width = 150
img_height = 150

def extract_features(directory, sample_count):
    #features = np.zeros(shape=(sample_count, 7, 7, 512))  # Must be equal to the output of the convolutional base
    features = np.zeros(shape=(sample_count, img_height, img_width, 3))
    labels = np.zeros(shape=(sample_count, 4))
    # Preprocess data
    generator = datagen.flow_from_directory(directory,
                                            target_size=(img_width,img_height),
                                            batch_size = batch_size,
                                            class_mode='categorical',shuffle=False)
    # Pass data through convolutional base
    i = 0
    for inputs_batch, labels_batch in generator:
        #features_batch = conv_base.predict(inputs_batch)
        #features[i * batch_size: (i + 1) * batch_size] = features_batch
        features[i * batch_size: (i + 1) * batch_size] = inputs_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels
    
train_features, train_labels = extract_features(train_dir, 95)  # Agree with our small dataset size
validation_features, validation_labels = extract_features(validation_dir, 24)
#test_features, test_labels = extract_features(test_dir, test_size)

# Concatenate training and validation sets
X_train = np.concatenate((train_features, validation_features))
Y_train = np.concatenate((train_labels, validation_labels))
number_labels = np.concatenate((np.argmax(train_labels, axis=1), np.argmax(validation_labels, axis=1)))
print('merged : ',X_train.shape)
print('merged : ',Y_train.shape)
print('merged : ',number_labels.shape)

skf = StratifiedKFold(n_splits=3, random_state=41, shuffle=True)

train_test_filename = "../data_kfold/dll_{}.pkl"
count = 0
for train_index, test_index in skf.split(X_train, number_labels):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_tr, X_te = X_train[train_index], X_train[test_index]
    y_tr, y_te = Y_train[train_index], Y_train[test_index]
    tmp_dict = {'X_tr':X_tr,'X_te':X_te,'y_tr':y_tr,'y_te':y_te}
    
    with open(train_test_filename.format(count), 'wb') as file:
        pickle.dump(tmp_dict, file)
    
    count += 1