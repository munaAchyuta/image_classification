from numpy.random import seed

seed(1)
import os
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np
import pandas as pd
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


# Input feature map is 150x150x3: 150x150 for the image pixels, and 3 for R, G, and B
img_input = layers.Input(shape=(150, 150, 3))

# First convolution extracts 16 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(16, 3, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)

# Second convolution extracts 32 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Dropout(0.4)(x)

# Third convolution extracts 64 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Dropout(0.4)(x)

# Fourth convolution extracts 128 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(128, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Dropout(0.4)(x)

# Fifth convolution extracts 256 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(256, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Dropout(0.4)(x)

# Flatten feature map to a 1-dim tensor to add fully connected layers
x = layers.Flatten()(x)

# Create a fully connected layer with ReLU activation
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.4)(x)

# Create output layer with 4 node(4-class) and sigmoid activation
output = layers.Dense(4, activation='sigmoid')(x)


# Train model
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
    
    # Clear model, and create it
    model = None
    
    # Create model:
    model = Model(img_input, output)

    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.001),
                  metrics=['acc'])
    
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('../model/mdl_wts_{}.hdf5'.format(k_fold), save_best_only=True, monitor='val_loss', mode='min')
    history = model.fit(X_train,Y_train, batch_size=6, epochs=50, validation_data=(X_val,Y_val), validation_steps=3, verbose=2, callbacks=[earlyStopping,mcp_save])
    
    accuracy_history = history.history['acc']
    val_accuracy_history = history.history['val_acc']
    print("Last training accuracy: " + str(accuracy_history[-1]) + ", last validation accuracy: " + str(val_accuracy_history[-1]))
    
    model.save(os.path.join(os.path.dirname(__file__), '../model/image_class_{}.h5'.format(k_fold)))
    model.save_weights(os.path.join(os.path.dirname(__file__), '../model/image_class_{}_weights.h5'.format(k_fold)))
