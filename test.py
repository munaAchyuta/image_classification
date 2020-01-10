from numpy.random import seed

seed(1)
import os
#from tensorflow.keras import layers
#from tensorflow.keras import Model
#from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

print(os.listdir(test_class_dir))

print('total training blank images:', len(os.listdir(train_blank_dir)))
print('total training cancelled images:', len(os.listdir(train_cancel_dir)))
print('total training handwritten images:', len(os.listdir(train_hand_dir)))
print('total training printed images:', len(os.listdir(train_print_dir)))

print('total validation blank images:', len(os.listdir(validation_blank_dir)))
print('total validation cancelled images:', len(os.listdir(validation_cancel_dir)))
print('total validation handwritten images:', len(os.listdir(validation_hand_dir)))
print('total validation printed images:', len(os.listdir(validation_print_dir)))
print('total test images:', len(os.listdir(test_class_dir)))



'''
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

# Create model:
model = Model(img_input, output)

# Compile model
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])
'''
# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# training images
train_generator = train_datagen.flow_from_directory(
    train_dir,  # This is the source directory for training images
    target_size=(150, 150),  # All images will be resized to 150x150
    batch_size=2,
    class_mode='categorical')
#'''
# validation images
validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=1,
    class_mode='categorical',shuffle=False)
#'''
'''
# test images
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=1,
    class_mode='categorical',
    shuffle=False)
'''
# Train model
'''
history = model.fit_generator(
    train_generator,
    steps_per_epoch=12, epochs=50, validation_data=validation_generator, validation_steps=3, verbose=2)

# Reset to avoid ambiguity
train_generator.reset()
'''

model = load_model(os.path.join(os.path.dirname(__file__), '../modelmdl_wts.hdf5'))#cheque_class.h5
#model._make_predict_function()  # make keras thread-safe

filenames = validation_generator.filenames
""" Predict probabilities for each class 
{0: 'Blank', 1: 'Cancelled', 2: 'Handwritten', 3: 'Printed'}"""

pred = model.predict_generator(validation_generator, steps=len(filenames))
predicted_class_indices = np.argmax(pred, axis=1)
print(predicted_class_indices)
print(len(predicted_class_indices))
labels = (train_generator.class_indices)
labels = dict((v, k) for k, v in labels.items())
print(labels)
predictions = [labels[k] for k in predicted_class_indices]
print(filenames)
print(predictions)
print(len(filenames),len(predictions))
print(predictions)
filenames = [i.split('/')[0] for i in filenames]
print(filenames)
print([i for i in zip(filenames,predictions)])
print([i[0]==i[1] for i in zip(filenames,predictions)])
xx = [i[0]==i[1] for i in zip(filenames,predictions)]
print(xx.count(True))
print(len(filenames))
print(xx.count(True)/len(filenames))

results_test = pd.DataFrame({"Filename": filenames,
                             "Predictions": predictions})

## Save test results

results_test.to_csv(os.path.join(os.path.dirname(__file__), '../model_bkp/test.csv'))
#model.save(os.path.join(os.path.dirname(__file__), '../model/cheque_class.h5'))
#model.save_weights(os.path.join(os.path.dirname(__file__), '../model/cheque_class_weights.h5'))
