import os
import numpy as np
from pathlib import Path

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import RMSprop

import matplotlib.pyplot as plt
import seaborn as sns

"""
Feature extraction with data augmentation

- Training is much more expensive

"""

plt.style.use('bmh')

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

base_dir = Path('C:/Users/Luke/Desktop/CodeAcademy Cheat Sheets/Deep Learning With Python/cats_dogs_small')
train_dir = base_dir / "train"
validation_dir = base_dir / "validation"
test_dir = base_dir / "test"

conv_base.trainable = False

model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=2e-5),
              metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)

# Model Performance Plotting
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'o', label='Training Acc')
plt.plot(epochs, val_acc, label='Validation Acc')
plt.title("Training and Validation Accuracy")
plt.savefig("Training and Validation Accuracy (VGC16) Expensive")
plt.legend()
plt.show()

plt.figure()

plt.plot(epochs, loss, 'o', label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title("Training and Validation Loss")
plt.savefig("Training and Validation Loss (VGC16) Expensive")
plt.legend()
plt.show()
