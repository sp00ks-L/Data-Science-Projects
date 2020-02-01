import os
import numpy as np
from pathlib import Path

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
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

conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

train_datagen = ImageDataGenerator(
    # rescale=1. / 255,
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# test_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

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
              optimizer=RMSprop(lr=1e-5),
              metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50)


def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


# Model Performance Plotting
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, smooth_curve(acc), 'o', label='Training Acc')
plt.plot(epochs, smooth_curve(val_acc), label='Validation Acc')
plt.title("Training and Validation Accuracy")
plt.legend()
plt.savefig("Training and Validation Accuracy (VGC16) Tuned")
plt.show()

plt.figure()

plt.plot(epochs, smooth_curve(loss), 'o', label='Training Loss')
plt.plot(epochs, smooth_curve(val_loss), label='Validation Loss')
plt.title("Training and Validation Loss")
plt.legend()
plt.savefig("Training and Validation Loss (VGC16) Tuned")
plt.show()

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print("Test Acc: {} \nTest loss: {}".format(test_acc, test_loss))
