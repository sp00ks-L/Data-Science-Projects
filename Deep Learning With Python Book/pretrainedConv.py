import os
import numpy as np
from pathlib import Path

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

import matplotlib.pyplot as plt
import seaborn as sns

"""
Fast Feature extraction without data augmentation

- Training is fast because only 2 dense layers have to be dealt with
- Can see rapid over-fitting due to lack of data augmentation

"""


plt.style.use('bmh')

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

base_dir = Path('C:/Users/Luke/Desktop/CodeAcademy Cheat Sheets/Deep Learning With Python/cats_dogs_small')
train_dir = base_dir / "train"
validation_dir = base_dir / "validation"
test_dir = base_dir / "test"

datagen = ImageDataGenerator(rescale=1. / 255)
batch_size = 20


def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=sample_count)
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')

    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels


train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

# Model definition
model = Sequential()
model.add(Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))

# Model Performance Plotting
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'o', label='Training Acc')
plt.plot(epochs, val_acc, label='Validation Acc')
plt.title("Training and Validation Accuracy")
plt.savefig("Training and Validation Accuracy (VGC16)")
plt.legend()
# plt.show()

plt.figure()

plt.plot(epochs, loss, 'o', label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title("Training and Validation Loss")
plt.savefig("Training and Validation Loss (VGC16)")
plt.legend()
plt.show()
