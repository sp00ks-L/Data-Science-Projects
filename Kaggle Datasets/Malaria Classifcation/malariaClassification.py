import numpy as np
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

import cv2
from PIL import Image
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

infected = os.listdir('cell_images/cell_images/Parasitized/')
uninfected = os.listdir('cell_images/cell_images/Uninfected/')

data = []
labels = []

for i in infected:
    try:

        image = cv2.imread("cell_images/Parasitized/" + i)
        image_array = Image.fromarray(image, 'RGB')
        resize_img = image_array.resize((50, 50))
        rotated45 = resize_img.rotate(45)
        rotated75 = resize_img.rotate(75)
        blur = cv2.blur(np.array(resize_img), (10, 10))
        data.append(np.array(resize_img))
        data.append(np.array(rotated45))
        data.append(np.array(rotated75))
        data.append(np.array(blur))
        labels.append(1)
        labels.append(1)
        labels.append(1)
        labels.append(1)

    except AttributeError:
        print('')

for u in uninfected:
    try:

        image = cv2.imread("cell_images/Uninfected/" + u)
        image_array = Image.fromarray(image, 'RGB')
        resize_img = image_array.resize((50, 50))
        rotated45 = resize_img.rotate(45)
        rotated75 = resize_img.rotate(75)
        data.append(np.array(resize_img))
        data.append(np.array(rotated45))
        data.append(np.array(rotated75))
        labels.append(0)
        labels.append(0)
        labels.append(0)

    except AttributeError:
        print('')

cells = np.array(data)
labels = np.array(labels)

s = np.arange(cells.shape[0])
np.random.shuffle(s)
cells = cells[s]
labels = labels[s]
num_classes = len(np.unique(labels))

# Normalisation using np broadcasting then split
cells = cells.astype('float32') / 255
(train_x, test_x, train_y, test_y) = train_test_split(cells, labels,
                                                      test_size=0.2, stratify=labels)

# Doing One hot encoding as classifier has multiple classes
train_y = to_categorical(train_y, num_classes)
test_y = to_categorical(test_y, num_classes)

# Keras Sequential model
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu", input_shape=(50, 50, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=256, kernel_size=3, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(2, activation="softmax"))
# model.summary()


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
plot_model = model.fit(train_x, train_y, validation_split=0.2, batch_size=128, epochs=3, verbose=0)


y_pred = model.predict(test_x)
predictions = [np.argmax(pred) for pred in y_pred]
true = [np.argmax(true) for true in test_y]


def evaluate_model(true_val, pred_val):
    """
    This is just a manual classification report to revise understanding of metrics
    :param true_val: list of true labels
    :param pred_val: list of predicted labels
    :return: formatted output of metrics
    """
    true_n, false_p, false_n, true_p = confusion_matrix(true, predictions).ravel()
    pred_prob = model.predict_proba(test_x)

    # Just for practice
    accuracy = round((true_p + true_n) / (true_p + true_n + false_p + false_n), 3)
    recall = round(true_p / (true_p + false_n), 3)
    precision = round(true_p / (true_p + false_p), 3)
    f1 = round(2 * ((precision * recall) / (precision + recall)), 3)
    auc_score = round(roc_auc_score(test_y, pred_prob), 3)

    print("\n Accuracy: {}\n Recall: {}\n Precision: {}\n F1 Score: {}\n AUC Score: {}".format(accuracy, recall,
                                                                                               precision, f1,
                                                                                               auc_score))


evaluate_model(true, predictions)
"""
 Accuracy: 0.968
 Recall: 0.962
 Precision: 0.982
 F1 Score: 0.972
 AUC Score: 0.993
"""


# print(classification_report(true, predictions))
