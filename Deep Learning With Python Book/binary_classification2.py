# Model
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import losses
from keras import metrics

# Number crunching
import numpy as np

# Graphing
import matplotlib.pyplot as plt
import seaborn as sns

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


def vectorise_data(data, dimension=10000):
    """
    Manual implementation of one-hot encoding
    :param data: the data which you wish to encode e.g. train data
    :param dimension: a constant to initialise an array of zeros of correct size
    :return: returns vectorised input
    """
    results = np.zeros((len(data), dimension))
    for i, seq in enumerate(data):
        results[i, seq] = 1.
    return results


# Vectorise Data
x_train = vectorise_data(train_data)
x_test = vectorise_data(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(10000,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid', input_shape=(10000,)))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)

# Accuracy = 88% using naive approach

print(results)
