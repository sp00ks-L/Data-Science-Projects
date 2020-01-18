# Model building
from keras.datasets import reuters
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

# Number crunching
import numpy as np

# Graphing
import matplotlib.pyplot as plt
import seaborn as sns

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)


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


x_train = vectorise_data(train_data)
x_test = vectorise_data(test_data)
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

# Define model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(10000,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(46, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Create validation sets (manual implementation of test train split?)
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=9,
                    batch_size=512,
                    validation_data=(x_val, y_val))


# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(1, len(loss) + 1)
#
# plt.plot(epochs, loss, 'o', label='Training Loss')
# plt.plot(epochs, val_loss, label='Validation Loss')
# plt.title("Training and Validation Loss \nReuters Dataset")
# plt.xlabel('Epochs')
# plt.ylabel("Loss")
# plt.legend()
# plt.savefig("Training and Validation Loss (Reuters).png")
# plt.show()
#
# plt.clf()
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
#
# plt.plot(epochs, acc, 'o', label='Training Accuracy')
# plt.plot(epochs, val_acc, label='Validation Accuracy')
# plt.title("Training and Validation Accuracy \nReuters Dataset")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.savefig("Training and Validation Accuracy (Reuters).png")
# plt.show()

results = model.evaluate(x_test, y_test)
# ~ 80% accuracy after 9 epochs

"""
##### Different ways to handle the labels and the loss #####

The approach used here is one hot encoding and thus uses the loss function 'categorical_crossentropy'

You can also cast the labels as an integer tensor:

y_train = np.array(train_labels)
y_test = np.array(test_labels)


IMPORTANT: categorical_crossentropy will not work for this. Instead use sparse_categorical_crossentropy
"""

