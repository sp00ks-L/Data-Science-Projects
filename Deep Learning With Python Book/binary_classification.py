# Model
from keras.datasets import imdb
from keras import models
from keras.layers import Dense
from keras import optimizers
from keras import losses
from keras import metrics

# Number crunching
import numpy as np

# Graphing
import matplotlib.pyplot as plt
import seaborn as sns

# sns.set_context('poster')
plt.style.use('seaborn-colorblind')

"""
AIM: Binary classification of movies from IMDb
        Classifies whether a movies review is positive (label = 1) or negative (label = 0)
"""

# Load data into train / test.
# num_words keeps top 'n' most frequently occurring words in training data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Indicates that num_words works (max_words = 9999)
max_words = max([max(sequence) for sequence in train_data])

"""
Decoding review back to readable english. This shows the first review which happens to be positive
"""


# word_index = imdb.get_word_index()
# reverse_word_index = dict(
#     [(value, key) for (key, value) in word_index.items()]
# )
# decoded_review = ' '.join(
#     [reverse_word_index.get(i - 3, '?') for i in train_data[0]]
# )
# print(decoded_review)


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

"""
Key architecture decisions
- How many layers to use
- How many hidden units to choose for each layer
"""

model = models.Sequential()
model.add(Dense(16, activation='relu', input_shape=(10000,)))
model.add((Dense(16, activation='relu')))
model.add(Dense(1, activation='sigmoid'))
#  For models that output probabilities, crossentropy is a good choice for loss
#  Cross entropy measures the distance between the ground truth and your predictions

#  Standard implementation of optimisers, loss function and metric
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

# You can also config the optimiser and use custom loss functions + metrics
# model.compile(optimizer=optimizers.RMSprop(lr=0.001),
#               loss=losses.binary_crossentropy,
#               metrics=[metrics.binary_accuracy])


#  Creating a validation set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, 21)
plt.plot(epochs, loss_values, 'o', label='Training Loss')
plt.plot(epochs, val_loss_values, label='Validation Loss')
plt.title("Training and Validation Loss \nIMBd Dataset")
plt.xlabel('Epochs')
plt.ylabel("Loss")
plt.legend()
plt.savefig("Training and Validation Loss (IMBd).png")
plt.show()


plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc_values, 'o', label='Training Accuracy')
plt.plot(epochs, val_acc_values, label='Validation Accuracy')
plt.title("Training and Validation Accuracy \nIMBd Dataset")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("Training and Validation Accuracy (IMBd).png")
plt.show()
