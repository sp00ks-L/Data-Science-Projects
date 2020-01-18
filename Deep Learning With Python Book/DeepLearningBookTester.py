from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

import numpy as np
import math
import matplotlib.pyplot as plt


#
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#
# network = models.Sequential()
# network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
# network.add(layers.Dense(10, activation='softmax'))
#
# network.compile(optimizer='adam',
#                 loss='categorical_crossentropy',
#                 metrics=['accuracy'])
#
# train_images = train_images.reshape((60000, 28 * 28))
# train_images = train_images.astype('float32') / 255
#
# test_images = test_images.reshape((10000, 28 * 28))
# test_images = test_images.astype('float32') / 255
#
#
# train_labels = to_categorical(train_labels)
# test_labels = to_categorical(test_labels)
#
# network.fit(train_images, train_labels, epochs=5, batch_size=128)
#
# test_loss, test_acc = network.evaluate(test_images, test_labels)
# print("test loss: {} \ntest acc: {}".format(test_loss, test_acc))


def naive_relu(x):
    """
    Effectively, this function checks if a number is positive
    If the number is not positive, the number is replaced by 0
    :param x: this is the input 2D tensor
    :return: element-wise, if less than 0, element == 0
    """
    assert len(x.shape) == 2

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)

    return x


def naive_add(x, y):
    assert len(x.shape) == 2
    assert x.shape == y.shape

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]

    return x


def naive_add_matrix_and_vector(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[j]

    return x


def naive_vector_dot(x, y):
    """
    Vectors have to have same number of elements hence assert x.shape[0] == y.shape[0]
    :param x: A vector (1D tensor)
    :param y: A vector (1D tensor)
    :return: A scalar (0D tensor)
    """
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]

    z = 0
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z


def naive_matrix_vector_dot(x, y):
    """
    Returns a vector where the coefficients are the dot products between vector y and the rows of x
    :param x: A matrix (2D tensor)
    :param y: A vector (1D tensor)
    :return: A vector
    """
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]

    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i, j] * y[j]
    return z


def naive_matrix_dot(x, y):
    """
    Matrix dot requires x.shape[1] == y.shape[0]
    :param x: A matrix (2D tensor)
    :param y: A matrix (2D tensor)
    :return: A matrix with shape (x.shape[0], y.shape[1]) where coeffs. are vector products
                between rows of x and columns of y
    """
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]

    z = np.zeros((x.shape[0], y.shape[1]))
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            row_x = x[i, :]
            column_y = y[:, j]
            z[i, j] = naive_vector_dot(row_x, column_y)

    return z


ar = np.array([[2, 3, 4],
               [5, 2, 8],
               [3, 7, -1],
               [-5, 0, 10]])

ar2 = np.array([[2, 3, 4],
                [5, 2, 8],
                [3, 7, -1]])

vec = np.array([4, 2, 22])

vec2 = np.array([4, 2, 22])


def f(x):  # sample function
    return x * np.sin(np.power(x, 2))


# evaluation of the function
x = np.linspace(-2, 4, 150)
y = f(x)

a = 3.301
h = 0.1
fprime = (f(a + h) - f(a)) / h  # derivative
tan = f(a) + fprime * (x - a)  # tangent

# plot of the function and the tangent
# plt.plot(x, y, 'b', a, f(a), 'om', x, tan, '--r')
plt.plot(x, y, a, f(a), 'om', x, tan, '--r')
plt.show()

print(fprime)
