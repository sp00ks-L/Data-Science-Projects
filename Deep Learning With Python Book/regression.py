# Building model
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense

# Number crunching
import numpy as np

# Graphing
import matplotlib.pyplot as plt

plt.style.use('seaborn-colorblind')

(train_data, train_target), (test_data, test_target) = boston_housing.load_data()

# Data Normalisation
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std


# Define Model
def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))

    """   This output layer has no activation function and is therefore linear (common for scalar regression)
          If you applied sigmoid, the network would predict values 0 to 1
          Here, due to the linearity, the network is free to learn to predict values in any range
    """
    model.add(Dense(1))

    model.compile(optimizer='rmsprop',
                  loss='mse',  # MSE is common for regression problems
                  metrics=['mae'])  # MAE of 0.5 in this problem would indicate your predictions are off by $500 avg.
    return model


"""
Evaluating the model

- Due to the small number of data points, the usual train test val split would result in val scores with high variance
- Best practice here is to use K-fold cross validation

K-fold
- Split data into K partitions (typically 4 or 5)
- Instantiate K identical models and train each one on K - 1 while evaluating on remaining partitions
- Average of K validation scores
"""

# k = 4
# num_val_samples = len(train_data) // k
# num_epochs = 80
# all_mae_histories = []
#
# for i in range(k):
#     print("Processing fold #", i)
#     val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
#     val_targets = train_target[i * num_val_samples: (i + 1) * num_val_samples]
#
#     partial_train_data = np.concatenate([
#         train_data[:i * num_val_samples],
#         train_data[(i + 1) * num_val_samples:]
#     ], axis=0)
#
#     partial_train_targets = np.concatenate([
#         train_target[:i * num_val_samples],
#         train_target[(i + 1) * num_val_samples:]
#     ], axis=0)
#
#     model = build_model()
#     history = model.fit(partial_train_data, partial_train_targets,
#                         validation_data=(val_data, val_targets),
#                         epochs=num_epochs, batch_size=1, verbose=0)
#     mae_history = history.history['val_mae']
#     all_mae_histories.append(mae_history)
#
# avg_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]


# def smooth_curve(points, factor=0.9):
#     smoothed_points = []
#     for point in points:
#         if smoothed_points:
#             previous = smoothed_points[-1]
#             smoothed_points.append(previous * factor + point * (1 - factor))
#         else:
#             smoothed_points.append(point)
#     return smoothed_points


# smoothed_mae_history = smooth_curve(avg_mae_history[10:])
#
# plt.plot(range(1, len(avg_mae_history) + 1), avg_mae_history)
# plt.title("Boston House Prices Network Performance")
# plt.xlabel("Epochs")
# plt.ylabel("Validation MAE")
# plt.savefig("Boston House Prices Network Performance.png")
# plt.show()
#
# plt.clf()
# plt.plot(range(1, len(smoothed_mae_history) + 1), smoothed_mae_history)
# plt.title("Boston House Prices Network Performance (Smoothed)")
# plt.xlabel("Epochs")
# plt.ylabel("Validation MAE")
# plt.savefig("Boston House Prices Network Performance (Smoothed).png")
# plt.show()

model = build_model()
model.fit(train_data, train_target,
          epochs=80, batch_size=16, verbose=0)
test_mse, test_mae = model.evaluate(test_data, test_target)

print("MSE: {} \nMAE: {}".format(test_mse, test_mae))


"""
Network log

100 epochs
- 2.1357951164245605, 2.94952654838562, 2.7440061569213867, 2.474660873413086
- Average: 2.5759971737861633

After 80 epochs
- MAE = 2.68 (predictions are off by $2,680)
- MSE = 18.9
"""
