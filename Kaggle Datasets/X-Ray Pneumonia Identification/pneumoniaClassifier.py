import numpy as np

# Image manipulation
import cv2
from PIL import Image, ImageEnhance
import os

# Plotting confusion matrix and testing image transformation using imshow()
import matplotlib.pyplot as plt
import seaborn as sns

# Data splitting and model evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

sns.set(context='poster', font_scale=1.2)

# import images into train, test, validation sets
normal = os.listdir('xrays/Normal')
bac_pneumonia = os.listdir('xrays/Pneumonia/Bacterial')
viral_pneumonia = os.listdir('xrays/Pneumonia/Viral')
# Desktop.ini found in list at index:0 and was throwing an Attribute Error
bac_pneumonia.pop(0)
viral_pneumonia.pop(0)


def image_import(lst, label, path):
    """
    :param lst: Target list of image names to import
    :param label: An integer to label the images: 0 for normal; 1 for bacterial; 2 for viral
    :param path: Path to the image locations
    :return: list of data and labels
    """
    data = []
    labels = []
    for i in lst:
        image = cv2.imread(path + i)
        image_array = Image.fromarray(image)
        resize_img = image_array.resize((100, 100))
        # Engineering new images to increase model data set and introduce variance in image quality
        blur = cv2.blur(np.array(resize_img), (3, 3))
        bright_up = ImageEnhance.Brightness(resize_img).enhance(1.3)
        bright_down = ImageEnhance.Brightness(resize_img).enhance(0.8)
        data.append(np.array(resize_img))
        data.append(np.array(blur))
        data.append(np.array(bright_up))
        data.append(np.array(bright_down))
        labels.append(label)
        labels.append(label)
        labels.append(label)
        labels.append(label)

    return data, labels


normal_data, normal_labels = image_import(normal, 0, 'xrays/Normal/')
bPneu_data, bPneu_labels = image_import(bac_pneumonia, 1, 'xrays/Pneumonia/Bacterial/')
vPneu_data, vPneu_labels = image_import(viral_pneumonia, 2, 'xrays/Pneumonia/Viral/')

data = np.array(normal_data + bPneu_data + vPneu_data)
labels = np.array(normal_labels + bPneu_labels + vPneu_labels)

s = np.arange(data.shape[0])
np.random.shuffle(s)
xrays = data[s]
labels = labels[s]
num_classes = len(np.unique(labels))

# Normalise images and split data
xrays = xrays.astype('float32') / 255
(train_x, test_x, train_y, test_y) = train_test_split(xrays, labels,
                                                      test_size=0.2, stratify=labels)
# One hot encoding
train_y = to_categorical(train_y, num_classes)
test_y = to_categorical(test_y, num_classes)

# Keras Sequential model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=(100, 100, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))

# Testing using a conv layer with strides to replace a Pooling layer
# model.add(Conv2D(filters=256, kernel_size=3, padding="same", activation="relu", strides=3))

model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(3, activation="softmax"))
# model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
plot_model = model.fit(train_x, train_y, validation_split=0.2, batch_size=32, epochs=16, verbose=1)

y_pred = model.predict(test_x)
predictions = [np.argmax(pred) for pred in y_pred]
true = [np.argmax(true) for true in test_y]

pred_prob = model.predict_proba(test_x)
print(classification_report(true, predictions))

norm, bac, vir = multilabel_confusion_matrix(true, predictions)

norm_tn, norm_fp = norm[0]
norm_fn, norm_tp = norm[1]
bac_tn, bac_fp = bac[0]
bac_fn, bac_tp = bac[1]
vir_tn, vir_fp = vir[0]
vir_fn, vir_tp = vir[1]

norm_sens = round(norm_tp / (norm_fn + norm_tp), 3)
norm_spec = round(norm_tn / (norm_tn + norm_fp), 3)

bac_sens = round(bac_tp / (bac_fn + bac_tp), 3)
bac_spec = round(bac_tn / (bac_tn + bac_fp), 3)

vir_sens = round(vir_tp / (vir_fn + vir_tp), 3)
vir_spec = round(vir_tn / (vir_tn + vir_fp), 3)

print(" Normal\n --------------------\n "
      "Normal Sensitivity: {0}\n "
      "Normal Specificity: {1}\n "
      "\n Bacterial\n -----------------\n "
      "Bacterial Sensitivity: {2}\n "
      "Bacterial Specificity: {3}\n "
      "\n Viral\n ---------------------\n "
      "Viral Sensitivity: {4}\n "
      "Viral Specificity: {5}\n".format(norm_sens, norm_spec, bac_sens, bac_spec, vir_sens, vir_spec))


def plot_confusion_matrix(cm,
                          target_names,
                          picname,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(30, 30))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.tight_layout()
    plt.savefig(picname)
    plt.show()


# target_names = ['Normal', 'Bacterial', 'Viral']
# plot_confusion_matrix(confusion_mat, target_names=target_names, picname="Confusion Matrix", normalize=False,
#                       title="Confusion Matrix \nNormal vs Bacterial vs Viral", cmap='GnBu')


"""
               precision    recall  f1-score   
           0       0.99      0.99      0.99      
           1       0.98      0.99      0.98      
           2       0.97      0.95      0.96
           
 Normal
 --------------------
 Normal Sensitivity: 0.989
 Normal Specificity: 0.995
 
 Bacterial
 -----------------
 Bacterial Sensitivity: 0.987
 Bacterial Specificity: 0.978
 
 Viral
 ---------------------
 Viral Sensitivity: 0.951
 Viral Specificity: 0.99
"""


