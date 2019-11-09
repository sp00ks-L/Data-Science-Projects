"""
Based on the Titanic dataset
Where:
- forest is my random forest classifer (forest = RandomForestClassifier())
- train_features and train_labels is the dataset split using 'from sklearn.model_selection import train_test_split'

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, \
    f1_score, precision_recall_curve, roc_curve, roc_auc_score

"""
Can be used to create a dataframe comparing the importance of each feature to training
- Can allow reduction of features and to see which really impact training
"""
importances = pd.DataFrame({'feature': train_features.columns, 'importance': np.round(forest.feature_importances_, 3)})
importances = importances.sort_values('importance', ascending=False).set_index('feature')

"""
Confusion Matrix
- Indicates models accuracy through displaying True Negatives, False Positives, False Negatives, True Positives
  as an array 
- Combine confusion matrix with Precision, Recall and F1 score
"""
predictions = cross_val_predict(forest, train_features, train_labels, cv=3)
matrix = confusion_matrix(train_labels, predictions)
# True Negatives, False Positives, False Negatives, True Positives
print(matrix)
print("Precision: {}".format(precision_score(train_labels, predictions)))
print("Recall: {}".format(recall_score(train_labels, predictions)))
print("F1 Score: {}".format(f1_score(train_labels, predictions)))

"""
Plotting a precision vs recall crossover curve in 2 ways
"""
# getting the probabilities of our predictions
train_label_scores = forest.predict_proba(train_features)
train_label_scores = train_label_scores[:, 1]
precision, recall, threshold = precision_recall_curve(train_labels, train_label_scores)

# getting the probabilities of our predictions
y_scores = forest.predict_proba(train_features)
y_scores = y_scores[:, 1]


def plot_precision_and_recall(precision, recall, threshold):
    """
    :param precision: The number of correct predictions compared to the number your classifier predicted would positive
    :param recall: The percentage of relevant items found by the classifier
    :param threshold: threshold if when binary classifier is 1 (true / survived) or 0 (false / not survived)
    :return: graph visualising crossover point of recall and precision
    """
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])


plt.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)


def plot_precision_vs_recall(precision, recall):
    """
    :param precision: The number of correct predictions compared to the number your classifier predicted would positive
    :param recall: The percentage of relevant items found by the classifier
    :return: graph indicating fall of precision as recall increases
    """
    plt.plot(recall, precision, "g--", linewidth=2.5)
    plt.ylabel("recall", fontsize=19)
    plt.xlabel("precision", fontsize=19)
    plt.axis([0, 1.5, 0, 1.5])


plt.figure(figsize=(14, 7))
plot_precision_vs_recall(precision, recall)

# compute true positive rate and false positive rate
false_positive_rate, true_positive_rate, thresholds = roc_curve(train_labels, train_label_scores)


def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)


plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()

r_a_score = roc_auc_score(train_labels, y_scores)
print("ROC-AUC-Score:", r_a_score)
