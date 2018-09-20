import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit, validation_curve, GridSearchCV, \
    cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, precision_recall_curve, \
    average_precision_score, f1_score, log_loss, classification_report, confusion_matrix, make_scorer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# local items
sys.path.insert(0, os.path.dirname(os.getcwd()))
from utilities import plot_learning_curve, plot_validation_curve


# Set up directory and file path
jpt_dir = os.getcwd()
base_dir = os.path.dirname(jpt_dir)
data_dir = os.path.join(base_dir, 'data')
data_fp = os.path.join(data_dir, 'php8Mz7BG.csv')

# Read the data set
data = pd.read_csv(data_fp)

# Split data set into training set and test set
X_cols = [col for col in data.columns if col != 'Class']
X = data[X_cols]
y = data['Class']
y = y.where(y==1, other=0)  # change class label 2 to 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

