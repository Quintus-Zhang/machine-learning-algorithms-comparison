import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit, validation_curve, GridSearchCV, \
    cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, precision_recall_curve, \
    average_precision_score, f1_score, log_loss, classification_report, confusion_matrix, make_scorer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras import regularizers
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import Nadam
from keras.layers import Dropout
from keras import backend as K


# local items
sys.path.insert(0, os.path.dirname(os.getcwd()))
from utilities import plot_learning_curve, plot_validation_curve


# Set up directory and file path
base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, 'data')
temp_dir = os.path.join(base_dir, 'temp')

data_fp = os.path.join(data_dir, 'php8Mz7BG.csv')
sp_data_fp = os.path.join(data_dir, 'dataset_44_spambase.csv')

X_train_fp = os.path.join(temp_dir, 'php_X_train')
X_test_fp = os.path.join(temp_dir, 'php_X_test')
y_train_fp = os.path.join(temp_dir, 'php_y_train')
y_test_fp = os.path.join(temp_dir, 'php_y_test')

# read data from pickle file
X_train = pd.read_pickle(X_train_fp)
X_test = pd.read_pickle(X_test_fp)
y_train = pd.read_pickle(y_train_fp)
y_test = pd.read_pickle(y_test_fp)

#

X_train_fp = os.path.join(temp_dir, 'sp_X_train')
X_test_fp = os.path.join(temp_dir, 'sp_X_test')
y_train_fp = os.path.join(temp_dir, 'sp_y_train')
y_test_fp = os.path.join(temp_dir, 'sp_y_test')

# read data from pickle file
sp_data = pd.read_pickle(os.path.join(temp_dir, 'sp_data'))
sp_X_train = pd.read_pickle(X_train_fp)
sp_X_test = pd.read_pickle(X_test_fp)
sp_y_train = pd.read_pickle(y_train_fp)
sp_y_test = pd.read_pickle(y_test_fp)
