import os
import pandas as pd
from sklearn.model_selection import train_test_split
from config import data_fp, credit_data_fp, temp_dir

# Read the data set
data = pd.read_csv(data_fp)
credit_data = pd.read_csv(credit_data_fp)

# Split data set into training set and test set
X_cols = [col for col in data.columns if col != 'Class']
X = data[X_cols]
y = data['Class']
y = y.where(y==1, other=0)  # change class label 2 to 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# save data to temp directory
X_train.to_pickle(path=os.path.join(temp_dir, 'php_X_train'))
X_test.to_pickle(path=os.path.join(temp_dir, 'php_X_test'))
y_train.to_pickle(path=os.path.join(temp_dir, 'php_y_train'))
y_test.to_pickle(path=os.path.join(temp_dir, 'php_y_test'))
