import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

train = pd.read_csv('https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv')
y_train = train['meal']
x_train = train.drop(columns=['meal', 'id'])

test = pd.read_csv('https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv')
x_test = test.drop(columns=['meal', 'id'])

x_train['DateTime'] = pd.to_datetime(x_train['DateTime'])
x_train['day_of_week'] = x_train['DateTime'].dt.day_of_week
x_train['day'] = x_train['DateTime'].dt.day
x_train['month'] = x_train['DateTime'].dt.month
x_train['year'] = x_train['DateTime'].dt.year
x_train.drop(columns=['DateTime'], inplace=True)

x_test['DateTime'] = pd.to_datetime(x_test['DateTime'])
x_test['day_of_week'] = x_test['DateTime'].dt.day_of_week
x_test['day'] = x_test['DateTime'].dt.day
x_test['month'] = x_test['DateTime'].dt.month
x_test['year'] = x_test['DateTime'].dt.year
x_test.drop(columns=['DateTime'], inplace=True)

param_dist = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth': [5, 10, 20, 30],
    'learning_rate': [0.01, 0.1, 0.25, 0.5, 0.75, 1]
}

model = XGBClassifier()

model_tuned = RandomizedSearchCV(model, param_dist, n_iter=100, n_jobs=-1, cv=3)

model_tuned.fit(x_train, y_train)

modelFit = model_tuned.best_estimator_

pred = modelFit.predict(x_test)