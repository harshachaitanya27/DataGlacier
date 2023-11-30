# model.py

import numpy as np
import pandas as pd
import joblib

dataset = pd.read_csv('hiring.csv')

# Handle missing values
dataset['experience'].fillna(0, inplace=True)
dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

X = dataset.iloc[:, :3]

# Convert words to integer values consistently
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]

# Splitting Training and Test Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

# Fitting model with training data
regressor.fit(X, y)

# Saving model to disk using joblib
joblib.dump(regressor, 'model.joblib')
