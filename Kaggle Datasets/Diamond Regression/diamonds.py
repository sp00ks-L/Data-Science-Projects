import numpy as np
from math import sqrt

# Data handling and evaluation
import pandas as pd
import pandas_profiling

import matplotlib.pyplot as plt
import seaborn as sns

# Algorithms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("diamonds.csv")
data = data.drop('Unnamed: 0', axis=1)

# Some dimensions were 0 and thus were unsuitable
data[['x', 'y', 'z']] = data[['x', 'y', 'z']].replace(0, np.NaN)
data = data.dropna().reset_index()

# profile = pp.ProfileReport(data)
# profile.to_file('profile_report.html')

one_hot = pd.get_dummies(data)
diamond_data = pd.DataFrame(one_hot, columns=one_hot.columns)
diamond_data = diamond_data.drop(columns=['index', 'depth', 'table'], axis=1)

# Feature engineering
diamond_data['volume'] = diamond_data.x * diamond_data.y * diamond_data.z

# Scale the data then split
scaler = StandardScaler()
numericals = pd.DataFrame(scaler.fit_transform(diamond_data[['carat', 'x', 'y', 'z', 'volume']]),
                          columns=['carat', 'x', 'y', 'z', 'volume'], index=diamond_data.index)
diamond_clean_data = diamond_data.copy(deep=True)
diamond_clean_data[['carat', 'x', 'y', 'z', 'volume']] = numericals[['carat', 'x', 'y', 'z', 'volume']]

features = diamond_clean_data.drop('price', axis=1)
price = diamond_clean_data['price']
x_train, x_test, y_train, y_test = train_test_split(features, price, train_size=0.8, test_size=0.2)

# Random Forest Regressor
rf_regr = RandomForestRegressor(n_estimators=100, oob_score=True)
rf_regr.fit(x_train, y_train)
prediction = rf_regr.predict(x_test)

print('R^2 Training Score: {:.2f} '
      '\nOOB Score: {:.2f} '
      '\nR^2 Validation Score: {:.2f} '.format(rf_regr.score(x_train, y_train), rf_regr.oob_score_,
                                               rf_regr.score(x_test, y_test)))


