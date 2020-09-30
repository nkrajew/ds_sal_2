# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:05:05 2020

@author: nkraj
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('eda_data.csv')


# turn rating_bin into categorical
df['rating_bin'] = df['rating_bin'].apply(lambda x: str(x))

# choose relevant columns
df_model = df[['avg_salary', 'rating_bin', 'Type of ownership', 'Industry', 'Sector', 'Revenue', 'job_state', 'age', 'python_yn', 'spark_yn', 'aws_yn', 'excel_yn', 'SQL_yn', 'tableau_yn', 'job_simp', 'seniority', 'desc_len']]

# get dummy data
df_dum = pd.get_dummies(df_model)

# train test split
from sklearn.model_selection import train_test_split

X = df_dum.drop('avg_salary',axis=1)
y = df_dum['avg_salary'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# multiple linear regression
import statsmodels.api as sm

X_sm = X = sm.add_constant(X)
model=sm.OLS(y, X_sm)
model.fit().summary()

# use sklearn
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

# standardize variables to use with linear model
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_lm = scaler.fit_transform(X_train)
X_test_lm = scaler.transform(X_test)

lm = LinearRegression()
lm.fit(X_train_lm, y_train)
lm.coef_

# using MAE to show how far off on average we are
np.mean(cross_val_score(lm, X_train_lm, y_train, scoring='neg_mean_absolute_error', cv=10))

# lasso regression
lm_l = Lasso(alpha=0.01)
lm_l.fit(X_train,y_train)

np.mean(cross_val_score(lm_l, X_train, y_train, scoring='neg_mean_absolute_error', cv=10))

alpha = []
error = []

for i in range(1,100):
    alpha.append(i/100)
    lml = Lasso(alpha=(i/10))
    error.append(np.mean(cross_val_score(lml, X_train, y_train, scoring='neg_mean_absolute_error', cv=10)))

plt.plot(alpha,error)

err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns=['alpha', 'error'])
df_err[df_err.error == max(df_err.error)]

# random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

np.mean(cross_val_score(rf, X_train, y_train, scoring = 'neg_mean_absolute_error', cv=10))

# tune using GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
params = {
    'n_estimators': range(100,200,10),
    'max_depth': range(1, 11, 1),
    'min_samples_split': range(2,10,2),
    'criterion': ['mae']
    }

rs = RandomizedSearchCV(rf, params, scoring='neg_mean_absolute_error', cv=10, random_state=42)
# gs = GridSearchCV(rf, params, scoring='neg_mean_absolute_error', cv=5)
rs.fit(X_train, y_train)

rs.best_score_
rs.best_estimator_

# test ensembles
tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_rf = rs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error
print("Linear Model:",mean_absolute_error(y_test, tpred_lm))
print("Lasso model:", mean_absolute_error(y_test, tpred_lml))
print("RF Model:", mean_absolute_error(y_test, tpred_rf))

# productionize model
# start by pickling the model
# import pickle
# pickl = {'model': rf.best_estimator_}
# pickle.dump(pickl, open('model_file' + ".p", "wb"))

# file_name = "model_file.p"
# with open(file_name, 'rb') as pickled:
#     data = pickle.load(pickled)
#     model = data['model']


# test that it worked
# model.predict(X_test.iloc[1,:].values.reshape(1,-1))

# list(X_test.iloc[1,:])
