#!/usr/bin/env python
# coding: utf-8

## Classification-based Collaborative Filtering Systems, Logistic Regression as a Classifier

import pandas as pd

from sklearn.linear_model import LogisticRegression

bank_full = pd.read_csv('bank_full_w_dummy_vars.csv')
bank_full.head()

bank_full.info()

X = bank_full.ix[:, (18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36)].values

y = bank_full.ix[:, 17].values

LogReg = LogisticRegression()
LogReg.fit(X, y)

new_user = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
y_pred = LogReg.predict(new_user)
y_pred
