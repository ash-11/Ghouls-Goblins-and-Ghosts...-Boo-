# importing basic libraries
import numpy as np
import pandas as pd

# loading data
train = pd.read_csv('./train.csv', low_memory=False)
test = pd.read_csv('./test.csv', low_memory=False)
saam = pd.read_csv('./sample_submission.csv', low_memory = False)

# dropping id as it's not required
train.drop('id', axis = 1, inplace=True)
test.drop('id', axis=1, inplace=True)

# we have finite number of colors, therefore converting color to integer data using pandas dummies
tr_clr = pd.get_dummies(train['color'], drop_first=True)
te_clr = pd.get_dummies(test['color'], drop_first=True)

# adding dummies to training and testing data
train = pd.concat([train, tr_clr], axis = 1)
test = pd.concat([test, te_clr], axis = 1)

# dropping previous column of color
train.drop('color', axis = 1, inplace=True)
test.drop('color', axis = 1, inplace=True)

# separating features and labels
y = train['type']
x = train.drop('type', axis = 1)

# using gradient boosting classifier as it gives maximum accuracy when tested on 20% of training set
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()
clf.fit(x, y)
pred = clf.predict(test)

# printing predictions in csv file
saam['type'] = pred
saam.to_csv('pred.csv', index = False)