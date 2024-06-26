from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def transform(df, *args, **kwargs):

    categorical = ['PULocationID', 'DOLocationID']

    train_dict = df[categorical].to_dict(orient='records')

    dv = DictVectorizer()

    X_train = dv.fit_transform(train_dict)

    target = 'duration'

    y_train = df[target].values

    lr = LinearRegression()

    lr.fit(X_train, y_train)

    print("Intercept:", lr.intercept_)

    return dv, lr
