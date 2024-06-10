from typing import List, Tuple
from pandas import DataFrame, Series
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

@data_exporter
def export(
    data: Tuple[DataFrame, DataFrame, DataFrame], *args, **kwargs
) -> Tuple[csr_matrix, csr_matrix, csr_matrix, Series, Series, Series, BaseEstimator, DictVectorizer]:
    df, df_train, df_val = data
    target = kwargs.get('target', 'duration')

    # Selecting features and target
    features = ['PULocationID', 'DOLocationID']

    # Vectorize features
    dv = DictVectorizer(sparse=True)
    X = dv.fit_transform(df[features].to_dict(orient='records'))
    y: Series = df[target]

    X_train = dv.transform(df_train[features].to_dict(orient='records'))
    y_train = df_train[target]
    X_val = dv.transform(df_val[features].to_dict(orient='records'))
    y_val = df_val[target]

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Print the intercept of the model
    print("Intercept of the model:", model.intercept_)

    return X, X_train, X_val, y, y_train, y_val, model, dv
