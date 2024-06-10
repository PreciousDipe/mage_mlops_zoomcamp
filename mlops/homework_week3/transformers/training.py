from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def transform(df, *args, **kwargs):
    df = df[0]

    df[['PULocationID', 'DOLocationID']] = df['PU_DO'].str.split('_', expand=True)

    categorical = df[['PULocationID', 'DOLocationID']]
    train_dict = categorical.to_dict(orient='records')

    dv = DictVectorizer()

    X_train = dv.fit_transform(train_dict)

    # Define the target variable
    target = 'duration'
    y_train = df[target].values

    # Initialize and fit the linear regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Retrieve the intercept
    intercept = lr.intercept_

    print("Intercept:", intercept)

    return(lr, dv)