import pandas as pd
import numpy as np


def get_datetime_features(df):
    df['Date'] = pd.to_datetime(df['Date'])
    new_df = pd.DataFrame()
    new_df['Month'] = df['Date'].dt.month
    new_df['day'] = df['Date'].dt.day
    new_df['year'] = df['Date'].dt.year
    new_df['quarter'] = df['Date'].dt.quarter
    new_df['dayofweek'] = df['Date'].dt.dayofweek
    df = df.drop('Date',axis=1)
    
    res=pd.concat([df,new_df],axis=1)
    assert res.shape[0]==df.shape[0]
    return res


def basic_categorical_encoding(df):
    return pd.get_dummies(df)

