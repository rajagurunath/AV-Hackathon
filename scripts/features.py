from re import S
from loguru import logger
import pandas as pd
import numpy as np
from pandas.core.algorithms import mode
from sklearn.preprocessing import LabelBinarizer,LabelEncoder

TRANSFORMER = {}

def get_datetime_features(df,**kwargs):
    ddf = pd.to_datetime(df['Date'])
    new_df = pd.DataFrame()
    new_df['Month'] = ddf.dt.month
    new_df['day'] = ddf.dt.day
    new_df['year'] = ddf.dt.year
    new_df['quarter'] =ddf.dt.quarter
    new_df['dayofweek'] = ddf.dt.dayofweek
    df = df.drop(['Date'],axis=1)
    if "#Order" in df.columns:
        logger.info("removing #Order")
        df= df.drop("#Order",axis=1)
    res=pd.concat([df,new_df],axis=1)
    assert res.shape[0]==df.shape[0]
    return res


def basic_categorical_encoding(df,**kwargs):
    return pd.get_dummies(df)


class TransFormer(object):

    def __init__(self,transformer) -> None:
        self.transformers = {}
        self.transformer =transformer
        super().__init__()

    def fit_transform(self,model_id,X,**kwargs):
        trans = self.transformer(**kwargs)
        X_transformed = trans.fit(X)
        if model_id not in self.transformers:
            self.transformers[model_id] = trans
        else:
            return self.transform(model_id=model_id,X=X)

        return X_transformed

    def transform(self,model_id,X,**kwargs):
        X_transformed = self.transformers[model_id].transform(X)
        return X_transformed


def custom_label_binarizer(df,**kwargs):
    model_id = kwargs.pop("model_id",None)
    if model_id is None:
        logger.error("Please provide model_id for encoding the categorical columns")
        raise Exception("Please provide model_id for encoding the categorical columns")

    transf = TransFormer(LabelBinarizer)
    print(df.columns)
    return transf.fit_transform(X=df,model_id=model_id)


def custom_label_encoder(df,**kwargs):
    model_id = kwargs.pop("model_id",None)
    if model_id is None:
        logger.error("Please provide model_id for encoding the categorical columns")
        raise Exception("Please provide model_id for encoding the categorical columns")
    
    categ = ["Store_Type","Location_Type","Region_Code","Holiday","Discount"]
    # Encode Categorical Columns
    le = LabelEncoder()
    df[categ] = df[categ].apply(le.fit_transform)
    return df