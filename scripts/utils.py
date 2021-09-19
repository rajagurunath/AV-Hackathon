import pandas as pd 
import numpy as np 
from metrics import mean_squared_log_error

ERROR_METRIC = []

def _train(model,**kwargs):
    return model.fit(**kwargs)


def _prediction(model,*args,**kwargs):
    return model.predict(*args,**kwargs)

def train_test_split(X,y):
    split = int(len(X)*0.8)
    X_train = X[:split]
    y_train = y[:split]
    X_test = X[split:]
    y_test = y[split:]

    return X_train,y_train,X_test,y_test


def _eval(model,X_test,y_test):

    preds = model.predict(X_test)
    # print(preds)
    preds = np.where(preds<0,-preds,preds)
    return preds,mean_squared_log_error(y_test,preds)*1000

def report_error(error,model_id,id_column):
    ERROR_METRIC.append([id_column,model_id,error])
    pd.DataFrame(ERROR_METRIC).to_markdown(open("report2.md","w"))