from logging import log
import pandas as pd
from pandas.core import series
from pandas.core.algorithms import mode
from logger import logger
from eval import plot_eval
from metrics import mean_squared_log_error
import numpy as np
from tabulate import tabulate
from utils import (_train,_eval,_prediction,
                    train_test_split,report_error)

class Ensembler(object):

    def __init__(self,model_ids,model,model_params:dict={},
                    id_column:str="Store_id",
                    transformers = {}) -> None:
        self.model_ids = model_ids
        self.id_column = id_column
        self.model = model
        self.model_params = model_params
        self.model_map = {}
        self.transformers  = transformers
        super().__init__()

    def train(self,train_X,train_y,**kwargs):

        for model_id in self.model_ids:
            logger.info(f"Training started for {self.id_column} = {model_id}")
            df = train_X[train_X[self.id_column]==model_id]
            y = train_y[train_X[self.id_column]==model_id]
            model = self.model(**self.model_params)
            ID = df.pop("ID")
            Sales = df.pop("Sales")
            for transformer in self.transformers:
                logger.info(f"Applying Transformer {transformer.__name__}")
                df = transformer(df,model_id=model_id)
            
            # logger.info(f"Trained with {df.columns}")
            
            # assert "Sales" not in df.columns, f"Dont cheat , Target should not \
            #                                     be present in Training data \
            #                                     {df.columns.tolist()}"

            trained_model = _train(model,X=df,y=y,**kwargs)
            self.model_map[model_id] = trained_model
            if hasattr(trained_model,"score"):
                score  = trained_model.score(X=df,y=y)
                logger.debug(f"Score {score}")
        assert len(self.model_map) == len(self.model_ids), "Some models are missing"
    
    def predict(self,test_X,req_cols,**kwargs):
        total_prediction = []
        
        for model_id in self.model_ids:
            logger.info(f"Prediction started for {self.id_column} = {model_id}")
            df = test_X[test_X[self.id_column]==model_id]
            ID = df.pop("ID")
            for transformer in self.transformers:
                logger.debug(f"Applying Transformer {transformer.__name__}")
                df = transformer(df,model_id=model_id)
            
            # logger.info(f"Prediction with {df.columns}")
            preds = _prediction(self.model_map[model_id],df)
            total_prediction.append(
                pd.DataFrame(
                    {
                        "ID":ID,
                        "Sales":preds
                    }
                )
            )
        total_prediction = pd.concat(total_prediction)
        assert total_prediction.shape[0] == test_X.shape[0],f"Some rows are missing \
                                                            in prediction {total_prediction.shape} \
                                                            !={test_X.shape}"
        return total_prediction

    def eval(self,train_X,train_y,plot=False,**kwargs):
        total_error = 0
        for model_id in self.model_ids:
            logger.info(f"Training started for {self.id_column} = {model_id}")
            df = train_X[train_X[self.id_column]==model_id]
            y = train_y[train_X[self.id_column]==model_id]

            ID = df.pop("ID")
            for transformer in self.transformers:
                logger.debug(f"Applying Transformer {transformer.__name__}")
                df = transformer(df,model_id=model_id)

            X_train,y_train,X_test,y_test = train_test_split(df,y)
            model = self.model(**self.model_params)
            trained_model = _train(model,X=X_train,y=y_train,**kwargs)
            self.model_map[model_id] = trained_model
            if hasattr(trained_model,"score"):
                score  = trained_model.score(X=X_train,y=y_train)
                logger.info(f"Score {score}")
            # eval
            preds,error = _eval(trained_model,X_test,y_test=y_test)
            report_error(error,model_id,self.id_column)
            total_error +=error
            logger.error(f"Error for {self.id_column} {model_id} = {error}")
            if plot:
                # print(preds)
                plot_eval(
                    preds=preds,
                    y_test=y_test.reset_index(drop=True),
                    store_id=model_id,
                    error =error
                )

        assert len(self.model_map) == len(self.model_ids), "Some models are missing"
        return total_error

class Blender(object):
    def __init__(self,operation:str="mean") -> None:
        self.model_results =[]
        self.operation = operation
        self._first=True
        super().__init__()

    def add(self,submission_df):
        if self._first:
            sub_df = submission_df.sort_values(by='ID').reset_index(drop=True)
            self._first = False
        else:
            sub_df = submission_df.sort_values(by='ID').reset_index(drop=True).drop("ID",axis=1)
        self.model_results.append(sub_df)

    def blend(self):
        blending_res =  pd.concat(self.model_results,axis=1)
        if self.operation =="mean":
            series= blending_res.mean(axis=1)
        elif self.operation =="min":
            series = blending_res.min(axis=1)
        elif self.operation =="max":
            series = blending_res.max(axis=1)
        res = series.to_frame(name="Sales")
        res["ID"]= blending_res.ID
        return res