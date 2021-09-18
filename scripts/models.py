import pandas as pd
from logger import logger
from eval import plot_eval
from metrics import mean_squared_log_error

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
    return preds,mean_squared_log_error(y_test,preds)

class Ensembler(object):

    def __init__(self,model_ids,model,model_params:dict={},id_column:str="Store_id") -> None:
        self.model_ids = model_ids
        self.id_column = id_column
        self.model = model
        self.model_params = model_params
        self.model_map = {}
        super().__init__()

    def train(self,train_X,train_y,**kwargs):

        for model_id in self.model_ids:
            logger.info(f"Training started for Store id = {model_id}")
            df = train_X[train_X[self.id_column]==model_id]
            y = train_y[train_X[self.id_column]==model_id]
            model = self.model(**self.model_params)
            trained_model = _train(model,X=df,y=y,**kwargs)
            self.model_map[model_id] = trained_model
            if hasattr(trained_model,"score"):
                score  = trained_model.score(X=df,y=y)
                logger.info(f"Score {score}")
        assert len(self.model_map) == len(self.model_ids), "Some models are missing"
    
    def predict(self,test_X,req_cols,**kwargs):
        total_prediction = []
        
        for model_id in self.model_ids:
            logger.info(f"Prediction started for Store ID = {model_id}")
            df = test_X[test_X[self.id_column]==model_id]
            ID = df.pop("ID")
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
            # logger.info(f"Training started for Store id = {model_id}")
            df = train_X[train_X[self.id_column]==model_id]
            y = train_y[train_X[self.id_column]==model_id]
            X_train,y_train,X_test,y_test = train_test_split(df,y)
            model = self.model(**self.model_params)
            trained_model = _train(model,X=X_train,y=y_train,**kwargs)
            self.model_map[model_id] = trained_model
            if hasattr(trained_model,"score"):
                score  = trained_model.score(X=X_train,y=y_train)
                logger.info(f"Score {score}")
            # eval
            preds,error = _eval(trained_model,X_test,y_test=y_test)
            total_error +=error
            logger.error(f"Error for Store_id {model_id} = {error}")
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

