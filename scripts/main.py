from re import T
from loguru import logger
from models import Ensembler
from metrics import mean_squared_log_error
from features import get_datetime_features,basic_categorical_encoding
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso,Ridge,LinearRegression
from xgboost import XGBRegressor,XGBRFRegressor
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import typer



def main(eval:bool=False,plot_eval:bool=False):

    train = pd.read_csv("../data/TRAIN.csv")
    test = pd.read_csv("../data/TEST_FINAL.csv")

    model_ids = train['Store_id'].unique().tolist()

    req_cols =["Store_id","Store_Type","Location_Type","Region_Code","Holiday","Date","Discount"]

    model = Ensembler(
        model_ids=model_ids,
        model=Ridge,
        model_params={}
            # {
            #     "n_estimators":100,
            #     # "objective":"reg:squaredlogerror",
            #     "n_jobs":-1
            #         },
        

    )

    
    train_dataset = get_datetime_features(train[req_cols])
    print(train_dataset.shape)
    train_dataset = basic_categorical_encoding(train_dataset)
    print(train_dataset.shape)

    train_y = train["Sales"]
    if eval:
        logger.info("Evaluation started")
        total_error = model.eval(
        train_X=train_dataset,
        train_y=train_y,
        plot = plot_eval
    )
        logger.info("====================================")
        logger.info(f"Total Error, {total_error}")
        logger.info("====================================")

    else:
        model.train(
            train_X=train_dataset,
            train_y=train_y
        )

        test_dataset = get_datetime_features(test[req_cols])
        print(test_dataset.shape)
        test_dataset = basic_categorical_encoding(test_dataset)
        test_dataset['ID']=test.ID
        print(test_dataset.shape)
        prediction = model.predict(test_X=test_dataset,req_cols=req_cols).set_index("ID")

        # assert pd.testing.assert_series_equal(prediction['ID'],test['ID'],)
        print(prediction['Sales'].unique())
        print(prediction['Sales'].unique().shape)

        prediction.to_csv("submission.csv")



if __name__ == "__main__":
    typer.run(main)