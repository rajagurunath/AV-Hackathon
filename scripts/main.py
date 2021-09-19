from re import T
import json
from loguru import logger
from models import Ensembler,Blender,Combiner
from metrics import mean_squared_log_error
from features import (get_datetime_features,basic_categorical_encoding,
                     custom_label_binarizer,custom_label_encoder)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import (Lasso,Ridge,LinearRegression,
                                 RidgeCV,BayesianRidge,ARDRegression,
                                 SGDRegressor)
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor,XGBRFRegressor
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import typer
from tabulate import tabulate


def make_ensemble(model_to_fit,model_ids,model_params,transformers,id_column):
    model = Ensembler(
        model_ids=model_ids,
        model=model_to_fit,
        model_params=model_params,
        transformers = transformers,
        id_column=id_column
    )
    return model

def main(eval:bool=False,plot_eval:bool=False):

    train = pd.read_csv("../data/TRAIN.csv")
    test = pd.read_csv("../data/TEST_FINAL.csv")
    req_cols =["Store_id","Store_Type","Location_Type","Region_Code","Holiday","Date","Discount"]
    track_error = []
    model_blender = Blender(operation="mean") 
    for id_column in ["Store_id","Store_Type","Location_Type","Region_Code"]:
        ids = train[id_column].unique().tolist()
        print(len(ids))
        model = make_ensemble(
            model_to_fit = Combiner,
            model_ids=ids,
            model_params={"models":[Ridge,BayesianRidge,ARDRegression,LinearRegression]},
            transformers = [get_datetime_features,basic_categorical_encoding],
            id_column=id_column
        )
        print(model)
        train_y = train["Sales"]
        if eval:
            logger.info("Evaluation started")
            total_error = model.eval(
            train_X=train,
            train_y=train_y,
            plot = plot_eval
        )
            logger.info("====================================")
            logger.info(f"Total Error, {total_error}")
            logger.info("====================================")
            track_error.append([id_column,total_error])

        else:
            model.train(
                train_X=train,
                train_y=train_y
            )
            prediction = model.predict(test_X=test,req_cols=req_cols)

            print(prediction['Sales'].unique())
            print(prediction['Sales'].unique().shape)

            model_blender.add(prediction)
            prediction.set_index("ID").to_csv(f"../submissions/submission_{id_column}.csv")
            
    
    if eval:
        logger.error(
            tabulate(tabular_data=track_error)
        )
        metrics_json = {}
        for err in track_error:
            key,value = err
            metrics_json[key] = value # To show metrics in pull request otherwise it will show 0.0
        json.dump(metrics_json,open("metrics.json","w"))
        metrics_df = pd.DataFrame([metrics_json]).T.reset_index()
        metrics_df.columns = ['entities',"error"]
        print(metrics_df)
        metrics_df.to_markdown(open("report.md","w"))
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set_color_codes("dark")
        ax = sns.barplot(x="entities",
                         y='error',
                         data=metrics_df,
                         palette = "Greens_d")
        ax.set(xlabel="entities", ylabel = "ERROR")
        plt.savefig("score.png",dpi=80)

    else:
        model_blending_submission = model_blender.blend()
        model_blending_submission.set_index("ID").to_csv(f"../submissions/submission_blending.csv")

if __name__ == "__main__":
    typer.run(main)
