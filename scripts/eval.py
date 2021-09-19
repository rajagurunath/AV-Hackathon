import matplotlib.pyplot as plt
import pandas as pd

def plot_eval(preds,y_test,store_id,error):
    if len(preds)<1000:
        plt.figure()
        # print(preds)
        # print(y_test)
        plt.plot(preds,label="prediction")
        plt.plot(y_test,label="original")
        plt.title(f"Error of {store_id} = {error}")
        plt.legend()
        plt.savefig(f"../imgs/{store_id}.png")

    else:
        ploter = pd.DataFrame({
            "prediction":preds,
            "original":y_test
        })
        ploter.plot(subplots=True,figsize=(15,8))
        plt.title(f"Error of {store_id} = {error}")
        plt.legend()
        plt.savefig(f"../imgs/{store_id}.png")
