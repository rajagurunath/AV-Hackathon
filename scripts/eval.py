

import matplotlib.pyplot as plt

def plot_eval(preds,y_test,store_id,error):
    plt.figure()
    # print(preds)
    # print(y_test)
    plt.plot(preds,label="prediction")
    plt.plot(y_test,label="original")
    plt.title(f"Error of {store_id} = {error}")
    plt.legend()
    plt.savefig(f"../imgs/{store_id}.png")
