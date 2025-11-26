import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Load all the dataframes
data_frames = []
for dataset in ["bike", "kin8nm"]:
    for model in ["rf", "gbt", "mlp"]:
        for seed in range(5):
            data_frames.append(pd.read_csv(os.path.join("models", dataset, f"{model}_{seed}", "results.csv")))
dataframe = pd.concat(data_frames, axis=0, ignore_index=True)
dataframe.to_csv("aggregated_results.csv")
