import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

res_df = pd.read_csv("alpha_beta_results.csv")

os.makedirs("alpha_beta_heatmaps", exist_ok=True)

for ds_name, ds in res_df.groupby("dataset"):
    # create a heatmap for each dataset with alpha and beta as the axes and score as the value
    ds = ds.pivot(index="alpha", columns="beta", values="score")
    sns.heatmap(ds, annot=True, fmt=".3f", cmap="viridis")
    plt.title(ds_name)
    plt.savefig(f"alpha_beta_heatmaps/{ds_name.lower()}.png")
    plt.show()

