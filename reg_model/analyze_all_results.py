import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

if __name__ == '__main__':
    res_df = pd.read_csv("all_results.csv")

    # convert use_aug, use_layer_norm, and feats_weighting to meaningful short names
    res_df["use_aug"] = res_df["use_aug"].apply(lambda x: "aug" if x else "n_aug")
    res_df["use_layer_norm"] = res_df["use_layer_norm"].apply(lambda x: "ln" if x else "n_ln")
    res_df["feats_weighting"] = res_df["feats_weighting"].apply(lambda x: "fw" if x else "n_fw")

    # plot the results as a bar subplots, with each bar being different configurations and each subplot being a dataset - there are 9 datasets
    # a configuration is a combination of use_aug, use_layer_norm, and feats_weighting
    # create subplots for each dataset, and a color for each configuration

    # create a new column for the configuration
    res_df["config"] = res_df.apply(lambda row: f"{row['use_aug']}-{row['use_layer_norm']}-{row['feats_weighting']}",
                                    axis=1)

    plt.subplots(3, 3, figsize=(20, 20))
    color_palette = plt.get_cmap("tab10").colors[:8]

    for i, (ds_name, ds) in enumerate(res_df.groupby("dataset")):
        plt.subplot(3, 3, i + 1)
        plt.title(ds_name)
        plt.bar(ds["config"], ds["score_partial_mean"] - ds["score_partial_mean"].min(), color=color_palette, yerr=ds["score_partial_std"])
        plt.xticks(rotation=45)
        # plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.xlabel("Configuration")

    plt.tight_layout()
    plt.savefig("all_results.png")
    plt.show()
