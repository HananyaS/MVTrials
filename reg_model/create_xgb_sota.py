import xgboost as xgb
from main_experimens import get_split_data
import os
import pandas as pd


if __name__ == '__main__':
    datasets = os.listdir("data/Tabular")
    all_results = {}

    for dataset in datasets:
        print("Starting", dataset)
        if dataset in ["Sensorless", "Credit"]:
            continue
        train_ds, val_ds, test_ds,  = get_split_data(dataset, use_aug=True, as_loader=False, norm=False)

        X_train, y_train = train_ds.X, train_ds.y
        X_val, y_val = val_ds.X, val_ds.y
        X_test, y_test = test_ds.X, test_ds.y

        xgb_model = xgb.XGBClassifier()
        xgb_model.fit(X_train, y_train)
        score = xgb_model.score(X_test, y_test)

        all_results[dataset] = score
        print(f"{dataset}: {score}")

    pd.DataFrame.from_dict(all_results, orient="index").to_csv("xgb_sota.csv")