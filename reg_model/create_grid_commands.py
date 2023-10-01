# datasets = ["Iris", "Accent", "Ecoli"]
datasets = [
    "analcatdata_germangss",
    "analcatdata_vineyard",
    "arsenic_male_bladder",
    "blood_transfusion",
    "climate",
    "credit_approval",
    "credit_g",
    "diabetes",
    "rmftsa_ctoarrivals",
    "tokyo",
]

commands_file = "grid_search_commands"

with open(commands_file, "w") as f:
    for dataset in datasets:
        for i in range(7):
            f.write(
                f"python main_experimens.py --run_grid True --dataset {dataset.capitalize()} --resfile grid_search_results/{dataset.lower()}_grid_val.csv --run_ts False --run_grid True --show_figs False\n"
            )
