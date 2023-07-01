datasets = ["Parkinson", "Accent", "Iris"]
commands_file = "grid_search_commands"

with open(commands_file, "w") as f:
    for dataset in datasets:
        for i in range(7):
            f.write(
                f"python main_experimens.py --run_grid True --dataset {dataset} --resfile grid_search_results/{dataset.lower()}_grid_val.csv \n")
