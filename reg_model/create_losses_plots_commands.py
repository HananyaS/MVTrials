datasets = ["Iris", "Accent", "Ecoli", "Breast", "Wifi", "Parkinson", "Wine", "Banknote"]
commands_file = "losses_plots_commands"

with open(commands_file, "w") as f:
    for dataset in datasets:
        f.write(
            f"python main_experimens.py --run_grid False --dataset {dataset}  --run_ts False --run_grid False --load_params True --full_test False\n"
        )
