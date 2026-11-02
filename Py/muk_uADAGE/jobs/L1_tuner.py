"""
Script to run the hyperparameter search to identify the optimal hyperparameter combination
for L1 ADAGE models on this compendium.

Date: October 31, 2026

:author: Mukundan Thanigaivelan
"""

import sys
from pathlib import Path

# Prepend the correct prefix to load modules from adage/
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from adage import SeqAdage
from json import dump, load

# def load_hps(filename):
#     """
#     Given the filepath to the optimal baseline hyperparameter values,
#     return them in a dictionary.

#     :param filename: a string containing the relative path to the file
#     :return: a dictionary with the optimal values
#     """
#     with open(filename, "r") as f:
#         optimal_hps = load(f)
#     return optimal_hps

def run_search(filename): # hps):
    """
    Run the actual hyperparameter search solely for L1 weight on the 
    given compendium.

    :param filename: a string containing the relative path to the file
    :return: a list of HyperParameters objects
    """
    model_trainer = SeqAdage.SeqAdage(str(filename))
    best_hps, tuner = model_trainer.tune_model(
        seed = 42,
        # u = [hps["units"], hps["units"], 1],
        d = [0, 0, 0.1],
        # a1 = [hps["act1"]],
        # a2 = [hps["act2"]],
        # i = [hps["init"]],
        k1 = [0.04, 1, 0.04], # *** Hyperparameter of interest ***
        k2 = [0, 0, 0.1],
        # l = [hps["lr"], hps["lr"], 0.1],
        # b = [hps["bs"], hps["bs"], 1],
        # m = [hps["mm"], hps["mm"], 0.1],
        # sh = [hps["shuffle"]]
    )
    return best_hps

def save_results(best_hps, output_path):
    """
    Given the optimal hyperparameter combinations, save them to
    a JSON file.

    :param best_hps: a list of HyperParameters objects
    :param output_path: a string containing the path to output file
    """
    best_hps_values = [hp.values for hp in best_hps]
    with open(output_path, "w") as f:
        dump(best_hps_values, f, indent = 4)

def main():
    data_dir = project_root / "muk_uADAGE" / "data"
    # hps_file = data_dir / "ec_unreg_model_hps.json"
    filename = data_dir / "ecmg_lcn01_medmean_normal_muk.csv"
    output_path = data_dir / "L1_tuner_results.json"

    # prev_hps = load_hps(hps_file)
    best_hps = run_search(filename) # prev_hps)
    save_results(best_hps, output_path)

if __name__ == "__main__":
    main()
