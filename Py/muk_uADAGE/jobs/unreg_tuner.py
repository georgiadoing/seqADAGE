"""
Script to run the hyperparameter search to identify the optimal hyperparameter combination
for unregularized ADAGE models on this compendium.

Date: October 30, 2026

:author: Mukundan Thanigaivelan
"""

import sys
from pathlib import Path

# Prepend the correct prefix to load modules from adage/
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from adage import SeqAdage
from json import dump

def run_search(filename):
    """
    Run the actual hyperparameter search on the given compendium.

    :param filename: a string containing the relative path to the file
    :return: a list of HyperParameters objects
    """
    model_trainer = SeqAdage.SeqAdage(str(filename))
    best_hps, tuner = model_trainer.tune_model(
        seed = 42,
        k1 = [0, 0, 0.1],
        k2 = [0, 0, 0.1],
        d = [0, 0, 0.1]
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
    filename = data_dir / "ecmg_lcn01_medmean_normal_muk.csv"
    output_path = data_dir / "unreg_tuner_results.json"
    
    best_hps = run_search(filename)
    save_results(best_hps, output_path)

if __name__ == "__main__":
    main()
