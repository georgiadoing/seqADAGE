"""
Script to train one L1 model based on the optimal hyperparameter combination
and save it and the loss history locally.

Date: October 31, 2026

:author: Mukundan Thanigaivelan
"""

import sys
from pathlib import Path

# Prepend the correct prefix to load modules from adage/
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from json import load
from numpy import savetxt
from adage import SeqAdage

def load_hps(filename):
    """
    Load the optimal hyperparameter combination saved locally
    and return it in a dictionary.

    :param filename: a string containing the relative path to the file
    :return: a dictionary with the optimal combination
    """
    with open(filename, 'r') as f:
        optimal_hps = load(f)

    return optimal_hps

def run_model(run_id, filename, optimal_hps):
    """
    Train one ADAGE model based on the given hyperparameter values.

    :param run_id: the current model index
    :param filename: a string containing the relative path to the file
    :param optimal_hps: the hyperparameter values to use
    :return: a SeqAdage object containing the autoencoder and loss history
    """
    sa = SeqAdage.SeqAdage(
        str(filename),
        seed = run_id + 100,
        enc_dim = optimal_hps['units'],
        kl1 = optimal_hps['kl1'],
        kl2 = optimal_hps['kl2'],
        act = optimal_hps['act1'],
        act2 = optimal_hps['act2'],
        epochs = 50,
        init = optimal_hps['init'],
        batch_size = optimal_hps['bs'],
        dropout = optimal_hps['dropout'],
        mm = optimal_hps['mm'],
        lr = optimal_hps['lr'],
        tied = optimal_hps['tied']
    )
    sa.train_model()
    return sa

def save_model(filename, sa):
    """
    Save the model locally.

    :param filename: a string containing the relative path to the file
    :param sa: a SeqAdage object containing the autoencoder and loss history
    """
    sa.autoencoder.save(filename)

def save_loss(run_id, data_dir, sa):
    """
    Save the model loss history locally.

    :param run_id: the current model index
    :param data_dir: a Pathlib object containing the path to data/
    :param sa: a SeqAdage object containing the autoencoder and loss history
    """
    for loss_type in ['loss', 'val_loss']:
        savetxt(
            data_dir / f"ec_L1_model_{loss_type}_{run_id}.csv",
            sa.history.history[loss_type],
            delimiter=','
        )

def main():
    run_id = int(sys.argv[1])
    data_dir = project_root / "muk_uADAGE" / "data"
    
    hps_file = data_dir / "ec_L1_model_hps.json"
    comp_file = data_dir / "ecmg_lcn01_medmean_normal_muk.csv"
    model_file = data_dir / f"ec_L1_model_{run_id}.keras"

    hps = load_hps(hps_file)
    sa_model = run_model(run_id, comp_file, hps)
    save_model(model_file, sa_model)
    save_loss(run_id, data_dir, sa_model)

if __name__ == "__main__":
    main()
