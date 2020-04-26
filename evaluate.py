""" Script to evaluate LSCD predictions. """

from scipy.stats import spearmanr

import pandas as pd
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="Evaluate an LSCD prediction for a dataset.")

parser.add_argument("dataset_dir", type=str, help="Dataset folder")
parser.add_argument("experiment_dir", type=str, help="Experiment folder")

parser.add_argument("--subfolders", dest="subfolders", action="store_true", help="Evaluate all subfolders and create csv with all scores")


def evaluate(dataset_dir="", experiment_dir="", subfolders=False):
    """ Evaluates an LSCD prediction for a dataset. """

    if subfolders:
        pass

    else:

        corr, p_val = evaluate_experiment(dataset_dir, experiment_dir)

        print("The Spearman correlation for experiment {} is:".format(experiment_dir))
        print("{:.6f} at a p-value of {:.6f}.".format(corr, p_val))



def evaluate_experiment(dataset_dir, experiment_dir):

    if not dataset_dir.endswith("/"):
        dataset_dir += "/"

    truth_fp = dataset_dir + "truth.tsv"
    assert os.path.exists(truth_fp), "No truth.tsv found in {}!".format(dataset_dir)

    if not experiment_dir.endswith("/"):
        experiment_dir += "/"

    pred_fp = experiment_dir + "prediction.tsv"
    assert os.path.exists(pred_fp), "No prediction.tsv found in {}!".format(experiment_dir)

    assert dataset_dir.split("/")[-2] == experiment_dir.split("/")[-2].split("_")[-1], "Experiment folder does not belong to the given dataset!"

    pred_df = pd.read_csv(pred_fp, sep="\t", names=["word", "change"], header=None).sort_values("word")
    y_df = pd.read_csv(truth_fp, sep="\t", names=["word", "change"], header=None).sort_values("word")
    assert np.all(y_df.word.values == pred_df.word.values)

    pred = np.argsort(np.argsort(pred_df.change.values))
    y = y_df.change.values

    corr, p_val = spearmanr(pred, y)

    return corr, p_val


if __name__ == "__main__":

    args = parser.parse_args()
    params = vars(args)
    evaluate(**params)
