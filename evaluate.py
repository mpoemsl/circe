""" Evaluates LSCD predictions made with predict.py or ensemble.py. """

from scipy.stats import spearmanr

import pandas as pd
import numpy as np
import argparse
import os


parser = argparse.ArgumentParser(description="Evaluates LSCD predictions made with predict.py or ensemble.py.")

parser.add_argument("experiment_dir", type=str, help="Experiment folder")
parser.add_argument("--subfolders", dest="subfolders", action="store_true", help="Evaluate all subfolders")


def evaluate(experiment_dir="", subfolders=False):
    """ Evaluates an LSCD prediction for a corresponding dataset in datasets/. """

    if not experiment_dir.endswith("/"):
        experiment_dir += "/"

    if subfolders:

        results = []
        
        # evaluate all experiments in subfolders of experiment_dir
        for experiment_name in os.listdir(experiment_dir):

            model_name, dataset_name = experiment_name.split("_")

            corr, p_val = evaluate_experiment("datasets/"+  dataset_name + "/", experiment_dir + experiment_name + "/")
            results.append({"dataset": dataset_name, "model": model_name, "correlation": corr, "p-value": p_val})

        results_df = pd.DataFrame(results).sort_values("dataset").reset_index(drop=True)
        results_df.to_csv(experiment_dir.split("/")[-2] + "_results.csv")

        print(results_df)

    else:

        dataset_name = experiment_dir.split("/")[-2].split("_")[-1]
        corr, p_val = evaluate_experiment("datasets/" + dataset_name + "/", experiment_dir)

        print("The Spearman correlation for experiment {} is:".format(experiment_dir))
        print("{:.4f} at a p-value of {:.4f}.".format(corr, p_val))


def evaluate_experiment(dataset_dir, experiment_dir):
    """ Evaluates the prediction in a experiment folder against the true ranks in the dataset foldder. """

    truth_fp = dataset_dir + "truth.tsv"
    assert os.path.exists(truth_fp), "No truth.tsv found in {}!".format(dataset_dir)

    pred_fp = experiment_dir + "prediction.tsv"
    assert os.path.exists(pred_fp), "No prediction.tsv found in {}!".format(experiment_dir)
    assert dataset_dir.split("/")[-2] == experiment_dir.split("/")[-2].split("_")[-1], "Experiment folder does not belong to dataset!"

    pred_df = pd.read_csv(pred_fp, sep="\t", names=["word", "change"], header=None).sort_values("word")
    y_df = pd.read_csv(truth_fp, sep="\t", names=["word", "change"], header=None).sort_values("word")
    assert np.all(y_df.word.values == pred_df.word.values), "Predictions do not correspond exactly to target words in dataset!"

    pred = np.argsort(np.argsort(pred_df.change.values))
    y = y_df.change.values

    corr, p_val = spearmanr(pred, y)

    return corr, p_val


if __name__ == "__main__":

    args = parser.parse_args()
    params = vars(args)
    evaluate(**params)

