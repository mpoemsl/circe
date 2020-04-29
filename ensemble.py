""" Ensembles context-free and context-dependent LSCD predictions made with predict.py. """

# only used for visualizations
from scipy.stats import spearmanr

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os


parser = argparse.ArgumentParser(description="Ensembles context-free and context-dependent LSCD predictions made with predict.py.")

parser.add_argument("context_free_dir", type=str, help="Context-free experiment folder")
parser.add_argument("context_dependent_dir", type=str, help="Context-dependent experiment folder")
parser.add_argument("--plot_all", dest="plot_all", action="store_true", help="Plots and evaluates all possible weights")


def ensemble(dataset_dir="", context_free_dir="", context_dependent_dir="", plot_all=False):
    """ Ensembles a context-free and a context-dependent lexical semantic change ranking. """

    # organisational data and directory checks

    if not context_free_dir.endswith("/"):
        context_free_dir += "/"

    if not context_dependent_dir.endswith("/"):
        context_dependent_dir += "/"

    cf_model_name, cf_dataset_name = context_free_dir.split("/")[-2].split("_")
    cd_model_name, cd_dataset_name = context_dependent_dir.split("/")[-2].split("_")

    assert cf_model_name == "context-free" and cd_model_name == "context-dependent", "Experiments labeld wrongly, check argument order!"
    assert cf_dataset_name == cd_dataset_name, "Not experiments on the same dataset!"

    if not os.path.exists("experiments"):
        os.mkdir("experiments")

    dataset_dir = "datasets/{}/".format(cf_dataset_name)
    experiment_dir = "experiments/ensemble-circe_{}/".format(cf_dataset_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print("Experiment data will be stored in {} ...".format(experiment_dir))

    # ensemble experiment execution

    cf_pred_df = pd.read_csv(context_free_dir + "prediction.tsv", sep="\t", names=["word", "change"], header=None).sort_values("word")
    cd_pred_df = pd.read_csv(context_dependent_dir + "prediction.tsv", sep="\t", names=["word", "change"], header=None).sort_values("word")
    assert np.all(cf_pred_df.word.values == cd_pred_df.word.values), "Predictions do not correspond to the same target words!"

    rank_cf = np.argsort(np.argsort(cf_pred_df.change.values))
    rank_cd = np.argsort(np.argsort(cd_pred_df.change.values))

    acc_bert = np.load(context_dependent_dir + "bert/classification_accuracy.npy")
    w_circe = 2 * acc_bert - 1 

    r_circe = w_circe * rank_cd + (1 - w_circe) * rank_cf
    rank_circe = np.argsort(np.argsort(r_circe))

    circe_pred_df = pd.DataFrame({"word": cf_pred_df.word.values, "change": rank_circe})
    circe_pred_df.to_csv(experiment_dir + "prediction.tsv", sep="\t", index=False, header=False)

    print("Finished ensembling. Prediction can be found in {}.".format(experiment_dir + "prediction.tsv"))
        
    # visualization    

    if plot_all:

        print("Plotting all possible weights ...")

        truth_fp = dataset_dir + "truth.tsv"
        assert os.path.exists(truth_fp), "No truth.tsv found in {}!".format(dataset_dir)

        y_df = pd.read_csv(truth_fp, sep="\t", names=["word", "change"], header=None).sort_values("word")
        assert np.all(y_df.word.values == cd_pred_df.word.values), "Predictions do not correspond to truth target words!"

        y = y_df.change.values

        # value space of w_circe = 2.0 * acc_bert - 1.0 contains 51 weigths since acc_bert is rounded to 2 decimals
        weights = np.linspace(0.0, 1.0, 51)

        correlations = []

        for w in weights:

            r_pred = w * rank_cd + (1 - w) * rank_cf
            rank_pred = np.argsort(np.argsort(r_pred))
            corr, _ = spearmanr(rank_pred, y)
            correlations.append(corr)

        visualize_weights(weights, correlations, w_circe, experiment_dir.split("/")[-2], experiment_dir + "all_weights.png")


def visualize_weights(weights, correlations, w_circe, experiment_name, export_fp):
    """ Visualizes all possible ensemble weights for CIRCE. """

    plt.plot(weights, correlations, color="b", label="CIRCE Performance")
    plt.axhline(y=max(correlations[0], correlations[-1]), color="r", linestyle="--", label="Best Component Performance")
    plt.scatter(w_circe, correlations[np.argmin(np.abs(weights - w_circe))], c="g", s=50, label="Submission Weight")

    plt.legend()

    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 1.0)

    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=["Context-Free", 0.2, 0.4, 0.6, 0.8, "Context-Dependent"])

    plt.xlabel("CIRCE Weight")
    plt.ylabel("Spearman Correlation")

    plt.suptitle(experiment_name)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    plt.savefig(export_fp)
    plt.show()


if __name__ == "__main__":

    args = parser.parse_args()
    params = vars(args)
    ensemble(**params)

