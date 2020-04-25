""" Script to evaluate predictions of lexical semantic change made by the W2V, BERT or ensemble method. """

from scipy.stats import spearmanr

import pandas as pd
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="Evaluate rank semantic change scores on DURel data set.")
parser.add_argument("pred_fp", type=str, help="Path to tsv file with predictions")
args = parser.parse_args()


def main():

    corpus = "_".join(args.pred_fp.split("/")[-1].split("_")[2:4])

    truth_fp = "datasets/{}/truth.tsv".format(corpus)
    assert os.path.exists(truth_fp), "Truth for corpus {} not found!".format(corpus)

    pred_df = pd.read_csv(args.pred_fp, sep="\t", names=["word", "change"], header=None).sort_values("word")
    y_df = pd.read_csv(truth_fp, sep="\t", names=["word", "change"], header=None).sort_values("word")
    assert np.all(y_df.word.values == pred_df.word.values)

    pred = np.argsort(np.argsort(pred_df.change.values))
    y = y_df.change.values

    corr, p_val = spearmanr(pred, y)

    print("The Spearman correlation between {} and the truth is:".format(args.pred_fp))
    print("{:.6f} at a p-value of {:.6f}.".format(corr, p_val))

    return corr


if __name__ == "__main__":
    main()
