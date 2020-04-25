""" Script to create an ensemble of two existing predictions of lexical semantic change degree. """

from scipy.stats import spearmanr
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="Make ensemble prediction from two predictions of change.")
parser.add_argument("w2v_fp", type=str, help="Path to BERT prediction file")
parser.add_argument("bert_fp", type=str, help="Path to W2V prediction file")
parser.add_argument("weight", type=float, help="Weight for BERT prediction")
parser.add_argument("--export", action="store_true", help="Export resulting png")

args = parser.parse_args()


def main():

    corpus_1 = "_".join(args.bert_fp.split("/")[-1].split("_")[2:4])
    corpus_2 = "_".join(args.w2v_fp.split("/")[-1].split("_")[2:4])
    assert corpus_1 == corpus_2, "Not predictions of the same corpus!"
    corpus = corpus_1

    bert_df = pd.read_csv(args.bert_fp, sep="\t", names=["word", "change"], header=None).sort_values("word")
    w2v_df = pd.read_csv(args.w2v_fp, sep="\t", names=["word", "change"], header=None).sort_values("word")
    assert np.all(bert_df.word.values == w2v_df.word.values)

    truth_fp = "datasets/{}/truth.tsv".format(corpus)
    assert os.path.exists(truth_fp), "Truth for corpus {} not found!".format(corpus)

    y_df = pd.read_csv(truth_fp, sep="\t", names=["word", "change"], header=None).sort_values("word")
    assert np.all(y_df.word.values == w2v_df.word.values)

    rank_bert = np.argsort(np.argsort(bert_df.change.values))
    rank_w2v = np.argsort(np.argsort(w2v_df.change.values))

    calc_args(y_df, rank_bert, rank_w2v, corpus)


def calc_args(y_df, rank_bert, rank_w2v, corpus):

    out_fp = "predictions/ensemble-circe_{}_prediction.tsv".format(corpus)

    ensemble_prediction = args.weight * rank_bert + (1 - args.weight) * rank_w2v
    rank_ensemble = np.argsort(np.argsort(ensemble_prediction)).astype(float)

    print("Chosen weight is {:.2f} with score {:.4f}.".format(args.weight, spearmanr(rank_ensemble, y_df.change.values)[0]))

    ens_df = pd.DataFrame({"word": y_df.word.values, "change": rank_ensemble})
    ens_df.to_csv(out_fp, sep="\t", header=False, index=False)

    

if __name__ == "__main__":
    main()
