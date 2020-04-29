""" Model for unsupervised lexical semantic change ranking based on context-free word representations. """

from models.utils.general_utils import get_word_freqs
from models.utils.io_utils import load_vector_dict

import pandas as pd
import numpy as np
import subprocess
import word2vec
import os


def preprocess_texts(dataset_dir, experiment_dir, filtered):
    """ Removes words below the frequency threshold in both corpora. """

    with open(dataset_dir + "c1.txt", "r") as fh:
        c1 = fh.read()

    with open(dataset_dir + "c2.txt", "r") as fh:
        c2 = fh.read()

    with open(dataset_dir + "targets.tsv", "r") as fh:
        targets = fh.read().splitlines()

    prep_dir = experiment_dir + "preprocessed_texts/"
    os.makedirs(prep_dir, exist_ok=True)

    preprocess_text(c1, targets, prep_dir + "c1.txt", filtered)
    preprocess_text(c2, targets, prep_dir + "c2.txt", filtered)


def preprocess_text(text, targets, export_fp, filtered):
    """ Preprocesses a single text by filtering words for frequency and sentences for length. """

    lines = text.split("\n")
    all_words = text.split()

    if filtered:

        # determine threshold
        min_freq = len(lines) // 50_000

        word_freqs = get_word_freqs(all_words)

        keep_words = set([word for word, freq in word_freqs.items() if freq >= min_freq])
        target_words = set(targets)

        missing_targets = target_words - keep_words

        if len(missing_targets) > 0:
            print("Keeping the following target words despite frequencies below {}:\n".format(min_freq), missing_targets)
            keep_words = keep_words.union(target_words)

    else: 
    
        keep_words = set(all_words)

    new_lines = []

    for line in lines:

        new_words = [word for word in line.split() if word in keep_words]

        if len(new_words) > 1:
            new_lines.append(" ".join(new_words))

    with open(export_fp, "w+") as fh:
        fh.write("\n".join(new_lines))


def train_word2vec(experiment_dir, n_window=10, n_negative=1, dim=300, **kwargs):
    """ Vectorizes all words in the two corpora separately with Word2Vec. """

    vec_dir = experiment_dir + "word_representations/"
    os.makedirs(vec_dir, exist_ok=True)

    prep_dir = experiment_dir +  "preprocessed_texts/"

    word2vec.word2vec(prep_dir + "c1.txt", vec_dir + "c1.vec", size=dim, negative=n_negative, window=n_window, cbow=0, binary=0, min_count=0, verbose=True)
    print()

    word2vec.word2vec(prep_dir + "c2.txt", vec_dir + "c2.vec", size=dim, negative=n_negative, window=n_window, cbow=0, binary=0, min_count=0, verbose=True)
    print()


def align_embeddings(experiment_dir):
    """ Calls VecMap alignment script to align word embeddings. """

    vec_dir = experiment_dir + "word_representations/"

    args = [
        "--normalize", "unit", "center",
        "--init_identical",
        "--orthogonal"
    ]

    files = [
        vec_dir + "c1.vec",
        vec_dir + "c2.vec",
        vec_dir + "c1_aligned.vec",
        vec_dir + "c2_aligned.vec"
    ]
    
    subprocess.call(["python3", "models/vecmap/map_embeddings.py"] + args + files)
    

def compare_context_free_representations(dataset_dir, experiment_dir):
    """ Compares aligned embeddings for all target words and makes a prediction. """

    with open(dataset_dir + "targets.tsv", "r") as fh:
        targets = fh.read().splitlines()

    vec_dir = experiment_dir + "word_representations/"
    
    c1_dict = load_vector_dict(vec_dir + "c1_aligned.vec", targets)
    c2_dict = load_vector_dict(vec_dir + "c2_aligned.vec", targets)

    dists = [np.linalg.norm(c1_dict[target] - c2_dict[target]) for target in targets]
    
    pd.DataFrame({"word": targets, "change": dists}).to_csv(experiment_dir + "prediction.tsv", sep="\t", index=False, header=False)

