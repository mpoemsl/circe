# CIRCE

A Classification-Informed Representation Comparison Ensemble for Lexical Semantic Change Detection (LSCD).  

This repository contains code and predictions for the winning contribution to SemEval-2020 Task 1 (Subtask 2). 

Models, experiments and results are described in the upcoming system description paper "CIRCE at SemEval-2020 Task 1: Ensembling Context-Free and Context-Dependent Word Representations".

## Overview

* `predict.py` can be used to make predictions of lexical semantic change ranking with a context-free and a context-dependent model
* `ensemble.py` can be used to ensemble predictions from a context-free and a context-dependent model
* `evaluate.py` can be used to evaluate a prediction of lexical semantic change rank against true ranks
* `datasets/` contains testsets for the development and submission experiments from the paper
* `models/` contains the code for the context-free and context-dependent model
* `submission_experiments/` contains experiment folder with predictions for the submission experiments from the paper
* `submission_experiments_results.csv` contains the results of evaluating all experiments in `submission_experiments/`

## Requirements

This system runs on Python 3.6. The required packages can best be installed with `pip install -r requirements.txt`. It might be necessary to install Cython separately, you can do this with `pip install Cython==0.29.14`.

Additionally, you need to clone the [VecMap](https://github.com/artetxem/vecmap) submodule in `models/vecmap/`. This can be achieved with `git submodule update --init --recursive`.

If you want to make predictions, you will need to complement the testsets in `datasets/` with the corresponding corpora. If your shell has the utilities `wget`, `unzip`, `gunzip` and `sed`, you can use the bash scripts `download_semeval_data.sh` and `download_development_data.sh` for this.


## Usage

Run `python predict.py [context-free|context-dependent] <dataset-folder>` to make a prediction. This will create a corresponding experiment folder in `experiments`.

Run `python evaluate.py <experiment-folder>` to evaluate a prediction. Add the flag `--subfolders` to look in subfolders of `<experiment-folder>` instead. This will store the results in the file `<experiment-folder>_results.csv`.

Run `python ensemble.py <context-free-experiment-folder> <context-dependent-experiment-folder>` to make an ensemble prediction. This will create a corresponding experiment folder in `experiments`. Add the flag `--plot_all` to create a graph with evaluations of all possible weights, which is stored in the experiment folder.

To learn more about any script and its parameters, run `python <script>.py -h`.

