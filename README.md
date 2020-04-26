# CIRCE

A Classification-Informed Representation Comparison Ensemble for Lexical Semantic Change Detection.  

This repository contains code and predictions for the winning contribution to SemEval-2020 Task 1 (Subtask 2). 

Models, experiments and results are described in the upcoming system description paper "CIRCE at SemEval-2020 Task 1: Ensembling Context-Free and Context-Dependent Word Representations".

## Overview

* `predict.py` can be used to make predictions of lexical semantic change ranking with a context-free and a context-dependent model
* `ensemble.py` can be used to ensemble predictions from a context-free and a context-dependent model
* `evaluate.py` can be used to evaluate a prediction of lexical semantic change rank against true ranks
* `datasets/` contains testsets for the development and submission experiments from the paper
* `models/` contains the code for the context-free and context-dependent model
* `submission_experiments/` contains predictions for the submission experiments from the paper


## Requirements

This system runs on Python 3.6. The required packages can best be installed with `pip install -r requirements.txt`.

Additionally, you need to have the submodule in `models/vecmap/`. This can be achieved with `git submodule update --init --recursive`.

If you want to make predictions, you will need to complement the testsets in `datasets/` with the corresponding corpora. If your shell has the utilities `wget`, `unzip` and `gunzip`, you can use the scripts `download_semeval_data.sh` and `download_development_data.sh` for this.


## Usage

Run `python predict.py [context-free|context-dependent] DATASET_FOLDER` to make a prediction. This will create a corresponding experiment folder in `experiments`.

Run `python evaluate.py EXPERIMENT_FOLDER` to evaluate a prediction. Add the flag `--subfolders` to look in subfolders of EXPERIMENT\_FOLDER instead, which stores results in a corresponding *.csv file.

Run `python ensemble.py CONTEXT_FREE_EXPERIMENT_FOLDER CONTEXT_DEPENDENT_EXPERIMENT_FOLDER` to make an ensemble prediction. This will create a corresponding experiment folder in `experiments`. Add the flag `--plot_all` to create a graph with evaluations of all possible weights.

To learn more about any script and its parameters, run `python SCRIPT.py -h`
