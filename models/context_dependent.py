""" Model for unsupervised lexical semantic change ranking based on context-dependent word representations. """

from models.utils.io_utils import make_masked_copy, load_dataset, load_pretrained_bert, load_local_bert, collect_sentences, load_rep_dict
from models.utils.general_utils import find_first_seq_ics, apply2dicts, dict2array

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, WarmupLinearSchedule
from scipy.spatial.distance import cdist
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
import os


def make_classification_dataset(dataset_dir, experiment_dir):
    """ Creates a balanced time classification dataset from a diachronic LSCD dataset. """

    prep_dir = experiment_dir + "preprocessed_texts/"
    os.makedirs(prep_dir, exist_ok=True)

    with open(dataset_dir + "c1.txt", "r") as fh:
        sents_c1 = fh.read().splitlines()

    with open(dataset_dir + "c2.txt", "r") as fh:
        sents_c2 = fh.read().splitlines()

    # detemine thresholds
    n_samples_per_class = (min(len(sents_c1), len(sents_c2)) // 1_000) * 1_000 # ensure round number
    n_train = int(n_samples_per_class * 2 * 0.8)
    n_test = n_samples_per_class * 2 - n_train

    # collect all samples in DataFrames
    df_0 = pd.DataFrame({"text": sents_c1, "label": 0}).sample(n=n_samples_per_class, replace=False)
    df_1 = pd.DataFrame({"text": sents_c2, "label": 1}).sample(n=n_samples_per_class, replace=False)

    # sample train and test data for each label without overlap
    perm = np.random.permutation(n_samples_per_class)
    train_df = pd.concat([df_0.iloc[perm[:(n_train // 2)]], df_1.iloc[perm[:(n_train // 2)]]], ignore_index=True)
    test_df = pd.concat([df_0.iloc[perm[(n_train // 2):]], df_1.iloc[perm[(n_train // 2):]]], ignore_index=True)

    # check balance of labels
    assert np.all(train_df.label.value_counts() / n_train == test_df.label.value_counts() / n_test), "Classification dataset is not balanced!"

    # shuffle train and test data
    train_df = train_df.sample(frac=1)
    test_df = test_df.sample(frac=1)

    # eliminate this stuff later in data processor

    for df in [train_df, test_df]:
        df["text"] = df["text"].str.rsplit("\t", expand=True)[0]
        df["alpha"] = ["a"] * len(df.index)
        df["id"] = range(len(df.index))

    train_df[["id", "label", "alpha", "text"]].to_csv(prep_dir + "train.tsv", sep="\t", index=False, header=False)
    test_df[["id", "label", "alpha", "text"]].to_csv(prep_dir + "test.tsv", sep="\t", index=False, header=False)

    make_masked_copy(prep_dir + "train.tsv")
    make_masked_copy(prep_dir + "test.tsv")


def finetune_bert(experiment_dir, limited, device="cpu", bert_name="bert-base-multilingual-cased", masked=True, **params):
    """ Finetunes a pretrained BERT model on a sentence time classification objective. """

    bert_dir = experiment_dir + "bert/"
    os.makedirs(bert_dir, exist_ok=True)

    device = torch.device(device)
    tokenizer, model = load_pretrained_bert(bert_name, device)

    if masked:
        train_fp = experiment_dir + "preprocessed_texts/train_masked.tsv"
    else:
        train_fp = experiment_dir + "preprocessed_texts/train.tsv"

    if limited:
        max_sents = 100
    else:
        max_sents = -1
        
    # train bert model
    train_dataset = load_dataset(tokenizer, bert_name, train_fp, max_sents=max_sents)
    train_bert(train_dataset, model, tokenizer, device, **params)

    # save bert model
    model.save_pretrained(bert_dir)
    tokenizer.save_pretrained(bert_dir)

    # reload bert model and test dataset
    tokenizer, model = load_local_bert(bert_dir, device)
    test_dataset = load_dataset(tokenizer, bert_name, experiment_dir + "preprocessed_texts/train.tsv", max_sents=max_sents)

    # evaluate bert model and save results
    acc = test_bert(test_dataset, model, tokenizer, device, **params)
    np.save(bert_dir + "classification_accuracy.npy", np.round(acc, decimals=2))


def train_bert(train_dataset, model, tokenizer, device, n_epochs=1, batch_size=10, learning_rate=4e-5, warmup_ratio=0.05, **kwargs):
    """ Trains a BERT model on a train dataset. """

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

    t_total = len(train_dataloader) * n_epochs

    optimizer_params = [{"params": [p for n, p in model.named_parameters()], "weight_decay": 0.0}]
    optimizer = AdamW(optimizer_params, lr=learning_rate, eps=1e-8)

    warmup_steps = int(warmup_ratio * t_total)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)

    model.zero_grad()
    model.train()

    for _ in tqdm(range(n_epochs), desc="BERT Training"):

        for step, batch in enumerate(tqdm(train_dataloader, desc="Current Epoch")):

            batch = tuple(t.to(device) for t in batch)

            inputs = {"input_ids":      batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2],
                      "labels":         batch[3]}

            outputs = model(**inputs)

            loss = outputs[0]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            model.zero_grad()


def test_bert(test_dataset, model, tokenizer, device, batch_size=10, **kwargs):
    """ Evaluates a BERT model on a test dataset. """

    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

    preds = None
    model.eval()

    for batch in tqdm(test_dataloader, desc="BERT Testing"):

        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():

            inputs = {"input_ids":      batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2],
                      "labels":         batch[3]}

            logits = model(**inputs)[1]

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=1)
    accuracy = (out_label_ids == preds).sum() / preds.shape[0]

    return accuracy


def extract_representations(dataset_dir, experiment_dir, limited, device="cpu", **kwargs):
    """ Extracts last hidden layer values for all target words in a datasets."""

    rep_dir_c1 = experiment_dir + "word_representations/c1/"
    rep_dir_c2 = experiment_dir + "word_representations/c2/"

    os.makedirs(rep_dir_c1, exist_ok=True)
    os.makedirs(rep_dir_c2, exist_ok=True)

    with open(dataset_dir + "targets.tsv", "r") as fh:
        targets = fh.read().splitlines()

    sents_c1 = collect_sentences(targets, dataset_dir + "c1.txt")
    sents_c2 = collect_sentences(targets, dataset_dir + "c2.txt")

    bert_dir = experiment_dir + "bert/"
    device = torch.device(device)

    tokenizer, model = load_local_bert(bert_dir, device, output_hidden_states=True)
    model.eval()

    if limited:
        sents_c1 = {key: value[:40] for key, value in sents_c1.items()}
        sents_c2 = {key: value[:40] for key, value in sents_c2.items()}

    save_representations(sents_c1, model, tokenizer, device, rep_dir_c1)
    save_representations(sents_c2, model, tokenizer, device, rep_dir_c2)


def save_representations(word_sents, model, tokenizer, device, rep_dir):
    """ Saves representations extracted from a number of sentences in a given folder. """

    for word, sents in word_sents.items():

        word_hidden_states = []
        word_tokens = tokenizer.encode(word)

        for sent in tqdm(sents, desc="Extracting Representations for '{}'".format(word)):

            hidden_states, encoded = get_hidden_from_sent(sent, model, tokenizer, device)

            if len(word_tokens) > 0:
                word_token_ics = find_first_seq_ics(encoded, word_tokens)
            else:
                word_token_ics = [np.argmax(encoded == word_tokens[0])]

            if len(word_token_ics) > 0:
                mean_state_last_layer = hidden_states[-1, word_token_ics, :].mean(axis=0)
                word_hidden_states.append(mean_state_last_layer)

        np.save(rep_dir + word + ".npy", np.array(word_hidden_states))


def get_hidden_from_sent(sent, model, tokenizer, device, max_sql=128):
    """ Returns a tokenized sentence and its last hidden layer matrix of a BERT model with dimensions (layers, heads, tokens, tokens). """

    encoded = tokenizer.encode(sent, max_length=max_sql)
    padded = np.array([tokenizer.prepare_for_model(encoded)["input_ids"]])
    token_type_ids = np.zeros_like(padded)
    attention_mask = (~(padded == 0)).astype(int)

    inputs = {
        "input_ids": torch.from_numpy(padded).to(device),
        "attention_mask": torch.from_numpy(attention_mask).to(device),
        "token_type_ids": torch.from_numpy(token_type_ids).to(device),
        "labels": torch.from_numpy(np.array([1]).astype(int)).to(device)
    }

    hidden_matrix = np.vstack([v.cpu().detach().numpy() for v in model(**inputs)[-1]])

    return hidden_matrix[:, :attention_mask.sum(), :], np.array(encoded)


def compare_context_dependent_representations(dataset_dir, experiment_dir):
    """ Compares extracted representations for all target words and makes a prediction. """

    with open(dataset_dir + "targets.tsv", "r") as fh:
        targets = fh.read().splitlines()
  
    c1_reps = load_rep_dict(experiment_dir + "word_representations/c1/", targets)
    c2_reps = load_rep_dict(experiment_dir + "word_representations/c2/", targets)

    dist_func = lambda x, y: np.mean(cdist(x, y, metric="euclidean"))
    dist_dict = apply2dicts(c1_reps, c2_reps, dist_func)
    dists = dict2array(dist_dict, targets)
    
    pd.DataFrame({"word": targets, "change": dists}).to_csv(experiment_dir + "prediction.tsv", sep="\t", index=False, header=False)
    