""" File operation utilities for LSCD models. """

from models.utils.bert_utils import BinaryProcessor, convert_examples_to_features
from models.utils.general_utils import mask_sent

from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from torch.utils.data import  TensorDataset
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
import io


def load_vector_dict(vec_fp, words):
    """ Utility to load word embeddings from a .vec file into a dictionary. """

    vdict = {}

    fh = io.open(vec_fp, "r", encoding="utf-8", newline="\n", errors="ignore")
    fh.readline()

    for line in fh:

        tokens = line.rstrip().split(" ")

        if tokens[0] in words:

            vdict[tokens[0]] = np.array(tokens[1:], dtype=float)

            if len(vdict) == len(words):
                break

    assert len(vdict) == len(words), "Not all target words were found in the *.vec file {}!".format(vec_fp)

    return vdict


def make_masked_copy(filepath):
    """ Makes a copy of a data set in which all words that are distinct for a class are masked. """

    df = pd.read_csv(filepath, sep="\t", header=None, names=["id", "label", "alpha", "text"])
    df.text = df.text.astype(str)

    text_1 = " ".join(df[df.label == 1].text.values)
    text_2 = " ".join(df[df.label == 0].text.values)

    vocab_1 = set(text_1.split())
    vocab_2 = set(text_2.split())

    unique_words = (vocab_1 - vocab_2).union(vocab_2 - vocab_1)
    df.text = df.text.apply(lambda x: mask_sent(x, unique_words))

    new_fp = filepath[:-4] + "_masked.tsv"
    df.to_csv(new_fp, sep="\t", index=False, header=False)


def load_dataset(tokenizer, bert_name, filepath, max_sql=128):
    """ Loads and prepares a local dataset for a BERT model. """

    set_type = filepath.split("/")[-1][:-4]

    processor = BinaryProcessor()

    label_list = processor.get_labels()
    examples = processor.get_examples(filepath, set_type)

    features = convert_examples_to_features(examples, label_list, max_sql,
        tokenizer,
        cls_token_at_end=False,
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    return dataset


def load_local_bert(bert_dir, device, output_hidden_states=False):
    """ Loads a finetuned BERT model from local storage. """

    tokenizer = BertTokenizer.from_pretrained(bert_dir)
    model = BertForSequenceClassification.from_pretrained(bert_dir, output_hidden_states=output_hidden_states)

    model.to(device)

    return tokenizer, model


def load_pretrained_bert(bert_name, device):
    """ Loads a pretrained BERT model from cloud storage for binary classification finetuning. """

    config = BertConfig.from_pretrained(bert_name, num_labels=2, finetuning_task="binary")
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    model = BertForSequenceClassification.from_pretrained(bert_name, config=config)

    model.to(device)

    return tokenizer, model


def collect_sentences(words, sents_fp):
    """ Collects all sentences containing words from a corpus. """

    with open(sents_fp, "r") as fh:
        lines = fh.read().split("\n")

    word_sents = {word: [] for word in words}

    for sent in tqdm(lines, desc="Collecting Sentences"):
        for word in words:
            if word in sent:
                word_sents[word].append(sent)

    return word_sents


def load_rep_dict(rep_dir, targets):
    """ Loads word representations from a directory of *.npy files into a dictionary. """

    rep_dict = {}

    for target in targets:

        rep_dict[target] = np.load(rep_dir + target + ".npy")
        assert rep_dict[target].size > 0, "No representations saved for word '{}' - check both corpora for occurrences!".format(target)

    return rep_dict

