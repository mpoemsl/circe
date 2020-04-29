""" General utilities for LSCD models. """

import numpy as np


def get_word_freqs(words):
    """ Returns dictionary frequency of word frequencies of words in a list. """

    freqs = {w: 0 for w in words}

    for word in words:
        freqs[word] += 1

    return freqs


def find_first_seq_ics(arr, seq):
    """ Finds the indices of the first occurrence of a sequence of ints in another sequence. """

    seq_pos = 0
    current_start = 0

    for ix, elem in enumerate(arr):

        if elem == seq[seq_pos]:
            seq_pos += 1
            if seq_pos == len(seq):
                return np.arange(current_start, current_start + seq_pos)
        else:
            current_start = ix + 1
            seq_pos = 0

    return []


def apply2dicts(data_dict_1, data_dict_2, func):
    """ Applies a function pairwise to values with the same key in two dicts. """

    return {key: func(data_dict_1[key], data_dict_2[key]) for key in data_dict_1.keys()}


def dict2array(data_dict, keys):
    """ Turns the values of dict into a numpy array in an order specified by keys. """

    return np.array([data_dict[key] for key in keys])


def mask_sent(sent, uniques):
    """ Replaces unique words with [MASK]-token. """

    return " ".join(["[MASK]" if word in uniques else word for word in sent.split()])

