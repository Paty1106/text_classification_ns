#!/usr/bin/python
# -*- coding: utf-8 -*-
from collections import Counter
import numpy as np
import re
import os
import csv
import pandas


class CorpusHelper(object):
    @staticmethod
    def open_file(filename, mode='r'):
        """
        Commonly used file reader and writer, change this to switch between python2 and python3.
        :param filename: filename
        :param mode: 'r' and 'w' for read and write respectively
        """
        return open(filename, mode, encoding='utf-8', errors='ignore')

    @staticmethod
    def clean(text):
        text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
        text = re.sub(r",", " , ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\(", " \( ", text)
        text = re.sub(r"\)", " \) ", text)
        text = re.sub(r"\?", " \? ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip().lower()

    @staticmethod
    def build_vocab(data, vocab_file, vocab_size=0):
        """
           Build vocabulary file from training data.
           """
        print('Building vocabulary...')
        all_data = []  # group all data
        for content in data:
            all_data.extend(content.split())

        counter = Counter(all_data)  # count and get the most common words
        count_pairs = counter.most_common(vocab_size - 1)
        words, _ = list(zip(*count_pairs))

        words = ['<PAD>'] + list(words)  # add a padding with id 0 to pad the sentence to same length
        CorpusHelper.open_file(vocab_file, 'w').write('\n'.join(words) + '\n')

    @staticmethod
    def read_vocab(vocab_file):
        """
        Read vocabulary from file.
        One word per line.
        """
        words = CorpusHelper.open_file(vocab_file).read().strip().split('\n')
        word_to_id = dict(zip(words, range(len(words))))
        return words, word_to_id

    @staticmethod
    def process_text(text, word_to_id, max_length, clean=True):
        """tokenizing and padding"""
        if clean:  # if the data needs to be cleaned
            text = CorpusHelper.clean(text)
        text = text.split()

        text = [word_to_id[x] for x in text if x in word_to_id]
        if len(text) < max_length:
            text = [0] * (max_length - len(text)) + text
        return text[:max_length]


class Corpus(object):

    def __init__(self, sent_max_length=40, vocab_size=8000):
        self.sentence_size = sent_max_length
        self.vocab_size = vocab_size
        self.word_id = None
        self.words = None  # REVER
        self.size = None

        self.x_train = None
        self.y_train = None
        self.x_validation = None
        self.y_validation = None
        self.x_test = None
        self.y_test = None

    def prepare(self, callback, args=None):
        return callback(args)  # x_data, y_data

    @staticmethod
    def distribution(y):  # TODO
        dt = pandas.DataFrame(y.reshape(-1, 1), columns=['class'])
        print(dt)
        dist = dt.groupby('class').get_group(1)
        print(dist)
        nd = dist.to_numpy().sum()
        dist = (1 / len(y)) * nd
        return dist

    def shuffle(self, x_data, y_data, dev_split=0.3, train_split=0.7, test_split=None):
        # shuffle
        indices = np.random.permutation(np.arange(len(x_data)))
        x_data = x_data[indices]
        y_data = y_data[indices]

        # train/validation/dev split
        dtsize = len(x_data)
        num_train = int(train_split * dtsize)

        self.x_train = x_data[:num_train]
        self.y_train = y_data[:num_train]

        if dev_split is not None:
            num_validation = int(dtsize * dev_split)
            self.x_validation = x_data[num_train:num_train + num_validation]
            self.y_validation = y_data[num_train:num_train + num_validation]
        else:
            num_validation = 0

        if test_split is not None:  # REVER
            self.x_test = x_data[num_train + num_validation:]
            self.y_test = y_data[num_train + num_validation:]

    def join(self):  # TO REMOVE

        x = np.append(self.x_train, self.x_validation, axis=0)
        y = np.append(self.y_train, self.y_validation)

        x = np.append(x, self.x_test, axis=0)
        y = np.append(y, self.y_test)
        return x, y

    def __str__(self):
        test_size = 0 if self.x_test is None else self.x_test
        return 'Training: {},Validation: {},Testing: {}, Vocabulary: {}'.format(len(self.x_train),
                                                                                len(self.x_validation),
                                                                                test_size, len(self.words))
