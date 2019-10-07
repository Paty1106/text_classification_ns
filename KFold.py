# K-Fold CV ou  Random Split
from Trainer import *
import numpy as np


class KFold (object):

    def __init__(self, corpus, k=3, rand=0):
        self.i = 0
        self.k = k
        self.corpus = corpus
        self.offset = int(corpus.size / k)
        self.rand = rand
        if rand == 0:
            self.fold = self.new_fold
        else:
            self.fold = self.rand_new_fold
        self.x_data = None
        self.y_data = None

    def prepare_fold(self, x_data, y_data): #rever
        self.x_data = x_data
        self.y_data = y_data

    def new_fold(self):
        offset_b = (self.i - 1)*self.offset
        offset_e = self.i*self.offset
        self.corpus.x_validation = self.x_data[offset_b:offset_e, :]
        self.corpus.x_train = np.delete(self.x_data, np.arange(offset_b, offset_e), 0)
        self.corpus.y_validation = self.y_data[offset_b:offset_e]
        self.corpus.y_train = np.delete(self.y_data, np.arange(offset_b, offset_e), 0)

    def rand_new_fold(self):
        # shuffle
        print("Shuffling data..")
        indices = np.random.permutation(np.arange(len(self.x_data)))
        self.x_data = self.x_data[indices]
        self.y_data = self.y_data[indices]
        self.new_fold()

    def compute_metrics(self): #TODO
        pass

    def __next__(self):
        if self.i == 0:
            self.i += 1
            return self.corpus
        if self.i < self.k:
            self.fold()
            self.i += 1
            return self.corpus
        else:
            self.i = 0
            raise StopIteration

    def __iter__(self):
        self.i = 0
        train_off = (self.k-1)*self.offset

        if self.rand == 1:
            #shuffle
            indices = np.random.permutation(np.arange(len(self.x_data)))
            self.x_data = self.x_data[indices]
            self.y_data = self.y_data[indices]

        self.corpus.x_train = self.x_data[:train_off]
        self.corpus.x_validation = self.x_data[train_off:]
        self.corpus.y_train = self.y_data[:train_off]
        self.corpus.y_validation = self.y_data[train_off:]
        return self

    def sub_sampling(self):
        #TODO: implementar balanceamento de classes.
        pass