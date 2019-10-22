

class RandomSplit(object):
    def __init__(self, corpus, n, splits=[0.7, 0.3, 0.0], sub=0, keep_prop=False):
        self.corpus = corpus
        self.n = n
        self.train_split = splits[0]
        self.val_split = splits[1]
        self.test_split = splits[2]
        self.x = None
        self.y = None
        self.sampling = sub
        self.keep_proportion = keep_prop
        self.i = 0

    def __iter__(self):
        self.i = 0
        self.corpus.shuffle(self.x, self.y, train_split=self.train_split, dev_split=self.val_split,
                            test_split=self.test_split)
        if self.sampling > 0:
            self.sub_sampling()
        return self

    def __next__(self):
        if self.i == 0:
            self.i += 1
            return self.corpus
        if self.i < self.n:
            self.corpus.shuffle(self.x, self.y, train_split=self.train_split, dev_split=self.val_split,
                                test_split=self.test_split)
            if self.sampling > 0:
                self.sub_sampling()
            self.i += 1
            return self.corpus
        else:
            self.i = 0
            raise StopIteration

    def sub_sampling(self):
        self.corpus.sub_sampling(self.sampling)

        if self.keep_proportion:
            # Keep the same split of data after sub-sampling
            train_dim = len(self.corpus.y_train)
            val_dim = int((train_dim * self.val_split) / self.train_split)
            test_dim = int((train_dim * self.test_split) / self.train_split)

            self.corpus.x_validation = self.corpus.x_validation[:val_dim, :]
            self.corpus.y_validation = self.corpus.y_validation[:val_dim]

            self.corpus.x_test = self.corpus.x_test[:test_dim, :]
            self.corpus.y_test = self.corpus.y_test[:test_dim]



