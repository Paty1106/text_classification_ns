#Corpus referente ao dataset de reconhecimento de entidades(Sao Paulo, Bahia, etc.)

from CorpusHelper import *
from torch import tensor
import random

class CorpusTE(Corpus):
    def __init__(self, train_file, vocab_file, dev_split=0.2, test_split=None, sent_max_length=50, vocab_size=8000):
        super().__init__()
        self.train_file = train_file
        self.vocab_file = vocab_file

        self.dev_split = dev_split
        self.test_split = test_split
        self.label_to_id = {'Nao': 0, 'Sim': 1}
        self.max_labels = len(self.label_to_id)

    def prepare_data(self, args):
        x_data = []
        y_data = []
        with CorpusHelper.open_file(self.train_file) as f:
            for l in f:
                l = l.strip()
                ftrs = l.split('\t')
                text = ftrs[1]
                label = ftrs[-1]
                x_data.append(text)
                ex_label = self.label_to_id[label.split()[0]]
                y_data.append(ex_label) # Get just the first one hashtag

        self.words, self.word_id = CorpusHelper.read_vocab(self.vocab_file)

        for i in range(len(x_data)):  # tokenizing and padding
            x_data[i] = CorpusHelper.process_text(x_data[i], self.word_id,
                                                  self.sentence_size, clean=False)

        # print(x_data)
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        self.size = len(y_data)

        return x_data, y_data

    def train_distribution(self):
        pos = np.sum(self.y_train)
        pos = pos/len(self.y_train)
        return pos


    def prepare_sample(self, x, y, size):
        self.shuffle(x, y)
        pos = np.sum(self.y_train)
        while(pos < size):
            self.shuffle(x, y)
            pos = np.sum(self.y_train)

    def sub_sampling(self, size, val=False):
        #pega o num de exem da classe menos contemplada no cjt de treino, seleciona a mesma qtd
        #da outra classe e entao pega 2n exemplos do dev e test. um novo split de tamanho 2n é gerado
        n_mask = np.logical_not(np.array(self.y_train, dtype=bool))
        mask = np.array(self.y_train, dtype=bool)

        pos = self.x_train[mask].reshape(-1, 50)#[:size]
        neg = self.x_train[n_mask].reshape(-1, 50)#[: pos.shape[0]]

        index_p = np.random.permutation(np.arange(pos.shape[0]))[:size]
        index_n = np.random.permutation(np.arange(neg.shape[0]))[:size]
        pos = pos[index_p]
        neg = neg[index_n]

        pos_y = np.ones(pos.shape[0])
        neg_y = np.zeros(neg.shape[0])

        x = np.append(pos, neg, axis=0)
        y = np.append(pos_y, neg_y, axis=0)

        #Shuffle train
        super().shuffle(x, y, dev_split=None, train_split=1)
        if val:
            if self.y_test is not None:#rever
                self.y_test = self.y_test[:len(y)]
                self.x_test = self.x_test[:len(y), :]
            if self.y_validation is not None:
                self.y_validation = self.y_validation[:int(len(y)/2)]
                self.x_validation = self.x_validation[:int(len(y)/2), :]






    def prepare(self):
        return super().prepare(self.prepare_data)

    def build_vocab(self):
        pass  # TODO
