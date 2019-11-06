from Trainer import *
from Trainer_NegSamp import *
from CorpusTH import *
from CorpusTE import CorpusTE
from KFold import KFold
from Tuner import *
import numpy as np


def exp_hashtags_NegSamp():

    file_config = FilesConfig(vocab_file='twitter_hashtag/twitterhashtags.vocab', dataset_file= 'twitter_hashtag/multiple.txt')

    corpus = TwitterHashtagCorpus(files=['train.txt', 'validation.txt'], vocab_file=file_config.vocab_file) # corpus
    # abre, lê, troca o indice...

    cnn_config = TCNNConfig(num_epochs=200) # Configurações da rede

    #corpus.x_train = corpus.x_train[:1000, :]
    #corpus.y_train = corpus.y_train[:1000]
    #corpus.x_validation = corpus.x_validation[:500, :]
    #corpus.y_validation = corpus.y_validation[:500]

    text_cnn = TextCNN_NegSamp(config = cnn_config) # rede

    #print(corpus.label_to_id.keys(), "\n")
    #print(corpus.label_to_id)


    t = Trainer_NegSamp(model = text_cnn, config=cnn_config, corpus = corpus, file_config = file_config)
    d = []

    result = t.train(train_data=d)


def exp_hashtags():

    file_config = FilesConfig(vocab_file='twitter_hashtag/twitterhashtags.vocab', dataset_file= 'twitter_hashtag/multiple.txt')

    corpus = TwitterHashtagCorpus(files=['train.txt', 'validation.txt'], vocab_file=file_config.vocab_file) # corpus
    # abre, lê, troca o indice...

    cnn_config = TCNNConfig(num_epochs=200) # Configurações da rede

    #corpus.x_train = corpus.x_train[:1000, :]
    #corpus.y_train = corpus.y_train[:1000]
    #corpus.x_validation = corpus.x_validation[:500, :]
    #corpus.y_validation = corpus.y_validation[:500]

    text_cnn = TextCNN(config = cnn_config) # rede

    t = Trainer(model = text_cnn, config=cnn_config, corpus = corpus, file_config = file_config)
    d = []

    result = t.train(train_data=d)


def separate_data_set():
    f = open('twitter_hashtag/multiple.txt', mode = 'r')
    train = open('train.txt', mode = 'w')
    validation = open('validation.txt', mode = 'w')

    f.readline()

    x = []
    for l in f:
        x.append(l)

    indices = np.random.permutation(np.arange(len(x)))
    size = int(0.9*len(x))
    for i in range(size):
         train.write(x[indices[i]])
    for j in range(len(x)- size):
        s = j+size
        validation.write(x[indices[s]])


#separate_data_set()
















