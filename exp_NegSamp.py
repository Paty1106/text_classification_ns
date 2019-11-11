from Trainer import *
from Trainer_NegSamp import *
from CorpusTH import *
from CorpusTE import CorpusTE
from KFold import KFold
from Tuner import *
import numpy as np


def exp_hashtags_NegSamp():

    file_config = FilesConfig(vocab_file='twitter_hashtag/1kthashtag.vocab', dataset_file= 'twitter_hashtag/multiple.txt')

    corpus = TwitterHashtagCorpus(files=['train.txt', 'validation.txt'], vocab_file=file_config.vocab_file) # corpus

  #  file_config = FilesConfig(vocab_file='twitter_hashtag/1kthashtag.vocab', dataset_file= 'twitter_hashtag/17mi-dataset_clean.txt')
   # corpus = TwitterHashtagCorpus(files=['train_clean.txt', 'validation_clean.txt'], vocab_file=file_config.vocab_file) # corpus

    cnn_config = TCNNConfig_NegSamp(num_epochs=150, n_classes = corpus.max_labels_train) # Configurações da rede

    text_cnn = TextCNN_NegSamp(config = cnn_config) # rede

    t = Trainer_NegSamp(model = text_cnn, config=cnn_config, corpus = corpus, file_config = file_config)
    d = []

    result = t.train(train_data=d)

def exp_hashtags():

 #   file_config = FilesConfig(vocab_file='twitter_hashtag/1kthashtag.vocab', dataset_file= 'twitter_hashtag/multiple.txt')
    #file_config = FilesConfig(vocab_file='twitter_hashtag/1kthashtag.vocab', dataset_file= 'twitter_hashtag/17mi-dataset_clean.txt')
    file_config = FilesConfig(vocab_file='twitter_hashtag/twitterhashtags.vocab', dataset_file= 'twitter_hashtag/multiple.txt')
    corpus = TwitterHashtagCorpus(files=['train.txt', 'validation.txt'], vocab_file=file_config.vocab_file) # corpus

    cnn_config = TCNNConfig(num_epochs=150) # Configurações da rede

    text_cnn = TextCNN(config = cnn_config) # rede

    t = Trainer(model = text_cnn, config=cnn_config, corpus = corpus, file_config = file_config)
    d = []

    result = t.train(train_data=d)


def separate_data_set():
    f = open('twitter_hashtag/multiple.txt', mode = 'r')
    train = open('train.txt', mode = 'w')
    validation = open('validation.txt', mode = 'w')
    f.readline()

    #f = open('twitter_hashtag/17mi-dataset_clean.txt', mode = 'r')
    #train = open('train_clean.txt', mode = 'w')
    #validation = open('validation_clean.txt', mode = 'w')

    x = []
    for l in f:
        x.append(l)

    indices = np.random.permutation(np.arange(len(x)))
    size = int(0.8*len(x))
    for i in range(size):
        #print(i)
        train.write(x[indices[i]])
    for j in range(len(x)- size):
        s = j+size
        #print(s)
        validation.write(x[indices[s]])


#separate_data_set()
















