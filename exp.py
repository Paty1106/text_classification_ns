from Trainer import *
from CorpusTH import *
from CorpusTE import CorpusTE
from KFold import KFold
from Tuner import *

def pre_rs_supernatural():
    file_config = FilesConfig(vocab_file='twitter_hashtag/twitterhashtags.vocab',
                              dataset_file='twitter_hashtag/out.txt', task='supernatural')
    c = CorpusTE(train_file='DataSetsEraldo/dataSetSupernatural.txt',
                 vocab_file='twitter_hashtag/twitterhashtags.vocab')
    x, y = c.prepare()
    print(c.size)
    f = KFold(c, 3, rand=1)
    f.prepare_fold(x, y)

    myTuner = Tuner(c, file_config)
    epochs = (20, 6)
    lrs = (1e-4, 1e-2)
    myTuner.random_search_cv(execs=1, epoch_limits=epochs, lr_limits=lrs, cv=1, folds=f, freeze_lr=True,
                             freeze_epochs=True)
    print("RS finished!\n")


def supernatural_rs():

    file_config = FilesConfig(vocab_file='twitter_hashtag/twitterhashtags.vocab',
                              dataset_file='twitter_hashtag/out.txt', task='supernatural')
    c = CorpusTE(train_file='DataSetsEraldo/dataSetSupernatural.txt',
                 vocab_file='twitter_hashtag/twitterhashtags.vocab')
    x, y = c.prepare()
    print(c.size)
    f = KFold(c, 3, rand=1)
    f.prepare_fold(x, y)

    myTuner = Tuner(c, file_config)
    epochs = (20, 80)
    lrs = (1e-5, 1e-2)
    myTuner.random_search_cv(execs=5, epoch_limits=epochs, lr_limits=lrs, cv=5, folds=f, freeze_lr=False)
    print("RS finished!\n")