from Trainer import *
from CorpusTH import *
from CorpusTE import CorpusTE
from KFold import KFold
from Tuner import *




def test_cnn_th():
    cnn_config = TCNNConfig()
    cnn_config.num_epochs = 10
    cnn_config.num_classes = 3
    cnn_config.dev_split = 0.3
    cnn_config.num_epochs = 4
    cnn_config.learning_rate = 1e-4

    dt_file = ['twitter_henrico/trainTT.neg', 'twitter_henrico/trainTT.pos', 'twitter_henrico/trainTT.neu']

    file_config = FilesConfig(vocab_file='twitter_hashtag/twitterhashtags.vocab', dataset_file='twitter_henrico/trainTT.neg')
    corpus_th = CorpusTH(dt_file, file_config.vocab_file, cnn_config.dev_split,
                                  cnn_config.seq_length, cnn_config.vocab_size)
    corpus_th.prepare()
    model_th = TextCNN(cnn_config)
    trainer_th = Trainer(corpus=corpus_th, model=model_th, config=cnn_config, file_config=file_config, verbose=True)
    train_acc, train_loss, val_acc, val_loss, best_epoch = trainer_th.train()
    print(val_acc)


def test_kfold():
    c = CorpusTE(train_file='DataSetsEraldo/dataSetSupernatural.txt', vocab_file='twitter_hashtag/twitterhashtags.vocab')
    x, y = c.prepare()
    print(c.size)
    f = KFold(c, 3, rand=1)
    f.prepare_fold(x, y)

    cnn_config = TCNNConfig()
    cnn_config.num_epochs = 10

    file_config = FilesConfig(vocab_file='twitterhashtags.vocab', dataset_file='DataSetsEraldo/dataSetBahia.txt')
    for i in range(5):
        for cf in f:
            model0 = TextCNN(cnn_config)
            print(c.train_distribution())
            c.prepare_sample(x, y, size=300)
            c.sub_sampling(size=300)
            print(c.x_train.shape)
            t = Trainer(corpus=cf, model=model0, config=cnn_config, file_config=file_config, verbose=True)
            train_acc, train_loss, val_acc, val_loss, best_epoch = t.train()


def test_loads(): #same problem
    results0 = []
    results1 = []

    cnn_config = TCNNConfig()
    cnn_config.num_epochs = 4
    file_config = FilesConfig(vocab_file='twitter_hashtag/twitterhashtags.vocab', dataset_file='twitter_hashtag/out.txt')
    corpus = TwitterHashtagCorpus(file_config.train_file, file_config.vocab_file, cnn_config.dev_split,
                                              cnn_config.seq_length, cnn_config.vocab_size)
    model0 = TextCNN(cnn_config)
    trainer0 = Trainer(corpus=corpus, model=model0, config=cnn_config, file_config=file_config, verbose=True)
    trainer0.train(results0)

    torch.save(model0.embedding.state_dict(), "dict0.out") #pegar do melhor.
    torch.save(model0.convs.state_dict(), "dict1.out")

    test2 = model0.state_dict()

    model2 = TextCNN(cnn_config)
    model2.use_pre_trained_layers("dict0.out", "dict1.out")


    trainer1 = Trainer(corpus=corpus, model=model2, config=cnn_config, file_config=file_config, verbose=True)
    trainer1.train(results1)
    #print(acc_train, loss_train, acc_test, loss_test, test_acc, bepoch)
    test3 = model2.state_dict()

    print("teste finalizado")

    #corpus.shuffle()
    #trainer2 = Trainer(corpus=corpus, config=cnn_config, file_config=file_config, verbose=False)
    #acc_train, loss_train, acc_test, loss_test, test_acc, bepoch = trainer2.train()"""

def test_load(): #dif models
    results0 = []
    results1 = []

    cnn_config = TCNNConfig()
    cnn_config.num_epochs = 4
    file_config = FilesConfig(vocab_file='twitter_hashtag/twitterhashtags.vocab',
                              dataset_file='twitter_hashtag/out.txt', task='125hashtags')
    corpus = TwitterHashtagCorpus(file_config.train_file, file_config.vocab_file, cnn_config.dev_split,
                                  cnn_config.seq_length, cnn_config.vocab_size)
    model0 = TextCNN(cnn_config)
    trainer0 = Trainer(corpus=corpus, model=model0, config=cnn_config, file_config=file_config, verbose=True)
    trainer0.train(results0)

    model0.save(["dict0.out", "dict1.out"])
    t0 = model0.state_dict()

    file_configs = FilesConfig(vocab_file='twitter_hashtag/twitterhashtags.vocab',
                              dataset_file='twitter_hashtag/out.txt', task='supernatural')
    c = CorpusTE(train_file='DataSetsEraldo/dataSetSupernatural.txt',
                 vocab_file='twitter_hashtag/twitterhashtags.vocab')
    x, y = c.prepare()
    print(c.size)
    #f = KFold(c, 3, rand=1)
    #f.prepare_fold(x, y)
    #f.new_fold()
    c.x_validation=x[int(0.7*len(x)):]
    c.y_validation=y[int(0.7*len(x)):]
    c.x_train=x[:int(0.7*len(x))]
    c.y_train=y[:int(0.7*len(x))]
    cnn_config_s = TCNNConfig()
    cnn_config_s.num_epochs = 4
    cnn_config_s.num_classes = 2
    model_supernatural = TextCNN(config=cnn_config_s)
    model_supernatural.use_pre_trained_layers("dict0.out", "dict1.out")
    trainer_s = Trainer(corpus=c, model=model_supernatural, config=cnn_config_s,file_config=file_configs, verbose=True)
    trainer_s.train(results1)

    ts = model_supernatural.state_dict()

    print("acabou")
#TODO REMOVE
def load_embedding(glove_file, size=100):
    #print("loading pre-treined emb..")
    data = []
    g_file = open(glove_file, 'r')
    reader = csv.reader(g_file, delimiter=' ')
    reader.__next__()
    for l in reader:
        data.append([l])
    #print(data)
    data = np.asarray(data, dtype=float)
    emb = torch.FloatTensor(data.reshape((-1, size)))
    print(emb)
    return emb

def test_tuner_1KHashtags():
    emb = load_embedding('twitter_hashtag/1kthashtag.glove')

    file_config = FilesConfig(vocab_file='twitter_hashtag/1kthashtag.vocab',
                              dataset_file='twitter_hashtag/multiple.txt',
                              task='1khashtags')
    corpus = TwitterHashtagCorpus(train_file=file_config.train_file,
                                  vocab_file=file_config.vocab_file)
    cnn_config = TCNNConfig()
    corpus = TwitterHashtagCorpus(train_file=file_config.train_file, vocab_file=file_config.vocab_file) # arrumar parametros


    myTuner = Tuner(corpus, file_config, rand=False)
    epochs = (50, 6)
    lrs = (1e-3, 1e-1)
    myTuner.random_search(5, epochs, lrs, rep=10, freeze_lr=True, freeze_epochs=True)
    print("RS finished!\n")


def test_kfold_rs():

    cnn_config = TCNNConfig()
    cnn_config.num_epochs = 4
    file_config = FilesConfig(vocab_file='twitter_hashtag/twitterhashtags.vocab',
                              dataset_file='twitter_hashtag/out.txt')
    c = CorpusTE(train_file='DataSetsEraldo/dataSetSupernatural.txt',
                 vocab_file='twitter_hashtag/twitterhashtags.vocab')
    x, y = c.prepare()
    print(c.size)
    f = KFold(c, 3, rand=1)
    f.prepare_fold(x, y)

    myTuner = Tuner(c, file_config)
    epochs = (10, 30)
    lrs = (0.0001, 0.01)
    myTuner.random_search_cv(execs=5, epoch_limits=epochs, lr_limits=lrs, cv=4, folds=f, freeze_lr=True)
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
    epochs = (4, 6)
    lrs = (1e-4, 1e-2)
    myTuner.random_search_cv(execs=2, epoch_limits=epochs, lr_limits=lrs, cv=4, folds=f, freeze_lr=True)
    print("RS finished!\n")


def supernatural_acc_loss():
    file_config = FilesConfig(vocab_file='twitter_hashtag/twitterhashtags.vocab',
                              dataset_file='DataSetsEraldo/dataSetSupernatural.txt', task='supernatural')
    c = CorpusTE(train_file='DataSetsEraldo/dataSetSupernatural.txt',
                 vocab_file='twitter_hashtag/twitterhashtags.vocab')
    x, y = c.prepare()
    print(c.size)
    f = KFold(c, 3, rand=1)
    f.prepare_fold(x, y)

    myTuner = Tuner(c, file_config)
    epochs = (4, 6)
    lrs = (1e-4, 1e-2)
    myTuner.random_search_cv(execs=1, epoch_limits=epochs, lr_limits=lrs, cv=4, folds=f, freeze_lr=True, freeze_epochs=True)
    print("AL finished!\n")


def corpus():
    file_config = FilesConfig(vocab_file='twitter_hashtag/twitterhashtags.vocab',
                              dataset_file='twitter_hashtag/out.txt', task='supernatural')
    c = CorpusTE(train_file='DataSetsEraldo/dataSetSupernatural.txt',
                 vocab_file='twitter_hashtag/twitterhashtags.vocab')
    x, y = c.prepare()
    dc = Corpus.distribution(y)
    print(c.size)
    print(dc)

def train_cv():
    ultimos_r =[]
    dt =[]
    file_config = FilesConfig(vocab_file='twitter_hashtag/twitterhashtags.vocab',
                              dataset_file='DataSetsEraldo/dataSetSupernatural.txt')
    c = CorpusTE(train_file='DataSetsEraldo/dataSetSupernatural.txt',
                 vocab_file='twitter_hashtag/twitterhashtags.vocab')
    x, y = c.prepare()
    print(c.size)
    f = KFold(c, 3, rand=1)
    f.prepare_fold(x, y)

    for cv in f:
        t = Trainer(corpus=cv, file_config=file_config, verbose=True)
        ultimos_r.append(t.train(dt))

    print(ultimos_r)
    print(':)')


def cv_rsplit():
    file_config = FilesConfig(vocab_file='twitter_hashtag/twitterhashtags.vocab',
                              dataset_file='DataSetsEraldo/dataSetSupernatural.txt', task='supernatural')
    c = CorpusTE(train_file='DataSetsEraldo/dataSetSupernatural.txt',
                 vocab_file='twitter_hashtag/twitterhashtags.vocab')
    x, y = c.prepare()
    print(c.size)
    f = RandomSplit(corpus=c, n=5, sub=350)
    f.x = x
    f.y = y

    t = Tuner(c, file_config)
    epochs = (5, 6)
    lrs = (1e-4, 1e-2)
    t.random_search_rsplit(2, f, epochs, lrs, freeze_epochs=True, freeze_lr=True)
    print(':)')


def test_new_multihashtags():
    file_config = FilesConfig(vocab_file='twitterhashtags.vocab', dataset_file='multiple.txt', base_dir='twitter_hashtag',
                              task='1labelthashtag')
    c = TwitterHashtagCorpus(train_file=file_config.train_file, vocab_file=file_config.vocab_file)

    #x = np.append(c.x_train, c.x_validation, axis=0)

    y = np.append(c.y_train, c.y_validation)
    ones = np.ones_like(y)

    dt = pandas.DataFrame(y, columns=['class'])
    dt.insert(1, 'c', ones)
    x = dt.groupby('class').sum()
    print(x)



def my_model(args):

    return ETextCNN(config=args[0], pre_trained_emb=args[1])
def test_pretrained_emb():
    embedding = load_embedding('twitter_hashtag/1kthashtag.glove')
    file_config = FilesConfig(vocab_file='../helpers/1kthashtag.vocab',
                              dataset_file='DataSetsEraldo/dataSetSupernatural.txt', task='supernatural')
    c = CorpusTE(train_file='DataSetsEraldo/dataSetSupernatural.txt',
                 vocab_file='../helpers/1kthashtag.vocab')
    x, y = c.prepare()
    print(c.size)
    f = RandomSplit(corpus=c, n=10, sub=350)
    f.x = x
    f.y = y
    cnn_config = TCNNConfig()
    args = [cnn_config, embedding]
    t = Tuner(c, file_config, rand=False, callback=my_model, args=args)
    epochs = (10, 6)
    lrs = (1e-3, 1e-2)
    t.random_search_rsplit(execs=2, rsplits=f, epoch_limits=epochs, lr_limits=lrs,
                           freeze_epochs=True, freeze_lr=False)
    print(embedding[1])
    print('Done supernatural.')



