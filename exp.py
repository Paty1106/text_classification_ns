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
    epochs = (100, 6)
    lrs = (1e-5, 1e-2)
    myTuner.random_search_cv(execs=1, epoch_limits=epochs, lr_limits=lrs, cv=1, folds=f, freeze_lr=True,
                             freeze_epochs=True)
    print("PRS finished!\n")


def supernatural_rs():

    file_config = FilesConfig(vocab_file='twitter_hashtag/twitterhashtags.vocab',
                              dataset_file='DataSetsEraldo/dataSetSupernatural.txt', task='supernatural')
    c = CorpusTE(train_file='DataSetsEraldo/dataSetSupernatural.txt',
                 vocab_file='twitter_hashtag/twitterhashtags.vocab')
    x, y = c.prepare()
    ndy = np.array(y)
    print('exemplos positivos, todo dataset')
    print(ndy.sum())
    print(c.size)
    f = KFold(c, 3, rand=1)
    f.prepare_fold(x, y)

    myTuner = Tuner(c, file_config)
    epochs = (100, 0)
    lrs = (1e-5, 1e-1)
    myTuner.random_search_cv(execs=6, epoch_limits=epochs, lr_limits=lrs, cv=10, folds=f, freeze_epochs=True, freeze_lr=False)
    print("RS finished!\n")


def pre_1khashtags_rs():
    file_config = FilesConfig(vocab_file='twitterhashtags.vocab', dataset_file='out.txt', base_dir='twitter_hashtag',
                              task='1labelthashtag')
    c = TwitterHashtagCorpus(train_file=file_config.train_file, vocab_file=file_config.vocab_file)
    cnn_config = TCNNConfig(num_epochs=4)

    x = np.append(c.x_train, c.x_validation, axis=0)
    y = np.append(c.y_train, c.y_validation)

    #remocao das classes nao presentes
    count = np.zeros(len(c.label_to_id), dtype=np.int)
    classes_ade = np.zeros_like(count, dtype=np.int)
    for i in y:
        count[i] += 1
    keys = list(c.label_to_id)

    i = 0

    for l in range(len(count)):
        if count[l] == 0:
            k = keys[l]
            del c.label_to_id[k]

    #remover classe 100- muitos exemplos
    del c.label_to_id[keys[100]]
    count[100] = 0

    indexes = [index for index in range(len(y)) if y[index] == 100]

    x_line = np.delete(x, indexes, 0)
    y_line = np.delete(y, indexes, 0)

    # Ajuste das classes
    i = 0
    for l in range(len(count)):
        if count[l] == 0:
            classes_ade[l] = -1
        else:
            classes_ade[l] = i
            i += 1
    c.max_labels = len(c.label_to_id)

    print(classes_ade)
    for s in range(len(y_line)):
        y_line[s] = classes_ade[y_line[s]]
    values = c.label_to_id.values()

    #Split
    c.x_train = x_line[:int(0.7*len(y_line)), :]
    c.x_validation = x_line[int(0.7 * len(y_line)):, :]
    c.y_train = y_line[:int(0.7 * len(y_line))]
    c.y_validation = y_line[int(0.7 * len(y_line)):]

    #Treino
    data = []
    d = []
    f1s_val = []
    train = Trainer(file_config=file_config, config=cnn_config, corpus=c, verbose=True)
    d = train.train(data, f1s_val)
    dt = np.array(data).reshape(-1, 7)
    # Tensorboard
    ResultsHandler.al_tensorboard(dt[:, 1:5], [cnn_config.num_epochs, cnn_config.learning_rate],
                                  file_config.main_dir)
    ResultsHandler.s_tensorboard(dt[:,  1:5], [cnn_config.num_epochs, cnn_config.learning_rate],
                                 file_config.main_dir)
    # Other files
    ResultsHandler.write_result_resume_row(d, file_config, cnn_config)
    ResultsHandler.simple_write(f1s_val, '{}/f1val.csv'.format(file_config.result_path))