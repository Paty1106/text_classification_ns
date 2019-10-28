##A ideia é que o contrutor receba um corpus, sob o qual os treinamentos serão realizados, um objeto com as configurações
##dos arquivos de entrada e saída, além do handler desejado para os resultados, responsável pela formatação e escrita de
#resultados nos arquivos indicados pela configuração dada.
import random

from RandomSplit import *
from CorpusTE import *
from KFold import *
import pandas
from TensorBoardHelper import TensorBoardHelper
class Tuner(object): #review class' name

    def __init__(self, corpus, files_config, results_handler=None, callback=None, args=None, rand=True):
        self.corpus = corpus
        self.files_config = files_config
        self.files_handler = results_handler

        self.model_callback = callback
        self.model_args = args
        self.rand = rand
        #TODO

    # epoch_limits: se freeze_epoch = False, então espera-se dois valores, cc, apenas o epoch_limits[0] precisa estar def.
    # lr_limits: =.
    def random_search(self, execs, epoch_limits, lr_limits, freeze_lr=False, freeze_epochs=False, rep=1):

        cnn_config = TCNNConfig(num_epochs=epoch_limits[0], learning_rate=lr_limits[0])
        lr_list = Tuner.lr_list(lr_limits)

        ######temp###########
        kernels = [1, 2, 3, 4, 5]
        #kernels = [6, 7, 8, 9, 10]

        for e in range(execs):
            cnn_config.kernel_sizes = [kernels[e]]
            if self.rand:
                if not freeze_lr:
                    cnn_config.learning_rate = lr_list[random.randint(0, len(lr_list) - 1)]
                if not freeze_epochs:
                    cnn_config.num_epochs = random.randint(epoch_limits[0], epoch_limits[1])
            else:
                if not freeze_lr:
                    cnn_config.learning_rate = lr_list[e]
                if not freeze_epochs:
                    cnn_config.num_epochs = random.randint(epoch_limits[0], epoch_limits[1])

            print("LR{0} EP{1}\n".format(cnn_config.learning_rate, cnn_config.num_epochs))
            res = []
            train_data, f1, dp = [], [], []
            for r in range(rep):
                trainer = Trainer(corpus=self.corpus, config=cnn_config, file_config=self.files_config, verbose=True)
                result = trainer.train(train_data, f1)
                res.append(result[:4])

            dt_frame = pandas.DataFrame(np.array(train_data)[:, :5],
                                        columns=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
            cv_res = dt_frame.groupby('epoch').mean()
            ncv = cv_res.to_numpy()
            dp.append(np.std(res, axis=0))
            # Other files
            ResultsHandler.write_result_resume_row(result, self.files_config, cnn_config)
            # Tensorboard
            ResultsHandler.al_tensorboard(ncv, [cnn_config.num_epochs, cnn_config.learning_rate],
                                          self.files_config.main_dir)
            ResultsHandler.s_tensorboard(ncv, [cnn_config.num_epochs, cnn_config.learning_rate],
                                         self.files_config.main_dir)
            ResultsHandler.simple_write(dp, '{}/dp.csv'.format(self.files_config.result_path))
           # ResultsHandler.simple_write(f1, '{}/f1val.csv'.format(self.files_config.result_path))

            TensorBoardHelper.hparams(ncv, folder=self.files_config.main_dir + '/hparam_tuning',
                                            hparams={'lr': cnn_config.learning_rate,
                                                     'ep': cnn_config.num_epochs, 'num_filtros':cnn_config.num_filters, 'kernel1':cnn_config.kernel_sizes[0]}
                                      , ex=cnn_config.learning_rate)
        #ResultsHandler.write_test_acc(test_accs, self.files_config)

    def random_search_cv(self, execs,  folds, epoch_limits, lr_limits, cv=1, freeze_lr=False, freeze_epochs=False):
        cnn_config = TCNNConfig(num_epochs=epoch_limits[0], learning_rate=lr_limits[0])

        cv_result = []
        dp = []
        lr_list = Tuner.lr_list(lr_limits)
        for e in range(execs):
            if not freeze_lr:
                cnn_config.learning_rate = lr_list[random.randint(0, len(lr_list)-1)]
            if not freeze_epochs:
                cnn_config.num_epochs = random.randint(epoch_limits[0], epoch_limits[1])

            print("LR{0} EP{1}\n".format(cnn_config.learning_rate, cnn_config.num_epochs))

            train_data = []

            for i in range(cv): #nao rand cv==1
                #self.corpus.prepare_sample(folds.x_data, folds.y_data, size=400)  # revisar se é aqui
                for cf in folds:
                    #model = TextCNN(cnn_config)
                    print('DISTRIBUTION')
                    print(self.corpus.train_distribution())
                    #Subsampling
                    self.corpus.sub_sampling(size=350)

                    print(self.corpus.x_train.shape)

                    if self.model_callback is not None:
                        m = self.model_callback(self.model_args)
                    else:
                        m = None
                    t = Trainer(corpus=cf, model=m, config=cnn_config, file_config=self.files_config, verbose=True)
                    r = t.train(train_data) #train_acc, train_loss, val_acc, val_loss, best_epoch
             #Average results
                    cv_result.append(r[:-1])
            cv_r = np.array(cv_result)
            av_cv_r = np.average(cv_r, axis=0)
            dp.append(np.std(cv_r, axis=0)) #TODO REVER
            dt_frame = pandas.DataFrame(np.array(train_data)[:,:5], columns=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
            cv_res = dt_frame.groupby('epoch').mean()
            ncv = cv_res.to_numpy()

            #Tensorboard
            ResultsHandler.al_tensorboard(ncv, [cnn_config.num_epochs, cnn_config.learning_rate],
                                          self.files_config.main_dir)
            ResultsHandler.s_tensorboard(ncv, [cnn_config.num_epochs, cnn_config.learning_rate],
                                          self.files_config.main_dir)
        #Other files
            ResultsHandler.write_result_resume_row(av_cv_r, self.files_config, cnn_config)
        ResultsHandler.simple_write(dp, '{}/dp.csv'.format(self.files_config.result_path))

    def random_search_rsplit(self, execs, rsplits, epoch_limits, lr_limits, r=1, freeze_lr=False, freeze_epochs=False):
        cnn_config = TCNNConfig(num_epochs=epoch_limits[0], learning_rate=lr_limits[0])

        dp = []
        lr_list = Tuner.lr_list(lr_limits)

        for e in range(execs):
            cv_result = []
            if self.rand:
                if not freeze_lr:
                    cnn_config.learning_rate = lr_list[random.randint(0, len(lr_list) - 1)]
                if not freeze_epochs:
                    cnn_config.num_epochs = random.randint(epoch_limits[0], epoch_limits[1])
            else:
                if not freeze_lr:
                    cnn_config.learning_rate = lr_list[e]
                if not freeze_epochs:
                    cnn_config.num_epochs = random.randint(epoch_limits[0], epoch_limits[1])

            print("LR{0} EP{1}\n".format(cnn_config.learning_rate, cnn_config.num_epochs))

            train_data, f1 = [], []

            for i in range(r):  #est.
                for c in rsplits:

                    #print('DISTRIBUTION')
                    #print(self.corpus.train_distribution())

                    if self.model_callback is not None:
                        m = self.model_callback(self.model_args)
                    else:
                        m = None
                    t = Trainer(corpus=c, model=m, config=cnn_config, file_config=self.files_config,
                                verbose=True, metric='f1-score')
                    result = t.train(train_data, f1)  # train_acc, train_loss, val_acc, val_loss, best_epoch

                    cv_result.append(result[:-1])
            #Average - Results
            cv_r = np.array(cv_result)
            dp.append(np.std(cv_r, axis=0))  # TODO REVER
            av_cv_r = np.average(cv_r, axis=0)
            dt_frame = pandas.DataFrame(np.array(train_data)[:, :5],
                                            columns=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
            cv_res_byepoch = dt_frame.groupby('epoch').mean()
            ncv = cv_res_byepoch.to_numpy()




            dt_frame_fscore = pandas.DataFrame(np.array(f1),
                                            columns=['0', '1', 'epoch', 'train/val']) #Arrumar
            mask_t = dt_frame_fscore["train/val"] == 0.0
            mask_v = dt_frame_fscore["train/val"] == 1.0

            train_dtframe = dt_frame_fscore[mask_t]
            train_byepoch = train_dtframe.groupby('epoch').mean()
            fscore_train = train_byepoch.to_numpy()[:, :-1]

            val_dtframe = dt_frame_fscore[mask_v]
            val_byepoch = val_dtframe.groupby('epoch').mean()
            fscore_val = val_byepoch.to_numpy()[:, :-1]
            d =fscore_val[:, 1].reshape(len(fscore_val), 1)
            d1 = ncv[:, 3].reshape(len(fscore_val), 1)
            ncv_f = np.append(d1, d, axis=1)




                # Tensorboard
            ResultsHandler.al_tensorboard(ncv, [cnn_config.num_epochs, cnn_config.learning_rate],
                                              self.files_config.main_dir)
            ResultsHandler.s_tensorboard(ncv, [cnn_config.num_epochs, cnn_config.learning_rate],
                                             self.files_config.main_dir)
            ResultsHandler.fscore_tensorboard(fscore_val, [cnn_config.num_epochs, cnn_config.learning_rate],
                                             self.files_config.main_dir, t='val', type='validation')
            ResultsHandler.fscore_tensorboard(fscore_train, [cnn_config.num_epochs, cnn_config.learning_rate],
                                                  self.files_config.main_dir, t='train', type='train')
            # Other files

            ResultsHandler.write_result_resume_row(av_cv_r, self.files_config, cnn_config)

            TensorBoardHelper.hparams_write(ncv_f, folder=self.files_config.main_dir+'/hparam_tuning', hparams={'lr':cnn_config.learning_rate,
                                                                 'ep':cnn_config.num_epochs}, ex=cnn_config.learning_rate)

        ResultsHandler.simple_write(dp, '{}/dp.csv'.format(self.files_config.result_path))




    @staticmethod
    def lr_list(limits):
        list = []
        value = limits[0]
        while value <= limits[1]:
            list.append(value)
            value *= 10.0
        return list