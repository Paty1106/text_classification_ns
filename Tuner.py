##A ideia é que o contrutor receba um corpus, sob o qual os treinamentos serão realizados, um objeto com as configurações
##dos arquivos de entrada e saída, além do handler desejado para os resultados, responsável pela formatação e escrita de
#resultados nos arquivos indicados pela configuração dada.
import random


from CorpusTE import *
from KFold import *
import pandas


class Tuner(object): #review class' name

    def __init__(self, corpus, files_config, results_handler=None):
        self.corpus = corpus
        self.files_config = files_config
        self.files_handler = results_handler
        #TODO

    # epoch_limits: se freeze_epoch = False, então espera-se dois valores, cc, apenas o epoch_limits[0] precisa estar def.
    # lr_limits: =.
    def random_search(self, execs, epoch_limits, lr_limits, freeze_lr=False, freeze_epochs=False):

        cnn_config = TCNNConfig(num_epochs=epoch_limits[0], learning_rate=lr_limits[0])

    #tentar já gerar o vetor... depois so aplicar.
        test_accs = []
        for e in range(execs):

            if not freeze_lr:
                cnn_config.learning_rate = random.uniform(lr_limits[0], lr_limits[1])
            if not freeze_epochs:
                cnn_config.num_epochs = random.randint(epoch_limits[0], epoch_limits[1])

            print("LR{0} EP{1}\n".format(cnn_config.learning_rate, cnn_config.num_epochs))

            trainer = Trainer(corpus=self.corpus, config=cnn_config, file_config=self.files_config, verbose=False)
            result = trainer.train()

            test_accs.append(result[4])
        # Other files
            ResultsHandler.write_result_resume_row(result, self.files_config, cnn_config)
        ResultsHandler.write_test_acc(test_accs, self.files_config)

    def random_search_cv(self, execs,  folds, epoch_limits, lr_limits, cv=1, freeze_lr=False, freeze_epochs=False):
        cnn_config = TCNNConfig(num_epochs=epoch_limits[0], learning_rate=lr_limits[0])

        cv_result = []
        dp = []

        for e in range(execs):
            if not freeze_lr:
                cnn_config.learning_rate = random.uniform(lr_limits[0], lr_limits[1]) #todo
            if not freeze_epochs:
                cnn_config.num_epochs = random.randint(epoch_limits[0], epoch_limits[1])

            print("LR{0} EP{1}\n".format(cnn_config.learning_rate, cnn_config.num_epochs))

            train_data = []

            for i in range(cv): #nao rand cv==1
                self.corpus.prepare_sample(folds.x_data, folds.y_data, size=400)  # revisar se é aqui
                for cf in folds:
                    #model = TextCNN(cnn_config)
                    print('DISTRIBUTION')
                    print(self.corpus.train_distribution())
                    #Subsampling
                    self.corpus.sub_sampling(size=400)

                    print(self.corpus.x_train.shape)

                    t = Trainer(corpus=cf, model=None, config=cnn_config, file_config=self.files_config, verbose=True)
                    r = t.train(train_data) #train_acc, train_loss, val_acc, val_loss, best_epoch
             #Average results
                    cv_result.append(r[:-1])
            cv_r = np.array(cv_result)
            av_cv_r = np.average(cv_r, axis=0)
            dp.append(np.std(cv_r, axis=0)) #TODO REVER
            dt_frame = pandas.DataFrame(np.array(train_data), columns=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
            cv_res = dt_frame.groupby('epoch').mean()
            ncv = cv_res.to_numpy()

            #Tensorboard
            ResultsHandler.al_tensorboard(ncv, [cnn_config.num_epochs, cnn_config.learning_rate],
                                          self.files_config.main_dir)
            ResultsHandler.s_tensorboard(ncv, [cnn_config.num_epochs, cnn_config.learning_rate],
                                          self.files_config.main_dir)
        #Other files
            ResultsHandler.write_result_resume_row(av_cv_r, self.files_config, cnn_config)
        #ResultsHandler.simple_write(dp, 'supernatural_rs/results/dp.csv')




