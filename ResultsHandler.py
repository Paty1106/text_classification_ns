#Handler class responsible for write or eventually read formatted results, especially.

#usar tags pra indicar formatos(ex.:long_form-> seaborn precisa pra ficar tudo bonitinho)
#https://seaborn.pydata.org/examples/errorband_lineplots.html
#https://pytorch.org/docs/stable/tensorboard.html

import csv
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import datetime
class ResultsHandler(object):#TODO a+


    # Escreve uma linha por vez dos resultados do treinamento.
    # CONFIG é o objeto que carrega as informações do modelo.
    # DATA é uma tupla com [].
    @staticmethod
    def write_train_row(config_model, data, config_files, long_form=True):
        if not long_form:
            #TODO call another function
            print("TODO")
        else:
            train_file = config_files.open_long_results_train_file()
            csvW = csv.writer(train_file, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvW.writerow([config_model.num_epochs, config_model.learning_rate, data[0], data[1], data[2], 'train'])
            csvW.writerow([config_model.num_epochs, config_model.learning_rate, data[0], data[3], data[4], 'test'])


    @staticmethod
    def write_test_acc(data, config_files):
        test_acc_file = config_files.open_test_acc_file()
        csvW = csv.writer(test_acc_file, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for acc in data:
            csvW.writerow([acc])


    @staticmethod
    def write_result_resume_row(data, config_files, config_model):
        resume_file = config_files.open_resume_file()
        csvW = csv.writer(resume_file, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
       # csvW.writerow([config_model.num_epochs, config_model.learning_rate, data[0], data[1], data[2], data[3], data[4]])
        csvW.writerow([config_model.num_epochs, config_model.learning_rate, data[0], data[1], data[2], data[3]])
    @staticmethod
    def simple_write(data, file):
        f = open(file, mode='w+')
        csvW = csv.writer(f, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for r in data:
            csvW.writerow(r)
        f.close()

    @staticmethod
    def s_tensorboard(dt, c_data, folder=None):#[e,lr]
        s = "epc{}lr{}".format(c_data[0], c_data[1])
        current_time = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
        log_dirf = folder + '/' + current_time + s
        w = SummaryWriter(log_dir=log_dirf, comment=s)
        i = 1
        for d in dt:

            w.add_scalar('Loss/train', d[0], i)
            w.add_scalar('Loss/validation', d[2], i)

            w.add_scalar('Accuracy/train', d[1], i)
            w.add_scalar('Accuracy/validation', d[3], i)
            i+=1
        w.close()

    @staticmethod
    def al_tensorboard(dt, c_data, folder=None, t=''):  # [e,lr]

        s = "lossacc.epc{}lr{}".format(c_data[0], c_data[1])
        current_time = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
        log_dirf = folder + '/' + t + s + 'validation/acc-loss'
        w = SummaryWriter(log_dir=log_dirf, comment=s)
        i = 1
        for d in dt:
            w.add_scalars('validation/loss-acc', {'loss': d[2], 'acc': d[3]}, i)
            i += 1
        w.close()

