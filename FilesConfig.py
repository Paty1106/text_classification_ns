import os
from datetime import date
import csv


class FilesConfig(object):
    #definir todos os nomes de arquivos de entrada de dados e saída de resultados...
    def __init__(self, vocab_file, dataset_file, model='model', result_dir='results', base_dir='', task=''):
        #do something
        self.train_file = os.path.join(base_dir, dataset_file) #'out.txt' - arquivo de treino
        self.vocab_file = os.path.join(base_dir, vocab_file) #'twitterhashtags.vocab'

        self.main_dir = self.generate_name(task)
        if not os.path.exists(self.main_dir):
            os.mkdir(self.main_dir)

        save_path = os.path.join(self.main_dir, 'checkpoints') # model save path

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.model_file = os.path.join(save_path, model)#'hashtag_cnn_pytorch.pt'

        self.result_path = os.path.join(self.main_dir, result_dir)  # model save path
        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path)
        ##Results File##
        self.results_train = os.path.join(self.result_path, "long_resul_train.csv")
        self.results_train_file = None

        self.test_acc = os.path.join(self.result_path, "resul_acc_test.csv")
        self.test_acc_file = None

        self.resume_results = os.path.join(self.result_path, "resul_resume.csv")
        self.resume_results_file = None

    """" grid_file = os.path.join(path, 'dados-grid.txt')
    grid_resume = os.path.join(path, 'dados-grid-resume.txt')
    curva_aprend = os.path.join(path, 'dados-curva-aprend.txt')"""

        #TODO add métodos de abertura para os arquivos.- o obj vai guardar o arquivo aberto também.

    def open_long_results_train_file(self, append=True):
        if not append:
            self.results_train_file = open(self.results_train, mode='w', encoding='utf8')
        else:
            if self.results_train_file is None:
                self.results_train_file = open(self.results_train, mode='w+', encoding='utf8')

        return self.results_train_file

    def open_test_acc_file(self, append=True):
        if not append:
            self.test_acc_file = open(self.test_acc, mode='w', encoding='utf8')
        else:
            if self.test_acc_file is None:
                self.test_acc_file = open(self.test_acc, mode='w+', encoding='utf8')

        return self.test_acc_file

    def open_resume_file(self, append=True):
        if not append:
            self.resume_results_file = open(self.resume_results, mode='w', encoding='utf8')
        else:
            if self.resume_results_file is None:
                self.resume_results_file = open(self.resume_results, mode='a+', encoding='utf8')

        return self.resume_results_file

    #cria um nome a partir dos parametros passados.
    def generate_name(self, dt_name):
        name = '../experiments/{}.{}'.format(dt_name, date.today())
        return name
    
    #TODO close files..