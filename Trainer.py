#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This example demonstrates the use of Conv1D for CNN text classification.
Original paper could be found at: https://arxiv.org/abs/1408.5882

This is the baseline model: CNN-rand.

The implementation is based on PyTorch.

We didn't implement cross validation,
but simply run `python cnn_mxnet.py` for multiple times,
the average accuracy is close to 78%.

It takes about 2 minutes for training 20 epochs on a GTX 970 GPU.
"""


import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from sklearn import metrics


from CorpusHelper import *
from CorpusTwitterHashtag import TwitterHashtagCorpus

from tqdm import tqdm
import time
import datetime
from TextCNN import *
from FilesConfig import FilesConfig
from ResultsHandler import ResultsHandler


use_cuda = torch.cuda.is_available()


class Trainer(object):

    def __init__(self, file_config, config=None, model=None, corpus=None, verbose=True, opt_param=None,
                 metric='accuracy'):

        self.metrics = {'accuracy': 0, 'f1-score': 1}
        self.metric = self.metrics[metric]

        print('Loading data...')
        self.verbose = verbose
        if config is None:
            config = TCNNConfig()

        self.config = config
        self.model_file = "{}epc{}lr{}".format(datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
                                                 ,self.config.num_epochs, self.config.learning_rate)
        self.file_config = file_config
        if corpus is None:
            corpus = TwitterHashtagCorpus(file_config.train_file, file_config.vocab_file, self.config.dev_split,
                                          self.config.seq_length, self.config.vocab_size)
    #DOING#####- Mover p um helper
        self.file_config.model_file = '{}/{}'.format(self.file_config.save_path, self.model_file)
        self.file_config.results_train ='{}/{}.epochs'.format(self.file_config.result_path, self.model_file)
        self.file_config.results_train_file = None
    ##########

        self.corpus = corpus
        self.config.vocab_size = len(self.corpus.words)
        self.config.target_names = self.corpus.label_to_id.keys()
        self.config.num_classes = len(self.corpus.label_to_id)
        self.train_data = TensorDataset(torch.LongTensor(self.corpus.x_train), torch.LongTensor(self.corpus.y_train))
        self.validation_data = TensorDataset(torch.LongTensor(self.corpus.x_validation), torch.LongTensor(self.corpus.y_validation))
        if corpus.x_test is not None:
            self.test_data = TensorDataset(torch.LongTensor(self.corpus.x_test), torch.LongTensor(self.corpus.y_test))
        print('Configuring CNN model...')
        if model is None:
            model = TextCNN(self.config)
        self.model = model
        if opt_param is None:
            opt_param = self.model.parameters()

        if self.verbose:
            print(self.corpus)
            print(self.model)

        if use_cuda:
            self.model.cuda()

        #Optimizer and Loss Function
        self.criterion = nn.CrossEntropyLoss(size_average=False)  # nn.MultiLabelSoftMarginLoss()
        self.optimizer = optim.Adam(opt_param, lr=self.config.learning_rate, weight_decay=0.0)

    def get_time_dif(self, start_time):
        end_time = time.time()
        time_dif = end_time - start_time
        return datetime.timedelta(seconds=int(round(time_dif)))

    def evaluate(self, data):
        task_labels = np.arange(self.corpus.max_labels)

        """
        Evaluation, return accuracy and loss
        """
        self.model.eval()  # set mode to evaluation to disable dropout
        data_loader = DataLoader(data, batch_size=40)

        data_len = len(data)
        total_loss = 0.0

        y_true, y_pred = [], []
        
        for data, label in data_loader:
            data, label = torch.tensor(data), torch.tensor(label)
            if use_cuda:
                data, label = data.cuda(), label.cuda()

            with torch.no_grad():
                output = self.model(data)
                losses = self.criterion(output, label)

            #total_loss += losses.data[0]
            total_loss += losses.item()
            pred = torch.max(output.data, dim=1)[1].cpu().numpy().tolist() #torch.max -> (value, index)
            y_pred.extend(pred)
            y_true.extend(label.data.cpu().numpy().tolist())

        if self.metric != 0:
            f = metrics.f1_score(y_true, y_pred, labels=task_labels, average=None)
        else:
            f = None
        acc = (np.array(y_true) == np.array(y_pred)).sum()

        return acc / data_len, f, total_loss / data_len

    """
          Train and evaluate the model with training and validation data.
    """
    def train(self, train_data, f1=None): # mudar o nome
        print("Training and evaluating...")
        start_time = time.time()
        # set the mode to train
        it_loss = []
        best_loss = [0.0, 0.0]
        best_metrics_train = [0.0, 0.0]
        best_metrics_val = [0.0, 0.0]

        best_epoch = 0
        for epoch in tqdm(range(self.config.num_epochs)):
            # load the training data in batch
            self.model.train()
            train_loader = DataLoader(self.train_data, batch_size=self.config.batch_size)

            for x_batch, y_batch in train_loader:
                inputs, targets = Variable(x_batch), Variable(y_batch)
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                self.optimizer.zero_grad()
                outputs = self.model(inputs)  # forward computation
                loss = self.criterion(outputs, targets)

                # backward propagation and update parameters
                loss.backward()
                self.optimizer.step()
                # evaluate on both training and test dataset

            #train_acc, train_loss, f1_train = self.evaluate(self.train_data)
            #train_acc,  f1_train, train_loss = self.evaluate(self.train_data)
            train_metrics = self.evaluate(self.train_data)
            val_metrics = self.evaluate(self.validation_data)

            """if f1 is not None and best_metrics_val[1] is not None:
                ###testing_ TODO- arrumar##
                f1.append([val_metrics[1][0], val_metrics[1][1], epoch, 1])
                f1.append([train_metrics[1][0], train_metrics[1][1], epoch, 0])"""

            if val_metrics[self.metric] > best_metrics_val[self.metric]:
                # store the best result
                """if best_metrics_val[1] is not None:
                    best_metrics_train[self.metric] = train_metrics[1][1]
                    best_metrics_val[self.metric] = val_metrics[1][1]"""

                best_metrics_train[self.metrics['accuracy']] = train_metrics[0]
                best_metrics_val[self.metrics['accuracy']] = train_metrics[0]
                best_loss[0] = train_metrics[2]
                best_loss[1] = val_metrics[2]
                best_epoch = epoch
                improved_str = '*'
                torch.save(self.model.state_dict(), self.file_config.model_file+'.all')
                self.model.save([self.file_config.model_file+'.emb', self.file_config.model_file + '.convs'])

            else:
                improved_str = ''

            time_dif = self.get_time_dif(start_time)
            msg = "Epoch {0:3}, Train_loss: {1:>7.2}, Train_acc {2:>6.2%}, " \
                  + "Test_loss: {3:>6.2}, Test_acc {4:>6.2%}, Time: {5} {6}"
            if self.verbose:
                print(msg.format(epoch + 1, train_metrics[2], train_metrics[0], val_metrics[2], val_metrics[0],
                                 time_dif, improved_str))
            #train.sum()/self.corpus.max_labels, f1_val.sum()/self.corpus.max_labels
            if val_metrics[1] is None:
                v = [0.0, 0.0]
            else:
                v = [0.0, 0.0]#[f1[-1][1], f1[-2][1]]
                ###testing##
                ResultsHandler.write_row(train_metrics[1],
                                         '{}/{}.fscoretrain'.format(self.file_config.result_path, self.model_file))
                ResultsHandler.write_row(val_metrics[1],
                                         '{}/{}.fscoreval'.format(self.file_config.result_path, self.model_file))

            data = [epoch + 1, train_metrics[2], train_metrics[0], val_metrics[2], val_metrics[0], v[0], v[1]]
            ResultsHandler.write_train_row(self.config, data, self.file_config)



            train_data.append(data)

        if self.verbose:
            print("F1 final train: {} F1 final validation {}".format(train_metrics[1], val_metrics[1]))
            print("Best Metric {}".format(best_metrics_val[self.metric]))


        return best_metrics_train[0], best_loss[0], best_metrics_val[0], best_loss[1], best_epoch

    def test(self, test_data):
        """
        Test the model on test dataset.
        """
        if self.verbose:
            print(self.config.num_epochs)
            print(self.config.learning_rate)

        print("Testing...")
        start_time = time.time()
        test_loader = DataLoader(test_data, batch_size=self.config.batch_size)

        # load config and vocabulary


        # restore the best parameters
        self.model.load_state_dict(torch.load(self.model_file, map_location=lambda storage, loc: storage))

        y_true, y_pred = [], []
        for data, label in test_loader:
            data, label = torch.tensor(data), torch.tensor(label)
            if use_cuda:
                data, label = data.cuda(), label.cuda()

            with torch.no_grad():
                output = self.model(data)
                pred = torch.max(output, dim=1)[1].cpu().numpy().tolist()
            y_pred.extend(pred)
            y_true.extend(label.cpu().numpy().tolist())

        test_acc = metrics.accuracy_score(y_true, y_pred)
        test_f1 = metrics.f1_score(y_true, y_pred, average='macro')
        if self.verbose:
            print("Test accuracy: {0:>7.2%}, F1-Score: {1:>7.2%}".format(test_acc, test_f1))
        #g_file.write("Test accuracy: {0:>7.2%}, F1-Score: {1:>7.2%}".format(test_acc, test_f1))

            print("Precision, Recall and F1-Score...")

        labels = np.array(range(len(self.config.target_names)))
        cm = metrics.confusion_matrix(y_true, y_pred)

        if self.verbose:
            print(metrics.classification_report(y_true, y_pred, labels=labels, target_names=self.config.target_names))
            print('Confusion Matrix...')
            print(cm)
            print("Time usage:", self.get_time_dif(start_time))

        return test_acc






