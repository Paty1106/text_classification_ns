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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn import metrics

from CorpusHelper import *
from CorpusTwitterHashtag import TwitterHashtagCorpus

from tqdm import tqdm
import time
import datetime
from TextCNN_NegSamp import *
from FilesConfig import FilesConfig
from ResultsHandler import ResultsHandler

import matplotlib.pyplot as plt

from NegativeSampling import *


use_cuda = torch.cuda.is_available()


class Trainer_NegSamp(object):

    def __init__(self, file_config, config=None, model=None, corpus=None, verbose=True, opt_param=None):
        print('Loading data...')
        self.verbose = verbose
        if config is None:
            config = TCNNConfig_NegSamp()

        self.config = config
        self.NegSamp = NegativeSampling(self.config)


        self.model_file = "{}epc{}lr{}".format(datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
                                                 ,self.config.num_epochs, self.config.learning_rate)
        self.file_config = file_config
        if corpus is None:
            corpus = TwitterHashtagCorpus(file_config.train_file, file_config.vocab_file, self.config.dev_split,
                                          self.config.seq_length, self.config.vocab_size)
    #DOING#####
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
            model = TextCNN_NegSamp(self.config)
        self.model = model
        if opt_param is None:
            opt_param = self.model.parameters()

        if self.verbose:
            print(self.corpus)
            print(self.model)

        if use_cuda:
            self.model.cuda()

        #Optimizer and Loss Function

        #self.criterion = nn.CrossEntropyLoss(size_average=False) # Lais  # nn.MultiLabelSoftMarginLoss()
        #self.criterion = self.NegSamp.negative_sampling_loss
        self.criterion = F.binary_cross_entropy_with_logits
        self.optimizer = optim.Adam(opt_param, lr=self.config.learning_rate)

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
        y_pred_value = []
        for data, label in data_loader:
            data, label = torch.tensor(data), torch.tensor(label)
            if use_cuda:
                data, label = data.cuda(), label.cuda()

            with torch.no_grad():

                negative_all_class_vector = torch.tensor(data.shape[0]*[list(self.corpus.label_to_id.values())])
                logits = self.model(data, negative_all_class_vector) # Na validação, quais classes negativas eu uso?

                targets_zero_one = torch.zeros([logits.shape[0], self.config.num_classes], dtype=torch.float32)
                targets_zero_one[:, self.config.n_negatives_class] = torch.ones(logits.shape[0])

                losses = self.criterion(logits, targets_zero_one)

            total_loss += losses.item()
            output = torch.sigmoid(logits)
            pred = torch.max(output.data, dim=1)[1].cpu().numpy().tolist() #torch.max -> (value, index)
            pred_value = torch.max(output.data, dim=1)[0].cpu().numpy().tolist() #torch.max -> (value, index)

            y_pred.extend(pred)
            y_pred_value.extend(pred_value)
            y_true.extend(label.data.cpu().numpy().tolist())

        f = metrics.f1_score(y_true, y_pred, labels=task_labels, average=None)
        #acc = metrics.accuracy_score(y_true, y_pred)
        acc = (np.array(y_true) == np.array(y_pred)).sum()

        return acc / data_len, total_loss / data_len, f, y_pred

    """
          Train and evaluate the model with training and validation data.
    """
    def train(self, train_data, f1=None): # mudar o nome
        print("Training and evaluating...")
        start_time = time.time()
        # set the mode to train
        it_loss = []
        bt_acc = bt_loss = bv_loss = 0.0
        best_acc = 0.0
        best_epoch = 0

        losses = []
        total_train_acc = []
        total_val_acc = []

        #print("here", self.train_data.tensors[1].shape)

       # hashtag_class = np.asarray(self.train_data.tensors[1])
       # print(len(hashtag_class))
       # unique, counts = np.unique(hashtag_class, return_counts=True)
       # frequency = dict(zip(unique, counts))

        frequency = Counter(np.array(self.train_data.tensors[1]))
        #print("Quantidade de hashtags: ", len(hashtags_counts))
        #print(hashtags_counts.keys())

      #  print(frequency)

        for epoch in tqdm(range(self.config.num_epochs)):

            total_loss = 0
            # load the training data in batch
            self.model.train()
            #train_data_subsampling = self.NegSamp.subsampling(self.train_data, frequency)
            #self.train_data = TensorDataset(torch.LongTensor(train_data_subsampling[0]),
            #                                torch.LongTensor(train_data_subsampling[1]))

            train_loader = DataLoader(self.train_data, batch_size=self.config.batch_size)

            for x_batch, y_batch in train_loader:

                inputs, targets = Variable(x_batch), Variable(y_batch)
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                self.optimizer.zero_grad()
                negative_class_vector = self.NegSamp.negative_class(targets)

                logits = self.model(inputs, negative_class_vector)  # forward computation

                targets_zero_one = torch.zeros([logits.shape[0], self.config.n_negatives_class + 1], dtype=torch.float32)
                targets_zero_one[:, self.config.n_negatives_class] =  torch.ones(logits.shape[0])

            #    w = []
            #    weight = []
            #    for batch in range(negative_class_vector.shape[0]):
            #        w = []
            #        for i in range(negative_class_vector.shape[1]):
            #            el = negative_class_vector[batch,i]
            #            w.append(1/frequency[int(el)])
            #        weight.append(w)

            #    weight = torch.tensor(weight)
              #  loss = self.criterion(logits, targets_zero_one, weight=weight) # Lais
                loss = self.criterion(logits, targets_zero_one)

                # backward propagation and update parameters
                loss.backward()
                self.optimizer.step()

                total_loss = total_loss + loss.item()
                # evaluate on both training and test dataset


            train_acc, train_loss, f1_train, y_pred_train = self.evaluate(self.train_data)
            val_acc, val_loss, f1_val, y_pred_val = self.evaluate(self.validation_data)

            total_train_acc.append(train_acc)
            total_val_acc.append(val_acc)
            losses.append(total_loss)

            if f1 is not None:
                ###testing##
                f1.append([f1_val[0], f1_val[1], epoch, 1])
                f1.append([f1_train[0], f1_train[1], epoch, 0])
            if val_acc > best_acc:
                # store the best result
                best_acc = val_acc
                bt_acc = train_acc
                bt_loss = train_loss
                bv_loss = val_loss
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
                print(msg.format(epoch + 1, total_loss, train_acc, val_loss, val_acc, time_dif, improved_str))

            data = [epoch + 1, train_loss, train_acc, val_loss, val_acc, f1_train.sum()/self.corpus.max_labels,
                    f1_val.sum()/self.corpus.max_labels]
            ResultsHandler.write_train_row(self.config, data, self.file_config)

            ###testing##
            ResultsHandler.write_row(f1_train, '{}/{}.fscoretrain'.format(self.file_config.result_path, self.model_file))
            ResultsHandler.write_row(f1_val, '{}/{}.fscoreval'.format(self.file_config.result_path, self.model_file))

            train_data.append(data)

        #if self.verbose:
            #print("F1 final train: {} F1 final validation {}".format(f1_train, f1_val))


        #plt.figure(figsize=(12,10))

        #plt.subplot(211)
        plt.plot(total_train_acc, color = 'blue',  label = 'Train')
        plt.plot(total_val_acc, color = 'red', label = 'Validation')
        plt.legend(loc=4)
        plt.yticks(np.arange(0.15, 1.05, 0.05))
        plt.grid()
        plt.title("15-Negative Sampling Accuracy")

      #  plt.subplot(212)
       # plt.figure(figsize=(20,10))
      #  plt.plot(losses, color = 'blue')
       # plt.title("Negative Sampling Loss")

        #plt.tight_layout()

        plt.savefig('images/acc_15negsamp_0001_bal_new_loss_150ep_10_11_.png')

        return bt_acc, bt_loss, best_acc, bv_loss, best_epoch







