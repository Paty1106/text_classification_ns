import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import math as m
import torch
import random
from FilesConfig import FilesConfig
from TextCNN_NegSamp import *
from Trainer_NegSamp import *

class NegativeSampling():

    def __init__(self, config):

        self.config = config

    def subsampling(self, train, frequency):

        threshold = 1e-05
        train_x = []
        train_y = []

        for i in range(len(train)):

            hashtag_class = train.tensors[1][i]

            class_frequency = frequency[int(hashtag_class)]/self.config.num_classes

            prob_stay = np.sqrt(threshold/class_frequency)

            if(prob_stay <= random.random()):
                train_x.append(list(train.tensors[0][i,:]))
                train_y.append(int(train.tensors[1][i]))

        return (torch.tensor(train_x), torch.tensor(train_y))

    def negative_class_(self, targets):  # Está pegando indice que pode ter sido
                                                                    #excluido no subsampling
        negative_class_vector = []

        for i in range(len(targets)):

            #train_indices = []

            neg = np.arange(self.config.num_classes)

            aux = neg[self.config.num_classes - 1]
            neg[self.config.num_classes - 1] = int(targets[i])
            neg[targets[i]] = aux

            negative_class_vector.append(neg)

        return torch.tensor(negative_class_vector)

    def negative_class(self, targets):  # Está pegando indice que pode ter sido
                                                                    #excluido no subsampling
        negative_class_vector = []
        k = self.config.n_negatives_class

        for i in range(len(targets)):

            train_indices = []

            while(k > 0):
                neg_class = random.randint(0, self.config.num_classes - 1)
                if(targets[i] != neg_class):
                    train_indices.append(neg_class)
                    k-=1

            k = self.config.n_negatives_class
            train_indices.append(int(targets[i]))
            negative_class_vector.append(train_indices)

        return torch.tensor(negative_class_vector)

    def negative_sampling_loss(self, class_probs):

        n_negative_class = self.config.n_negatives_class

        e0 = torch.tensor((class_probs[:,n_negative_class].shape[0])*[1e-30])  # Para não zerar
        e1 = torch.tensor((n_negative_class)*[1e-30])  # Para não zerar
        one = torch.ones([n_negative_class], dtype=torch.float32)

        w_i = torch.log(class_probs[:,n_negative_class] + e0)

        w_ij = torch.log(one - class_probs[:,0:n_negative_class] + e1)
        w_ij = torch.sum(w_ij, dim = 1)

        s = -(w_i + w_ij)

        loss = torch.sum(s)/class_probs.shape[0]

        return loss
