import torch
import torch.nn as nn
import torch.nn.functional as F
from NegativeSampling import *


class TextCNN_NegSamp(nn.Module):
    """
    CNN text classification model, based on the paper.
    """

    def __init__(self, config):
        super(TextCNN_NegSamp, self).__init__()

        V = config.vocab_size
        E = config.embedding_dim
        Nf = config.num_filters
        Ks = config.kernel_sizes
        C = config.num_classes
        Dr = config.dropout_prob
        self.config = config

        self.embedding = nn.Embedding(V, E, padding_idx=0)  # embedding layer
        self.class_embedding = nn.Embedding(C, 3 * Nf)   # Lais embedding

        # three different convolutional layers
        self.convs = nn.ModuleList([nn.Conv1d(E, Nf, k) for k in Ks])
        self.dropout = nn.Dropout(Dr)  # a dropout layer

        #self.fc1 = nn.Linear(3 * Nf, C)  # a dense layer for classification

    @staticmethod
    def conv_and_max_pool(x, conv):
        """Convolution and global max pooling layer"""
        return F.relu(conv(x).permute(0, 2, 1).max(1)[0])

    def forward(self, inputs, negative_class_vector):
        # Conv1d takes in (batch, channels, seq_len), but raw embedded is (batch, seq_len, channels)
        embedded = self.embedding(inputs).permute(0, 2, 1)
        x = [self.conv_and_max_pool(embedded, k) for k in self.convs]  # convolution and global max pooling

        x_cat = self.dropout(torch.cat(x, 1))
        #x = self.fc1(x_cat)

        class_embedding = self.class_embedding(negative_class_vector) # Pega os embbedings do ruido e da palavra certa

        x_cat = x_cat.view(class_embedding.shape[0],-1,1)  # Para ficar 3d

        logit = torch.matmul(class_embedding, x_cat).view(class_embedding.shape[0], -1) # theta * e_c

        #class_probs = torch.sigmoid(logit)

        return logit

    def load_pre_trained(self, emb_file, conv_file): # sem a ultima camada
        #Embeddings
        emb = torch.load(emb_file)
        self.embedding.load_state_dict(emb)
        #Conv. Filters
        conv = torch.load(conv_file)
        self.convs.load_state_dict(conv)

    def use_pre_trained_layers(self, emb_file, conv_file):

        self.load_pre_trained(emb_file, conv_file)

        self.embedding.weight.requires_grad = False
        for param in self.convs.parameters():
            param.requires_grad = False

    def save(self, paths): #[emb,conv...]
        torch.save(self.embedding.state_dict(), paths[0])
        torch.save(self.convs.state_dict(), paths[1])
#____________________________________________________________________________________________________________________#


class TCNNConfig_NegSamp(object): #TODO - pegar automaticamente alguns param?
    """
    CNN Parameters
    """

    def __init__(self, num_epochs=10, learning_rate=0.001, n_classes= 125):
        self.learning_rate = learning_rate  # learning rate
        self.num_epochs = num_epochs  # total number of epochs
        self.num_classes = n_classes # number of classes 125


    embedding_dim = 50  # embedding vector size
    seq_length = 50  # maximum length of sequence
    vocab_size = 8000  # most common words

    num_filters = 100  # number of the convolution filters (feature maps)
    kernel_sizes = [3, 4, 5]  # three kind of kernels (windows)

    hidden_dim = 50  # hidden size of fully connected layer

    dropout_prob = 0.5  # how much probability to be dropped

    batch_size = 8  # batch size for training

    target_names = ['--', '-', '=', '+', '++']

    dev_split = 0.1  # percentage of dev data

    cuda_device = 0  # cuda device to be used when available

    n_negatives_class = 15 # Classes negativas a serem atualizadas em cada iteração


    def __str__(self):
        pass #TODO
