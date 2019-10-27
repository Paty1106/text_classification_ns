import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from torch.utils.tensorboard import SummaryWriter
import numpy as np
class TensorBoardHelper(object):

    @staticmethod
    def hparams_write(dt, hparams, folder=None, ex=''):
        with tf.summary.create_file_writer('{}/l{}'.format(folder,ex)).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            for i in range(len(dt)):
                tf.summary.scalar('acc', dt[i][0], step=i+1)
                tf.summary.scalar('f1', dt[i][1], step=i+1)
            v = dt[:,0]
            m =np.max(v)
            tf.summary.scalar('acc',m , step=len(dt) + 1)
            tf.summary.scalar('f1', np.max(dt[:,1]), step=len(dt) + 1)

    @staticmethod
    def hparams(dt, hparams, folder=None, ex=''):
        with tf.summary.create_file_writer('{}/l{}'.format(folder, ex)).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            for i in range(len(dt)):
                tf.summary.scalar('acc', dt[i][3], step=i + 1)

            v = dt[:, 3]
            m = np.max(v)
            tf.summary.scalar('acc', m, step=len(dt) + 1)
