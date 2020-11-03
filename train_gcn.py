###############################################################################
# TF 2.0 implementation of GCN.
# This version doesn't use the the Keras Model API
#
###############################################################################

from absl import app
from absl import flags
from models.gcn import GCN

from models.utils import preprocess_graph, load_data, load_data_planetoid, smooth_plot, del_all_flags
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import warnings




os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

try:
    import tensorflow.compat.v2 as tf
except ImportError as e:
    print(e)

print("Using TF {}".format(tf.__version__))
SEED = 15
np.random.seed(SEED)
tf.random.set_seed(SEED)

del_all_flags(flags.FLAGS)

# let hyperpaprameters to be accessible in multiple modules
FLAGS = flags.FLAGS
FLAGS(sys.argv)
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('hidden1', 6, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('early_stopping', 20, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_bool('verbose', True, 'Toogle the verbose.')
flags.DEFINE_bool('logging', False, 'Toggle the logging.')
flags.DEFINE_integer('gpu_id', None, 'Specify the GPU id')


# config the CPU/GPU in TF, assume only one GPU is in use.
# For multi-gpu setting, please refer to
#   https://www.tensorflow.org/guide/gpu#using_multiple_gpus

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) == 0 or FLAGS.gpu_id is None:
    device_id = "/device:CPU:0"
else:
    tf.config.experimental.set_visible_devices(gpus[FLAGS.gpu_id], 'GPU')
    device_id = '/device:GPU:0'

A_mat, X_mat, z_vec, train_idx, val_idx, test_idx = load_data_planetoid(FLAGS.dataset)
An_mat = preprocess_graph(A_mat)

# N = A_mat.shape[0]
K = z_vec.max() + 1

with tf.device(device_id):
    gcn = GCN(An_mat, X_mat, [FLAGS.hidden1, K])
    train_stats = gcn.train(train_idx, z_vec[train_idx], val_idx, z_vec[val_idx])
    train_losses = train_stats[0]
    val_losses = train_stats[1]
    train_accuracies = train_stats[2]
    val_accuracies = train_stats[3]
    
    plt.figure()
    smooth_plot(train_losses, label='Train Loss')
    smooth_plot(val_losses,  label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Citeseer')
    plt.legend()
    
    plt.figure()
    
    smooth_plot(train_accuracies, label='Train Acc')
    smooth_plot(val_accuracies, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Citeseer')
    plt.legend()
    
    
    test_res = gcn.evaluate(test_idx, z_vec[test_idx], training=False)
    # gcn = GCN(An_mat_diag, X_mat_stack, [FLAGS.hidden1, K])
    # gcn.train(train_idx_recal, z_vec[train_idx], val_idx_recal, z_vec[val_idx])
    # test_res = gcn.evaluate(test_idx_recal, z_vec[test_idx], training=False)
    print("Dataset {}".format(FLAGS.dataset),
          "Test loss {:.4f}".format(test_res[0]),
          "test acc {:.4f}".format(test_res[1]))

