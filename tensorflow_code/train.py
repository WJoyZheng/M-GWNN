#coding:utf-8
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from utils import *
from models import MGWNN
from coarsen_louvain import *
import copy
import matplotlib.pyplot as plt
import pickle as pkl



# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')   
flags.DEFINE_string('model', 'mgwnn', 'Model string.')   
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')
flags.DEFINE_integer('seed1', 123, 'random seed for numpy.')
flags.DEFINE_integer('seed2', 123, 'random seed for tf.')
flags.DEFINE_integer('hidden', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('node_wgt_embed_dim', 5 , 'Number of units for mgwnn node weight embedding.')  
flags.DEFINE_float('dropout', 0.8, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 9e-3, 'Weight for L2 loss on embedding matrix.')  
flags.DEFINE_integer('early_stopping', 100, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('coarsen_level', 2 , 'Maximum coarsen level.')
flags.DEFINE_integer('max_node_wgt', 50, 'Maximum node_wgt to avoid super- node being too large.')
flags.DEFINE_integer('channel_num', 4, 'Number of channels') 
flags.DEFINE_float('wavelet_s',0.9, 'wavelet s .')  
flags.DEFINE_bool('weight_normalize', False, 'weight normalize string.')
flags.DEFINE_bool('laplacian_normalize', True, 'laplacian normalize string.')
flags.DEFINE_bool('sparse_ness', True, 'wavelet sparse_ness string.')
flags.DEFINE_float('threshold', 9e-6, 'sparseness threshold .') 

# Set random seed
seed1 = FLAGS.seed1
seed2 = FLAGS.seed2
np.random.seed(seed1)
tf.set_random_seed(seed2)

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset) 

# Some preprocessing
features = preprocess_features(features)

if FLAGS.model == 'mgwnn':

    num_supports = 2
    model_func = MGWNN
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

graph, mapping = read_graph_from_adj(adj, FLAGS.dataset)   
print('total nodes:', graph.node_num)


# Step-1: Graph Coarsening.
original_graph = graph
transfer_list = []
adj_list = [copy.copy(graph.A)]
node_wgt_list = [copy.copy(graph.node_wgt)]  
node_num_list = [copy.copy(graph.node_num)]  

for i in range(FLAGS.coarsen_level):
    
    match, coarse_graph_size = louvain(FLAGS.max_node_wgt,graph)
    coarse_graph = create_coarse_graph(graph, match, coarse_graph_size)

    transfer_list.append(copy.copy(graph.C))  
    graph = coarse_graph
    adj_list.append(copy.copy(graph.A)) 
    node_wgt_list.append(copy.copy(graph.node_wgt)) 
    node_num_list.append(copy.copy(graph.node_num))  
    
    print('There are %d nodes in the %d coarsened graph' %(graph.node_num, i+1))
    print("\n")

print("\n")
print('layer_index ', 1)
print('input shape:   ', features[-1])


# Define placeholders
placeholders = {
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)), 
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])), 
    'labels_mask': tf.placeholder(tf.int32), 
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32), # helper variable for sparse dropout
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
}

# Create model
weight_normalize = FLAGS.weight_normalize

model = model_func( weight_normalize, placeholders, input_dim=features[2][1], logging=True,transfer_list = transfer_list, adj_list = adj_list, node_wgt_list = node_wgt_list,node_num_list = node_num_list)


# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)

# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []
acc_val = []


cost_train = []
acc_train = []

cost_test = []
acc_test = []
best_test_acc = 0
t = time.time()
# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features,  y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)


    # Validation
    cost, acc, duration = evaluate(features, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Test
    test_cost, test_acc, test_duration = evaluate(features, y_test, test_mask, placeholders)
    cost_train.append(outs[1])
    acc_train.append(outs[2])
    cost_test.append(test_cost)
    acc_test.append(test_acc)
    acc_val.append(acc)
    if test_acc > best_test_acc:
        best_test_acc = test_acc

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "test_acc=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(time.time() - t),"      best test_acc=", "{:.5f}".format(best_test_acc),)

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")
run_time = time.time() - t

############################### test acc for every epoch
mat = np.array(acc_test)
# print(np.max(mat))

if FLAGS.dataset == 'cora' or FLAGS.dataset == 'citeseer' or FLAGS.dataset == 'pubmed':
    val_index_best =  np.argmax(np.array(acc_test))
    print('best epoch:   ',val_index_best)
    print('test result:  ',mat[val_index_best])
    print('run_time: ',"{:.4f}".format(run_time))






