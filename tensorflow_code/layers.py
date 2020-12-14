#coding:utf-8
from inits import *
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
import scipy.sparse as sp
import numpy as np 
import copy
import math

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)

def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)    

class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs[0])
            return outputs
    #weight_3,2,1,0
    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])




#gwnn
class Wavelet_Convolution(Layer):
    """Graph convolution layer."""
    def __init__(self, node_num,weight_normalize,input_dim, output_dim, placeholders,support, transfer, node_emb, mod, layer_index,dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(Wavelet_Convolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.node_num = node_num
        self.weight_normalize = weight_normalize
        self.act = act
        #add
        self.support = support
        #print("support:" , self.support)
        self.transfer = transfer
        self.node_emb = node_emb
        self.mod = mod
        self.layer_index = layer_index
        
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        #print(len(self.support))
        # helper variable for sparse dropout

        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            if self.mod == 'coarsen' or self.mod == 'refine':
                for i in range(4):   #通道数为4
                    self.vars['weights_' + str(i)] = glorot([input_dim + FLAGS.node_wgt_embed_dim, output_dim],
                                                        name='weights_' + str(i))  #glorot用于权值初始化
            else:
                for i in range(4): #通道数为4
                    self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],

                                                        name='weights_' + str(i))
            self.vars['kernel'] = ones([self.node_num], name='kernel')

            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')


        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        # input (add)
        if self.mod == 'coarsen' or self.mod == 'refine':
            x = tf.concat([inputs, self.node_emb], 1)
            print('layer_index ', self.layer_index + 1)
            print('input shape:   ', inputs.get_shape().as_list())
            self.sparse_inputs = False
        elif self.mod == 'input' or self.mod == 'output':
            x = inputs

        # dropout
        #print(self.sparse_inputs)
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        #convolve
        supports = list()
        for i in range(4):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
        #print(self.support[0])
        #print(self.support[1])
        for i in range(0,8,2):
            row_1 = self.support[i][0][:, 0]  
            col_1 = self.support[i][0][:, 1]  
            data_1 = self.support[i][1]  
            sp_suport_1 = sp.csr_matrix((data_1, (row_1, col_1)), shape=self.support[i][2],
                                      dtype=np.float32)  
            sp_suport_tensor_1 = convert_sparse_matrix_to_sparse_tensor(sp_suport_1)

            row_2 = self.support[i+1][0][:, 0]  
            col_2 = self.support[i+1][0][:, 1]  
            data_2 = self.support[i+1][1]  
            sp_suport_2 = sp.csr_matrix((data_2, (row_2, col_2)), shape=self.support[i+1][2],
                                      dtype=np.float32)  
            sp_suport_tensor_2 = convert_sparse_matrix_to_sparse_tensor(sp_suport_2)

            support_ans = tf.matmul(tf.sparse_tensor_to_dense(sp_suport_tensor_1),tf.diag(self.vars['kernel']),a_is_sparse=True,b_is_sparse=True)
            support_ans = tf.matmul(support_ans,tf.sparse_tensor_to_dense(sp_suport_tensor_2),a_is_sparse=True,b_is_sparse=True)
            support_ans = dot(support_ans,pre_sup)
            supports.append(support_ans)

        supports = tf.convert_to_tensor(supports)
        supports = tf.transpose(supports, [1, 2, 0])  
        output = tf.squeeze(tf.layers.conv1d(supports, 1, 1, use_bias=False))

        #bias
        if self.bias:
            output += self.vars['bias']
        output = self.act(output)

    
        gwnn_out = output

        if self.mod == 'output':
            print('layer_index ', self.layer_index+1)
            print('input shape:   ', inputs.get_shape().as_list())
            print('output shape:    ',output.get_shape().as_list())
            return output,gwnn_out

        if self.mod == 'coarsen' or self.mod == 'input':
            transfer_opo = normalize(self.transfer.T, norm='l2', axis = 1).astype(np.float32)
            transfer_opo = convert_sparse_matrix_to_sparse_tensor(transfer_opo)
            output = dot(transfer_opo, gwnn_out, sparse=True) 

        elif self.mod == 'refine' :
            transfer_opo = convert_sparse_matrix_to_sparse_tensor(self.transfer.astype(np.float32))
            output = dot(transfer_opo, gwnn_out, sparse=True) 

        print('output shape:    ',output.get_shape().as_list())
        return output,gwnn_out

