import tensorflow as tf
from tensorflow.keras import activations, regularizers, constraints, initializers
spdot = tf.sparse.sparse_dense_matmul
dot = tf.matmul


class GCNConv(tf.keras.layers.Layer):

    def __init__(self,
                 units,
                 K = 3, #distance from the central node
                 activation=lambda x: x,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 activity_regularizer=None,
                 **kwargs):

        self.units = units
        self.K = K
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(GCNConv, self).__init__()
        #commento da pubblicare 

    def build(self, input_shape):
        
        """ GCN has two inputs : [shape(An), shape(X)]
        """
        # gsize = input_shape[0][0]  # graph size
        fdim = input_shape[1][1]  # feature dim

        if not hasattr(self, 'H'):
            self.H = []
            for k in range(self.K):
                weight = self.add_weight(name="weight_"+str(k),
                                              shape=(fdim, self.units),
                                              initializer=self.kernel_initializer,
                                              constraint=self.kernel_constraint,
                                              trainable=True)
                self.H.append(weight)
            
        if self.use_bias:
            if not hasattr(self, 'biases'):
                self.biases = []
                for k in range(self.K):
                    self.biases = []
                    bias = self.add_weight(name="bias_"+str(k),
                                                shape=(self.units, ),
                                                initializer=self.bias_initializer,
                                                constraint=self.bias_constraint,
                                                trainable=True)
                    self.biases.append(bias)
                    
        super(GCNConv, self).build(input_shape)

    def call(self, inputs):
        """ GCN has two inputs : [An, X]
        """
        self.An = inputs[0]
        self.X = inputs[1]
        output = None
        if not hasattr(self, 'Sk'):
            self.Sk = [tf.sparse.from_dense(tf.eye(self.An.shape[0])), self.An]
            for k in range(2,self.K+1):
                self.Sk.append(tf.sparse.from_dense(spdot(self.Sk[-1] , tf.sparse.to_dense(self.An))))
                
        for k in range(self.K):
            if isinstance(self.X, tf.SparseTensor):
                h = spdot(self.X, self.H[k])
            else:
                h = dot(self.X, self.H[k])
            output = output + spdot(self.Sk[k], h) if output is not None else spdot(self.Sk[k], h)

            if self.use_bias:
                output = tf.nn.bias_add(output, self.biases[k])

        if self.activation:
            output = self.activation(output)

        return output
