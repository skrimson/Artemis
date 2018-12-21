import keras.backend as K
from keras.engine.topology import Layer
from keras import initializers
from keras import regularizers
from keras import constraints
import numpy as np

class Attention(Layer):
    def __init__(self, W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True):
        """
        Keras Layer that implements an Content Attention mechanism.
        Supports Masking.
        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias

    def build(self, input_shape):
        assert type(input_shape) == list
        assert len(input_shape) == 2

        self.steps = input_shape[0][1]
        print("steps:{}".format(self.steps))

        #M: matrix mapping between global context embeddings and word embedding
        self.kernel = self.add_weight((input_shape[0][-1], input_shape[1][-1]),
                                    initializer=self.init,
                                    name='{}_W'.format(self.name),
                                    regularizer=self.W_regularizer,
                                    constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((1,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        super(MyLayer, self).build(input_shape)

    def call(self, input_tensor, mask=None):
        #e_w
        x = input_tensor[0]
        #y_s
        y = input_tensor[1]
        mask = mask[0]

        #M * y_s
        y = K.transpose(K.dot(self.kernel, K.transpose(y)))
        print("after transpose M * y_s:{}".format(y.shape))
        y = K.expand_dims(y, -2)
        #repeating the average 200 vectors for length of sequence(157)
        y = K.repeat_elements(y, self.steps, axis=1)
        print("repeated y:{}".format(y.shape))
        eij = K.sum(x*y, axis=-1)

        if self.bias:
            b = K.repeat_elements(self.b, self.steps, axis=0)
            eij += b

        #strengthing weighting by tanh
        eij = K.tanh(eij)
        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        return a

class Average(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Average, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask)
            x = x * mask
        return K.sum(x, axis=-2) / K.sum(mask, axis=-2)
