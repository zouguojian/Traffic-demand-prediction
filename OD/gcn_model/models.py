# -- coding: utf-8 --

from gcn_model.layers import *
from gcn_model.metrics import *


class Model(object):
    def __init__(self):

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope('gcn'):  # add two dense layers "self.layers"
            self._build()

    def predict(self,inputs):
        '''
        :return:  output each node result
        '''
        self.inputs = inputs  # input features
        # Build sequential layer model
        self.activations.append(self.inputs)

        for layer in self.layers:
            hidden = layer.forward(self.activations[-1])
            self.activations.append(hidden) # feed forward
        outputs = self.activations[-1] # the last layer output
        return outputs

class GCN(Model):
    def __init__(self, placeholders, input_dim, para):
        super(GCN, self).__init__()

        self.input_dim = input_dim  # input features dimension
        self.para=para

        # self.output_dim = placeholders['labels'].get_shape().as_list()[1]  # number of class
        self.output_dim = para.gcn_output_size  # number of features of gcn output
        self.placeholders = placeholders

        self.build()  # build model

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.para.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            bias=True,
                                            dropout=True,
                                            sparse_inputs=False,
                                            res_name='layer1'))

        self.layers.append(GraphConvolution(input_dim=self.para.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=False,
                                            res_name='layer2'))