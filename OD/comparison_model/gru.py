# -- coding: utf-8 --

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class gru(object):
    def __init__(self, batch_size, predict_time,layer_num=1, nodes=128, placeholders=None):
        '''

        :param batch_size:
        :param layer_num:
        :param nodes:
        :param is_training:
        '''
        self.batch_size=batch_size
        self.layer_num=layer_num
        self.nodes=nodes
        self.placeholders=placeholders
        self.predict_time=predict_time

        self.encoder()
        self.decoder()

    def encoder(self):
        '''
        :return:  shape is [batch size, time size, hidden size]
        '''

        def cell():
            gru_cell=tf.nn.rnn_cell.GRUCell(num_units=self.nodes)
            gru_cell_=tf.nn.rnn_cell.DropoutWrapper(cell=gru_cell,output_keep_prob=1.0)
            return gru_cell_
        self.e_mrnn=tf.nn.rnn_cell.MultiRNNCell([cell() for _ in range(self.layer_num)])
        self.e_initial_state = self.e_mrnn.zero_state(self.batch_size, tf.float32)

    def decoder(self):
        def cell():
            gru_cell=tf.nn.rnn_cell.GRUCell(num_units=self.nodes)
            gru_cell_=tf.nn.rnn_cell.DropoutWrapper(cell=gru_cell,output_keep_prob=1-self.placeholders['dropout'])
            return gru_cell_
        self.d_mrnn=tf.nn.rnn_cell.MultiRNNCell([cell() for _ in range(self.layer_num)])
        self.d_initial_state = self.d_mrnn.zero_state(self.batch_size, tf.float32)

    def encoding(self, inputs):
        '''
        :param inputs:
        :return: shape is [batch size, time size, hidden size]
        '''
        # out put the store data
        with tf.variable_scope('encoder_gru'):
            self.ouputs, self.state = tf.nn.dynamic_rnn(cell=self.e_mrnn, inputs=inputs,initial_state=self.e_initial_state,dtype=tf.float32)
        return self.ouputs

    def decoding(self,  encoder_hs):
        '''
        :param encoder_hs:
        :return:  shape is [batch size, prediction size]
        '''

        h = []
        h_state = encoder_hs[:, -1, :]
        h_state = tf.expand_dims(input=h_state, axis=1)

        for i in range(self.predict_time):
            with tf.variable_scope('decoder_gru'):
                h_state, state = tf.nn.dynamic_rnn(cell=self.d_mrnn, inputs=h_state,
                                                   initial_state=self.d_initial_state, dtype=tf.float32)
                self.initial_state = state
                results = tf.layers.dense(inputs=tf.squeeze(h_state), units=1, name='layer', reuse=tf.AUTO_REUSE)
            h.append(results)

        return tf.squeeze(tf.transpose(tf.convert_to_tensor(h), [1, 2, 0]), axis=1)

import numpy as np
if __name__ == '__main__':
    train_data=np.random.random(size=[32,3,16])
    x=tf.placeholder(tf.float32, shape=[32, 3, 16])
    r=gru(32,10,2,128)
    hs=r.encoding(x)
    print(hs.shape)

    pre=r.decoding(encoder_hs=hs)
    print(pre.shape)
