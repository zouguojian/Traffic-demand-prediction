# -- coding: utf-8 --

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class mlrnn(object):
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
        self.predict_time=predict_time
        self.placeholders=placeholders
        self.multi_layers=162

        self.glo_encoder()
        self.glo_decoder()

    def encoder(self):
        '''
        :return:  shape is [batch size, time size, hidden size]
        '''

        def cell():
            lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=self.nodes)
            lstm_cell_=tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell,output_keep_prob=1-self.placeholders['dropout'])
            return lstm_cell_
        e_mlstm=tf.nn.rnn_cell.MultiRNNCell([cell() for _ in range(self.layer_num)])
        e_initial_state = e_mlstm.zero_state(self.batch_size, tf.float32)
        return e_mlstm,e_initial_state

    def decoder(self):
        def cell():
            lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=self.nodes)
            lstm_cell_=tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell,output_keep_prob=1-self.placeholders['dropout'])
            return lstm_cell_
        d_mlstm=tf.nn.rnn_cell.MultiRNNCell([cell() for _ in range(self.layer_num)])
        d_initial_state = d_mlstm.zero_state(self.batch_size, tf.float32)
        return d_mlstm,d_initial_state

    def glo_decoder(self):
      self.d_mlstm, self.d_initial_state = self.decoder()

    def glo_encoder(self):
      self.e_mlstm, self.e_initial_state=self.encoder()
    
    def multi_encoding(self, inputs):
        '''
        :param inputs:
        :return: shape is [batch size, time size, site size = multi_layers, hidden size]
        '''
        # out put the store data
        outputs=list()
        for i in range(self.multi_layers):
            with tf.variable_scope(str(i)+'encoder'):
                e_mlstm, e_initial_state=self.encoder()
                output, state = tf.nn.dynamic_rnn(cell=e_mlstm, inputs=inputs[:,:,i,:],initial_state=e_initial_state,dtype=tf.float32)
                # outputs.append(tf.layers.dense(inputs=output,units=32))
                outputs.append(output)
        return tf.transpose(outputs,[1,2,0,3])

    def global_encoding(self, inputs):
        '''
        :param inputs:
        :return: shape is [batch size, time size, hidden size]
        '''
        # out put the store data
        with tf.variable_scope('global_encoder'):
            # e_mlstm, e_initial_state=self.encoder()
            ouputs, state = tf.nn.dynamic_rnn(cell=self.e_mlstm, inputs=inputs,initial_state=self.e_initial_state,dtype=tf.float32)
        return ouputs

    def mul_decoding(self,  encoder_hs, site):
        '''
        :param encoder_hs:
        :return:  shape is [batch size, prediction size]
        '''

        h = []
        h_state = encoder_hs[:, -1, :]
        h_state = tf.expand_dims(input=h_state, axis=1)
        with tf.variable_scope('decoder_lstm_', reuse=tf.AUTO_REUSE):
        # with tf.variable_scope('decoder_lstm_'+str(site)):
            d_mlstm, d_initial_state = self.decoder()
            for i in range(self.predict_time):
                h_state, state = tf.nn.dynamic_rnn(cell=d_mlstm,
                                                   inputs=h_state,
                                                   initial_state=d_initial_state,
                                                   dtype=tf.float32)
                # d_initial_state = state
                results = tf.layers.dense(inputs=tf.squeeze(h_state), units=1, name='layer')
                h.append(results)

        return tf.squeeze(tf.transpose(tf.convert_to_tensor(h), [1, 2, 0]), axis=1)

    def glo_decoding(self,  encoder_hs):
        '''
        :param encoder_hs:
        :return:  shape is [batch size, prediction size]
        '''

        h = []
        h_state = encoder_hs[:, -1, :]
        h_state = tf.expand_dims(input=h_state, axis=1)

        with tf.variable_scope('decoder_lstm'):
            # d_mlstm, d_initial_state = self.decoder()
            for i in range(self.predict_time):
                h_state, state = tf.nn.dynamic_rnn(cell=self.d_mlstm,
                                                   inputs=h_state,
                                                   initial_state=self.d_initial_state,
                                                   dtype=tf.float32)
                # self.d_initial_state = state
                results = tf.layers.dense(inputs=tf.squeeze(h_state), units=1, name='layer', reuse=tf.AUTO_REUSE)
                h.append(results)

        return tf.squeeze(tf.transpose(tf.convert_to_tensor(h), [1, 2, 0]), axis=1)

import numpy as np
if __name__ == '__main__':
    train_data=np.random.random(size=[32,3,16])
    x=tf.placeholder(tf.float32, shape=[32, 3, 16])
    r=lstm(32,10,2,128)
    hs=r.encoding(x)

    print(hs.shape)

    pre=r.decoding(hs)
    print(pre.shape)