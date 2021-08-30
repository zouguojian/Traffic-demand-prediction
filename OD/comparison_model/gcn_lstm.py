# -- coding: utf-8 --

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class gcn_lstm(object):
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
        self.encoder()
        self.decoder()

    def encoder(self):
        '''
        :return:  shape is [batch size, time size, hidden size]
        '''

        def cell():
            lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=self.nodes)
            lstm_cell_=tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell,output_keep_prob=1.)
            return lstm_cell_
        self.e_mlstm=tf.nn.rnn_cell.MultiRNNCell([cell() for _ in range(self.layer_num)])
        self.e_initial_state = self.e_mlstm.zero_state(self.batch_size, tf.float32)

    def decoder(self):
        def cell():
            lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=self.nodes)
            lstm_cell_=tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell,output_keep_prob=1-self.placeholders['dropout'])
            return lstm_cell_
        self.d_mlstm=tf.nn.rnn_cell.MultiRNNCell([cell() for _ in range(self.layer_num)])
        self.d_initial_state = self.d_mlstm.zero_state(self.batch_size, tf.float32)

    def encoding(self, inputs):
        '''
        :param inputs:
        :return: shape is [batch size, time size, hidden size]
        '''
        # out put the store data
        with tf.variable_scope('encoder_lstm'):
            self.ouputs, self.state = tf.nn.dynamic_rnn(cell=self.e_mlstm, inputs=inputs,initial_state=self.e_initial_state,dtype=tf.float32)
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
            with tf.variable_scope('decoder_lstm'):
                h_state, state = tf.nn.dynamic_rnn(cell=self.d_mlstm, inputs=h_state,
                                                   initial_state=self.d_initial_state, dtype=tf.float32)
                self.d_initial_state = state
                results = tf.layers.dense(inputs=tf.squeeze(h_state), units=1, name='layer', reuse=tf.AUTO_REUSE)
            h.append(results)

        return tf.squeeze(tf.transpose(tf.convert_to_tensor(h), [1, 2, 0]), axis=1)

    def attention(self, h_t, encoder_hs):
        '''
        h_t for decoder, the shape is [batch, h]
        encoder_hs for encoder, the shape is [batch, time ,h]
        :param h_t:
        :param encoder_hs:
        :return:
        '''
        #scores = [tf.matmul(tf.tanh(tf.matmul(tf.concat(1, [h_t, tf.squeeze(h_s, [0])]),
        #                    self.W_a) + self.b_a), self.v_a)
        #          for h_s in tf.split(0, self.max_size, encoder_hs)]
        #scores = tf.squeeze(tf.pack(scores), [2])
        scores = tf.reduce_sum(tf.multiply(encoder_hs, tf.tile(h_t,multiples=[1,encoder_hs.shape[1],1])), 2)
        # a_t    = tf.nn.softmax(tf.transpose(scores))  #[batch, time]
        a_t = tf.nn.softmax(scores)  # [batch, time]
        a_t    = tf.expand_dims(a_t, 2) #[batch, time ,1]
        c_t    = tf.matmul(tf.transpose(encoder_hs, perm=[0,2,1]), a_t) #[batch ,h , 1]
        c_t    = tf.squeeze(c_t) #[batch, h]]
        h_t=tf.squeeze(h_t)
        h_tld  = tf.layers.dense(tf.concat([h_t, c_t], axis=1),units=c_t.shape[-1],activation=tf.nn.relu) #[batch, h]
        # print('h_tld shape is : ',h_tld.shape)
        return h_tld

    # def decoding(self,encoder_hs):
    #     '''
    #     :param h_state:
    #     :return:
    #     '''
    #     h=[]
    #
    #     h_state=encoder_hs[:,-1,:]
    #     for i in range(self.predict_time):
    #         h_state = tf.expand_dims(input=h_state,axis=1)
    #
    #         with tf.variable_scope('decoder_lstm', reuse=tf.AUTO_REUSE):
    #             h_state, state = tf.nn.dynamic_rnn(cell=self.d_mlstm, inputs=h_state,initial_state=self.initial_state,dtype=tf.float32)
    #             # h_state=tf.squeeze(h_state)
    #             h_state=self.attention(h_t=h_state,encoder_hs=encoder_hs) # attention
    #             self.initial_state=state
    #
    #             results=tf.layers.dense(inputs=h_state,units=1,name='layer',reuse=tf.AUTO_REUSE)
    #         h.append(results)
    #
    #     return tf.squeeze(tf.transpose(tf.convert_to_tensor(h),[1,2,0]),axis=1)

import numpy as np
if __name__ == '__main__':
    train_data=np.random.random(size=[32,3,16])
    x=tf.placeholder(tf.float32, shape=[32, 3, 16])
    r=lstm(32,10,2,128)
    hs=r.encoding(x)

    print(hs.shape)

    pre=r.decoding(hs)
    print(pre.shape)