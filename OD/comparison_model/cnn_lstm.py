# -- coding: utf-8 --

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class cnn_lstm(object):
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
        self.h=15
        self.w=3
        self.position_size=162
        self.features=5
        self.encoder()
        self.decoder()

    def position_em(self):
        '''
        :return:  shape is [batch size, site num ,new feature size]
        '''
        position_e=tf.Variable(initial_value=tf.random_normal(shape=[self.position_size, 32]),name='filter1')
        pos_es=tf.nn.embedding_lookup(position_e,[i for i in range(self.position_size)])
        pos_es = tf.expand_dims(pos_es, axis=0)
        pos_es = tf.tile(pos_es, multiples=[self.batch_size, 1, 1])
        return pos_es

    def cnn(self,x):
        '''
        :param x: shape is [batch size, site num, features, channel]
        :return: shape is [batch size, height, channel]
        '''
        filter1=tf.Variable(initial_value=tf.random_normal(shape=[self.h,self.w,1,64]),name='filter1')
        layer1=tf.nn.conv2d(input=x,filter=filter1,strides=[1,1,1,1],padding='SAME')
        # bn1=tf.layers.batch_normalization(layer1,training=self.placeholders['is_training'])
        relu1=tf.nn.relu(layer1)
        # max_pool1=tf.nn.max_pool(relu1, ksize=[self.h,self.w, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

        # print('max_pool1 output shape is : ',max_pool1.shape)

        filter2 = tf.Variable(initial_value=tf.random_normal(shape=[self.h,self.w, 64, 64]), name='filter2')
        layer2 = tf.nn.conv2d(input=relu1, filter=filter2, strides=[1, 1, 1, 1], padding='SAME')
        # bn2=tf.layers.batch_normalization(layer2,training=self.placeholders['is_training'])
        relu2=tf.nn.relu(layer2)
        # max_pool2=tf.nn.max_pool(relu2, ksize=[self.h,self.w, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

        # print('max_pool2 output shape is : ',max_pool2.shape)

        filter3 = tf.Variable(initial_value=tf.random_normal(shape=[self.h,self.w, 64, 128]), name='filter3')
        layer3 = tf.nn.conv2d(input=relu2, filter=filter3, strides=[1, 1, 1, 1], padding='SAME')
        # bn3=tf.layers.batch_normalization(layer3,training=self.placeholders['is_training'])
        relu3=tf.nn.relu(layer3)
        # max_pool3=tf.nn.max_pool(relu3, ksize=[self.h,self.w, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

        # print('max_pool3 output shape is : ', max_pool3.shape)

        # cnn_shape = max_pool3.get_shape().as_list()
        # nodes = cnn_shape[1] * cnn_shape[2] * cnn_shape[3]
        # reshaped = tf.reshape(max_pool3, [cnn_shape[0], nodes])
        '''shape is  : [batch size, site num, features, channel]'''
        relu3=tf.reduce_mean(relu2,axis=3)
        print('relu3 shape is : ',relu3.shape)

        s=tf.layers.dense(inputs=relu3,units=128,activation=tf.nn.relu)

        # print('cnn output shape is : ',s.shape)
        return s

    def encoder(self):
        '''
        :return:  shape is [batch size, time size, hidden size]
        '''

        def cell():
            lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=self.nodes)
            lstm_cell_=tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell,output_keep_prob=1.0)
            return lstm_cell_
        self.e_mlstm=tf.nn.rnn_cell.MultiRNNCell([cell() for _ in range(self.layer_num)])
        self.initial_state = self.e_mlstm.zero_state(self.batch_size, tf.float32)

    def decoder(self):
        def cell():
            lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=self.nodes)
            lstm_cell_=tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell,output_keep_prob=1.0-self.placeholders['dropout'])
            return lstm_cell_
        self.d_mlstm=tf.nn.rnn_cell.MultiRNNCell([cell() for _ in range(self.layer_num)])
        self.initial_state = self.d_mlstm.zero_state(self.batch_size, tf.float32)

    def encoding(self, inputs):
        '''
        :param inputs:
        :return: shape is [batch size, time size, hidden size]
        '''
        # out put the store data
        with tf.variable_scope('encoder_lstm'):
            self.ouputs, self.state = tf.nn.dynamic_rnn(cell=self.e_mlstm, inputs=inputs,initial_state=self.initial_state,dtype=tf.float32)
        return self.ouputs

    def decoding(self,  encoder_h):
        '''
        :param encoder_hs:
        :return:  shape is [batch size, prediction size]
        '''

        h = []
        h_state = encoder_h
        h_state = tf.expand_dims(input=h_state, axis=1)


        for i in range(self.predict_time):
            with tf.variable_scope('decoder_lstm'):
                h_state, state = tf.nn.dynamic_rnn(cell=self.d_mlstm, inputs=h_state,
                                                   initial_state=self.initial_state, dtype=tf.float32)
                self.initial_state = state
                results = tf.layers.dense(inputs=tf.squeeze(h_state), units=1, name='layer', reuse=tf.AUTO_REUSE)
            h.append(results)

        return tf.squeeze(tf.transpose(tf.convert_to_tensor(h), [1, 2, 0]), axis=1)

import numpy as np
if __name__ == '__main__':
    train_data=np.random.random(size=[32,1,162,5])
    x=tf.placeholder(tf.float32, shape=[32, 3, 162,5])
    r=cnn_lstm(32,10,2,128)
    # hs=r.cnn(x)

    model_cnn_ = []
    for time in range(3):
        model_cnn_.append(r.cnn(tf.expand_dims(x[:,time,:,:],axis=1)))
    inputs = tf.concat(values=model_cnn_, axis=1)
    print(inputs.shape)

    pos_es=r.position_em()
    pos_es=tf.expand_dims(pos_es,axis=0)
    pos_es=tf.tile(pos_es,multiples=[32,3,1,1])
    pos_es = tf.layers.dense(inputs=pos_es, units=128, name='layer', activation=tf.nn.relu, reuse=tf.AUTO_REUSE)
    print(pos_es.shape)

    # print(hs.shape)
    #
    # pre=r.decoding(hs)
    # print(pre.shape)