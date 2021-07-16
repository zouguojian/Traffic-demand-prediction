import numpy as np
#
# import tensorflow as tf


# class ls(object):
#     def __init__(self):
#         self.decoder()
#
#     def decoder(self):
#         def cell():
#             lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
#             lstm_cell_ = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, output_keep_prob=1 )
#             return lstm_cell_
#
#         self.d_mlstm = tf.nn.rnn_cell.MultiRNNCell([cell() for _ in range(1)])
#         self.d_initial_state = self.d_mlstm.zero_state(32, tf.float32)
#         # return d_mlstm, d_initial_state
#
#     def mul(self,x,layer):
#         d_initial_state=self.d_initial_state
#         for i in range(3):
#
#             with tf.variable_scope(str(i)+'_decoder_lstm_'+str(layer)):
#                 h_state, state = tf.nn.dynamic_rnn(cell=self.d_mlstm,
#                                                    inputs=x,
#                                                    initial_state=d_initial_state,
#                                                    dtype=tf.float32)
#                 # print(self.d_mlstm.variables)
#                 print(d_initial_state)
#                 # print(tf.Session().run(d_initial_state))
#             d_initial_state=state
#
#
# x=tf.placeholder(tf.float32, shape=[32, 3, 16])
# # x=np.random.random(size=[32,3,16])
# l = ls()
# for i in range(3):
#     l.mul(x,i)

A=[1]

B=[[2,3],[4,5]]

import numpy as np

c=np.concatenate([A,B],axis=0)
print(c)

