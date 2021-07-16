import model.rlstm as lstm
import tensorflow as tf
class encoder(object):
    def __init__(self,batch_size,layer_num=1,nodes=128,is_training=True):
        '''
        :param batch_size:
        :param layer_num:
        :param nodes:
        :param is_training:

        We need to define the encoder of Encoder-Decoder Model,and the parameter will be
        express in the Rlstm.
        '''
        with tf.variable_scope('encoder'):
            self.encoder_lstm=lstm.rlstm(batch_size,layer_num,nodes,is_training)

    def encoding(self,input):
        '''
        we always use c_state as the input to decoder
        '''
        (self.c_state, self.h_state) = self.encoder_lstm.calculate(input)
        return (self.c_state,self.h_state)