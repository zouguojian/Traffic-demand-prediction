# -- coding: utf-8 --

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from gcn_model.data_read import *
import argparse
from gcn_model.hyparameter import parameter


# data=data_save(file_path,save_path)
#
# train_data(save_path,train_path)
#
# data_combine(train_path, combine_path)

def sudden_changed(city_dictionary_):
    '''
    用于处理突变的值
    Args:
        city_dictionary:
    Returns:
    '''
    if city_dictionary_:
        for key in city_dictionary_.keys():
            dataFrame=city_dictionary_[key].values
            shape=city_dictionary_[key].shape
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if i!=0:
                        if dataFrame[i][j]-dataFrame[i-1][j]>200:
                            dataFrame[i][j] = dataFrame[i - 1][j]
            city_dictionary_[key]=pd.DataFrame(dataFrame)
    return city_dictionary_

class DataIterator():             #切记这里的训练时段和测试时段的所用的对象不变，否则需要重复加载数据
    def __init__(self,
                 site_id=0,
                 is_training=True,
                 time_size=3,
                 prediction_size=1,
                 data_divide=0.9,
                 window_step=1,
                 normalize=False,
                 hp=None):
        '''
        :param is_training: while is_training is True,the model is training state
        :param field_len:
        :param time_size:
        :param prediction_size:
        :param target_site:
        '''
        self.site_id=site_id                   # ozone ID
        self.time_size=time_size               # time series length of input
        self.prediction_size=prediction_size   # the length of prediction
        self.is_training=is_training           # true or false
        self.data_divide=data_divide           # the divide between in training set and test set ratio
        self.window_step=window_step           # windows step
        self.para=hp
        self.source_data=self.get_source_data(combine_path)

        # self.data=self.source_data.loc[self.source_data['ZoneID']==self.site_id]
        self.data=self.source_data
        self.length=self.data.values.shape[0]  #data length
        self.max,self.min=self.get_max_min()   # max and min are list type, used for the later normalization
        self.normalize=normalize
        if self.normalize:self.normalization() #normalization

    def get_source_data(self,file_path):
        '''
        :return:
        '''
        data = pd.read_csv(file_path, encoding='utf-8')
        return data

    def get_max_min(self):
        '''
        :return: the max and min value of input features
        '''
        self.min_list=[]
        self.max_list=[]
        print('the shape of features is :',self.data.values.shape[1])
        for i in range(self.data.values.shape[1]):
            self.min_list.append(round(float(min(self.data[list(self.data.keys())[i]].values)),3))
            self.max_list.append(round(float(max(self.data[list(self.data.keys())[i]].values)),3))
        print('the max feature list is :',self.max_list)
        print('the min feature list is :', self.min_list)
        return self.max_list,self.min_list

    def normalization(self):
        for i,key in enumerate(list(self.data.keys())):
            if i: self.data[key]=round((self.data[key] - np.array(self.min[i])) / (np.array(self.max[i]) - np.array(self.min[i])), 6)

    def generator_(self):
        '''
        :return: yield the data of every time,
        shape:input_series:[time_size,field_size]
        label:[predict_size]
        '''
        para=self.para
        shape=self.data.values.shape
        
        if self.is_training:
            low,high=0,int(shape[0]//para.site_num * self.data_divide)*para.site_num
        else:
            low,high=int(shape[0]//para.site_num * self.data_divide) *para.site_num, shape[0]

        while low+para.site_num*(para.input_length + para.output_length)<= high:
            label=self.data.values[low + self.time_size * para.site_num: low + self.time_size * para.site_num + self.prediction_size * para.site_num,-1:shape[1]]
            label=np.concatenate([label[i * para.site_num:(i + 1) * para.site_num, :] for i in range(self.prediction_size)], axis=1)

            time=self.data.values[low + self.time_size * para.site_num: low + self.time_size * para.site_num + self.prediction_size * para.site_num,1:4]
            time=np.concatenate([time[i * para.site_num:(i + 1) * para.site_num, :] for i in range(self.prediction_size)], axis=1)
            yield (np.array(self.data.values[low:low+self.time_size*para.site_num]),
                   label,time)
            if self.is_training: low += self.window_step*para.site_num
            else:low+=self.prediction_size*para.site_num
        return

    def next_batch(self,batch_size,epochs, is_training=True):
        '''
        :return the iterator!!!
        :param batch_size:
        :param epochs:
        :return:
        '''
        self.is_training=is_training
        dataset=tf.data.Dataset.from_generator(self.generator_,output_types=(tf.float32,tf.float32,tf.float32))

        if self.is_training:
            dataset=dataset.shuffle(buffer_size=int(self.data.values.shape[0]//self.para.site_num * self.data_divide-self.time_size-self.prediction_size)//self.window_step)
            dataset=dataset.repeat(count=epochs)
        dataset=dataset.batch(batch_size=batch_size)
        iterator=dataset.make_one_shot_iterator()

        return iterator.get_next()

        
def re_current(line, max, min):
    return [[line_sub[i]*(max[i]-min[i])+min[i]+0.1 for i in range(len(line_sub))] for line_sub in line]
#
if __name__=='__main__':
    para = parameter(argparse.ArgumentParser())
    para = para.get_para()

    iter=DataIterator(site_id=0,is_training=True,normalize=True,time_size=8,prediction_size=3,hp=para)
    print(iter.data.keys())
    # print(iter.data.loc[iter.data['ZoneID']==0])

    next=iter.next_batch(32,1,False)
    with tf.Session() as sess:
        for _ in range(3):
            x,y,time=sess.run(next)
            # print(time.shape)
            # print(time)
            rows = np.reshape(time, [-1, 3,3])
            print(rows.shape)
            rows = np.array([re_current(row_data, [30.0, 23.0, 60.0], [1.0, 0.0, 15.0]) for row_data in rows],dtype=int)
            print(rows)