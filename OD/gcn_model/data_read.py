# -- coding: utf-8 --
import pandas as pd
import csv

file_path=r'/Users/guojianzou/PycharmProjects/OD/data/Order_all.csv'
save_path=r'/Users/guojianzou/PycharmProjects/OD/data/data_all.csv'

train_path=r'/Users/guojianzou/PycharmProjects/OD/data/train_data.csv'
combine_path=r'/Users/guojianzou/PycharmProjects/OD/data/combine_data.csv'
data_colum=["ZoneID","Area","Slon","Slat","Elon","Elat","day","hour","min","second"]

def data_save(file_path,save_pave):
    '''
    :param file_name:
    :return:
    dtype pd.datafrme
    '''

    data = pd.read_csv(file_path, encoding='utf-8')
    data=data.values

    file = open(save_path, 'w', encoding='utf-8')
    writer=csv.writer(file)
    writer.writerow(data_colum)

    for line in data:
        # line = char.split(',')
        data_line=[int(line[1])]+[float(ch) for ch in line[2:7]]+[int(line[11]),int(line[12])]+[int(line[9][14:16]),int(line[9][17:19])]
        writer.writerow(data_line)
    file.close()
    print('data_save finish')


def train_data(save_path,train_path):
    train_colum = ["ZoneID", "day", "hour", "min","label"]
    file = open(train_path, 'w', encoding='utf-8')
    writer=csv.writer(file)
    writer.writerow(train_colum)

    data = pd.read_csv(save_path, encoding='utf-8')
    for d in range(1,31):
        data1=data.loc[data['day'] == d]
        if data1.values.shape[0]==0:
            print('day empty')
            continue
        for h in range(0,24):
            data2 = data1.loc[data1['hour'] == h]
            if data2.values.shape[0] == 0:
                print('hour empty')
                continue
            for m in range(0,60):
                data3 = data2.loc[data2['min'] == m]
                if data3.values.shape[0] == 0:
                    print('min empty')
                    continue
                for id in range(162):
                    data4 = data3.loc[data3['ZoneID'] == id]
                    if data4.values.shape[0] == 0:
                        print('zone empty')
                        continue
                    line=[id,d,h,m,data4.values.shape[0]]
                    writer.writerow(line)
    file.close()
    print('train_data finish!!!!')
    return


def data_combine(train_path, combine_path):
    train_colum = ["ZoneID", "day", "hour", "min-15","label"]
    file = open(combine_path, 'w', encoding='utf-8')
    writer=csv.writer(file)
    writer.writerow(train_colum)

    data = pd.read_csv(train_path, encoding='utf-8')

    for d in range(1,31):
        data1=data.loc[data['day'] == d]
        if data1.values.shape[0]==0:
            print(d,' day empty')
            continue
        for h in range(0,24):
            data2 = data1.loc[data1['hour'] == h]
            if data2.values.shape[0] == 0:
                print(d,h,' hour empty')
                continue
            for i in range(4):
                for id in range(162):
                    data3 = data2.loc[data2['ZoneID'] == id]
                    if data3.values.shape[0] == 0:
                        print(d, h, (i + 1) * 15, id,' zone empty')
                        line = [id, d, h, (i + 1) * 15, 0]
                        writer.writerow(line)
                        continue
                    sum_ = sum([data3.loc[(data['min'] == (j + i * 15))].values.shape[0] for j in range(15)])
                    line=[id,d,h,(i+1)*15,sum_]
                    writer.writerow(line)
    file.close()
    print('data_combine finish!!!!')
    return