#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def Semi_FedAvg(w_cluster):
    w_avg = copy.deepcopy(w_cluster[0])
    for k in w_avg.keys():
        for i in range(1, len(w_cluster)):
            w_avg[k] += w_cluster[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w_cluster))
    return w_avg

def FedCorrect(w1,w2,alpha):
    w_cor = copy.deepcopy(w1)
    for k1,k2 in zip(w1.keys(),w2.keys()):
        w_cor[k1] = alpha * w1[k1] + (1-alpha) * w2[k2]
    return w_cor

def text_save(filename, data):#filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename,'w+')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存文件成功") 

