'''
 * @Author: zhikunch 
 * @Date: 2021-03-19 20:39:28 
 * @Last Modified by: zhikunch
 * @Last Modified time: 2021-03-19 20:43:39
 '''

import matplotlib
import matplotlib.pyplot as plt
from utils.options import args_parser
from utils.sfl_init import _Init_
from utils.sampling import mnist_iid, cifar_iid
import copy
import torch
from tqdm import tqdm
import random
import numpy as np
from models.Update import LocalUpdate
from models.Fed import FedAvg, text_save
from models.test import test_img
from models.Nets import *
from torchvision import datasets,transforms
import pickle
import time

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    acc_threshold = 0
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # assign data to 1 user
        dict_users = mnist_iid(dataset_train, 1)
        acc_threshold = 80 # for logging convergence time
    elif args.dataset == 'fashion-mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.FashionMNIST('../data/fashion_mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.FashionMNIST('../data/fashion_mnist/', train=False, download=True, transform=trans_mnist)
        dict_users = mnist_iid(dataset_train, 1)
        acc_threshold = 65 
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        # assign data to 1 user
        dict_users = cifar_iid(dataset_train, 1)
        acc_threshold = 35
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'mnist':
        # net_glob = CNNMnist(args=args).to(args.device)
        net_glob = torch.load('./model_init/CNNMnist.pkl').to(args.device)
    elif args.model == 'cnn' and args.dataset == 'fashion-mnist':
        # net_glob = CNNFashion_Mnist(args=args).to(args.device)
        net_glob = torch.load('./model_init/CNNFashion_Mnist.pkl').to(args.device)
    elif args.model == 'cnn' and args.dataset == 'cifar':
        # net_glob = CNNCifar(args=args).to(args.device)
        net_glob = torch.load('./model_init/CNNCifar.pkl').to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    # train mode
    net_glob.train()
    w_glob = net_glob.state_dict()
    
    acc_test_lst, loss_train_lst = [], []
    duration = []
    delta_time = 0
    step_size = args.lr
    # training
    
    for iter in range(args.epochs):
        w_locals, loss_locals = [], []
        net_glob.train()
        if iter%10 == 0:
            step_size *= args.decay_rate
            if step_size <= 1e-5:
                step_size = 1e-5
        # FedAvg training
        start_time = time.time()
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[0])
        w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device),stepsize=step_size)
        end_time = time.time()
        delta_time = delta_time + (end_time-start_time)
        # copy weight to net_glob
        net_glob.load_state_dict(w)
        # test
        net_glob.eval()
        acc_test, loss_test = test_img(net_glob.to(args.device), dataset_test, args)
        if acc_test >= acc_threshold:
            duration.append(delta_time)
            duration.append(iter+1)
            print("duration: {}s".format(duration[0]))
            text_save('./log/cl/Time_{}.csv'.format(args.dataset),duration)
            break
        print('Round:{} --loss:{:.2f} --acc:{:.2f}%'.format(iter, loss, acc_test))
        loss_train_lst.append(loss)
        acc_test_lst.append(acc_test)
    # text_save('./log/cl/AveLoss_{}.csv'.format(args.dataset),loss_train_lst)
    # text_save('./log/cl/TestAcc_{}.csv'.format(args.dataset),acc_test_lst)
