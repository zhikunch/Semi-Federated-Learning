'''
 * @Author: zhikunch 
 * @Date: 2021-03-19 20:39:28 
 * @Last Modified by: zhikunch
 * @Last Modified time: 2021-03-19 20:43:39
 '''

import matplotlib
import matplotlib.pyplot as plt
from utils.options import args_parser
from utils.sampling import *
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
from tensorboardX import SummaryWriter
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
        acc_threshold = 80
        # sample users
        if args.iid:
            print('i.i.d')
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            print('Non-i.i.d')
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'fashion-mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.FashionMNIST('../data/fashion_mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.FashionMNIST('../data/fashion_mnist/', train=False, download=True, transform=trans_mnist)
        acc_threshold = 65
        # sample users
        if args.iid:
            print('i.i.d')
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            print('Non-i.i.d')
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        acc_threshold = 35
        if args.iid:
            print('i.i.d')
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            print('Non-i.i.d')
            dict_users = cifar_noniid2(dataset_train, args.num_users)
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
    step_size = args.lr
    duration = []
    delta_time = 0
    writer = SummaryWriter()
    # training
    
    for iter in range(args.epochs):
        comp_time = []
        w_locals, loss_locals = [], []
        net_glob.train()
        if iter%10 == 0:
            step_size *= args.decay_rate
            if step_size <= 1e-5:
                step_size = 1e-5
        # choose partially or all users
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False) # 重新选择m个用户
        # FedAvg training
        for idx in idxs_users:
            start_time = time.time()
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device),stepsize=step_size)
            end_time = time.time()
            comp_time.append(end_time-start_time)
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        loss_avg = sum(loss_locals)/len(loss_locals)
        # update global weights
        start_time = time.time()
        w_glob = FedAvg(w_locals)
        end_time = time.time()
        fed_time = end_time-start_time
        delta_time = delta_time + (max(comp_time)+fed_time)
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        # test
        net_glob.eval()
        acc_test, loss_test = test_img(net_glob.to(args.device), dataset_test, args)
        # if acc_test >= acc_threshold:
        #     duration.append(delta_time)
        #     duration.append(iter+1)
        #     print("duration: {}s".format(duration[0]))
        #     text_save('./log/fl/Time_{}_iid{}_partial.csv'.format(args.dataset, args.iid),duration)
        #     break
        print('Round:{} --averaged loss:{:.2f} --Test acc:{:.2f}%'.format(iter, loss_avg, acc_test))
        loss_train_lst.append(loss_avg)
        acc_test_lst.append(acc_test)
        # writer.add_scalar('scalar/test',acc_test,iter)
    # writer.close()
    text_save('./log/fl/AveLoss_{}_frac{}_iid{}.csv'.format(args.dataset, args.frac, args.iid),loss_train_lst)
    text_save('./log/fl/TestAcc_{}_frac{}_iid{}.csv'.format(args.dataset, args.frac, args.iid),acc_test_lst)
