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
import copy
from tqdm import tqdm
import random
from models.Update import LocalUpdate
from models.Fed import Semi_FedAvg, text_save
from models.test import test_img
import time
import torch
from torchvision import datasets,transforms
from utils.sampling import *
from models.Nets import *
import pickle

if __name__ == '__main__':
    # init
    args = args_parser()
    # net_glob, clusters, dataset_train, dataset_test, dict_users = _Init_(args, num_cluster=10) # init dataset,clustering,sampling,net
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
            # exit('Error: only consider IID setting in CIFAR10')
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

    # divide 100 clients into 10 clusters
    if args.pattern == 0:
        clusters = cluster_iid(dict_users, 10) # manual setup分簇
        print("manual setup")
    elif args.pattern == 1:
        clusters = cluster_random(dict_users, 10) # random setup分簇
        print("random setup")
    elif args.pattern == 2:
        clusters = cluster_leach(dict_users, 10) # leach setup分簇
        print("leach setup")
    else:
        exit('No clustering patterns')
    
    # for early stop
    patience = 20
    stop_cnt = 0
    monitor = []
    should_stop = False
    net_best = copy.deepcopy(net_glob)
    acc_best = 0

    # train mode
    net_tmp = copy.deepcopy(net_glob)
    net_glob.train()
    w_glob = net_glob.state_dict() # copy weights
    acc_test_lst, loss_train_lst = [], []
    step_size = args.lr
    duration = []
    delta_time = 0
    
    # training
    for iter in range(args.epochs):
        comp_time = []
        w_clusters, loss_clusters = [], []
        net_glob.train()
        # decaying learning rate
        # print("step_size:",step_size)
        if iter>=10 and iter%10 == 0:
            step_size *= args.decay_rate
            if step_size <= 1e-8:
                step_size = 1e-8
        # 遍历每个簇
        for idx_cluster, _users in clusters.items():
            idx_users, loss_local = [], []
            # 每个簇头用户 model初始状态
            net_tmp = copy.deepcopy(net_glob)
            # 遍历簇内的每个用户
            for user_key, user_val in _users.items():
                idx_users.append(int(user_key)) # 该簇内的所有用户idx
            # print(idx_users)

            # shuffle the in-cluster sequential order and randomly select a CH
            random.shuffle(idx_users)
            # each cluster is performed parallel
            start_time = time.time()
            for idx in idx_users:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, loss = local.train(net=copy.deepcopy(net_tmp).to(args.device), stepsize=step_size)
                loss_local.append(copy.deepcopy(loss))
                # 用相邻节点的 model初始化下一节点的 model
                net_tmp.load_state_dict(w) 
            # 一个簇内的用户按 seq 方式训练完成后，记录每个簇参与上传的 model
            end_time = time.time()
            w_clusters.append(copy.deepcopy(w))
            loss_clusters.append(sum(loss_local)/len(loss_local))
            comp_time.append(end_time-start_time)
        loss_avg = sum(loss_clusters)/len(loss_clusters)
        # 对每个簇产生的 model进行Aggregation
        start_time = time.time()
        w_glob = Semi_FedAvg(w_clusters)
        end_time = time.time()
        fed_time = end_time-start_time
        delta_time = delta_time + (max(comp_time) + fed_time)
        net_glob.load_state_dict(w_glob)
        # 保存每一轮的model
        # torch.save(net_glob.state_dict(),'./semi-fed/logmodel/1/model_'+'%d'%iter+'.pkl')
        
        # test
        net_glob.eval()
        # acc_train, loss_train = test_img(net_glob.to(args.device), dataset_train, args)
        acc_test, loss_test = test_img(net_glob.to(args.device), dataset_test, args)
        if acc_test >= acc_threshold:
            duration.append(delta_time)
            duration.append(iter+1)
            print("duration: {}s".format(duration[0]))
            text_save('./log/sfl/Time_{}_iid{}.csv'.format(args.dataset, args.iid),duration)
            break
        if should_stop == False:
            if (acc_test>acc_best):
                stop_cnt = 0
                acc_best = acc_test
                net_best = copy.deepcopy(net_glob)
            else:
                stop_cnt += 1
            if stop_cnt >= patience:
                should_stop = True
                net_glob = copy.deepcopy(net_best)
                acc_test = acc_best
                step_size = step_size * 0.1
                should_stop = False
                stop_cnt = 0
                print("Early stopping at round:{}, mutate learning rate".format(iter))

        print('Round:{}  --loss:{:.2f}  --acc:{:.2f}%'.format(iter, loss_avg, acc_test))
        loss_train_lst.append(loss_avg)
        acc_test_lst.append(acc_test)
    # text_save('./log/sfl/AveLoss_{}_c{}_iid{}.csv'.format(args.dataset, args.pattern, args.iid),loss_train_lst)
    # text_save('./log/sfl/TestAcc_{}_c{}_iid{}.csv'.format(args.dataset, args.pattern, args.iid),acc_test_lst)
