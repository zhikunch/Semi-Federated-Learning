'''
 * @Author: zhikunch 
 * @Date: 2021-03-19 20:39:28 
 * @Last Modified by: zhikunch
 * @Last Modified time: 2021-03-19 20:43:39
 '''
import torch
from torchvision import datasets,transforms
from utils.sampling import *
from models.Nets import *
import pickle

def _Init_(args, num_cluster):
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    acc_threshold = 0
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        acc_threshold = 0.8
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
        acc_threshold = 0.65
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
        acc_threshold = 0.35
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
        clusters = cluster_iid(dict_users, num_cluster) # manual setup分簇
        print("manual setup")
    elif args.pattern == 1:
        clusters = cluster_random(dict_users, num_cluster) # random setup分簇
        print("random setup")
    elif args.pattern == 2:
        clusters = cluster_leach(dict_users, num_cluster) # leach setup分簇
        print("leach setup")
    else:
        exit('No clustering patterns')

    return net_glob,clusters,dataset_train,dataset_test,dict_users, acc_threshold