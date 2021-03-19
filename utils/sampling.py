'''
 * @Author: zhikunch 
 * @Date: 2021-03-19 20:39:28 
 * @Last Modified by: zhikunch
 * @Last Modified time: 2021-03-19 20:43:39
 '''
# divide whole dataset into several shards with i.i.d or non-i.i.d manners

import random
import numpy as np
from torchvision import datasets, transforms
from utils.clustering import run_leach

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users) # length 长度为60000
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False)) # sample num_items from all_idxs, using uniform distribution
        all_idxs = list(set(all_idxs) - dict_users[i]) # 不放回抽样
    return dict_users


def mnist_noniid2(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs= 200, 300  # num_shards * num_imgs = 60000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    print(idxs) # 排序后的数据
    idxs_label = idxs_labels[1,:]
    print(idxs_label)

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

# 按排序后的标签将训练数据依次分给100个用户-MNIST
def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs= 100, 600  # num_shards * num_imgs = 60000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(dataset.targets)
    
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    print(idxs_labels)
    # divide and assign
    for i in range(num_users):
        rand_set = set([idx_shard[0]])
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            #print(idxs_labels[1,rand*num_imgs:(rand+1)*num_imgs])
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

# 按排序后的标签将训练数据依次分给100个用户-CIFAR10
def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 100, 500 # num_shards * num_imgs = 50000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels)) # 按垂直方向堆叠形成新的array
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()] # 将labels按序排列
    print(idxs_labels)
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        # rand_set = set([idx_shard[0]])
        # idx_shard = list(set(idx_shard) - rand_set)
        # for rand in rand_set:
            #print(idxs_labels[1,rand*num_imgs:(rand+1)*num_imgs])
        rand = idx_shard[i]
        dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def cifar_noniid2(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250 # num_shards * num_imgs = 50000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels)) # 按垂直方向堆叠形成新的array
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()] # 将labels按序排列
    print(idxs_labels)
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            #print(idxs_labels[1,rand*num_imgs:(rand+1)*num_imgs])
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

# 每个簇内的用户覆盖10种类别标签的训练数据，簇内imbalance，簇间iid
def cluster_iid(dict_user, cluster_num):
    cluster = {i:{} for i in range(cluster_num)}
    user_list = list(range(len(dict_user)))
    for i in range(cluster_num):
        tmp = {}
        # 将用户0，10，20，...，90作为第一个簇内用户; 1, 11, 21, ..., 91作为第二个簇内用户
        idxs = [j for j in range(i,len(dict_user),10)] 
        for idx in idxs:
            tmp[idx] = dict_user[idx]
        cluster[i] = tmp
    return cluster

# clustering followes LEACH
def cluster_leach(dict_user, cluster_num):
    dict_clusters = run_leach(cluster_num) # {0:[ch,cn1,cn2,...],1:[ch,cn1,cn2,...],...}
    cluster = {i:{} for i in range(len(dict_clusters))}
    i=0
    for val in dict_clusters.values():
        tmp = {}
        for idx in val:
            tmp[idx] = dict_user[idx]
        cluster[i] = tmp
        i+=1
    return cluster

# randomly grouping users into diff clusters
def cluster_random(dict_user, cluster_num):
    cluster = {i:{} for i in range(cluster_num)}
    idx_list = list(range(100))
    random.shuffle(idx_list)
    m = int(len(idx_list)/cluster_num)
    for i in range(cluster_num):
        rand_set = set(random.sample(idx_list,m))
        idx_list = list(set(idx_list)-rand_set)
        tmp = {}
        for idx in rand_set:
            tmp[idx] = dict_user[idx]
        cluster[i] = tmp
    return cluster

# for N, K tests
def cluster_rand(dict_user, cluster_num, user_num):
    """
    randomly divide all users into certain clusters
    params: cluster_num: number of clusters
    params: user_num: number of users
    return: cluster: dict of clusters-->{1:{dict_users{...}},2:{dict_users{...}},3:{dict_users{...}}}
    """
    cluster = {i:{} for i in range(cluster_num)}
    idx_list = list(range(user_num))
    random.shuffle(idx_list)
    m = int(len(idx_list)/cluster_num)
    for i in range(cluster_num):
        rand_set = set(random.sample(idx_list,m))
        idx_list = list(set(idx_list)-rand_set)
        tmp = {}
        for idx in rand_set:
            tmp[idx] = dict_user[idx]
        cluster[i] = tmp
    return cluster


# 每个簇里的用户仅拥有同一类标签，簇内balance，簇间non-iid
def cluster_noniid_1(dict_user, cluster_num):
    cluster = {i:{} for i in range(cluster_num)}
    user_list = list(range(len(dict_user)))
    for i in range(cluster_num):
        tmp = {}
         # 将0-9，10-19，...，90-99分别作为10个簇内的用户，
         # 每个簇内的用户拥有同一类别标签的训练数据
        idxs = [j for j in range(i*10,(i+1)*10)]
        for idx in idxs:
            tmp[idx] = dict_user[idx]
        cluster[i] = tmp
    return cluster

# 每个簇里的用户仅包含 2类标签，簇内 balance，簇间 non-iid
def cluster_noniid_2(dict_user, cluster_num):
    cluster = {i:{} for i in range(cluster_num)}
    user_list = list(range(len(dict_user)))
    for i in range(cluster_num):
        tmp = {}
         # 将[0,1],[1,2],[2,3],...,[8,9],[9,0]分别作为10个簇内的用户所含标签，
         # 每个簇内的用户拥有2种类别标签的训练数据
        if i < (cluster_num - 1):
            idxs = [j for j in range(i*10+5,(i+1)*10+5)]
        else:
            idxs = [j for j in range(i*10+5,(i+1)*10)] + [j for j in range(0,5)]
        # idxs = [j for j in range(i*10,(i+1)*10)]
        for idx in idxs:
            tmp[idx] = dict_user[idx]
        cluster[i] = tmp
    return cluster

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


if __name__ == '__main__':
    # dataset_train_mnist = datasets.MNIST('../data/mnist/', train=True, download=True,
    #                                transform=transforms.Compose([
    #                                    transforms.ToTensor(),
    #                                    transforms.Normalize((0.1307,), (0.3081,))
    #                                ]))
    # num = 100
    # d = mnist_noniid(dataset_train_mnist, num)
    dataset_train_cifar = datasets.CIFAR10('../data/cifar/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    e = cifar_noniid(dataset_train_cifar, num)
    # print(d,e)
    # clu = cluster_iid(e, 10)
    # clu = cluster_leach(e)
    clu = cluster_random(e,10)
    # print(clu)
    for i in range(len(clu)):
        print(clu[i].keys())