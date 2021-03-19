'''
 * @Author: zhikunch 
 * @Date: 2021-03-19 20:39:28 
 * @Last Modified by: zhikunch
 * @Last Modified time: 2021-03-19 20:43:39
 '''
 # devide users into several clusters following leach protocal
import numpy as np
import matplotlib.pyplot as plt

def dist(v_A, v_B):
    """
    判断两个节点之间的距离
    :param v_A: A 二维向量
    :param v_B: B 二维向量
    :return: 距离
    """
    return np.sqrt(np.power((v_A[0] - v_B[0]), 2) + np.power((v_A[1] - v_B[1]), 2))

def node_factory(N):
    """
    生成N个节点的集合
    :param N:节点的数目
    :param nodes: 节点的集合
    :param selected_flag: 标志，是否被选择为簇首-->初始化为0
    :return: 节点集合nodes=[[x,y],[x,y]...] + 标志 flag
    """
    nodes = []
    selected_flag = []
    for i in range(0, N):
        # 1*1生成[x,y]坐标
        node = [np.random.random(),np.random.random()]
        # print("生成的节点为：",node)
        nodes.append(node)
        # 初始化对应标志为0
        selected_flag.append(0)
    return nodes, selected_flag

def sel_heads(r, nodes, flags):
    """
    根据阈值选取簇头节点
    ：param r: 轮数
    ：param nodes：节点列表
    ：param flags：选择标志
    ：param P：比例因子
    ：return：簇首列表heads，簇成员列表members
    """
    # 阈值函数Tn使用leach计算
    P = 0.1 * (100 / len(nodes))
    Tn = P / (1 - P * (r % (1/P)))
    
    # 簇首列表
    heads = []
    # 成员列表
    members = []
    # 本轮簇首数
    n_head = 0
    
    # 对每个节点生成对应的随机数
    rands = [np.random.random() for _ in range(len(nodes))]
    
    # 遍历随机数列表，选取簇首
    for i in range(len(nodes)):
        # 若此节点未被选择为簇首
        if flags[i] == 0:
            # 随机数低于阈值-->选为簇首
            if rands[i] <= Tn:
                flags[i] = 1
                heads.append(nodes[i])
                n_head += 1
            # 随机数高于阈值
            else:
                members.append(nodes[i])
        # 若此节点已经被选择过
        else:
            members.append(nodes[i])
    
#     print("簇首为：", len(heads),"个")
#     print("簇成员为：",len(members),"个")
    
    return heads, members

def classify(nodes, flag, k=1):
    """
    进行簇分类
    ：param nodes：节点列表
    ：param flag：节点标记
    ：param k：轮数
    ：return：簇分类结果列表 classes[[类1..],[类2..],......] [类1...簇首...簇成员]
    """
    # k轮的集合
    global head_cla
    iter_classes = []
    # 迭代r轮
    for r in range(k):
        # 获取簇首列表，簇成员列表
        heads, members = sel_heads(r, nodes, flag)
        
        # 建立簇类的列表
        classes = [[] for _ in range(len(heads))]
        
        # 将簇首作为首节点添加到聚类列表中
        for i in range(len(heads)):
            classes[i].append(heads[i])
            ch = np.argwhere(np.array(nodes) == heads[i])[0][0]
            print("head:",ch)
        # 簇分类：遍历节点node
        for n in range(len(members)):
            # 选取距离最小的节点
            dist_min = 1
            
            for i in range(len(heads)):
                dist_heads = dist(members[n], heads[i])
                
                # 找到距离最小的簇头对应的heads下标i
                if dist_heads < dist_min:
                    dist_min = dist_heads
                    head_cla = i
            # 0 个簇首的情况
            if dist_min == 1:
                print("本轮没有簇首！")
                break
            # 添加到距离最小的簇首对应的聚类列表中
            classes[head_cla].append(members[n])
            
        iter_classes.append(classes)

        # dict
        dict_clusters = {j:{} for j in range(len(iter_classes[0]))}
        for j in range(len(iter_classes[0])):
            # print(len(iter_classes[0][j]))
            idx = []
            for jj in range(len(iter_classes[0][j])):
                idx.append(np.argwhere(np.array(nodes)==iter_classes[0][j][jj])[0][0])
            dict_clusters[j] = idx
    return iter_classes, dict_clusters

def show_plt(classes):
    """
    显示分类图
    ：param classes：[[类1..],[类2..],....]-->[簇首，成员，成员...]
    : return: 
    """
    fig = plt.figure()
    ax1 = plt.gca()
    
    ax1.set_title('WSN1')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    icon = ['o','*','.','x','+','s']
    color = ['r','b','g','c','y','m']
    
    # 对每个簇分类列表进行show
    for i in range(len(classes)):
        centor = classes[i][0]
        for point in classes[i]:
            ax1.plot([centor[0],point[0]],[centor[1],point[1]], c=color[i % 6], marker=icon[i % 5],alpha=0.4)
    plt.show()

def run_leach(cluster_num):
    """
    1、输入节点数N
    2、node_factory(N)：生成N个节点的列表
    3、classify(nodes, flags, k=10):进行k轮簇分类，flag标记的节点不再成为簇首，返回所有簇
    4、show_plt(classes):迭代每次聚类结果，显示连线图
    return：
    """
    N=100
    # 获取初始节点列表，选择标志列表
    nodes, flags = node_factory(N)
    # 对节点列表进行簇分类，k为迭代轮数
    iter_classes, dict_clusters = classify(nodes, flags, k=1)
    
    # for classes in iter_classes:
    #     show_plt(classes)
    return dict_clusters

if __name__ == '__main__':
    clusters = run_leach()
    # print(len(classes[0]))
    pass