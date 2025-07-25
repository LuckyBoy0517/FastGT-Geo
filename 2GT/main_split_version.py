import argparse
import copy
import os
import random
import random as rd
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from logger import Logger
from parse import parse_method, parser_add_default_args, parser_add_main_args
from torch_geometric.utils import (add_self_loops, remove_self_loops,
                                   to_undirected)

import itertools
import dgl
from math import radians, cos, sin, asin, sqrt
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast,GradScaler

warnings.filterwarnings('ignore')

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    dgl.random.seed(seed)

fix_seed(1024)


def geodistance(lng1, lat1, lng2, lat2):
    # lng1,lat1,lng2,lat2 = (120.12802999999997,30.28708,115.86572000000001,28.7427)
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])  # 经纬度转换成弧度
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000  # 地球平均半径，6371km
    distance = round(distance / 1000, 3)
    return distance

def mediannum(num):
    listnum = [num[i] for i in range(len(num))]
    listnum.sort()
    lnum = len(num)
    if lnum % 2 == 1:
        i = int((lnum + 1) / 2) - 1
        return listnum[i]
    else:
        i = int(lnum / 2) - 1
        return (listnum[i] + listnum[i + 1]) / 2

def get_node_attr(filepath):
    fr1 = open(filepath, 'r')
    node_list = []
    node_attr_list = []
    for line in fr1.readlines():
        str_list = line.strip('\r\n').split(sep=',')
        temp_ip = str(str_list[0])
        temp_nodeID = int(str_list[1])
        temp_node_attr = list(map(eval, str_list[2:7]))
        node_list.append(temp_nodeID)
        node_attr_list.append(temp_node_attr)
    node = np.array(node_list)
    node_attr_array = np.array(node_attr_list)  # .reshape(-1, 1)  # 这是一维的。。转成1x1维
    return node_attr_array

def build_graph(filepath):  # 可以运行的图组建代码 同时输出edge的特征
    # 定义两组np array,一列是头部，一列是end，这个要求我输出两列或者两行列表即可
    fr1 = open(filepath, 'r')
    src_list = []
    dst_list = []  # 不是最终地标 而是某条边的节点
    edge_attr_list = []
    for line in fr1.readlines():
        str_list = line.strip('\r\n').split(sep=',')
        temp_src = int(str_list[0])
        temp_dst = int(str_list[1])
        # temp_attr = float(str_list[2])
        temp_node_attr = list(map(eval, str_list[2:12]))

        src_list.append(temp_src)
        dst_list.append(temp_dst)
        edge_attr_list.append(temp_node_attr)

        # 手工重复一份，图本身还是原来的图，就是不知对其他的操作，比如traintest划分是否有影响
        src_list.append(temp_dst)
        dst_list.append(temp_src)
        edge_attr_list.append(temp_node_attr)

    src = np.array(src_list)
    dst = np.array(dst_list)
    edge_attr_array = np.array(edge_attr_list)  # .reshape(-1, 1)#这是一维的。。转成1x1维

    # Edges are directional in DGL; Make them bi-directional.
    # dgl中是有向的，搞成双向的 才是无向图 但这里涉及到对特征的处理，需要把特征也重复一份，比较麻烦，最好是直接在生成数据的时候，或者读取数据的时候就直接复制一份边和边特征，这里就不用专门再复制一遍了
    # u = np.concatenate([src, dst])#u和v？
    # v = np.concatenate([dst, src])
    u = src  # u和v？
    v = dst
    # Construct a DGLGraph
    return dgl.DGLGraph((u, v)), edge_attr_array  # 图和节点就创建完毕了

def get_min_err(err_list,err_list1):
    min_err = min(err_list)
    ind = err_list.index(min_err)
    min_err1 = err_list1[ind]
    return min_err,min_err1,ind

from sklearn import preprocessing

target_scaler1 = preprocessing.MinMaxScaler()
target_scaler2 = preprocessing.MinMaxScaler()
new_label_lat_array = []
new_label_lon_array = []
label_array = []
target_lat_label_dict = {}
target_lon_label_dict = {}
new_target_lat_label_dict = {}
new_target_lon_label_dict = {}


def train_test_val(filepath_train,filepath_val,filepath_test):
    fr3 = open(filepath_train, 'r', encoding='UTF-8')
    fr4 = open(filepath_val,'r',encoding = 'UTF-8')
    fr5 = open(filepath_test,'r',encoding = 'UTF-8')
    target_lat_label_dict = {}
    target_lon_label_dict = {}
    train_target_lat_label_dict = {}
    train_target_lon_label_dict = {}
    val_target_lat_label_dict = {}
    val_target_lon_label_dict = {}
    test_target_lat_label_dict = {}
    test_target_lon_label_dict = {}


    label_lat_array = []
    label_lon_array = []
    train_nodeid_list = []
    val_nodeid_list = []
    test_nodeid_list = []


    for line in fr3.readlines():
        str_list = line.strip('\r\n').split(sep=',')
        node_id = int(str_list[1])
        city_label = str_list[2]
        lat_label = float(str_list[3])
        lon_label = float(str_list[4])
        label_lat_array.append(lat_label)
        label_lon_array.append(lon_label)
        train_nodeid_list.append(node_id)

        target_lat_label_dict[node_id] = lat_label
        target_lon_label_dict[node_id] = lon_label
        train_target_lat_label_dict[node_id] = lat_label
        train_target_lon_label_dict[node_id] = lon_label

    for line in fr4.readlines():
        str_list = line.strip('\r\n').split(sep=',')
        node_id = int(str_list[1])
        city_label = str_list[2]
        lat_label = float(str_list[3])
        lon_label = float(str_list[4])
        label_lat_array.append(lat_label)
        label_lon_array.append(lon_label)
        val_nodeid_list.append(node_id)

        target_lat_label_dict[node_id] = lat_label
        target_lon_label_dict[node_id] = lon_label
        val_target_lat_label_dict[node_id] = lat_label
        val_target_lon_label_dict[node_id] = lon_label

    
    for line in fr5.readlines():
        str_list = line.strip('\r\n').split(sep=',')
        node_id = int(str_list[1])
        city_label = str_list[2]
        lat_label = float(str_list[3])
        lon_label = float(str_list[4])
        label_lat_array.append(lat_label)
        label_lon_array.append(lon_label)
        test_nodeid_list.append(node_id)

        target_lat_label_dict[node_id] = lat_label
        target_lon_label_dict[node_id] = lon_label
        test_target_lat_label_dict[node_id] = lat_label
        test_target_lon_label_dict[node_id] = lon_label

    new_label_lat_array = target_scaler1.fit_transform(
        np.array(label_lat_array).reshape(-1, 1))  # 对target也进行转换？ 这里按理说不应该对test和val进行转换 但是实际来看对准确率影响不大
    new_label_lon_array = target_scaler2.fit_transform(np.array(label_lon_array).reshape(-1, 1))  # 对target也进行转换？

    new_label_lat_array = new_label_lat_array.astype(np.float32)  # RuntimeError: Found dtype Double but expected Float
    new_label_lon_array = new_label_lon_array.astype(np.float32)  # RuntimeError: Found dtype Double but expected Float
    # http://cache.baiducontent.com/c?m=3YtYZLCCHN6vGjWIo3xGOwnFqMepYZfJFe4NpPAcQQqaBoCGtCOE4F9XEGQvtdqQ0QPw5xNyDN-n131gJkxfH8ug-cwQiNy9uQ3IbNQPtunH2PbzO2EyLU8XCzyb2aoD1skqBHzOdu_NE-A20i80iPe85FRReM2gjfFwcFWV9cY1SUYde09OKJvFVYgjuMP0tgVo2N3g6qMfjiTJmx4Hb-hZ--PsMa5Nnvnq7aMQ_WG&p=8b2a970d87934eac59edce21527a95&newp=aa3bc54ad5c241be0be296291659cd231610db2151d7d7146b82c825d7331b001c3bbfb423291504d9c577600bae4f5aeffb3079370923a3dda5c91d9fb4c5747999&s=cfcd208495d565ef&user=baidu&fm=sc&query=RuntimeError%3A+Found+dtype+Double+but+expected+Float++bakcward&qid=f9ae955f0011259a&p1=5






    i = 0
    for k, item in target_lat_label_dict.items():
        new_target_lat_label_dict[k] = new_label_lat_array[i]
        new_target_lon_label_dict[k] = new_label_lon_array[i]
        i += 1




    # 随机分割一下 目标IP的 train test集合

    train_node_id_list = []  # 7:2:1
    train_node_label_list = []
    lat_train_node_label_list = []
    lon_train_node_label_list = []
    new_lat_train_node_label_list = []
    new_lon_train_node_label_list = []

    val_node_id_list = []
    val_node_label_list = []  # 注意是一一对应的 label和id 这里为了快速验证写的
    lat_val_node_label_list = []  # 注意是一一对应的 label和id 这里为了快速验证写的
    lon_val_node_label_list = []  # 注意是一一对应的 label和id 这里为了快速验证写的
    new_lat_val_node_label_list = []  # 注意是一一对应的 label和id 这里为了快速验证写的
    new_lon_val_node_label_list = []  # 注意是一一对应的 label和id 这里为了快速验证写的

    test_node_id_list = []
    test_node_label_list = []
    lat_test_node_label_list = []
    lon_test_node_label_list = []
    new_lat_test_node_label_list = []
    new_lon_test_node_label_list = []



    for key, item in target_lat_label_dict.items():
        node_id = int(key)
        # node_label = int(item)
        lat_label = target_lat_label_dict[node_id]
        lon_label = target_lon_label_dict[node_id]
        new_lat_label = new_target_lat_label_dict[node_id]
        new_lon_label = new_target_lon_label_dict[node_id]

        if node_id in train_nodeid_list:
            #train_node_id_list.append(node_id)
            lat_train_node_label_list.append(lat_label)
            lon_train_node_label_list.append(lon_label)
            new_lat_train_node_label_list.append(new_lat_label)
            new_lon_train_node_label_list.append(new_lon_label)
        
        if node_id in val_nodeid_list:
            #val_nodeid_list.append(node_id)
            lat_val_node_label_list.append(lat_label)
            lon_val_node_label_list.append(lon_label)
            new_lat_val_node_label_list.append(new_lat_label)
            new_lon_val_node_label_list.append(new_lon_label)

        if node_id in test_nodeid_list:
            #test_node_id_list.append(node_id)
            lat_test_node_label_list.append(lat_label)
            lon_test_node_label_list.append(lon_label)
            new_lat_test_node_label_list.append(new_lat_label)
            new_lon_test_node_label_list.append(new_lon_label)


        """
        rd_number = rd.random()
        if (rd_number < 0.1):
            test_node_id_list.append(node_id)
            # test_node_label_list.append(node_label)
            lat_test_node_label_list.append(lat_label)
            lon_test_node_label_list.append(lon_label)
            new_lat_test_node_label_list.append(new_lat_label)
            new_lon_test_node_label_list.append(new_lon_label)



        elif (rd_number < 0.3) & (rd_number >= 0.1):
            val_node_id_list.append(node_id)
            # val_node_label_list.append(node_label)
            lat_val_node_label_list.append(lat_label)
            lon_val_node_label_list.append(lon_label)
            new_lat_val_node_label_list.append(new_lat_label)
            new_lon_val_node_label_list.append(new_lon_label)

        else:
            train_node_id_list.append(node_id)
            # train_node_label_list.append(node_label)
            lat_train_node_label_list.append(lat_label)
            lon_train_node_label_list.append(lon_label)
            new_lat_train_node_label_list.append(new_lat_label)
            new_lon_train_node_label_list.append(new_lon_label)
        """

    train_labeled_nodes = torch.tensor(train_nodeid_list)
    # train_labels = torch.tensor(train_node_label_list)  # their labels are different
    lat_train_labels = torch.tensor(lat_train_node_label_list)  # their labels are different
    lon_train_labels = torch.tensor(lon_train_node_label_list)  # their labels are different
    new_lat_train_labels = torch.tensor(new_lat_train_node_label_list)  # their labels are different
    new_lon_train_labels = torch.tensor(new_lon_train_node_label_list)  # their labels are different

    val_labeled_nodes = torch.tensor(val_nodeid_list)
    # val_labels = torch.tensor(val_node_label_list)  # their labels are different
    lat_val_labels = torch.tensor(lat_val_node_label_list)  # their labels are different
    lon_val_labels = torch.tensor(lon_val_node_label_list)  # their labels are different
    new_lat_val_labels = torch.tensor(new_lat_val_node_label_list)  # their labels are different
    new_lon_val_labels = torch.tensor(new_lon_val_node_label_list)  # their labels are different

    test_labeled_nodes = torch.tensor(test_nodeid_list)
    # test_labels = torch.tensor(test_node_label_list)  # their labels are different
    lat_test_labels = torch.tensor(lat_test_node_label_list)  # their labels are different
    lon_test_labels = torch.tensor(lon_test_node_label_list)  # their labels are different
    new_lat_test_labels = torch.tensor(new_lat_test_node_label_list)  # their labels are different
    new_lon_test_labels = torch.tensor(new_lon_test_node_label_list)  # their labels are different

    return [train_labeled_nodes, new_lat_train_labels, new_lon_train_labels, val_labeled_nodes, new_lat_val_labels,
            new_lon_val_labels, test_labeled_nodes, new_lat_test_labels, new_lon_test_labels
        , lat_train_labels, lon_train_labels, lat_val_labels, lon_val_labels, lat_test_labels, lon_test_labels]

def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=3):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']
    # acc 如果输入值大于给定值，那么就继续，同时更新给定值为输入值，这个的目的是获得不断上升的数据，dec则相反，如果获得的数值小于给定值，则不断压缩，直到反过来
    # 每次反过来都计数一次，连续几次则中断
    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step,
                                                                    best_value))  # best_value 原来的是log_value 可能适用于找最大值
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
parser_add_default_args(args)
print(args)
#******************一些参数的赋值******************
args = parser.parse_args()
aggregator_type = args.aggregator_type
node_embedding_size = args.node_embedding_size  # [64,32,16,4,1]
node_hidden_size = args.node_hidden_size
edge_hidden_size = args.edge_hidden_size
fc_size = args.fc_size
regs = args.regs

gnn_dropout = args.gnn_dropout
edge_dropout = args.edge_dropout
nn_dropout1 = args.nn_dropout1
nn_dropout2 = args.nn_dropout2
loss1_weight = args.loss1_weight

epochs = args.epochs
lr = args.lr
init_method = args.init_method
print_flag = args.print_flag
num_step_message_passing = args.num_step_message_passing
####注意！！
edge_input_dim=10

filepath = os.path.split(os.path.realpath(__file__))[0]

from time import time

stamp = int(time())
#result_path1 = filepath + '/output_train70_0203/' + str(stamp) + '_loss.txt'  # 详细loss 演化过程

result_path2 = filepath + '/output_train_para_train70_0528/'+str(stamp)+'_ans.txt'

#if not os.path.exists("./output_train70_0210/"):
#    os.makedirs("./output_train70_0210/")


if not os.path.exists("./output_train_para_train70_0528/"):
    os.makedirs("./output_train_para_train70_0528/")

#fw1 = open(result_path1, 'w')

#fw1.write(str(args))

#fw1.flush()

fw2 = open(result_path2, 'w')

fw2.write(str(args))

fw2.flush()

loss_loger, error_loger = [], []
stopping_step = 0

G, edge_attr_array = build_graph(
    filepath + '/data_0201/ny_edge_feature_sim.txt')  # 同时返回图和边特征
node_attr_array = get_node_attr(
    filepath + '/data_0201/ny_ip_feature_sim.txt')  # 新建函数，返回节点特征

if print_flag > 0:
    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges.' % G.number_of_edges())
embed = nn.Embedding(G.number_of_nodes(), node_embedding_size)  # 23216 x 16维 初始化embedding维度是16 我随便定的 是超参 需要调参
G.ndata['feat'] = embed.weight  # 节点初始化

inputs = embed.weight  # 这里额外给模型的inputs再赋值一次各节点的特征

train_labeled_nodes, lat_train_labels, lon_train_labels, \
val_labeled_nodes, lat_val_labels, lon_val_labels, \
test_labeled_nodes, lat_test_labels, lon_test_labels \
    , old_lat_train_labels, old_lon_train_labels, old_lat_val_labels, old_lon_val_labels, old_lat_test_labels, old_lon_test_labels = \
    train_test_val(filepath + '/data_0201/' + "ny_dstip_id_allinfo_sim_train_0.1.txt",filepath + '/data_0201/' + "ny_dstip_id_allinfo_sim_val_0.2.txt",filepath + '/data_0201/' + "ny_dstip_id_allinfo_sim_test_0.7.txt")

all_logits = []
loss_fn = nn.MSELoss()
node_attr_tensor = torch.from_numpy(node_attr_array).to(torch.float32)
edge_attr_tensor = torch.from_numpy(edge_attr_array).to(torch.float32)
node_feat = torch.cat([inputs, node_attr_tensor], dim=1)

#fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)
                          ) if torch.cuda.is_available() else torch.device("cpu")
    G=G.to(device)

if not args.cpu:
    loss_fn = loss_fn.to(device)
    node_feat = node_feat.to(device)
    edge_attr_tensor = edge_attr_tensor.to(device)
    train_labeled_nodes_cpu = train_labeled_nodes
    val_labeled_nodes_cpu = val_labeled_nodes
    test_labeled_nodes_cpu = test_labeled_nodes
    train_labeled_nodes, lat_train_labels, lon_train_labels, val_labeled_nodes, lat_val_labels, lon_val_labels, test_labeled_nodes, lat_test_labels, lon_test_labels = \
            train_labeled_nodes.to(device), lat_train_labels.to(device), lon_train_labels.to(device), \
            val_labeled_nodes.to(device), lat_val_labels.to(device), lon_val_labels.to(device), \
            test_labeled_nodes.to(device), lat_test_labels.to(device), lon_test_labels.to(device)
cur_best_pre_0 = np.inf



####n = dataset.graph['num_nodes']
n = G.num_nodes

d = node_embedding_size + 5#不知道对不对，不知道代不代表节点特征维度，先这么写着

### Load method ###
model = parse_method(args.method, args, d, aggregator_type,edge_input_dim,edge_hidden_size,num_step_message_passing,gnn_dropout, edge_dropout,
               nn_dropout1,device).to(device)

scaler = GradScaler()


### Training loop ###
patience = 0
if args.method == 'ours' and args.use_graph:
    optimizer = torch.optim.Adam([
        {'params': model.params1, 'weight_decay': args.ours_weight_decay},
        {'params': model.params2, 'weight_decay': args.weight_decay},
        #我修改的代码
        {'params': model.params3, 'weight_decay': args.ours_weight_decay},
        {'params':embed.parameters(),'weight_decay':args.ours_weight_decay},
        {'params': model.params4, 'weight_decay': args.ours_weight_decay},
        #{'params': model.params5, 'weight_decay': args.ours_weight_decay},
        #{'params': embed.parameters(),'weight_decay':args.ours_weight_decay},
        #{'params': model.params6,'weight_decay':args.ours_weight_decay},
        {'params': model.params7, 'weight_decay': args.ours_weight_decay},
    ],
        lr=args.lr)
else:
    optimizer = torch.optim.Adam(
        model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

scheduler = ReduceLROnPlateau(optimizer, 'min',factor =0.5,patience = 100)
###最优值
med_err_res = []
ave_err_res = []

med_err_res_eval = []
ave_err_res_eval = []


for epoch in range(epochs):
    model.train()

    logits = model(G, node_feat, edge_attr_tensor,node_feat)  # 模型(图和节点特征)

    y1_predict = logits[:, 0]
    y2_predict = logits[:, 1]

    y1_train_loss = loss_fn(y1_predict[train_labeled_nodes], lat_train_labels.squeeze(-1))
    y2_train_loss = loss_fn(y2_predict[train_labeled_nodes], lon_train_labels.squeeze(-1))

    # total_loss = args.loss1_weight * y1_train_loss + y2_train_loss#額外調一個參數

    total_loss = loss1_weight * y1_train_loss + y2_train_loss
    '''
    这个也不是最优
    纬度变化一度，球面南北方向距离变化：πR/180 ........111.7km
    经度变化一度，球面东西方向距离变化：πR/180*cosB ....111.7*cosB
    比如北京 B = 40、cosB = 0.766，经度变化1度，则东西方向距离变化 85.567km

    lon和lat对误差影响不同，客观上，不同地区，lat和lon作为两个不同的优化目标，难度也不同。alaph可以控制总loss中的权重，也就是优先照顾谁
    最好是自己做个loss函数 这个留给下一步工作吧
    作为一个小创新点       

    此外既然两者不同，两者就不该共享dropout以及其他超参数 比如DNN网络的深度宽度等

    '''

    scheduler.step(total_loss)

    # old_logits = target_scaler.inverse_transform(logits)  # 将y_pedict转换回去原来的代码

    optimizer.zero_grad()  # 固定3步 复制即可
    #scaler.scale(total_loss).backward()
    total_loss.backward()
    #scaler.step(optimizer)
    #scaler.update()
    optimizer.step()

    model.eval()
        # ----------转换为原始lat lon 计算error distance
    temp_y1_predict = y1_predict.detach().cpu().numpy()
    temp_y2_predict = y2_predict.detach().cpu().numpy()
    old_y1_predict = target_scaler1.inverse_transform(temp_y1_predict.reshape(-1, 1))
    old_y2_predict = target_scaler2.inverse_transform(temp_y2_predict.reshape(-1, 1))


    with torch.no_grad():
        if args.cpu:
            model.cpu()
        if 1:  # (epoch%10==1):

            y1_val_loss = loss_fn(y1_predict[val_labeled_nodes], lat_val_labels.squeeze(-1))  # 计算loss
            y2_val_loss = loss_fn(y2_predict[val_labeled_nodes], lon_val_labels.squeeze(-1))
            total_val_loss = loss1_weight * y1_val_loss + y2_val_loss
            # if print_flag > 0:
            #     print('Epoch %d | Loss: %.4f' % (epoch, total_val_loss.item()))#

            sum_dis = 0
            dis_list=[]
            for i, temp in enumerate(old_lat_val_labels):
                # temp_dis = 0
                temp_lat_label = temp.item()
                temp_lon_label = old_lon_val_labels[i].item()
                temp_lat_pre = old_y1_predict[val_labeled_nodes_cpu][i].item()
                temp_lon_pre = old_y2_predict[val_labeled_nodes_cpu][i].item()
                temp_dis = geodistance(temp_lon_label, temp_lat_label, temp_lon_pre, temp_lat_pre)

                sum_dis += temp_dis
                dis_list.append(temp_dis)

            error_thisepoch_val = sum_dis/len(old_lat_val_labels)
            mediannum_dis_val = mediannum(dis_list)

            perf_str = 'ValEpoch:%d:avg_error:%.5f:median_error:%.5f\n' % (epoch, error_thisepoch_val, mediannum_dis_val)
            if print_flag > 0:
                print(perf_str)
            #fw1.write(perf_str)
            #fw1.flush()
            med_err_res_eval.append(mediannum_dis_val)
            ave_err_res_eval.append(error_thisepoch_val)

            cur_best_pre_0, stopping_step, should_stop = early_stopping(error_thisepoch_val, cur_best_pre_0,
                                                                            stopping_step, expected_order='dec',
                                                                            flag_step=500)
            # ---TEST
            sum_dis = 0
            dis_list=[]
            for i, temp in enumerate(old_lat_test_labels):
                # temp_dis = 0
                temp_lat_label = temp.item()
                temp_lon_label = old_lon_test_labels[i].item()
                temp_lat_pre = old_y1_predict[test_labeled_nodes_cpu][i].item()
                temp_lon_pre = old_y2_predict[test_labeled_nodes_cpu][i].item()
                temp_dis = geodistance(temp_lon_label, temp_lat_label, temp_lon_pre, temp_lat_pre)

                sum_dis += temp_dis
                dis_list.append(temp_dis)

            error_thisepoch = sum_dis/len(old_lat_test_labels)
            mediannum_dis = mediannum(dis_list)
            perf_str = 'TestEpoch:%d:avg_error:%.5f:median_error:%.5f\n' % (epoch, error_thisepoch, mediannum_dis)
            med_err_res.append(mediannum_dis)
            ave_err_res.append(error_thisepoch)
            #fw1.write(perf_str)
            #fw1.flush()
            #for temp in dis_list:
                #fw1.write(str(temp) + ',')
            #fw1.write('\n')
            if print_flag > 0:
                print(perf_str)
            if not args.cpu:
                model.to(device)
            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            if should_stop == True:
                break

min_err_eval , min_err1_eval ,ind= get_min_err(ave_err_res_eval,med_err_res_eval)
min_err , min_err1  = ave_err_res[ind],med_err_res[ind]
#fw1.write('ave_err:'+str(min_err)+'\n')
#fw1.write('med_err:'+str(min_err1)+'\n')

#fw1.write('\n')

print('ave medium err:%.5f'%(min_err))
print('med medium err:%.5f'%(min_err1))
print('eval_ave err:%.5f'%(min_err_eval))
print('eval_med err:%.5f'%(min_err1_eval))

fw2.write("eval"+'\n')
for i in ave_err_res_eval:
    fw2.write(str(i)+' ')

fw2.write('\n')


fw2.write("test:"+"\n")
for i in ave_err_res:
    fw2.write(str(i)+' ')

fw2.write('\n')
fw2.write('eval_ave_err:'+str(min_err_eval)+'\n')
fw2.write('eval_med_err:'+str(min_err1_eval)+'\n')
fw2.write('ave_err:'+str(min_err)+'\n')
fw2.write('med_err:'+str(min_err1)+'\n')

fw2.close()
