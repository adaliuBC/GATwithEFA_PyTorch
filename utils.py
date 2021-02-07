import numpy as np
import scipy.sparse as sp
import torch
import os
import pdb
import pickle


def encode_onehot(labels):
    classes = set(labels)  #创建不重复元素集
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot
    
def encode_onehot_fcinkml(label, label_type=0):
    label_onehot = []
    for i in label:
        pos = []
        for j in range(label_type):
            pos.append(0)
        pos[(int(i))%label_type] = 1
        label_onehot.append(pos)
        del pos
    return np.array(label_onehot)

def modetostr(mode):
    if mode==0:
        return "train"
    elif mode==1:
        return "valid"
    elif mode==2:
        return "test"
    else:
        return NULL;

def load_data_cora(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    pdb.set_trace()

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])  #one_hot encode labels
    
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    #features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def file_list(dirname, ext=''):
    """获取目录下所有特定后缀的文件
    @param dirname: str 目录的完整路径
    @param ext: str 后缀名, 必须以点号开头
    @return: list(str) 所有子文件名(不包含路径)组成的列表
    """
    return list(filter(
        lambda filename: os.path.splitext(filename)[1] == ext,
        os.listdir(dirname)))

def load_data_yun(path="./data/", dataset="fcinkml", chart_num=[0,0,0], feat_num = 0, edge_feat_num=0, label_type = 0):
    """Load flowchart dataset (fc only for now)"""
    #mode = "train"/"valid"/"test"
    print('Loading {} dataset...'.format(dataset))
    #print('Only one document now')
    pos_list = []
    feats = []
    labels = []
    adjs = []
    edge_feats = []
    #all feats and adjs save in above lists, lists' len = train+valid+test data num = 200+48+171
    
    dir_train = path + dataset + "_yun/train/"
    dir_valid = path + dataset + "_yun/valid/"
    dir_test = path + dataset + "_yun/test/"
    nodefeat_filename_train = file_list(dirname = dir_train, ext = ".stroke_feature")
    nodefeat_filename_valid = file_list(dirname = dir_valid, ext = ".stroke_feature")
    nodefeat_filename_test = file_list(dirname = dir_test, ext = ".stroke_feature")
    nodefeat_filename_train.sort()
    nodefeat_filename_valid.sort()
    nodefeat_filename_test.sort()
    nodefeat_filename = nodefeat_filename_train + nodefeat_filename_valid + nodefeat_filename_test  #stroke features
    nodelabel_filename_train = file_list(dirname = dir_train, ext = ".label")
    nodelabel_filename_valid = file_list(dirname = dir_valid, ext = ".label")
    nodelabel_filename_test = file_list(dirname = dir_test, ext = ".label")
    nodelabel_filename_train.sort()
    nodelabel_filename_valid.sort()
    nodelabel_filename_test.sort()
    nodelabel_filename = nodelabel_filename_train + nodelabel_filename_valid + nodelabel_filename_test  #stroke labels
    edge_filename_train = file_list(dirname = dir_train, ext = ".edge")
    edge_filename_valid = file_list(dirname = dir_valid, ext = ".edge")
    edge_filename_test = file_list(dirname = dir_test, ext = ".edge")
    edge_filename_train.sort()
    edge_filename_valid.sort()
    edge_filename_test.sort()
    edge_filename = edge_filename_train + edge_filename_valid + edge_filename_test  #edges
    edgefeat_filename_train = file_list(dirname = dir_train, ext = ".binary_feature")
    edgefeat_filename_valid = file_list(dirname = dir_valid, ext = ".binary_feature")
    edgefeat_filename_test = file_list(dirname = dir_test, ext = ".binary_feature")
    edgefeat_filename_train.sort()
    edgefeat_filename_valid.sort()
    edgefeat_filename_test.sort()
    edgefeat_filename = edgefeat_filename_train + edgefeat_filename_valid + edgefeat_filename_test  #edge features
    
    
    #pkl_file = open("D:/ada/大学/毕设/pyGAT-master/data/FC_B_yun/train/" + "writer000_fc_001.inkml.pkl", 'rb')
    #data1 = pickle.load(pkl_file)
    #print(data1)

    if chart_num[0]==0 or feat_num==0 or edge_feat_num==0:
        print("ERROR: Chart num / feat_num == 0!")
        return;
    for mode in range(3): #train, valid, test
        if mode == 0:
            base = 0
        elif mode == 1:
            base = chart_num[0]
        else: #mode = 2
            base = chart_num[0] + chart_num[1]
        for i in range(chart_num[mode]):  #for each document in tr/v/te set
            #read node_feats(stroke number*23)
            feat_str = np.genfromtxt("{}{}_yun/{}/{}".format(path, dataset, modetostr(mode), str(nodefeat_filename[base+i])), dtype=np.dtype(str))
            #"./data/fcinkml/train/file0_feats.txt"
            feat = np.zeros((len(feat_str), feat_num), dtype=float)  #stroke_num * 23
            for j in range(len(feat_str)):  #for each stroke
                for k in range(feat_num):  #for each feat
                    pos = feat_str[j,k]
                    feat[j, k] = float(pos)  
            feats.append(feat)  #append a feat mat in the list of feats
            del feat
            del pos_list[0:-1]
            
            #read stroke labels(stroke num*1-->stroke num*7(one-hot))
            label = []
            label_str = np.genfromtxt("{}{}_yun/{}/{}".format(path, dataset, modetostr(mode), str(nodelabel_filename[base+i])), dtype=np.dtype(str))
            label = encode_onehot_fcinkml(label_str, label_type)
            labels.append(label)  #add a int list of label in the list of labels
            del pos_list[0:-1]
            
            # build graph(adj = stroke num*stroke num)
            edge_str = np.genfromtxt("{}{}_yun/{}/{}".format(path, dataset, modetostr(mode), str(edge_filename[base+i])), dtype=np.int32)
            edge = []
            for j in range(len(edge_str)):  #for each edge
                edge_s = int(edge_str[j][0])
                edge_e = int(edge_str[j][1])
                edge.append([edge_s, edge_e])  #edge[[0,1], [1,2]]
            edge = np.array(edge).reshape(-1, 2)
            adj = sp.coo_matrix((np.ones(edge.shape[0]), (edge[:, 0], edge[:, 1])), shape=(label.shape[0], label.shape[0]), dtype=np.int32)
            # build symmetric adjacency matrix
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            adjs.append(adj)  #add a adj mat to the list of adjs
            del label
            del edge
            del pos_list[0:-1]
            
            #read edge features(edge num * 19)
            edge_feat = np.zeros((len(feat_str), len(feat_str), edge_feat_num), dtype=float)
            edge_feat_str = np.genfromtxt("{}{}_yun/{}/{}".format(path, dataset, modetostr(mode), str(edgefeat_filename[base+i])), dtype=np.float32)
            edge_feat_map_str = np.genfromtxt("{}{}_yun/{}/{}".format(path, dataset, modetostr(mode), str(edge_filename[base+i])), dtype=np.int32)
            assert(len(edge_feat_str)==len(edge_feat_map_str))
            for j in range(len(edge_feat_str)):
                #use map to fit edge and feat
                edge_s = int(edge_feat_map_str[j][0])
                edge_e = int(edge_feat_map_str[j][1])
                for k in range(edge_feat_num):
                    edge_feat[edge_s][edge_e][k] = float(edge_feat_str[j][k])
                    #change to: coo_matrix?
            edge_feats.append(edge_feat)
            del edge_feat
        #pdb.set_trace()
    
    
    for i in range(len(feats)):
        #feats[i] = normalize_features(feats[i])
        adjs[i] = normalize_adj(adjs[i] + sp.eye(adjs[i].shape[0])) 
        feats[i] = torch.FloatTensor(np.array(feats[i]))
        edge_feats[i] = torch.FloatTensor(np.array(edge_feats[i]))
        labels[i] = torch.LongTensor(np.where(labels[i])[1])
        adjs[i] = torch.FloatTensor(np.array(adjs[i].todense()))
        
    #pdb.set_trace()
    
    return adjs, feats, edge_feats, labels, nodelabel_filename

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)    

def accuracy_class(output, labels):
    preds = output.max(1)[1].type_as(labels)  #取向量中最大的作为预测结果的label
    correct = preds.eq(labels).double()  #求所有正确预测的位置
    correct_cnt = correct.sum()  #求正确预测总数
    type_num = 7
    data_cnt = []
    data_cor = []
    data_wrong = []
    for i in range(7):
        cnt = labels.eq(i)
        data_cnt.append(cnt.sum())
        cor = [correct[i] and cnt[i] for i in range(len(labels))]
        data_cor.append(cor.count(1))
        base = labels.eq(i).double()
        wrong_row = []
        for j in range(7):
            mask = preds.eq(j).double()
            output = [base[k] and mask[k] for k in range(len(labels))]
            wrong_row.append(output.count(1))
        data_wrong.append(wrong_row)  #i行j列是实际为i但认为是j的数据实例
    return correct_cnt / len(labels), data_cor, data_cnt, data_wrong

def combine_batch(features, edge_feats, adjs, labels, chart_num, mode, batch_size):
    if mode=='train':
        mode_base = 0
        num = chart_num[0]
    elif mode=='valid':
        mode_base = chart_num[0]
        num = chart_num[1]
    else:
        print("ERROR: no such mode! mode should be 'train' or 'valid'!")
    
    set_node_feats = []
    set_edge_feats = []
    set_adjs = []
    set_labels = []
    for i in range(num//batch_size+1):  #for each batch
        b_node_feats = []
        b_edge_feats = []
        b_adjs = []
        b_labels = []
        base = []
        node_num=0
        #pdb.set_trace()
        for j in range(batch_size):  #for each fc in a batch
            order = batch_size*i+j
            #print(i, j, order)
            b_node_feats.append(features[order%num])
            b_adjs.append(adjs[order%num])
            b_edge_feats.append(edge_feats[order%num])
            b_labels.append(labels[order%num])
            base.append(node_num)
            node_num += features[order%num].shape[0]
        base.append(node_num)
        b_node_feat = torch.cat(b_node_feats, dim=0)  #?????
        b_label = torch.cat(b_labels, dim=0)
        assert node_num==b_node_feat.shape[0]
        assert node_num==b_label.shape[0]
        b_adj = torch.zeros(size=(node_num, node_num))
        for j in range(len(b_adjs)):  #for each fc
            b_adj[base[j]:base[j+1],base[j]:base[j+1]] = b_adjs[j]
        edge_feat_num = edge_feats[order%num].shape[2]  #edge_num*edge_num*19
        b_edge_feat = torch.zeros(size=(node_num, node_num, edge_feat_num))
        
        #pdb.set_trace()
    
        for j in range(len(b_edge_feats)):  #for each fc
            b_edge_feat[base[j]:base[j+1],base[j]:base[j+1]] = b_edge_feats[j]
        set_node_feats.append(b_node_feat)
        set_edge_feats.append(b_edge_feat)
        set_adjs.append(b_adj)
        set_labels.append(b_label)
    
    return set_node_feats, set_edge_feats, set_adjs, set_labels