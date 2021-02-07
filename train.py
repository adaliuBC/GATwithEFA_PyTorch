from __future__ import division
from __future__ import print_function

import os
import sys
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import copy
import pdb

from utils import load_data_cora, load_data_yun, accuracy, accuracy_class, combine_batch
from models import GAT_EFA, GAT, GCN, MLP

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--load_checkpoint', action='store_true', default=False, help='Start training from middle.')
parser.add_argument('--dataset', default='fcinkml', help='The data set we work on')
parser.add_argument('--model', default='GAT', help='The model, GAT or MLP')
parser.add_argument('--seed', type=int, default=31, help='Random seed.')
parser.add_argument('--residual', action='store_true', default=False, help='GAT with residual connection or not.')
parser.add_argument('--nlayer', type=int, default=3, help='Layer num of GAT model.')
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')  #8 attention heads
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--nout_heads', type=int, default=2, help='Number of output head attentions.')
#parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay .')  #0.1
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')  #0.005
parser.add_argument('--batch_size', type=int, default=8, help='Initial batch size.')
parser.add_argument('--val_freq', type=int, default=10, help='Frequency of validation.')  #early stopping not applied now.
parser.add_argument('--patience', type=int, default=200, help='Patience')  #early stopping not applied now.


print("torch.__version__: ", torch.__version__)
args = parser.parse_args()  #解析args
print(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()  #确定cuda状态
fileptr = open('./log.txt', 'w')
fileptr.write(str(args) + '\n')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
#set same seed

if args.dataset == 'fcinkml':
    train_chartnum = 220 #200
    valid_chartnum = 140  #48
    test_chartnum = 171
    chart_num = train_chartnum + valid_chartnum + test_chartnum
    feat_num = 23
    edge_feat_num = 19
    label_type = 7
elif args.dataset == 'fc_b':
    train_chartnum = 280
    valid_chartnum = 196 
    test_chartnum = 196
    chart_num = train_chartnum + valid_chartnum + test_chartnum
    feat_num = 23
    edge_feat_num = 19
    label_type = 7
elif args.dataset == 'fa':
    train_chartnum = 132
    valid_chartnum = 84
    test_chartnum = 84
    chart_num = train_chartnum + valid_chartnum + test_chartnum
    feat_num = 23
    edge_feat_num = 19
    label_type = 7
else:
    print("ERROR: This dataset is not supported now!")
    quit()

#pdb.set_trace()

# Load data
print("Loading data...")
if args.dataset == 'cora':
    adj, features, labels, idx_train, idx_val, idx_test = load_data_cora()
elif args.dataset == 'fcinkml':
    adjs, features, edge_feats, labels, file_names = load_data_yun(dataset='fcinkml', 
                                                       chart_num = [train_chartnum, valid_chartnum, test_chartnum], 
                                                       feat_num = feat_num, edge_feat_num=edge_feat_num, label_type = label_type)
    #./data/fcinkml_yun  data:yun
elif args.dataset == 'fc_b':
    adjs, features, edge_feats, labels, file_names = load_data_yun(dataset='fc_b', 
                                                       chart_num = [train_chartnum, valid_chartnum, test_chartnum], 
                                                       feat_num = feat_num, edge_feat_num=edge_feat_num, label_type = label_type)
elif args.dataset == 'fa':
    adjs, features, edge_feats, labels, file_names = load_data_yun(dataset='fa', 
                                                       chart_num = [train_chartnum, valid_chartnum, test_chartnum], 
                                                       feat_num = feat_num, edge_feat_num=edge_feat_num, label_type = label_type)
else:
    print('ERROR: No such dataset!')
print("Data Loaded")


tr_features, tr_edge_feats, tr_adjs, tr_labels = combine_batch(features, edge_feats, adjs, labels,
                                                 chart_num = [train_chartnum, valid_chartnum, test_chartnum],
                                                 mode='train', batch_size=args.batch_size)  
#data in flowchart-->data in minibatch

# Model and optimizer
print("Creating model...")
if args.model=='GAT_EFA':
    print("Model is GAT_EFA!")
    model = GAT_EFA(nfeat=features[0].shape[1], 
                nedgef = edge_feats[0].shape[2],
                nhid=args.hidden, 
                nclass=label_type,   #7
                dropout=args.dropout,
                nheads=args.nb_heads,
                noutheads=args.nout_heads,
                nlayer = args.nlayer, 
                alpha=args.alpha)
elif args.model=='GAT':
    print("Model is GAT!")
    model = GAT(nfeat=features[0].shape[1], 
                nedgef=edge_feats[0].shape[2],
                nhid=args.hidden, 
                nclass=label_type,   #7
                dropout=args.dropout,
                nheads=args.nb_heads,
                noutheads=args.nout_heads,
                nlayer = args.nlayer, 
                alpha=args.alpha)
elif args.model=='GCN':
    print("Model is GCN!")
    model = GCN(nfeat=features[0].shape[1], 
                nhid=args.hidden, 
                nclass=label_type,   #7
                dropout=args.dropout,
                nlayer = args.nlayer, 
                alpha=args.alpha)
elif args.model=='MLP':
    print("Model is MLP!")
    #MLP model, 23-50-7, cannot change nlayer now
    model = MLP(nfeat=features[0].shape[1],
                nhid=50, 
                nclass=label_type,   #7
                dropout=args.dropout,
                alpha=args.alpha)
else:
    print("ERROR: This model is not supported now!")
    quit()

if args.cuda:
    model = model.cuda()
    for i in range(chart_num):
        features[i] = features[i].cuda()
        labels[i] = labels[i].cuda()
        adjs[i] = adjs[i].cuda()
        edge_feats[i] = edge_feats[i].cuda()    
    for i in range(train_chartnum//args.batch_size+1):
        tr_features[i] = tr_features[i].cuda()
        tr_labels[i] = tr_labels[i].cuda()
        tr_adjs[i] = tr_adjs[i].cuda()
        tr_edge_feats[i] = tr_edge_feats[i].cuda()
for i in range(chart_num):
    features[i] = Variable(features[i])
    labels[i] = Variable(labels[i])
    adjs[i] = Variable(adjs[i])
    edge_feats[i] = Variable(edge_feats[i])
for i in range(train_chartnum//args.batch_size+1):
    tr_features[i] = Variable(tr_features[i])
    tr_labels[i] = Variable(tr_labels[i])
    tr_adjs[i] = Variable(tr_adjs[i])
    tr_edge_feats[i] = Variable(tr_edge_feats[i])

#optim, sched
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
#optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.3)
print("Model created")

x_list = [] #epoch
y_train_list = [] #loss
y_valid_list = [] #loss
#loss fig

def train(epoch):
    #train
    t = time.time()
    model.train()
    #optimizer.zero_grad()
    losst_data = 0
    acct_data = 0
    
    global tr_features
    global tr_edge_feats
    global tr_adjs
    global tr_labels
    
    for i in range(train_chartnum//args.batch_size + 1):  #for each batch
        optimizer.zero_grad()
        output = model(tr_features[i], tr_edge_feats[i], tr_adjs[i])
        loss = F.nll_loss(output, tr_labels[i])
        acc = accuracy(output, tr_labels[i])
        loss.backward()
        optimizer.step()
        losst_data += loss.item()
        acct_data += acc.item()
        sys.stdout.write(str(i) + '/' + str(train_chartnum//args.batch_size + 1)+'\r')
        sys.stdout.flush()
    #optimizer.step()
    losst_data = losst_data/(train_chartnum//args.batch_size + 1)
    acct_data = acct_data/(train_chartnum//args.batch_size + 1)
   
    #valid
    lossv_data = 0 if epoch==0 else copy.deepcopy(y_valid_list[-1])
    accv_data = 0
    if epoch%args.val_freq==0:  #freq set as 1
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        lossv_data = 0
        valid_base = train_chartnum
        for i in range(valid_chartnum):  #for each flowchart in valid set
            output = model(features[valid_base+i], edge_feats[valid_base+i], adjs[valid_base+i])
            loss = F.nll_loss(output, labels[valid_base+i])
            acc = accuracy(output, labels[valid_base+i])
            lossv_data += loss.item()
            accv_data += acc.item()
            del loss
            del acc
        lossv_data = lossv_data/valid_chartnum
        accv_data = accv_data/valid_chartnum
        scheduler.step()
        
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(losst_data),
          'acc_train: {:.4f}'.format(acct_data),
          'loss_val: {:.4f}'.format(lossv_data),
          'acc_val: {:.4f}'.format(accv_data),
          'time: {:.4f}s'.format(time.time() - t), end=' ')
    fileptr.write('Epoch: {:04d}'.format(epoch+1) + 
          'loss_train: {:.4f}'.format(losst_data) + 
          'acc_train: {:.4f}'.format(acct_data) + 
          'time: {:.4f}s'.format(time.time() - t) + 
          'loss_val: {:.4f}'.format(lossv_data) + 
          'acc_val: {:.4f}'.format(accv_data) + '\n')
          
    print('\n')
    x_list.append(epoch)
    y_train_list.append(losst_data)
    y_valid_list.append(lossv_data)
    #save the loss of train & valid, for plt.figure
    return accv_data  #avg loss of valid data

def compute_test():
    model.eval()
    test_base = train_chartnum + valid_chartnum
    losste_data = 0
    accte_data = 0
    cor_class = [0,0,0,0,0,0,0]
    cnt_class = [0,0,0,0,0,0,0]
    wrong_class = [[0,0,0,0,0,0,0], [0,0,0,0,0,0,0], [0,0,0,0,0,0,0], [0,0,0,0,0,0,0], [0,0,0,0,0,0,0], [0,0,0,0,0,0,0], [0,0,0,0,0,0,0]]
    for i in range(test_chartnum):
        output = model(features[test_base+i], edge_feats[test_base + i], adjs[test_base + i])
        loss = F.nll_loss(output, labels[test_base + i])
        acc, cor_c, cnt_c, wrong_c = accuracy_class(output, labels[test_base + i])
        for j in range(7):
            cor_class[j] += cor_c[j]
            cnt_class[j] += cnt_c[j].item()
            for k in range(7):
                wrong_class[j][k] += wrong_c[j][k] 
        losste_data += loss.item()
        accte_data += acc.item()
        #paint
        #print(file_names)
        #if(file_names[test_base+i]=='writer2_1b.inkml.label'):
        pred_list = [[],[],[],[],[],[],[]]
        preds = output.max(1)[1].type_as(labels[test_base + i])
        for j in range(len(preds)):
            pred_list[preds[j]].append(j)
        with open('./pred_fig/{}.txt'.format(file_names[test_base+i]), 'w') as f:
            for j in range(7):
                for k in range(len(pred_list[j])):
                    f.write(str(pred_list[j][k]) + ',')
                f.write('\n')
                
    losste_data = losste_data/test_chartnum
    accte_data = accte_data/test_chartnum
    accte_class = [cor_class[j]/cnt_class[j] for j in range(7)]
    print("Test set results:",
          "loss= {:.4f}".format(losste_data),
          "accuracy= {:.4f}".format(accte_data))
    print("class accuracy: ")
    for i in range(7):
        print("{:.4f} ".format(accte_class[i]))
    print("↓True label →Pred label:")
    for i in range(7):
        for j in range(7):
             print(wrong_class[i][j], end = ' ')
        print("\n")
    
    fileptr.write("Test set results:" + 
          "loss= {:.4f}".format(losste_data) + 
          "accuracy= {:.4f}".format(accte_data) + '\n')
    fileptr.write("Test set class accuracy: " + '\n')
    for i in range(7):
        fileptr.write("{:.4f} ".format(accte_class[i]) + '\n')
    fileptr.write("↓True label →Pred label:" + '\n')
    for i in range(7):
        for j in range(7):
             fileptr.write("{:.0f}".format(wrong_class[i][j]) + ' ')
        fileptr.write('\n')

# Train model
print("Training model...")
t_total = time.time()
acc_values = []
bad_counter = 0
best = 0
best_epoch = 0

#load checkpoint
if args.load_checkpoint:
    if os.path.isfile('./checkpoint/checkpoint.pkl'):
        print("Loading checkpoint ......")
    else:
        print("No checkpoint exists!")
    checkpoint = torch.load('./checkpoint/checkpoint.pkl')
    start_epoch = checkpoint['epoch']
    x_list = checkpoint['x_list']
    y_train_list = checkpoint['y_train_list']
    y_valid_list = checkpoint['y_valid_list']
    best_epoch = checkpoint['best_epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print("Loaded checkpoint")
else:
    start_epoch = 0

for epoch in range(start_epoch, args.epochs):
    
    #rand resort train set each time
    index = [i for i in range(train_chartnum)] 
    index = np.random.permutation(index)
    features_pos = [features[i] for i in index]
    labels_pos = [labels[i] for i in index]
    adjs_pos = [adjs[i] for i in index]
    edge_feats_pos = [edge_feats[i] for i in index]
    tr_features, tr_edge_feats, tr_adjs, tr_labels = combine_batch(features_pos, edge_feats_pos, adjs_pos, labels_pos,
                                                 chart_num = [train_chartnum, valid_chartnum, test_chartnum],
                                                 mode='train', batch_size=args.batch_size)  
                                                 #data in flowchart-->data in minibatch
    del index, features_pos, labels_pos, adjs_pos, edge_feats_pos
    if args.cuda:
        for i in range(train_chartnum//args.batch_size+1):
            tr_features[i] = tr_features[i].cuda()
            tr_labels[i] = tr_labels[i].cuda()
            tr_adjs[i] = tr_adjs[i].cuda()
            tr_edge_feats[i] = tr_edge_feats[i].cuda()
    for i in range(train_chartnum//args.batch_size+1):
        tr_features[i] = Variable(tr_features[i])
        tr_labels[i] = Variable(tr_labels[i])
        tr_adjs[i] = Variable(tr_adjs[i])
        tr_edge_feats[i] = Variable(tr_edge_feats[i])
    
    acc = train(epoch)
    acc_values.append(acc)
    
    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if acc_values[-1] > best:  #如果最新的loss最好，则设为best
        best = acc_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:  #否则，count++
        bad_counter += 1

    if bad_counter == args.patience:
        break  #early_stopping

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)  #del all .pkl before best epoch
            
    torch.save({'epoch': epoch + 1,
        'state_dict': model.state_dict(), 
        'x_list': x_list,
        'y_train_list': y_train_list,
        'y_valid_list': y_valid_list,
        'best_epoch': best_epoch},
        './checkpoint/checkpoint.pkl')

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)
        

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

#figure
print("Painting...")
print("Red is train loss, green is valid loss!")
fig = plt.figure(figsize = (7,5))
pl.plot(x_list, y_train_list,'r-', label = u'train')
pl.plot(x_list, y_valid_list,'g-', label = u'valid')
pl.xlabel(u'iters')
pl.ylabel(u'loss')
pl.show()

print(y_valid_list[0])
# Testing
print("Testing model...")
compute_test()
fileptr.close()
