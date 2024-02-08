# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 20:58:25 2023

@author: Haiyang Jiang
"""

from networks import DAT_Net
from torch import tensor
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import argparse
import os
from torch.utils.data import random_split, WeightedRandomSampler
import time
from sklearn.metrics import roc_auc_score
import numpy as np


def get_labels_numbers(dataset):
    I,J = 0,0
    for data in dataset:
        #print(data.y.shape)
        if data.y==1:
            I+=1
        if data.y==0:
            J+=1
    print("@@@@@@@@@@@@@@@@")
    print("number of y=1:", I)
    print("number of y=0:", J)

def labels_weights(dataset):
    I,J = 0,0
    for data in dataset:
        if data.y==1:
            #print("label y=1")
            I+=1
        if data.y==0:
            #print("label y=0")
            J+=1
    return (I+J)/I,(I+J)/J
def get_sample_weight(dataset):
    weight = [labels_weights(dataset)[1],labels_weights(dataset)[0]]
    sample_weight = [weight[t.y] for t in dataset]
    return sample_weight

patience = 0

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777,
                    help='seed')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.005,  #0.0005
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay')
parser.add_argument('--nhid', type=int, default=128,
                    help='hidden size')
parser.add_argument('--dropout', type=float, default=0.,
                    help='dropout ratio on GNN')
parser.add_argument('--dropout_ratio1', type=float, default=0.1,
                    help='dropout ratio on GNN')
parser.add_argument('--dropout_ratio2', type=float, default=0.3, 
                    help='dropout ratio on MLP')
parser.add_argument('--pooling_ratio', type=float, default=0.5, 
                    help='dropout ratio on MLP')
parser.add_argument('--dataset', type=str, default='DD',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity/MOLT-4H/SF-295H')
parser.add_argument('--epochs', type=int, default=1000,
                    help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=50,
                    help='patience for earlystopping')
parser.add_argument('--K', type=int, default=2)
parser.add_argument('--alfa', type=float, default=0.8)
parser.add_argument('--random', type=str, default='uniform',
                    help='uniform, norm, poisson')
parser.add_argument('--runs', type=int, default=5)
parser.add_argument('--heads', type=int, default=4)
parser.add_argument('--model', type=str, default='DAT',
                    help='SAGPool, DAT, DAGNN')



def test(model,loader):
    model.eval()
    correct = 0.
    loss = 0.
    i=0
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out,data.y,reduction='sum').item()
    return correct / (len(loader.dataset)-i),loss / (len(loader.dataset)-i)

def train(model):
    
    val_list = []
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    min_loss = 1e10
    patience = 0
    train_time = []
    for epoch in range(args.epochs):
        time_0 = time.time()
        model.train()
        for i, data in enumerate(train_loader):
            data = data.to(args.device)
            out = model(data)
            
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        time_1 = time.time()
        train_time.append(time_1-time_0)
        
        val_acc,val_loss = test(model,val_loader)
        val_list.append(val_acc)
        #print("Validation loss:{}\taccuracy:{}".format(val_loss,val_acc))
        if val_loss < min_loss:
            torch.save(model.state_dict(),'latest2.pth')
            #print("Model saved at epoch{}".format(epoch))
            test_acc,test_loss = test(model,test_loader)
            #print("################Test accuarcy:{}".format(test_acc))
            min_loss = val_loss
            patience = 0
        else:
            patience += 1
        if patience > args.patience:
            #print("Early stop!")
            break 
        # 
    return test_acc, epoch, train_time
        
def ROC_AUC(model,loader):
    model.eval()
    True_ = []
    Pred = []
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        #print("out", out.shape)
        true = data.y
        Pred.append(np.array(out.cpu().detach()).reshape(-1)[-1])
        True_.append(np.array(true.cpu().detach()))
    Pred = np.array(Pred)
    True_ = np.array(True_)    
    #print(Pred.shape, True_.shape)
    auc = roc_auc_score(True_, Pred)
    #print('AUC', auc)
    return auc
    

args = parser.parse_args()
args.device = 'cpu'
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:0'


dataset = TUDataset(os.path.join('data',args.dataset),name=args.dataset)
args.num_classes = dataset.num_classes
args.num_features = dataset.num_features
print(args)
num_training = int(len(dataset)*0.8)
num_val = int(len(dataset)*0.1)
num_test = len(dataset) - (num_training+num_val)
training_set,validation_set,test_set = random_split(dataset,[num_training,num_val,num_test])

train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validation_set,batch_size=args.batch_size,shuffle=False)
test_loader = DataLoader(test_set,batch_size=1,shuffle=False)
model2 = DAT_Net(args).to(args.device)

print(model2)

t1 = time.time()
model2.reset_parameters()
test_acc, epoch, train_time = train(model2)
auc = ROC_AUC(model2, test_loader)
print("################Test accuarcy:{:.4f}, AUC:{:.3f}".
      format(test_acc*100, auc))


