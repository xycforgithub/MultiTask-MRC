'''
Highway layers and multitask modules
Author: yichongx@cs.cmu.edu
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from allennlp.modules.elmo import Elmo, batch_to_ids
import pickle

test_outputs=False
class PositionwiseNN(nn.Module):
    def __init__(self, idim, hdim, dropout=None):
        super(PositionwiseNN, self).__init__()
        self.w_0 = nn.Conv1d(idim, hdim, 1)
        self.w_1 = nn.Conv1d(hdim, hdim, 1)
        self.dropout = dropout

    def forward(self, x):
        output = F.relu(self.w_0(x.transpose(1, 2)))
        output = self.dropout(output)
        output = self.w_1(output)
        output = self.dropout(output).transpose(2, 1)
        return output

class HighwayLayer(nn.Module):
    def __init__(self, dim, target_dim=None, dropout=None, father=None):
        super(HighwayLayer,self).__init__()
        if target_dim is None:
            target_dim = dim
            self.linear_transform=False
        else:
            self.target_dim=target_dim
            self.linear_transform=True
        self.transform = nn.Conv1d(dim,target_dim, 1)
        self.gate = nn.Conv1d(dim, target_dim, 1)
        if self.linear_transform:
            self.linear=nn.Conv1d(dim, target_dim, 1)
        self.dropout = dropout
        self.father = [father]
    def forward(self,x):
        tx=x.transpose(1,2)

        gate=F.sigmoid(self.gate(tx))
        trans = F.relu(self.transform(tx))
        if self.linear_transform:
            linear = self.linear(tx)
        else:
            linear = tx
        res=(gate * trans + linear).transpose(2,1)
        if self.dropout:
            res=self.dropout(res)
        if test_outputs:
            print('test_outputs=', test_outputs)
            gate_cpu=gate.cpu().detach().numpy()
            with open('output_gate_{}.pt'.format(self.father[0].name),'wb') as f:
                pickle.dump(gate_cpu,f)
                print('written:output_gate_{}.pt'.format(self.father[0].name))

        return res

class GateLayer(nn.Module):
    def __init__(self, dim, target_dim=None, dropout=None):
        super(GateLayer,self).__init__()
        if target_dim is None:
            target_dim = dim
            self.linear_transform=False
        else:
            self.target_dim=target_dim
            self.linear_transform=True
        self.gate = nn.Conv1d(dim, target_dim, 1)
        if self.linear_transform:
            self.linear=nn.Conv1d(dim, target_dim, 1)
        self.dropout=dropout
    def forward(self,x):
        tx=x.transpose(1,2)

        gate=F.sigmoid(self.gate(tx))
        if self.linear_transform:
            linear = self.linear(tx)
        else:
            linear = tx
        res = (gate * linear).transpose(2,1)
        if self.dropout:
            res=self.dropout(res)
        return res
        
class HighwayNetwork(nn.Module):

    def __init__(self, dim, target_dim=None, num_layers=1, size_format='shrink_first', dropout=None):
        super(HighwayNetwork, self).__init__()

        infered_dim = dim if target_dim is None else target_dim

        module_list=[]

        if size_format =='shrink_first':
            module_list.append(HighwayLayer(dim, target_dim, dropout=dropout, father=self))
            for i in range(1, num_layers):
                module_list.append(HighwayLayer(infered_dim, None, dropout=dropout, father=self))
            self.comp=nn.Sequential(*module_list)

        elif size_format=="keep_first":
            for i in range(0, num_layers-1):
                module_list.append(HighwayLayer(dim, None, dropout=dropout))
            module_list.append(HighwayLayer(dim, target_dim, dropout=dropout))
            self.comp=nn.Sequential(*module_list)
        self.dropout=dropout
        self.name=None

    def forward(self,x):
        return self.comp(x)

class GateNetwork(nn.Module):

    def __init__(self, dim, target_dim=None, num_layers=1, size_format='shrink_first', dropout=None):
        super(GateNetwork, self).__init__()

        infered_dim = dim if target_dim is None else target_dim

        module_list=[]

        if size_format =='shrink_first':
            module_list.append(GateLayer(dim, target_dim, dropout=dropout))
            for i in range(1, num_layers):
                module_list.append(GateLayer(infered_dim, None, dropout=dropout))
            self.comp=nn.Sequential(*module_list)

        elif size_format=="keep_first":
            for i in range(0, num_layers-1):
                module_list.append(GateLayer(dim, None, dropout=dropout))
            module_list.append(GateLayer(dim, target_dim, dropout=dropout))
            self.comp=nn.Sequential(*module_list)

    def forward(self,x):
        return self.comp(x)

class MultiDatasetWrapper(nn.Module):
    def __init__(self, opt):
        super(MultiDatasetWrapper, self).__init__()
        self.layer_set = {'-1' : None}
        self.opt = opt

    def add_layer(self, specific_name, layertype, *args, **kwargs):
        for dataset in self.opt['train_datasets']:
            id_layer = self.opt['dataset_configs'][dataset][specific_name]
            if id_layer not in self.layer_set:
                self.layer_set[id_layer] = layertype(*args, **kwargs)
                self.layer_set[id_layer].name=specific_name+'_'+dataset
            self.__setattr__(specific_name+'_'+dataset, self.layer_set[id_layer])

    def forward(self, specific_name, dataset, *args):
        try:
            current_setup = self.__getattr__(specific_name+'_'+dataset)
        except:
            current_setup = self.__getattribute__(specific_name+'_'+dataset)
        if current_setup:
            return current_setup(*args)
        else:
            return args[0]


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-4):
        super(LayerNorm, self).__init__()
        self.alpha = Parameter(torch.ones(1,1,hidden_size)) # gain g
        self.beta = Parameter(torch.zeros(1,1,hidden_size)) # bias b
        self.eps = eps

    def forward(self, x):
        mu = torch.mean(x, 2, keepdim=True).expand_as(x)
        sigma = torch.std(x, 2, keepdim=True).expand_as(x)
        return (x - mu) / (sigma + self.eps) * self.alpha.expand_as(x) + self.beta.expand_as(x)


very_small_number=1e-40

class AttnSum(nn.Module):
    """Attention Sum Layer as in Kadlec et. al (2016):

    Optionally don't normalize output weights.
    """
    def __init__(self, x_size, y_size, identity=False):
        super(AttnSum, self).__init__()
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask, candidate_aggre):
        """
        x = batch * len * h1
        y = batch * h2
        x_ans_mask = batch * len
        candidate_aggre = batch * len * c
        """
        x_ans_mask = candidate_aggre.sum(dim=2).ge(0).float()
        Wy = self.linear(y) if self.linear is not None else y # batch * h1
        p = torch.bmm(x,Wy.unsqueeze(2)).squeeze(2) # batch * len
        p.data.masked_fill_(x_mask.data, -float('inf')) 
        pm = F.softmax(p, dim=1) * x_ans_mask # batch * len
        unnormalized_probs=torch.bmm(pm.unsqueeze(1), candidate_aggre).squeeze(1) # batch * c

        normalized_probs=unnormalized_probs/unnormalized_probs.sum(dim=1, keepdim=True)+very_small_number
        if self.training:
            return torch.log(normalized_probs)
        else:
            return normalized_probs
