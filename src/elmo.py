'''
ELMo Layer
Created August, 2018
Author: yichongx@cs.cmu.edu
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from allennlp.modules.elmo import batch_to_ids
from torch.nn.utils import weight_norm
from src.dropout_wrapper import DropoutWrapper
from .my_elmo import Elmo

class ElmoLayer(nn.Module):
    
    def __init__(self, opt, num_outputs=2):
        super(ElmoLayer, self).__init__()
        self.elmo=Elmo(opt['elmo_options_path'], opt['elmo_weight_path'], num_outputs, dropout=opt['elmo_dropout'])
        # self.elmo_config=json.load(open(opt['elmo_options_path']))
        # self.dim = self.elmo_config['lstm']['projection_dim']

    def forward(self, batch_text):
        # batch_char = batch_to_ids(batch_text)
        return self.elmo(batch_text)['elmo_representations']

    def l2norm(self):
        loss=0
        for scalar_mix in self.elmo._scalar_mixes:
            for para in scalar_mix.scalar_parameters:
                loss+=para.pow(2).sum()*0.5
        return loss

    def reset_states(self):
        self.elmo._elmo_lstm._elmo_lstm.reset_states()

