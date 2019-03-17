'''
MT-SAN model, modified from original SAN
'''

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .recurrent import OneLayerBRNN, ContextualEmbed
from .dropout_wrapper import DropoutWrapper
from .encoder import LexiconEncoder
from .similarity import DeepAttentionWrapper, FlatSimilarityWrapper, SelfAttnWrapper
from .similarity import AttentionWrapper
from .san import SAN
from .sub_layers import HighwayNetwork
from .sub_layers import GateNetwork
from .sub_layers import MultiDatasetWrapper, AttnSum
# from my_utils.utils import maybe_append, maybe_cat

class DNetwork(nn.Module):
    """Network for SAN doc reader."""


    def __init__(self, opt, embedding=None, padding_idx=0):
        super(DNetwork, self).__init__()
        my_dropout = DropoutWrapper(opt['dropout_p'], opt['vb_dropout'])
        self.dropout = my_dropout

        self.highway_dropout = DropoutWrapper(opt['highway_dropout'], opt['vb_dropout']) if opt['highway_dropout']>0 else None

        self.lexicon_encoder = LexiconEncoder(opt, embedding=embedding, dropout=my_dropout, highway_dropout = self.highway_dropout)
        query_input_size = self.lexicon_encoder.query_input_size
        doc_input_size = self.lexicon_encoder.doc_input_size
        self.add_elmo=opt['add_elmo']
        elmo_size = self.lexicon_encoder.elmo_output_size if self.add_elmo else 0

        covec_size = self.lexicon_encoder.covec_size
        embedding_size = self.lexicon_encoder.embedding_dim

        self.opt = opt

        # share net
        contextual_share = opt.get('contextual_encoder_share', False)
        prefix = 'contextual'
        doc_encoder_low_input_size = doc_input_size + covec_size + (opt['elmo_configs']['lower_input']>0) * elmo_size
        self.doc_encoder_low = OneLayerBRNN(doc_encoder_low_input_size, opt['contextual_hidden_size'], prefix=prefix, opt=opt, dropout=my_dropout)
        doc_encoder_high_input_size = self.doc_encoder_low.output_size + covec_size + (opt['elmo_configs']['higher_input']>0) * elmo_size
        self.doc_encoder_high = OneLayerBRNN(doc_encoder_high_input_size, opt['contextual_hidden_size'], prefix=prefix, opt=opt, dropout=my_dropout)
        if contextual_share:
            self.query_encoder_low = self.doc_encoder_low
            self.query_encoder_high = self.doc_encoder_high
        else:
            query_encoder_low_input_size = query_input_size + covec_size + (opt['elmo_configs']['lower_input']>0) * elmo_size
            self.query_encoder_low = OneLayerBRNN(query_encoder_low_input_size, opt['contextual_hidden_size'], prefix=prefix, opt=opt, dropout=my_dropout)
            query_encoder_high_input_size = self.query_encoder_low.output_size + covec_size + (opt['elmo_configs']['higher_input']>0) * elmo_size
            self.query_encoder_high = OneLayerBRNN(query_encoder_high_input_size, opt['contextual_hidden_size'], prefix=prefix, opt=opt, dropout=my_dropout)


        highway_type = HighwayNetwork if opt['highway_type']=='highway' else GateNetwork

        self.encoder_low_highway = MultiDatasetWrapper(opt)
        self.encoder_low_highway.add_layer('query_encoder_low_highway', 
            highway_type, self.query_encoder_low.output_size, num_layers = opt['highway_num'], dropout = self.highway_dropout)
        self.encoder_low_highway.add_layer('doc_encoder_low_highway', 
            highway_type, self.doc_encoder_low.output_size, num_layers = opt['highway_num'], dropout = self.highway_dropout)


        self.encoder_high_highway = MultiDatasetWrapper(opt)
        self.encoder_high_highway.add_layer('query_encoder_high_highway', 
            highway_type, self.query_encoder_high.output_size, num_layers = opt['highway_num'], dropout = self.highway_dropout)
        self.encoder_high_highway.add_layer('doc_encoder_high_highway', 
            highway_type, self.doc_encoder_high.output_size, num_layers = opt['highway_num'], dropout = self.highway_dropout)

        doc_encoder_low_size = self.doc_encoder_low.output_size + (opt['elmo_configs']['lower_output']>0) * elmo_size
        query_encoder_low_size = self.query_encoder_low.output_size + (opt['elmo_configs']['lower_output']>0) * elmo_size
        doc_encoder_high_size = self.doc_encoder_high.output_size + (opt['elmo_configs']['higher_output']>0) * elmo_size
        query_encoder_high_size = self.query_encoder_high.output_size + (opt['elmo_configs']['higher_output']>0) * elmo_size
        doc_hidden_size = doc_encoder_low_size + doc_encoder_high_size
        query_hidden_size = query_encoder_low_size + query_encoder_high_size


        self.query_understand = OneLayerBRNN(query_hidden_size, opt['msum_hidden_size'], prefix='msum', opt=opt, dropout=my_dropout)
        doc_attn_size = doc_hidden_size + covec_size + embedding_size + (opt['elmo_configs']['attention']>0) * elmo_size
        query_attn_size = query_hidden_size + covec_size + embedding_size + (opt['elmo_configs']['attention']>0) * elmo_size
        num_layers = 3 + (opt['elmo_configs']['attention']>0)
        doc_list_size = doc_hidden_size + (opt['elmo_configs']['attention']>0) * elmo_size
        query_list_size = query_hidden_size + self.query_understand.output_size + (opt['elmo_configs']['attention']>0) * elmo_size

        prefix = 'deep_att'
        self.deep_attn = DeepAttentionWrapper(doc_attn_size, query_attn_size, num_layers, prefix, opt, my_dropout)

        doc_und_size = doc_list_size + query_list_size
        self.doc_understand = OneLayerBRNN(doc_und_size, opt['msum_hidden_size'], prefix='msum', opt=opt, dropout=my_dropout)
        query_mem_hidden_size = self.query_understand.output_size
        doc_mem_hidden_size = self.doc_understand.output_size

        if opt['self_attention_on']:
            att_size = query_list_size + doc_list_size + self.doc_understand.output_size + covec_size + embedding_size
            self.doc_self_attn = AttentionWrapper(att_size, att_size, prefix='self_att', opt=opt, dropout=my_dropout)
            doc_mem_hidden_size = doc_mem_hidden_size * 2
            self.doc_mem_gen = OneLayerBRNN(doc_mem_hidden_size, opt['msum_hidden_size'], 'msum', opt, my_dropout)
            doc_mem_hidden_size = self.doc_mem_gen.output_size
        # Question merging

        self.mem_highway = MultiDatasetWrapper(opt)
        self.mem_highway.add_layer('query_mem_highway', highway_type, query_mem_hidden_size, dropout = self.highway_dropout)
        self.mem_highway.add_layer('doc_mem_highway', highway_type, doc_mem_hidden_size, dropout = self.highway_dropout)

        self.query_sum_attn = SelfAttnWrapper(query_mem_hidden_size, prefix='query_sum', opt=opt, dropout=my_dropout)

        if 'wdw' in self.opt['train_datasets']:
            print('mem hidden sizes:',doc_mem_hidden_size,query_mem_hidden_size)
            self.wdw_decoder = AttnSum(doc_mem_hidden_size, query_mem_hidden_size)

        if len(self.opt['train_datasets'])>1 or ('wdw' not in self.opt['train_datasets']):
            self.decoder = MultiDatasetWrapper(opt)
            self.decoder.add_layer('SAN', SAN, doc_mem_hidden_size, query_mem_hidden_size, 
                        opt, prefix='decoder', dropout=my_dropout)
        if self.opt['uncertainty_loss']:
            self.log_uncertainty = {dataset_name:nn.Parameter(torch.Tensor(1).fill_(0.0)) for dataset_name in self.opt['train_datasets']} # log sigma^2 values
            self.uncertainty_paras = nn.ParameterList([para for name,para in self.log_uncertainty.items()])

    def elmo_l2norm(self):
        return self.lexicon_encoder.elmo_l2norm()
    
    def reset_elmo_states(self):
        self.lexicon_encoder.elmo.reset_states()
    
    def maybe_append(self,hlist, elmo_repre, name):
        if self.add_elmo and elmo_repre[name] is not None:
            hlist.append(elmo_repre[name])
        return hlist

    def maybe_cat(self,ten1, elmo, name):
        return torch.cat([ten1,elmo[name]], dim=2) if self.add_elmo and elmo[name] is not None else ten1

    def forward(self, *args, dataset_name):
        doc_input, query_input,\
        doc_emb, query_emb,\
        doc_coves, query_coves, \
        doc_mask, query_mask, \
        doc_elmo, query_elmo = self.lexicon_encoder(*args[:12], dataset_name)
        
        query_list, doc_list = [], []
        query_list.append(query_input)
        doc_list.append(doc_input)

        # doc encode
        doc_low_input=self.maybe_append([doc_input, doc_coves[0]], doc_elmo, 'lower_input')
        doc_low = self.doc_encoder_low(torch.cat(doc_low_input, 2), doc_mask)
        doc_low = self.dropout(doc_low)
        doc_high_input = self.maybe_append([doc_low, doc_coves[1]], doc_elmo, 'higher_input')
        doc_high = self.doc_encoder_high(torch.cat(doc_high_input, 2), doc_mask)
        doc_high = self.dropout(doc_high)        
        doc_low = self.encoder_low_highway('doc_encoder_low_highway',dataset_name, doc_low)
        doc_high = self.encoder_high_highway('doc_encoder_high_highway',dataset_name, doc_high)
        doc_low = self.maybe_cat(doc_low, doc_elmo, 'lower_output')
        doc_high = self.maybe_cat(doc_high, doc_elmo, 'higher_output')


        # query
        query_low_input = self.maybe_append([query_input, query_coves[0]], query_elmo, 'lower_input')
        query_low = self.query_encoder_low(torch.cat(query_low_input, 2), query_mask)
        query_low = self.dropout(query_low)
        query_high_input = self.maybe_append([query_low, query_coves[1]], query_elmo, 'higher_input')
        query_high = self.query_encoder_high(torch.cat(query_high_input, 2), query_mask)
        query_high = self.dropout(query_high)
        query_low = self.encoder_low_highway('query_encoder_low_highway',dataset_name, query_low)
        query_high = self.encoder_high_highway('query_encoder_high_highway',dataset_name, query_high)
        query_low = self.maybe_cat(query_low, query_elmo, 'lower_output')
        query_high = self.maybe_cat(query_high, query_elmo, 'higher_output')


        query_mem_hiddens = self.query_understand(torch.cat([query_low, query_high], 2), query_mask)
        query_mem_hiddens = self.dropout(query_mem_hiddens)
        query_list = [query_low, query_high, query_mem_hiddens]
        query_list = self.maybe_append(query_list, query_elmo, 'attention')
        doc_list = [doc_low, doc_high]
        doc_list = self.maybe_append(doc_list, doc_elmo, 'attention')

        query_att_input = self.maybe_append([query_emb, query_coves[2], query_low, query_high], query_elmo, 'attention')
        query_att_input = torch.cat(query_att_input, 2)
        doc_att_input = self.maybe_append([doc_emb, doc_coves[2], doc_low, doc_high], doc_elmo, 'attention')
        doc_att_input = torch.cat(doc_att_input, 2)
        doc_attn_hiddens = self.deep_attn(doc_att_input, query_att_input, query_list, query_mask)
        doc_attn_hiddens = self.dropout(doc_attn_hiddens)
        doc_mem_hiddens = self.doc_understand(torch.cat([doc_attn_hiddens] + doc_list, 2), doc_mask)
        doc_mem_hiddens = self.dropout(doc_mem_hiddens)
        doc_mem_inputs = [doc_attn_hiddens] + doc_list
        if self.opt['self_attention_on']:
            doc_att = torch.cat(doc_mem_inputs + [doc_mem_hiddens, doc_coves[3], doc_emb], 2)
            doc_self_hiddens = self.doc_self_attn(doc_att, doc_att, doc_mask, x3=doc_mem_hiddens)
            doc_mem = self.doc_mem_gen(torch.cat([doc_mem_hiddens, doc_self_hiddens], 2), doc_mask)
        else:
            doc_mem = doc_mem_hiddens

        doc_mem = self.mem_highway('doc_mem_highway', dataset_name, doc_mem)
        if self.opt['query_mem_highway_pos']=='before':
            query_mem_hiddens = self.mem_highway('query_mem_highway', dataset_name, query_mem_hiddens)

        query_mem = self.query_sum_attn(query_mem_hiddens, query_mask)
        if self.opt['query_mem_highway_pos']=='after':
            query_mem = self.mem_highway('query_mem_highway', dataset_name, query_mem.unsqueeze(1))
            query_mem =query_mem.squeeze(1)
        if dataset_name=='wdw':
            choice_aggre=args[-1]
            pred=self.wdw_decoder(doc_mem, query_mem, doc_mask, choice_aggre)
        else:
            start_scores, end_scores = self.decoder('SAN', dataset_name, doc_mem, query_mem, doc_mask)
            pred = start_scores, end_scores

        return pred