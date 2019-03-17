'''
Adapted from original SAN
'''

import torch
import math
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from .recurrent import BRNNEncoder, ContextualEmbed
from .dropout_wrapper import DropoutWrapper
from .common import activation
from .similarity import AttentionWrapper
from .sub_layers import PositionwiseNN
from .sub_layers import HighwayNetwork
from .sub_layers import GateNetwork
from .sub_layers import MultiDatasetWrapper
from .elmo import ElmoLayer


class LexiconEncoder(nn.Module):
    def create_embed(self, vocab_size, embed_dim, padding_idx=0):
        return nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

    def create_word_embed(self, embedding=None, opt={}, prefix='wemb'):
        vocab_size = opt.get('vocab_size', 1)
        embed_dim = opt.get('{}_dim'.format(prefix), 300)
        self.embedding = self.create_embed(vocab_size, embed_dim)
        if embedding is not None:
            self.embedding.weight.data = embedding
            if opt['fix_embeddings'] or opt['tune_partial'] == 0:
                opt['fix_embeddings'] = True
                opt['tune_partial'] = 0
                for p in self.embedding.parameters():
                    p.requires_grad = False
            else:
                assert opt['tune_partial'] < embedding.size(0)
                fixed_embedding = embedding[opt['tune_partial']:]
                self.register_buffer('fixed_embedding', fixed_embedding)
                self.fixed_embedding = fixed_embedding
        return embed_dim

    def create_pos_embed(self, opt={}, prefix='pos'):
        vocab_size = opt.get('{}_vocab_size'.format(prefix), 56)
        embed_dim = opt.get('{}_dim'.format(prefix), 12)
        self.pos_embedding = self.create_embed(vocab_size, embed_dim)
        return embed_dim

    def create_ner_embed(self, opt={}, prefix='ner'):
        vocab_size = opt.get('{}_vocab_size'.format(prefix), 19)
        embed_dim = opt.get('{}_dim'.format(prefix), 8)
        self.ner_embedding = self.create_embed(vocab_size, embed_dim)
        return embed_dim

    def create_cove(self, vocab_size, embedding=None, embed_dim=300, 
        padding_idx=0, opt=None, dropout=None):
        self.ContextualEmbed= ContextualEmbed(opt['covec_path'], opt['vocab_size'], embedding=embedding, padding_idx=padding_idx)
        return self.ContextualEmbed.output_size

    def create_prealign(self, x1_dim, x2_dim, opt={}, prefix='prealign'):
        self.prealign = AttentionWrapper(x1_dim, x2_dim, prefix, opt, self.dropout)

    def __init__(self, opt, pwnn_on=True, embedding=None, padding_idx=0, dropout=None, highway_dropout = None):
        super(LexiconEncoder, self).__init__()
        doc_input_size = 0
        que_input_size = 0
        self.dropout = DropoutWrapper(opt['dropout_p']) if dropout == None else dropout
        self.dropout_emb = DropoutWrapper(opt['dropout_emb'])
        self.opt=opt
        # word embedding
        embedding_dim = self.create_word_embed(embedding, opt)
        self.embedding_dim = embedding_dim
        doc_input_size += embedding_dim
        que_input_size += embedding_dim

        elmo_output_size=0
        num_elmo_outputs=0
        self.add_elmo=opt['add_elmo']
        if opt['add_elmo']:
            for key in opt['elmo_configs']:
                if opt['elmo_configs'][key]>0:
                    num_elmo_outputs+=1
            self.elmo = ElmoLayer(opt, num_outputs=num_elmo_outputs)

            elmo_output_size = self.elmo.elmo.get_output_dim()
            if opt['elmo_configs']['linear_transform']:
                doc_input_size+=elmo_output_size
                que_input_size+=elmo_output_size
        self.elmo_output_size = elmo_output_size

        # pre-trained contextual vector
        covec_size = self.create_cove(opt['vocab_size'], embedding, opt=opt, dropout = self.dropout) if opt['covec_on'] else 0
        self.covec_size = covec_size

        highway_type = HighwayNetwork if opt['highway_type']=='highway' else GateNetwork

        self.cove_low_highway = MultiDatasetWrapper(opt)
        self.cove_low_highway.add_layer('cove_low_highway', highway_type, covec_size, num_layers=opt['cove_highway_num'], dropout=highway_dropout)

        self.cove_high_highway = MultiDatasetWrapper(opt)
        self.cove_high_highway.add_layer('cove_high_highway', highway_type, covec_size, num_layers = opt['cove_highway_num'], dropout=highway_dropout)

        prealign_size = 0
        if opt['prealign_on'] and embedding_dim > 0:
            prealign_size = embedding_dim
            self.create_prealign(embedding_dim, embedding_dim, opt)
        self.prealign_size = prealign_size
        pos_size = self.create_pos_embed(opt) if opt['pos_on'] else 0
        ner_size = self.create_ner_embed(opt) if opt['ner_on'] else 0
        feat_size = opt['num_features'] if opt['feat_on'] else 0
        doc_hidden_size = embedding_dim + covec_size + prealign_size + pos_size + ner_size + feat_size
        que_hidden_size = embedding_dim + covec_size
        que_hidden_size += pos_size + ner_size +feat_size
        if opt['add_elmo'] and opt['elmo_configs']['linear_transform']:
            doc_hidden_size+=elmo_output_size
            que_hidden_size+=elmo_output_size

        if opt['prealign_bidi']:
            que_hidden_size += prealign_size
        self.pwnn_on = pwnn_on
        if self.pwnn_on:
            if opt['embed_transform_highway']=='after':
                doc_input_size, que_input_size = opt['pwnn_hidden_size'], opt['pwnn_hidden_size']
                self.doc_pwnn = PositionwiseNN(doc_hidden_size, opt['pwnn_hidden_size'], dropout)
                if doc_hidden_size == que_hidden_size:
                    self.que_pwnn = self.doc_pwnn
                else:
                    self.que_pwnn = PositionwiseNN(que_hidden_size, opt['pwnn_hidden_size'], dropout)
            elif opt['embed_transform_highway']=='combine':
                doc_input_size, que_input_size = opt['pwnn_hidden_size'], opt['pwnn_hidden_size']
                self.input_highway = MultiDatasetWrapper(opt)
                self.input_highway.add_layer('doc_input_highway', HighwayNetwork, doc_hidden_size, 
                    target_dim=doc_input_size, num_layers=2, dropout=highway_dropout)
                self.input_highway.add_layer('query_input_highway', HighwayNetwork, que_hidden_size,
                    target_dim=que_input_size, num_layers=2, dropout=highway_dropout)

        if not self.pwnn_on or opt['embed_transform_highway']=='after':
            self.input_highway = MultiDatasetWrapper(opt)
            self.input_highway.add_layer('query_input_highway', highway_type, que_input_size, num_layers=opt['highway_num'], dropout=highway_dropout)
            self.input_highway.add_layer('doc_input_highway', highway_type, doc_input_size, num_layers=opt['highway_num'], dropout=highway_dropout)

        self.doc_input_size = doc_input_size
        self.query_input_size = que_input_size

    def elmo_l2norm(self):
        return self.elmo.l2norm()

    def transform_cove(self, doc_cove_low, doc_cove_high, query_cove_low, query_cove_high, dataset_name):
        doc_cove_low = self.cove_low_highway('cove_low_highway',dataset_name,doc_cove_low)
        doc_cove_high = self.cove_high_highway('cove_high_highway',dataset_name,doc_cove_high)
        query_cove_low = self.cove_low_highway('cove_low_highway',dataset_name,query_cove_low)
        query_cove_high = self.cove_high_highway('cove_high_highway',dataset_name,query_cove_high)

        return doc_cove_low, doc_cove_high, query_cove_low, query_cove_high

    def forward(self, doc_tok, doc_pos, doc_ner, doc_fea, 
                   doc_char, query_tok, query_pos, query_ner,
                   query_fea, doc_mask, query_mask,
                   query_char, dataset_name):
        drnn_input_list = []
        qrnn_input_list = []
        emb = self.embedding if self.training else self.eval_embed
        doc_emb, query_emb = emb(doc_tok), emb(query_tok)
        if self.opt['dropout_emb'] > 0:
            doc_emb = self.dropout_emb(doc_emb)
            query_emb = self.dropout_emb(query_emb)
        drnn_input_list.append(doc_emb)
        qrnn_input_list.append(query_emb)

        if self.add_elmo:
            doc_elmo_list, query_elmo_list = self.elmo(doc_char), self.elmo(query_char)
            counter=0
            doc_elmo = {}
            query_elmo = {}
            for key in self.opt['elmo_configs']:
                if self.opt['elmo_configs'][key]:
                    doc_elmo[key] = doc_elmo_list[counter]
                    query_elmo[key] = query_elmo_list[counter]
                    counter+=1
                else:
                    doc_elmo[key]=None
                    query_elmo[key]=None
            if self.opt['elmo_configs']['linear_transform']==1:
                doc_elmo_tran = doc_elmo['linear_transform']
                query_elmo_tran = query_elmo['linear_transform']
                drnn_input_list.append(doc_elmo_tran)
                qrnn_input_list.append(query_elmo_tran)
        else:
            doc_elmo=None
            query_elmo=None



        doc_cove_low, doc_cove_high = None, None
        query_cove_low, query_cove_high = None, None
        if self.opt['covec_on']:
            doc_cove_low, doc_cove_high = self.ContextualEmbed(doc_tok, doc_mask)
            query_cove_low, query_cove_high = self.ContextualEmbed(query_tok, query_mask)
            doc_cove_low = self.dropout(doc_cove_low)
            doc_cove_high = self.dropout(doc_cove_high)
            query_cove_low = self.dropout(query_cove_low)
            query_cove_high = self.dropout(query_cove_high)

            if self.opt['cove_highway_position']=='first':
                doc_cove_low, doc_cove_high, query_cove_low, query_cove_high = self.transform_cove(doc_cove_low, 
                                                                        doc_cove_high, query_cove_low, query_cove_high, dataset_name)
            doc_coves = [doc_cove_low, doc_cove_high, doc_cove_high, doc_cove_high]
            query_coves = [query_cove_low, query_cove_high, query_cove_high, query_cove_high]
            drnn_input_list.append(doc_coves[0])
            qrnn_input_list.append(query_coves[0])

        if self.opt['prealign_on']:
            q2d_atten = self.prealign(doc_emb, query_emb, query_mask)
            d2q_atten = self.prealign(query_emb, doc_emb, doc_mask)
            drnn_input_list.append(q2d_atten)
            if self.opt['prealign_bidi']:
                qrnn_input_list.append(d2q_atten)

        if self.opt['pos_on']:
            doc_pos_emb = self.pos_embedding(doc_pos)
            drnn_input_list.append(doc_pos_emb)
            query_pos_emb = self.pos_embedding(query_pos)
            qrnn_input_list.append(query_pos_emb)
        if self.opt['ner_on']:
            doc_ner_emb = self.ner_embedding(doc_ner)
            drnn_input_list.append(doc_ner_emb)
            query_ner_emb = self.ner_embedding(query_ner)
            qrnn_input_list.append(query_ner_emb)

        if self.opt['feat_on']:
            drnn_input_list.append(doc_fea)
            qrnn_input_list.append(query_fea)
        doc_input = torch.cat(drnn_input_list, 2)
        query_input = torch.cat(qrnn_input_list, 2)
        if self.pwnn_on and self.opt['embed_transform_highway']=='after':
            doc_input = self.doc_pwnn(doc_input)
            query_input = self.que_pwnn(query_input)
        doc_input = self.input_highway('doc_input_highway',dataset_name, doc_input)
        query_input = self.input_highway('query_input_highway', dataset_name, query_input)

        doc_input = self.dropout(doc_input)
        query_input = self.dropout(query_input)


        if self.opt['cove_highway_position']=='final':
            doc_cove_low, doc_cove_high, query_cove_low, query_cove_high = self.transform_cove(doc_cove_low, 
                                                                            doc_cove_high, query_cove_low, query_cove_high, dataset_name)
            doc_coves = [doc_cove_low, doc_cove_high, doc_cove_high, doc_cove_high]
            query_coves = [query_cove_low, query_cove_high, query_cove_high, query_cove_high]

        return doc_input, query_input, doc_emb, query_emb, doc_coves, query_coves, doc_mask, query_mask, doc_elmo, query_elmo
