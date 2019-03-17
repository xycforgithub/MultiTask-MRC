'''
High level of model for training and prediction
Adapted from original SAN
Author: yichongx@cs.cmu.edu
'''


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
import math
from collections import defaultdict
import random

from torch.optim.lr_scheduler import *
from torch.autograd import Variable
from my_utils.utils import AverageMeter
from .dreader import DNetwork
from .my_optim import EMA

logger = logging.getLogger(__name__)

class DocReaderModel(object):

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

    def __init__(self, opt, embedding=None, state_dict=None):
        self.opt = opt
        self.updates = state_dict['updates'] if state_dict and 'updates' in state_dict else 0
        self.eval_embed_transfer = True
        self.train_loss = AverageMeter()
        if state_dict and 'train_loss' in state_dict:
            self.train_loss.load_state_dict(state_dict['train_loss'])

        self.network = DNetwork(opt, embedding)
        self.forward_network = nn.DataParallel(self.network) if opt['multi_gpu'] else self.network
        self.state_dict = state_dict

        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if opt['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, opt['learning_rate'],
                                       momentum=opt['momentum'],
                                       weight_decay=opt['weight_decay'])
        elif opt['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          opt['learning_rate'],
                                          weight_decay=opt['weight_decay'])
        elif opt['optimizer'] == 'adam':
            self.optimizer = optim.Adam(parameters,
                                        opt['learning_rate'],
                                        weight_decay=opt['weight_decay'])
        elif opt['optimizer'] == 'adadelta':
            self.optimizer = optim.Adadelta(parameters,
                                            opt['learning_rate'],
                                            rho=0.95)
        else:
            raise RuntimeError('Unsupported optimizer: %s' % opt['optimizer'])


        if opt['fix_embeddings']:
            wvec_size = 0
        else:
            wvec_size = (opt['vocab_size'] - opt['tune_partial']) * opt['embedding_dim']
        if opt.get('have_lr_scheduler', False):
            if opt.get('scheduler_type', 'rop') == 'rop':
                self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=opt['lr_gamma'], patience=3)
            elif opt.get('scheduler_type', 'rop') == 'exp':
                self.scheduler = ExponentioalLR(self.optimizer, gamma=opt.get('lr_gamma', 0.5))
            else:
                milestones = [int(step) for step in opt.get('multi_step_lr', '10,20,30').split(',')]
                self.scheduler = MultiStepLR(self.optimizer, milestones=milestones, gamma=opt.get('lr_gamma'))
        else:
            self.scheduler = None
        self.total_param = sum([p.nelement() for p in parameters]) - wvec_size



    def update(self, batch, name_map, dataset_name):
        self.network.train()
        pred = self.forward_network(*batch[:name_map['input_len']], dataset_name=dataset_name)
        if dataset_name=='wdw':
            if self.opt['cuda']:
                y = Variable(batch[name_map['truth']].cuda(async=True))
                score = Variable(batch[name_map['score']].cuda(async=True))
            else:
                y = Variable(batch[name_map['truth']])
                score = Variable(batch[name_map['score']])
            loss = F.nll_loss(pred, y, reduction='none')
        else:
            if self.opt['cuda']:
                y = Variable(batch[name_map['start']].cuda(async=True)), Variable(batch[name_map['end']].cuda(async=True))
                score = Variable(batch[name_map['score']].cuda(async=True))
            else:
                y = Variable(batch[name_map['start']]), Variable(batch[name_map['end']])
                score = Variable(batch[name_map['score']])
            start,end = pred
            loss = F.cross_entropy(start, y[0], reduce=False) + F.cross_entropy(end, y[1], reduce=False)

        if self.opt['uncertainty_loss']:
            loss = loss * torch.exp(-self.network.log_uncertainty[dataset_name])/2+self.network.log_uncertainty[dataset_name]/2
        loss = torch.mean(loss * score)

        if self.opt['elmo_l2']>0:
            loss += self.network.elmo_l2norm()*self.opt['elmo_l2']
        self.train_loss.update(loss.item(), len(score))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(),self.opt['grad_clipping'])
        self.optimizer.step()
        if self.opt['ema']:
            self.ema.update()
        self.updates += 1
        self.reset_embeddings()
        self.eval_embed_transfer = True
        self.para_swapped=False
    def eval(self):
        if self.opt['ema']:
            self.ema.swap_parameters()
            self.para_swapped=True
    def train(self):
        if self.para_swapped:
            self.ema.swap_parameters()
            self.para_swapped=False
    def predict(self, batch, name_map, top_k=1, dataset_name='squad'):
        self.network.eval()
        if self.eval_embed_transfer:
            self.update_eval_embed()
            self.eval_embed_transfer = False
        self.network.drop_emb = False
        pred = self.forward_network(*batch[:name_map['input_len']], dataset_name=dataset_name)

        if dataset_name=='wdw':
            probs = pred.cpu()
            predictions = torch.max(probs,dim=1)[1].tolist()
            return (predictions, probs.tolist())
        else:
            start, end = pred
            if name_map['valid_size']!=-1:
                valid_size = name_map['valid_size']
                start = start[:valid_size,:]
                end = end[:valid_size,:]
            else:
                valid_size = len(batch[name_map['text']])

            start = F.softmax(start, dim=1)
            end = F.softmax(end, dim=1)
            start = start.data.cpu()
            end = end.data.cpu()
            text = batch[name_map['text']]
            spans = batch[name_map['span']]
            predictions = []
            best_scores = []

            if 'marco' in dataset_name:
                max_len = self.opt['marco_max_len'] or start.size(1)
            else:
                max_len = self.opt['max_len'] or start.size(1)
            doc_len = start.size(1)
            pos_enc = self.position_encoding(doc_len, max_len)
            for i in range(start.size(0)):
                scores = torch.ger(start[i], end[i])
                scores = scores * pos_enc
                scores.triu_()
                scores = scores.numpy()
                best_idx = np.argpartition(scores, -top_k, axis=None)[-top_k]
                best_score = np.partition(scores, -top_k, axis=None)[-top_k]
                s_idx, e_idx = np.unravel_index(best_idx, scores.shape)
                s_offset, e_offset = spans[i][s_idx][0], spans[i][e_idx][1]
                predictions.append(text[i][s_offset:e_offset])
                best_scores.append(best_score)

            start_scores_list = start.tolist()
            end_scores_list = end.tolist()

            return (predictions, best_scores, start_scores_list, end_scores_list)

    def setup_eval_embed(self, eval_embed, padding_idx = 0):
        self.network.lexicon_encoder.eval_embed = nn.Embedding(eval_embed.size(0),
                                               eval_embed.size(1),
                                               padding_idx = padding_idx)
        self.network.lexicon_encoder.eval_embed.weight.data = eval_embed
        for p in self.network.lexicon_encoder.eval_embed.parameters():
            p.requires_grad = False
        self.eval_embed_transfer = True

        if self.opt['covec_on']:
            self.network.lexicon_encoder.ContextualEmbed.setup_eval_embed(eval_embed)

    def update_eval_embed(self):
        if self.opt['tune_partial'] > 0:
            offset = self.opt['tune_partial']
            self.network.lexicon_encoder.eval_embed.weight.data[0:offset,:] \
                = self.network.lexicon_encoder.embedding.weight.data[0:offset,:]

    def reset_embeddings(self):
        if self.opt['tune_partial'] > 0:
            offset = self.opt['tune_partial']
            if offset < self.network.lexicon_encoder.embedding.weight.data.size(0):
                self.network.lexicon_encoder.embedding.weight.data[offset:,:] \
                    = self.network.lexicon_encoder.fixed_embedding

    def save(self, filename, epoch, best_em_score, best_f1_score):
        # strip cove
        network_state = dict([(k, v) for k, v in self.network.state_dict().items() if k[0:4] != 'CoVe' and '_elmo_lstm' not in k])
        if 'eval_embed.weight' in network_state:
            del network_state['eval_embed.weight']
        if 'lexicon_encoder.fixed_embedding' in network_state:
            del network_state['lexicon_encoder.fixed_embedding']
        params = {
            'state_dict': {'network': network_state,
                           'optimizer':self.optimizer.state_dict(),
                           'train_loss':self.train_loss.state_dict(),
                           'updates':self.updates,
                           'ema':self.ema.state_dict()},
            'config': self.opt,
            'random_state': random.getstate(),
            'torch_state': torch.random.get_rng_state(),
            'torch_cuda_state': torch.cuda.get_rng_state(),
            'epoch':epoch,
            'best_em_score':best_em_score,
            'best_f1_score':best_f1_score
        }
        if self.scheduler:
            params['scheduler_state']=self.scheduler.state_dict()
        for try_id in range(10):
            try:
                torch.save(params, filename)
                break
            except Exception as e:
                print('save failed. error:',e)


        logger.info('model saved to {}'.format(filename))

    def cuda(self):
        self.network.cuda()
        ema_state=None
        if self.state_dict:
            new_state = set(self.network.state_dict().keys())
            for k in list(self.state_dict['network'].keys()):
                if k not in new_state:
                    print('key dropped:',k)
                    del self.state_dict['network'][k]
            for k, v in list(self.network.state_dict().items()):
                if k not in self.state_dict['network']:
                    self.state_dict['network'][k] = v
            self.network.load_state_dict(self.state_dict['network'])
            if self.opt['tune_partial'] > 0:
                offset = self.opt['tune_partial']
                self.network.lexicon_encoder.embedding.weight.data[0:offset,:] \
                    = self.state_dict['network']['lexicon_encoder.embedding.weight'][0:offset,:]
            if 'optimizer' in self.state_dict and not self.opt['not_resume_optimizer']:
                self.optimizer.load_state_dict(self.state_dict['optimizer'])
            ema_state = self.state_dict['ema']
        if self.opt['ema']:
            self.ema = EMA(self.opt['ema_gamma'],self.network, ema_state)

    def position_encoding(self, m, threshold=4):
        encoding = np.ones((m, m), dtype=np.float32)
        for i in range(m):
            for j in range(i, m):
                if j - i > threshold:
                    encoding[i][j] = float(1.0 / math.log(j - i + 1))
        return torch.from_numpy(encoding)
