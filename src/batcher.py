import os
import sys
import json
import torch
import random
import string
import logging
import numpy as np
import pickle as pkl
from shutil import copyfile
from allennlp.modules.elmo import batch_to_ids

PAD_ID = 0
UNK_ID = 1
STA_ID = 2
END_ID = 3

def load_meta(opt, meta_path):
    with open(meta_path, 'rb') as f:
        meta = pkl.load(f)
    embedding = torch.Tensor(meta['embedding'])
    opt['pos_vocab_size'] = len(meta['vocab_tag'])
    opt['ner_vocab_size'] = len(meta['vocab_ner'])
    opt['vocab_size'] = len(meta['vocab'])
    opt['vocab']=meta['vocab']
    return embedding, opt

class BatchGen:
    def __init__(self, data_path, batch_size, gpu, is_train=True, doc_maxlen=1300, dataset_name = 'squad', drop_less=False,
                 num_gpu=1, dropout_w=0.0, dw_type=0, extra_score=None, extra_score_cap = 1.0):
        self.batch_size = batch_size
        self.doc_maxlen = doc_maxlen
        self.is_train = is_train
        self.gpu = gpu
        self.data_path = data_path
        if 'squad' in dataset_name or 'newsqa' in dataset_name or 'marco' in dataset_name:
            self.span_mode=True
        else:
            self.span_mode=False
        self.data = self.load(self.data_path, is_train, doc_maxlen, drop_less)
        if is_train:
            indices = list(range(len(self.data)))
            random.shuffle(indices)
            data = [self.data[i] for i in indices]
        data = [self.data[i:i + batch_size] for i in range(0, len(self.data), batch_size)]
        self.data = data
        self.ordered_data = data
        self.offset = 0
        self.dataset_name = dataset_name
        self.doc_maxlen = doc_maxlen if is_train else None
        self.num_gpu = num_gpu
        self.dropout_w = dropout_w
        self.dw_type=dw_type
        self.extra_score = extra_score
        if extra_score is not None:
            for k in self.extra_score:
                self.extra_score[k] = min(self.extra_score[k]/extra_score_cap, 1.0)

    def load(self, path, is_train=True, doc_maxlen=1300, drop_less = False):
        with open(path, 'r', encoding='utf-8') as reader:
            # filter
            data = []
            cnt = 0
            answer_counter=0
            sum_answer_tokens=0
            for line in reader:
                sample = json.loads(line)
                if self.is_train:
                    sample['context']=None
                    sample['span'] = None
                cnt += 1
                if is_train:
                    if self.span_mode and (sample['start'] is None or sample['end'] is None):
                        continue
                    if (not self.span_mode) and (not sample['has_answer']):
                        continue
                    answer_counter+=1
                    if len(sample['doc_tok']) > doc_maxlen:
                        if not drop_less:
                            continue
                        if self.span_mode:
                            if sample['start']>doc_maxlen-100 or sample['end']>doc_maxlen-100:
                                continue
                        else:
                            max_c_tid = max([max(indices) for indices in sample['choice_indices']])
                            if max_c_tid>doc_maxlen-100:
                                continue
                        sample['doc_tok']=sample['doc_tok'][:doc_maxlen]
                        sample['doc_pos']=sample['doc_pos'][:doc_maxlen]
                        sample['doc_ner']=sample['doc_ner'][:doc_maxlen]
                        sample['doc_text']=sample['doc_text'][:doc_maxlen]

                        if self.span_mode:
                            sample['span']=sample['span'][:doc_maxlen] if sample['span'] is not None else None
                if not self.span_mode and sample['choice_indices'] is None:
                    continue # There is one ambiguous question in test set, we remove it

                if isinstance(sample['doc_fea'],str):
                    sample['doc_fea'] = np.array(eval(sample['doc_fea']), dtype=np.float32)
                    sample['query_fea'] = np.array(eval(sample['query_fea']), dtype=np.float32)
                else:
                    sample['doc_fea'] = np.array(sample['doc_fea'], dtype=np.float32)
                    sample['query_fea'] = np.array(sample['query_fea'], dtype=np.float32)

                data.append(sample)
            print('Loaded {} samples out of {}'.format(len(data), cnt))
            return data


    def reset(self):
        if self.is_train:
            indices = list(range(len(self.data)))
            random.shuffle(indices)
            self.data = [self.ordered_data[i] for i in indices]
            # self.last_indices=indices
        self.offset = 0

    def __len__(self):
        return len(self.data)

    def patch(self, v):
        v = v.cuda(async=True)
        return v

    def __random_select__(self, arr):
        if self.dropout_w > 0:
            if self.dw_type > 0:
                ids = list(set(arr))
                ids_size = len(ids)
                random.shuffle(ids)
                ids = set(ids[:int(ids_size * self.dropout_w)])
                return [UNK_ID if e in ids else e for e in arr]
            else:
                return [UNK_ID if random.uniform(0, 1) < self.dropout_w else e for e in arr]
        else: return arr

    def __iter__(self):
        while self.offset < len(self):
            batch = self.data[self.offset]
            batch_size = len(batch)
            if batch_size < self.num_gpu**2:
                valid_size = batch_size
                real_batch = batch
                while len(batch) % self.num_gpu !=0:
                    # batch+=real_batch
                    batch.append(batch[0])

                batch_size = len(batch)
                real_valid_size = valid_size
                print('small batch size: gave special treatment.')
            else:
                valid_size=-1
                real_valid_size = len(batch)

            batch_dict = {}

            doc_len = max(len(x['doc_tok']) for x in batch)
            # feature vector
            feature_len = len(batch[0]['doc_fea'][0]) if len(batch[0].get('doc_fea', [])) > 0 else 0
            doc_id = torch.LongTensor(batch_size, doc_len).fill_(0)
            doc_tag = torch.LongTensor(batch_size, doc_len).fill_(0)
            doc_ent = torch.LongTensor(batch_size, doc_len).fill_(0)
            doc_feature = torch.Tensor(batch_size, doc_len, feature_len).fill_(0)
            doc_text = []

            query_len = max(len(x['query_tok']) for x in batch)
            query_id = torch.LongTensor(batch_size, query_len).fill_(0)
            query_tag = torch.LongTensor(batch_size, query_len).fill_(0)
            query_ent = torch.LongTensor(batch_size, query_len).fill_(0)
            query_feature = torch.Tensor(batch_size, query_len, feature_len).fill_(0)
            query_text = []

            for i, sample in enumerate(batch):
                select_len = min(len(sample['doc_tok']), doc_len)
                doc_id[i, :select_len] = torch.LongTensor(self.__random_select__(sample['doc_tok'][:select_len]))
                doc_tag[i, :select_len] = torch.LongTensor(sample['doc_pos'][:select_len])
                doc_ent[i, :select_len] = torch.LongTensor(sample['doc_ner'][:select_len])
                doc_text.append(sample['doc_text'])
                for j, feature in enumerate(sample['doc_fea']):
                    if self.doc_maxlen and j>=self.doc_maxlen:
                        break
                    doc_feature[i, j, :] = torch.Tensor(feature)
                # parse query
                select_len = min(len(sample['query_tok']), query_len)
                query_id[i, :len(sample['query_tok'])] = torch.LongTensor(self.__random_select__(sample['query_tok'][:select_len]))
                query_tag[i, :select_len] = torch.LongTensor(sample['query_pos'][:select_len])
                query_ent[i, :select_len] = torch.LongTensor(sample['query_ner'][:select_len])
                query_text.append(sample['query_text'])
                for j, feature in enumerate(sample['query_fea']):
                    query_feature[i, j, :] = torch.Tensor(feature)

            doc_mask = torch.eq(doc_id, 0)
            query_mask = torch.eq(query_id, 0)

            b_doc_tok = doc_id
            b_doc_pos = doc_tag
            b_doc_ner = doc_ent
            b_doc_fea = doc_feature
            b_doc_text = batch_to_ids(doc_text)
            b_query_tok = query_id
            b_query_pos = query_tag
            b_query_ner = query_ent
            b_query_fea = query_feature
            b_doc_mask = doc_mask
            b_query_mask = query_mask
            b_query_text = batch_to_ids(query_text)


            batch_list = [b_doc_tok, b_doc_pos, b_doc_ner, b_doc_fea,
                   b_doc_text, b_query_tok, b_query_pos, b_query_ner,
                   b_query_fea, b_doc_mask, b_query_mask,
                   b_query_text]

            batch_name_ids = {
                'valid_size': valid_size,
                'doc_tok': 0,
                'doc_pos': 1,
                'doc_ner': 2,
                'doc_fea': 3,
                'doc_text': 4,
                'query_tok': 5,
                'query_pos': 6,
                'query_ner': 7,
                'query_fea': 8,
                'doc_mask': 9,
                'query_mask': 10,
                'query_text': 11,
                'input_len': 12 # length of input to core reader
            }

            if not self.span_mode:
                choice_num = max(len(sample['choice_indices']) for sample in batch)
                choice_aggre = np.zeros((batch_size, doc_len, choice_num))
                for i, sample in enumerate(batch):
                    for j, ans_word in enumerate(sample['choice_indices']):
                        for word_idx in ans_word:
                            choice_aggre[i,word_idx,j]=1
                choice_aggre=torch.Tensor(choice_aggre)
                batch_list.append(choice_aggre)
                batch_name_ids['input_len']=13
            if self.extra_score is not None:
                score = [sample['score']*self.extra_score[str(sample['uid'])] for sample in batch]
            else:
                score = [sample['score'] for sample in batch]
            if valid_size>0:
                for idx in range(valid_size,batch_size):
                    score[idx]=0

            if not self.span_mode:
                b_score = torch.FloatTensor(score)
                b_truth = torch.LongTensor([sample['correct_answer'] for sample in batch])
                batch_list += [b_truth, b_score]
                batch_name_ids['truth'] = batch_name_ids['input_len']
                batch_name_ids['score'] = batch_name_ids['input_len']+1
                current_len = batch_name_ids['input_len']+2
            elif self.is_train:
                start = [sample['start'] for sample in batch]
                end = [sample['end'] for sample in batch]
                b_start = torch.LongTensor(start)
                b_end = torch.LongTensor(end)
                b_score = torch.FloatTensor(score)
                batch_list += [b_start, b_end, b_score]
                batch_name_ids['start'] = batch_name_ids['input_len']
                batch_name_ids['end'] = batch_name_ids['input_len']+1
                batch_name_ids['score'] = batch_name_ids['input_len']+2
                current_len = batch_name_ids['input_len']+3
            else:
                current_len = batch_name_ids['input_len']


            if self.gpu:
                for i, item in enumerate(batch_list):
                    batch_list[i] = self.patch(item.pin_memory())


            b_text = [sample['context'] for sample in batch]
            b_span = [sample['span'] for sample in batch] if self.span_mode else None
            b_uids = [sample['uid'] for sample in batch]

            b_text = b_text[:real_valid_size]
            b_span = b_span[:real_valid_size] if self.span_mode else None
            b_uids = b_uids[:real_valid_size]
            self.offset += 1

            batch_name_ids['text'] = current_len
            batch_name_ids['span'] = current_len+1
            batch_name_ids['uids'] = current_len+2


            yield batch_list + [b_text, b_span, b_uids], batch_name_ids
