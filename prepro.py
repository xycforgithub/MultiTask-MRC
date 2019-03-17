import re
import os
import json
import spacy
import unicodedata
import numpy as np
import argparse
import collections
import multiprocessing
import logging
import random
import tqdm
import pickle
from functools import partial
from collections import Counter
from my_utils.tokenizer import Vocabulary, reform_text
from my_utils.word2vec_utils import load_glove_vocab, build_embedding
from my_utils.utils import set_environment
from my_utils.log_wrapper import create_logger
from config import set_args
from multiprocessing import Pool as ThreadPool


"""
This script is to preprocess SQuAD dataset.
TODO: adding multi-thread, use spacy 2.0
"""
test_mode=False
n_threads=72 if not test_mode else 4

NLP = spacy.load('en', disable=['vectors', 'textcat', 'parser'])

def build_vocab(data, glove_vocab=None, sort_all=False, thread=n_threads, clean_on=False, args=None):
    nlp = spacy.load('en', disable=['vectors', 'textcat', 'tagger', 'ner', 'parser'])
    def token(sample, key=None):
        s = sample[key]
        if clean_on:
            s = reform_text(s)
        return [w.text for w in nlp(s) if len(w.text) > 0]
    def extract(data, key=None):
        if clean_on:
            all_doc = [reform_text(sample[key]) for sample in data]
        else:
            all_doc = [sample[key] for sample in data]        
        return all_doc

    logger.info('Collect vocab')

    if sort_all:
        counter = Counter()
        token_ = partial(token, key='context')
        for sample in tqdm.tqdm(data, total=len(data)):
            counter.update(token_(sample))
        token_ = partial(token, key='question')
        for sample in tqdm.tqdm(data, total=len(data)):
            counter.update(token_(sample))
        vocab = sorted([w for w in counter if w in glove_vocab], key=counter.get, reverse=True)
    else:
        query_counter = Counter()
        doc_counter = Counter()
        all_doc=extract(data, key='context')
        for doc in tqdm.tqdm(nlp.pipe(all_doc, batch_size=10000, n_threads=thread), total=len(all_doc)):
            result=[w.text for w in doc if len(w.text)>0]
            doc_counter.update(result)
        all_query=extract(data, key='question')
        for query in tqdm.tqdm(nlp.pipe(all_query, batch_size=10000, n_threads=thread), total=len(all_query)):
            result=[w.text for w in query if len(w.text)>0]
            query_counter.update(result)

        token_ = partial(token, key='question')
        for sample in tqdm.tqdm(data, total=len(data)):
            query_counter.update(token_(sample))
        counter = query_counter + doc_counter
        # sort query words
        vocab = sorted([w for w in query_counter if w in glove_vocab], key=query_counter.get, reverse=True)
        vocab += sorted([w for w in doc_counter.keys() - query_counter.keys() if w in glove_vocab], key=counter.get, reverse=True)
    total = sum(counter.values())
    matched = sum(counter[w] for w in vocab)
    logger.info('raw vocab size vs vocab in glove: {0}/{1}'.format(len(counter), len(vocab)))
    logger.info('OOV rate:{0:.4f}={1}/{2}'.format(100.0 * (total - matched)/total, (total - matched), total))
    vocab = Vocabulary.build(vocab)
    logger.info('final vocab size: {}'.format(len(vocab)))
    return vocab

def load_data(path, is_train=True):
    rows = []
    with open(path, encoding="utf8") as f:
        data = json.load(f)['data']
    # parse data
    for article in tqdm.tqdm(data, total=len(data)):
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                if 'squad' in path or 'newsqa' in path or 'marco' in path:
                    uid, question = qa['id'], qa['question']
                    answers = qa.get('answers', [])
                    is_impossible = qa.get('is_impossible',False)
                    if is_train:
                        all_answers=[]
                        # get most common answer
                        for aid,ans in enumerate(answers):
                            if len(ans['text'])>0:
                                all_answers.append(ans['text'])

                        if len(all_answers) < 1:
                            if not '2.0' in path: continue
                            # no answer
                            assert is_impossible
                            answer_start=-1
                            answer_end=-1
                            answer_score=1.0
                            answer=None
                        else:
                            ans_counter = Counter(all_answers)
                            most_common_answer_text, most_common_count=ans_counter.most_common(1)[0]
                            
                            this_aid=-1
                            for aid in range(len(answers)):
                                if answers[aid]['text']==most_common_answer_text:
                                    this_aid=aid
                                    break
                            assert this_aid!=-1

                            answer = answers[this_aid]['text']
                            answer_start = int(answers[this_aid]['answer_start'])
                            answer_end = answer_start + len(answer.strip())
                            if 'score' in answers[this_aid]:
                                answer_score=answers[this_aid]['score']
                            else:
                                answer_score=1.0
                        sample = {'uid': uid, 'context': context, 'question': question, 
                        'answer': answer, 'answer_start': answer_start, 'answer_end':answer_end,
                        'answer_score':answer_score}
                    else:
                        sample = {'uid': uid, 'context': context, 'question': question, 'answer': answers, 'answer_start': -1, 'answer_end':-1,
                        'answer_score':1.0}
                rows.append(sample)
    if test_mode and ('marco' not in path):
        print('using first 200 data for test')
        rows=rows[:200]
    print('total number of data:',len(rows))

    return rows

def postag_func(toks, vocab):
    return [vocab[w.tag_] for w in toks if len(w.text) > 0]

def nertag_func(toks, vocab):
    return [vocab['{}_{}'.format(w.ent_type_, w.ent_iob_)] for w in toks if len(w.text) > 0] 

def tok_func(toks, vocab):
    return [vocab[w.text] for w in toks if len(w.text) > 0]

def toktext_func(toks):
    return [w.text for w in toks if len(w.text) > 0]

def match_func(question, context):
    counter = Counter(w.text.lower() for w in context)
    total = sum(counter.values())
    freq = [counter[w.text.lower()] / total for w in context]
    question_word = {w.text for w in question}
    question_lower = {w.text.lower() for w in question}
    question_lemma = {w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in question}
    match_origin = [1 if w in question_word else 0 for w in context]
    match_lower = [1 if w.text.lower() in question_lower else 0 for w in context]
    match_lemma = [1 if (w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower()) in question_lemma else 0 for w in context]
    features = np.asarray([freq, match_origin, match_lower, match_lemma], dtype=np.float32).T.tolist()
    return features

def build_span(context, answer, context_token, answer_start, answer_end, is_train=True):
    p_str = 0
    p_token = 0
    t_start, t_end, t_span = -1, -1, []
    while p_str < len(context):
        if re.match('\s', context[p_str]):
            p_str += 1
            continue
        token = context_token[p_token]
        token_len = len(token)
        if context[p_str:p_str + token_len] != token:
            print('dropped:',context[p_str:p_str + token_len], '/ and / ', token)
            return (None, None, [])
        t_span.append((p_str, p_str + token_len))
        if is_train:
            if (p_str <= answer_start and answer_start < p_str + token_len):
                t_start = p_token
            if (p_str < answer_end and answer_end <= p_str + token_len):
                t_end = p_token
        p_str += token_len
        p_token += 1


    if is_train:
        if answer_start==-1 and answer_end==-1:
            assert answer is None
            return (-1,-1,t_span)
        if t_start == -1 or t_end == -1:

            print('none: ', 'answer=', answer, 'answer_start=',answer_start, 'corres context:', context[answer_start:answer_end])
            return (None, None, [])
    return (t_start, t_end, t_span)

    
def feature_func(sample, doc_tokend, query_tokend, vocab, vocab_tag, vocab_ner, is_train=True, dataset_name='squad'):
    # features
    fea_dict = {}
    fea_dict['uid'] = sample['uid']
    fea_dict['context'] = sample['context']
    fea_dict['query_tok'] = tok_func(query_tokend, vocab)
    fea_dict['query_pos'] = postag_func(query_tokend, vocab_tag)
    fea_dict['query_ner'] = nertag_func(query_tokend, vocab_ner)
    fea_dict['query_text'] = toktext_func(query_tokend)
    fea_dict['query_fea']='{}'.format(match_func(doc_tokend, query_tokend))
    fea_dict['doc_tok'] = tok_func(doc_tokend, vocab)
    fea_dict['doc_pos'] = postag_func(doc_tokend, vocab_tag)
    fea_dict['doc_ner'] = nertag_func(doc_tokend, vocab_ner)
    fea_dict['doc_fea'] = '{}'.format(match_func(query_tokend, doc_tokend)) # json don't support float
    fea_dict['doc_text'] = toktext_func(doc_tokend)

    assert len(fea_dict['doc_tok'])==len(fea_dict['doc_text'])

    doc_toks = [t.text for t in doc_tokend]
    if 'squad' in dataset_name or 'newsqa' in dataset_name or 'marco' in dataset_name:
        start, end, span = build_span(sample['context'], sample['answer'], doc_toks, sample['answer_start'], sample['answer_end'], is_train=is_train)
        fea_dict['span'] = span
        fea_dict['start'] = start
        fea_dict['end'] = end

    fea_dict['score']=sample['answer_score']
    return fea_dict

def build_data(data, vocab, vocab_tag, vocab_ner, fout, is_train, dataset_name='squad'):
    with open(fout, 'w', encoding='utf-8') as writer:
        dropped_sample = 0
        all_datas=[]
        all_context = [reform_text(sample['context']) for sample in data]
        all_query = [reform_text(sample['question']) for sample in data]
        context_parsed = NLP.pipe(all_context, batch_size=5000, n_threads=n_threads)
        query_parsed = NLP.pipe(all_query, batch_size=5000, n_threads=n_threads)

        for sample, doc_tokend, query_tokend in tqdm.tqdm(zip(data, context_parsed, query_parsed), total=len(data)):
            fd = feature_func(sample, doc_tokend, query_tokend, vocab, vocab_tag, vocab_ner, is_train, dataset_name=dataset_name)
            if fd is None:
                dropped_sample += 1
                continue
            all_datas.append(fd)
        print('writing data. filename=',fout, 'len=', len(data))
        for fd in all_datas:
            writer.write('{}\n'.format(json.dumps(fd)))

        logger.info('dropped {} in total {}'.format(dropped_sample, len(data)))

def main():
    args = set_args()
    args.datasets=args.datasets.split(',')
    global logger
    logger = create_logger(__name__, to_disk=True, log_file=args.log_file)

    
    all_data=[]
    all_datasets=[]
    for dataset_name in args.datasets:
        test_file_prefix='test'
        if test_mode:
            if 'marco' in dataset_name:
                train_file_prefix='train'
                dev_file_prefix='dev'
            else:
                train_file_prefix='dev'
                dev_file_prefix='dev'
        else:
            train_file_prefix='train'
            dev_file_prefix='dev'

        logger.info('Processing %s dataset' % dataset_name)
        this_data_dir=args.data_dir+dataset_name+'/'
        train_data=None
        train_path = os.path.join(this_data_dir, '%s.json' % train_file_prefix)
        logger.info('The path of training data: {}'.format(train_path))
        train_data = load_data(train_path)
        all_data+=train_data

        valid_path = os.path.join(this_data_dir, '%s.json' % dev_file_prefix)
        logger.info('The path of validation data: {}'.format(valid_path))
        valid_data = load_data(valid_path, False)
        all_data+=valid_data
        if args.include_test_set and 'squad' not in dataset_name and 'marco2.0' not in dataset_name:
            test_path = os.path.join(this_data_dir, '%s.json' % test_file_prefix)
            logger.info('The path of test data: {}'.format(test_path))
            test_data = load_data(test_path, False)
            all_data+=test_data
            all_datasets.append((train_data, valid_data, test_data))
        else:
            all_datasets.append((train_data,valid_data))


    logger.info('{}-dim word vector path: {}'.format(args.glove_dim, args.glove))
    glove_path = args.glove
    glove_dim = args.glove_dim
    nlp = spacy.load('en', parser=False)
    set_environment(args.seed)
    logger.info('Loading glove vocab.')
    glove_vocab = load_glove_vocab(glove_path, glove_dim)

    multitask_base_path='../data/mtmrc/'
    with open(multitask_base_path+'vocab_tag.pick','rb') as f:
        vocab_tag = pickle.load(f)
    with open(multitask_base_path+'vocab_ner.pick','rb') as f:
        vocab_ner = pickle.load(f)


    logger.info('Build vocabulary ')
    vocab = build_vocab(all_data, glove_vocab, sort_all=args.sort_all, clean_on=True, 
        args=args)
    meta_path = os.path.join(args.output_path, args.meta)
    logger.info('building embedding ')
    embedding = build_embedding(glove_path, vocab, glove_dim)
    meta = {'vocab': vocab, 'vocab_tag': vocab_tag, 'vocab_ner': vocab_ner, 'embedding': embedding}
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    for i, item in enumerate(all_datasets):
        dataset_name=args.datasets[i]
        if args.include_test_set and 'squad' not in dataset_name and 'marco2.0' not in dataset_name:
            train_data, valid_data, test_data = item
        else:
            train_data, valid_data = item
        print('building output file for ', dataset_name)
        train_fout = os.path.join(args.output_path, dataset_name+'_train.json')
        build_data(train_data, vocab, vocab_tag, vocab_ner, train_fout, True, dataset_name=dataset_name)
        dev_fout = os.path.join(args.output_path, dataset_name+'_dev.json')
        build_data(valid_data, vocab, vocab_tag, vocab_ner, dev_fout, False, dataset_name=dataset_name)
        if args.include_test_set and 'squad' not in dataset_name:
            test_fout = os.path.join(args.output_path, dataset_name+'_test.json')
            build_data(test_data, vocab, vocab_tag, vocab_ner, test_fout, False, dataset_name=dataset_name)

if __name__ == '__main__':
    main()
