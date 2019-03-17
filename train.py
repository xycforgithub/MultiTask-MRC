import re
import os
import sys
import random
import string
import logging
import argparse
import json
import torch
import msgpack
import pickle
import numpy as np
from shutil import copyfile, move
from datetime import datetime
from collections import Counter, defaultdict
from src.model import DocReaderModel
from src.batcher import load_meta, BatchGen
from config import set_args
from my_utils.utils import set_environment
from my_utils.log_wrapper import create_logger
from my_utils.squad_eval import evaluate_file, load_gold
from my_utils.marco_eval import load_rank_score, generate_submit
from my_utils.ms_marco_eval_pretoken import MAX_BLEU_ORDER, compute_metrics_from_files
import configparser
from my_utils.utils import repeat_save,load_jsonl

args = set_args()

if args.valid_batch_size is None:
    args.valid_batch_size=args.batch_size
if args.log_file is None:
    args.log_file = os.path.join(args.model_dir, 'mtmrc.log')
if args.dev_datasets is None:
    args.dev_datasets=args.train_datasets
if args.elmo_path is None:
    args.elmo_path=args.data_dir
if args.elmo_model=='big':
    args.elmo_options_path=os.path.join(args.elmo_path,'elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json')
    args.elmo_weight_path=os.path.join(args.elmo_path,'elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5')
elif args.elmo_model=='small':
    args.elmo_options_path=os.path.join(args.elmo_path,'elmo_2x1024_128_2048cnn_1xhighway_options.json')
    args.elmo_weight_path=os.path.join(args.elmo_path,'elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')
if args.optimizer =='sgd':
    args.not_resume_optimizer = True

# set model dir
print('output directory:',args.model_dir)
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)
model_dir = os.path.abspath(model_dir)

args.num_gpu = torch.cuda.device_count() if args.multi_gpu else 1
    
config_path = os.path.join(args.model_dir, 'config.json')
json.dump(vars(args),open(config_path,'w'), indent=4)


args.dataset_config_path='dataset_configs/dataset_config_%d.ini' % args.dataset_config_id
print('dataset config path:', args.dataset_config_path)
config = configparser.ConfigParser()
config.read(args.dataset_config_path)
args.dataset_configs=config
args.dataset_configs['marco_test']=args.dataset_configs['marco']
args.dataset_configs['squad_mt']=args.dataset_configs['squad']

extra_score = json.load(open(os.path.join(args.multitask_data_path,args.extra_score_file))) if args.extra_score_file is not None else {}



args.elmo_config_path='elmo_configs/elmo_config_%d.ini' % args.elmo_config_id
print('elmo config path:', args.elmo_config_path)
config = configparser.ConfigParser()
config.read(args.elmo_config_path)
args.elmo_configs={}
for key in config['elmo']:
    args.elmo_configs[key]=int(config['elmo'][key])
print('elmo config:', args.elmo_configs[key])

print(vars(args))

print('log directory:',args.log_file)

# parse multitask dataset names
args.train_datasets=args.train_datasets.split(',')
args.dev_datasets=args.dev_datasets.split(',')  




# set environment
set_environment(args.seed, args.cuda)
# setup logger
logger =  create_logger(__name__, to_disk=True, log_file=args.log_file)

def check(model, data, gold_data):
    data.reset()
    predictions = {}
    best_scores=[]
    start_scores=[]
    end_scores = []
    for batch_list, name_map in data:
        phrase, best_score_batch, start_batch, end_batch = model.predict(batch_list, name_map, dataset_name=data.dataset_name)
        uids = batch_list[name_map['uids']]
        best_scores+=best_score_batch
        start_scores+=start_batch
        end_scores+=end_batch
        for uid, pred in zip(uids, phrase):
            predictions[uid] = pred

    results = evaluate_file(gold_data, predictions)
    scores = {
    'best_scores':best_scores,
    'start_scores':start_scores,
    'end_scores':end_scores
    }
    return results['exact_match'], results['f1'], predictions, scores
def check_wdw(model,data):
    data.reset()
    correct_count = 0
    total_count = 0
    preds=[]
    scores=[]
    truths=[]
    for batch_list, name_map in data:
        pred, probs = model.predict(batch_list, name_map, dataset_name=data.dataset_name)
        truth = batch_list[name_map['truth']].tolist()
        correct_count += sum([pred[i]==truth[i] for i in range(len(truth))])
        total_count += len(truth)
        preds += pred
        truths+=truth
        scores += probs
    all_preds = [pair for pair in zip(preds,truths)]
    return correct_count/total_count*100, all_preds, scores


def eval_model_marco(model, data):
    data.reset()
    dev_predictions = []
    dev_best_scores = []
    dev_ids_list = []
    for batch_list, name_map in data:
        phrase, phrase_score, _, _ = model.predict(batch_list, name_map, dataset_name=data.dataset_name)
        dev_predictions.extend(phrase)
        dev_best_scores.extend(phrase_score)
        dev_ids_list.extend(batch_list[name_map['uids']])
    return dev_predictions, dev_best_scores, dev_ids_list

def main():
    logger.info('Launching the SAN')
    start_test=False
    dev_name = 'dev'

    opt = vars(args)
    logger.info('Loading data')
    embedding, opt = load_meta(opt, os.path.join(args.multitask_data_path, args.meta))
    gold_data = load_gold(args.dev_datasets, args.data_dir, dev_name=dev_name)
    best_em_score, best_f1_score = 0.0, 0.0
    
    all_train_batchgen=[]
    all_dev_batchgen=[]
    all_train_iters=[]
    for dataset_name in args.train_datasets:
        path=os.path.join(args.multitask_data_path, dataset_name+'_train.json')
        this_extra_score = extra_score.get(dataset_name, None)

        all_train_batchgen.append(BatchGen(path,
                              batch_size=args.batch_size,
                              gpu=args.cuda, dataset_name = dataset_name, 
                              doc_maxlen=args.doc_maxlen, drop_less = args.drop_less,
                              num_gpu = args.num_gpu,
                              dropout_w=args.dropout_w, dw_type=args.dw_type, 
                              extra_score = this_extra_score, extra_score_cap = args.extra_score_cap))
    all_train_iters=[iter(item) for item in all_train_batchgen]
    for dataset_name in args.dev_datasets:
        path=os.path.join(args.multitask_data_path, dataset_name+'_%s.json' % dev_name)
        all_dev_batchgen.append(BatchGen(path,
                              batch_size=args.valid_batch_size,
                              gpu=args.cuda, is_train=False, 
                              dataset_name = dataset_name, doc_maxlen=args.doc_maxlen,
                              num_gpu = args.num_gpu))
        if 'marco' in dataset_name:
            rank_path = os.path.join(args.data_dir,dataset_name)
            dev_rank_path = os.path.join(rank_path, 'dev_rank_scores.json')
            dev_rank_scores = load_rank_score(dev_rank_path)
            dev_yn=json.load(open(os.path.join(rank_path,'dev_yn_dict.json')))
            dev_gold_path = os.path.join(args.data_dir, dataset_name, 'dev_original.json')
            dev_gold_data_marco = load_jsonl(dev_gold_path)
    if args.resume_last_epoch:
        latest_time=0
        for o in os.listdir(model_dir):
            if o.startswith('checkpoint_') and 'trim' not in o:
                edit_time = os.path.getmtime(os.path.join(model_dir, o))
                if edit_time>latest_time:
                    latest_time=edit_time
                    args.resume_dir = model_dir
                    args.resume = o

    if args.resume_dir is not None:
        print('resuming model in ', os.path.join(args.resume_dir, args.resume))
        checkpoint = torch.load(os.path.join(args.resume_dir, args.resume))
        
        model_opt = checkpoint['config'] if args.resume_options else opt
        model_opt['multitask_data_path']=opt['multitask_data_path']
        model_opt['covec_path']=opt['covec_path']
        model_opt['data_dir']=opt['data_dir']

        if args.resume_options:
            logger.info('resume old options')
        else:
            logger.info('use new options.')
        model_opt['train_datasets']=checkpoint['config']['train_datasets']

        state_dict = checkpoint['state_dict']
        model = DocReaderModel(model_opt, embedding, state_dict)

        if not args.new_random_state:
            logger.info('use old random state.')
            random.setstate(checkpoint['random_state'])
            torch.random.set_rng_state(checkpoint['torch_state'])
            if args.cuda:
                torch.cuda.set_rng_state(checkpoint['torch_cuda_state'])

        if model.scheduler:
            if args.new_scheduler:
                model.scheduler = torch.optim.lr_scheduler.MultiStepLR(model.optimizer, milestones=[2,5,8], gamma=args.lr_gamma)
            elif 'scheduler_state' in checkpoint:
                model.scheduler.load_state_dict(checkpoint['scheduler_state'])
            else:
                print('warning: not loading scheduler state because didn\'t save.')
        start_epoch = checkpoint['epoch']+1
    else:
        model = DocReaderModel(opt, embedding)
        start_epoch=0
    logger.info('using {} GPUs'.format(torch.cuda.device_count()))
    headline = '############# Model Arch of SAN #############'
    # print network
    logger.info('\n{}\n{}\n'.format(headline, model.network))
    model.setup_eval_embed(embedding)

    logger.info("Total number of params: {}".format(model.total_param))
    if args.cuda:
        model.cuda()


    all_lens=[len(bg) for bg in all_train_batchgen]

    if args.continue_epoches is not None:
        args.epoches = start_epoch + args.continue_epoches
    num_all_batches=args.epoches * sum(all_lens)
    best_performance={name:0.0 for name in args.dev_datasets}
    best_performance['total']=0.0

    for epoch in range(start_epoch, args.epoches):
        logger.warning('At epoch {}'.format(epoch))

        # batch indices
        all_call_indices=[]
        for train_data in all_train_batchgen:
            train_data.reset()
        if args.dataset_include_ratio>=0:
            other_indices=[]
            for i in range(1,len(all_train_batchgen)):
                other_indices+=[i]*len(all_train_batchgen[i])
            if args.dataset_include_ratio>1:
                inverse_ratio = 1/args.dataset_include_ratio
                batch0_indices=[0]*(int(len(other_indices)*inverse_ratio))
            else:
                batch0_indices=[0]*len(all_train_batchgen[0])               
                other_picks=int(len(other_indices)*args.dataset_include_ratio)
                other_indices=random.sample(other_indices, other_picks)
            all_call_indices=batch0_indices+other_indices
        else:
            for i in range(len(all_train_batchgen)):
                all_call_indices+=[i]*len(all_train_batchgen[i])
        random.shuffle(all_call_indices)
        all_call_indices=all_call_indices[:10]
        start = datetime.now()
        for i in range(len(all_call_indices)):

            batch_list, name_map=next(all_train_iters[all_call_indices[i]])
            dataset_name = args.train_datasets[all_call_indices[i]]

            model.update(batch_list, name_map, dataset_name)
            if (model.updates) % args.log_per_updates == 0 or i == 0:
                logger.info('o(*^~^*) Task [{0:2}] #updates[{1:6}] train loss[{2:.5f}] remaining[{3}]'.format(
                    all_call_indices[i], model.updates, model.train_loss.avg,
                    str((datetime.now() - start) / (i + 1) * (len(all_call_indices) - i - 1)).split('.')[0]))

        em_sum=0
        f1_sum=0
        model.eval()
        this_performance={}
        for i in range(len(all_dev_batchgen)):
            dataset_name = args.dev_datasets[i]
            if dataset_name in ['squad','newsqa']:
                em, f1, results, scores = check(model, all_dev_batchgen[i], gold_data[dataset_name])
                output_path = os.path.join(model_dir, 'dev_output_{}_{}.json'.format(dataset_name,epoch))
                output_scores_path = os.path.join(model_dir, 'dev_scores_{}_{}.pt'.format(dataset_name,epoch))
                for repeat_times in range(10):
                    try:
                        with open(output_path, 'w') as f:
                            json.dump(results, f)
                        with open(output_scores_path, 'wb') as f:
                            pickle.dump(scores, f)
                        break
                    except Exception as e:
                        print('save predict failed. error:', e)
                em_sum+=em
                f1_sum+=f1
                this_performance[dataset_name]=em+f1
                logger.warning("Epoch {0} - Task {1:6} dev EM: {2:.3f} F1: {3:.3f}".format(epoch, dataset_name, em, f1))
            elif dataset_name=='wdw':
                acc, results, scores = check_wdw(model, all_dev_batchgen[i])
                output_path = os.path.join(model_dir, 'dev_output_{}_{}.json'.format(dataset_name,epoch))
                output_scores_path = os.path.join(model_dir, 'dev_scores_{}_{}.pt'.format(dataset_name,epoch))
                for repeat_times in range(10):
                    try:
                        with open(output_path, 'w') as f:
                            json.dump(results, f)
                        with open(output_scores_path, 'wb') as f:
                            pickle.dump(scores, f)
                        break
                    except Exception as e:
                        print('save predict failed. error:',e)
                em_sum+=acc
                f1_sum+=acc
                logger.warning("Epoch {0} - Task {1:6} dev ACC: {2:.3f}".format(epoch, dataset_name, acc))
                this_performance[dataset_name]=acc

            elif 'marco' in dataset_name:
                # dev eval
                output = os.path.join(model_dir, 'dev_pred_{}.json'.format(epoch))
                output_yn = os.path.join(model_dir, 'dev_pred_yn_{}.json'.format(epoch))
                span_output = os.path.join(model_dir, 'dev_pred_span_{}.json'.format(epoch))
                dev_predictions, dev_best_scores, dev_ids_list=eval_model_marco(model, all_dev_batchgen[i])
                answer_list, rank_answer_list, yesno_answer_list=generate_submit(dev_ids_list, 
                    dev_best_scores, dev_predictions, dev_rank_scores, dev_yn)

                dev_gold_path = os.path.join(args.data_dir, dataset_name, 'dev_original.json')
                metrics = compute_metrics_from_files(dev_gold_data_marco, \
                                                        rank_answer_list, \
                                                        MAX_BLEU_ORDER)
                rouge_score = metrics['rouge_l']
                blue_score = metrics['bleu_1']
                logger.warning("Epoch {0} - dev ROUGE-L: {1:.4f} BLEU-1: {2:.4f}".format(epoch, rouge_score, blue_score))

                for metric in sorted(metrics):
                    logger.info('%s: %s' % (metric, metrics[metric]))

                this_performance[dataset_name]=rouge_score+blue_score
        this_performance['total']=sum([v for v in this_performance.values()])
        model.train()
        # setting up scheduler
        if model.scheduler is not None:
            logger.info('scheduler_type {}'.format(opt['scheduler_type']))
            if opt['scheduler_type'] == 'rop':
                model.scheduler.step(f1, epoch=epoch)
            else:
                model.scheduler.step()
        # save
        for try_id in range(10):
            try:
                model_file = os.path.join(model_dir, 'checkpoint_epoch_{}.pt'.format(epoch))
                model.save(model_file, epoch, best_em_score, best_f1_score)
                if em_sum + f1_sum > best_em_score + best_f1_score:
                    copyfile(os.path.join(model_dir, model_file), os.path.join(model_dir, 'best_checkpoint.pt'))
                    best_em_score, best_f1_score = em_sum, f1_sum
                    logger.info('Saved the new best model and prediction')
                break
            except Exception as e:
                print('save model failed: outer step. error=',e)

if __name__ == '__main__':
    main()
