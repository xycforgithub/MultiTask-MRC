#/usr/bin/env python3
import argparse
import multiprocessing
import torch
import os
"""
Configuration file

"""

def str2bool(v):
    return v.lower() in ('true', '1')

def model_config(parser):
    parser.add_argument('--vocab_size', type=int, default=0)
    parser.add_argument('--wemb_dim', type=int, default=300)
    parser.add_argument('--covec_on', action='store_false')
    parser.add_argument('--embedding_dim', type=int, default=300)
    parser.add_argument('--doc_maxlen', type=int, default=1000, help='ans positions larger than this value is dropped.')
    parser.add_argument('--drop_less', type=str2bool, default=True, help='Do not drop the question if the answer lies before doc_maxlen.')

    # pos
    parser.add_argument('--no_pos', dest='pos_on', action='store_false')
    parser.add_argument('--pos_vocab_size', type=int, default=56)
    parser.add_argument('--pos_dim', type=int, default=12)
    parser.add_argument('--no_ner', dest='ner_on', action='store_false')
    parser.add_argument('--ner_vocab_size', type=int, default=19)
    parser.add_argument('--ner_dim', type=int, default=8)
    parser.add_argument('--no_feat', dest='feat_on', action='store_false')
    parser.add_argument('--num_features', type=int, default=4)
    # q->p
    parser.add_argument('--prealign_on', action='store_false')
    parser.add_argument('--prealign_head', type=int, default=1)
    parser.add_argument('--prealign_att_dropout', type=float, default=0)
    parser.add_argument('--prealign_norm_on', action='store_false')
    parser.add_argument('--prealign_proj_on', action='store_true')
    parser.add_argument('--prealign_bidi', action='store_false')
    parser.add_argument('--prealign_hidden_size', type=int, default=300)
    parser.add_argument('--prealign_share', action='store_false')
    parser.add_argument('--prealign_residual_on', action='store_true')
    parser.add_argument('--prealign_scale_on', action='store_false')
    parser.add_argument('--prealign_sim_func', type=str, default='dotproductproject')
    parser.add_argument('--prealign_activation', type=str, default='relu')

    parser.add_argument('--pwnn_on', action='store_false')
    parser.add_argument('--pwnn_hidden_size', type=int, default=125)

    ##contextual encoding
    parser.add_argument('--contextual_hidden_size', type=int, default=125)
    parser.add_argument('--contextual_cell_type', type=str, default='lstm')
    parser.add_argument('--contextual_weight_norm_on', action='store_true')
    parser.add_argument('--contextual_maxout_on', action='store_true')
    parser.add_argument('--contextual_residual_on', action='store_true')
    parser.add_argument('--contextual_encoder_share', action='store_true')
    parser.add_argument('--contextual_num_layers', type=int, default=2)

    ## mem setting
    parser.add_argument('--msum_hidden_size', type=int, default=125)
    parser.add_argument('--msum_cell_type', type=str, default='lstm')
    parser.add_argument('--msum_weight_norm_on', action='store_false')
    parser.add_argument('--msum_maxout_on', action='store_true')
    parser.add_argument('--msum_residual_on', action='store_true')
    parser.add_argument('--msum_lexicon_input_on', action='store_true')
    parser.add_argument('--msum_num_layers', type=int, default=1)

    # attention
    parser.add_argument('--deep_att_lexicon_input_on', action='store_false')
    parser.add_argument('--deep_att_hidden_size', type=int, default=250)
    parser.add_argument('--deep_att_sim_func', type=str, default='dotproductproject')
    parser.add_argument('--deep_att_activation', type=str, default='relu')
    parser.add_argument('--deep_att_norm_on', action='store_false')
    parser.add_argument('--deep_att_proj_on', action='store_true')
    parser.add_argument('--deep_att_residual_on', action='store_true')
    parser.add_argument('--deep_att_share', action='store_false')
    parser.add_argument('--deep_att_opt', type=int, default=0)

    # self attn
    parser.add_argument('--self_attention_on', action='store_false')
    parser.add_argument('--self_att_hidden_size', type=int, default=75)
    parser.add_argument('--self_att_sim_func', type=str, default='dotproductproject')
    parser.add_argument('--self_att_activation', type=str, default='relu')
    parser.add_argument('--self_att_norm_on', action='store_false')
    parser.add_argument('--self_att_proj_on', action='store_true')
    parser.add_argument('--self_att_residual_on', action='store_true')
    parser.add_argument('--self_att_dropout', type=float, default=0.1)
    parser.add_argument('--self_att_drop_diagonal', action='store_false')
    parser.add_argument('--self_att_share', action='store_false')

    # query summary
    parser.add_argument('--query_sum_att_type', type=str, default='linear',
                        help='linear/mlp')
    parser.add_argument('--query_sum_norm_on', action='store_false')

    parser.add_argument('--san_on', action='store_false')
    parser.add_argument('--max_len', type=int, default=5)
    parser.add_argument('--marco_max_len', type=int, default=15)
    parser.add_argument('--decoder_ptr_update_on', action='store_false')
    parser.add_argument('--decoder_num_turn', type=int, default=5)
    parser.add_argument('--decoder_mem_type', type=int, default=1)
    parser.add_argument('--decoder_mem_drop_p', type=float, default=0.1)
    parser.add_argument('--decoder_opt', type=int, default=0)
    parser.add_argument('--decoder_att_hidden_size', type=int, default=125)
    parser.add_argument('--decoder_att_type', type=str, default='bilinear',
                        help='bilinear/simple/default')
    parser.add_argument('--decoder_rnn_type', type=str, default='gru',
                        help='rnn/gru/lstm')
    parser.add_argument('--decoder_sum_att_type', type=str, default='bilinear',
                        help='bilinear/simple/default')
    parser.add_argument('--decoder_weight_norm_on', action='store_false')

    # switcher config
    parser.add_argument('--highway_type',default='highway', type=str, help='highway or gate.')
    parser.add_argument('--highway_num', default=1, type=int,help='highway layer number.')
    parser.add_argument('--cove_highway_num', default=1, type=int,help='highway layer number for cove.')
    # parser.add_argument('--')
    parser.add_argument('--cove_highway_position', default='final', type=str,
        help='position of cove highways. final=apply after embedding layer, first=apply before linear transformation.')
    parser.add_argument('--embed_transform_highway', type=str, default='after',
        help='position of embed linear transform highway. after=apply after linear transform, combine=combine with linear layer.')
    parser.add_argument('--query_mem_highway_pos', type=str, default='after',
        help='position of query mem highway. After = after sum attention, before = before sum attention.')
    parser.add_argument('--dataset_config_id', type=int, default=1,
        help='Path of dataset-specific config file.')
    parser.add_argument('--highway_dropout', type=float, default=0.0,
        help='highway dropout.')

    # elmo config
    parser.add_argument('--add_elmo', action='store_true', help='use elmo embeddings.')
    parser.add_argument('--elmo_model', default='big', type=str, help='which model to use for elmo. big or small right now.')
    # parser.add_argument('--elmo_hidden_size', default=125, type=int, help='elmo hidden size for transformed elmo.')
    parser.add_argument('--elmo_opt', default=0, type=int,
        help='elmo config for transformed elmo. 0=scalar combine, 1=linear transform combine, 2=gate combine.')
    parser.add_argument('--elmo_dropout', default=0.6, type=float, help='Dropout for elmo.')
    parser.add_argument('--elmo_config_id', default=0, type=int,
        help='config to put elmo.' )
    parser.add_argument('--elmo_l2', default=0.0, type=float, help='l2 regularization for elmo.')

    return parser

def data_config(parser):
    base_dir=os.getenv('PT_OUTPUT_DIR', '../model_data')
    # parser.add_argument('--log_file', default='%s/san/san.log' % base_dir, help='path for log file.')
    parser.add_argument('--log_file', default=None, help='path for log file.')
    parser.add_argument('--data_dir', default='../data/')
    parser.add_argument('--elmo_path', default=None, type=str, help='elmo path.')
    parser.add_argument('--meta', default='meta.pick', help='path to preprocessed meta file.')
    parser.add_argument('--output_path', default='../data/multitask/san_tt/', help='output files path for prepro.')
    parser.add_argument('--multitask_data_path',default='../data/multitask/san/', help='multitask data path')
    parser.add_argument('--include_test_set',action='store_true', help='include test set for processing.')

    parser.add_argument('--covec_path', default='../data/MT-LSTM.pt')
    parser.add_argument('--glove', default='../data/glove.840B.300d.txt',
                        help='path to word vector file.')
    parser.add_argument('--glove_dim', type=int, default=300,
                        help='word vector dimension.')
    parser.add_argument('--sort_all', action='store_true',
                        help='sort the vocabulary by frequencies of all words.'
                             'Otherwise consider question words first.')
    parser.add_argument('--threads', type=int, default=multiprocessing.cpu_count(),
                        help='number of threads for preprocessing.')
    parser.add_argument('--datasets', type=str, default='squad', help='datasets to process. Separate by commas.')
    parser.add_argument('--extra_score_file',type=str, default=None, help='extra score file for data selection.')
    parser.add_argument('--extra_score_cap', type=float, default=1.0, help='score cap for extra score; score >this -> 1.0, otherwise scaled.')

    parser.add_argument('--flatten_parameters', action='store_true', help='flatten parameters to reduce memory usage.')


    # multi-task
    parser.add_argument('--train_datasets',default='[squad]', type=str,
                        help='datasets to train on.')
    parser.add_argument('--dev_datasets',default=None, type=str,
                        help='datasets to validate on.')
    # parser.add_argument('--bridges',action='store_true',help='use bridges mode. Will use $SCRATCH before data dir and model dir.')


    return parser

def train_config(parser):
    base_dir=os.getenv('PT_OUTPUT_DIR', '../model_data/san/')
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                        help='Use GPU acceleration.')
    parser.add_argument('--multi_gpu', action='store_true', help='multi gpu training.')
    parser.add_argument('--log_per_updates', type=int, default=53)
    parser.add_argument('--progress_per_updates', type=int, default=500)
    parser.add_argument('--epoches', type=int, default=50)
    parser.add_argument('--continue_epoches', type=int, default=None, help='if continue train, add this amount of epoches. Default to use --epoches.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--valid_batch_size',type=int, default=16)

    parser.add_argument('--optimizer', default='adamax',
                        help='supported optimizer: adamax, sgd, adadelta, adam')
    parser.add_argument('--ema', action='store_false', help='use exponential moving average for testing.')
    parser.add_argument('--ema_gamma', type=float, default=0.995, help='gamma for ema.')
    parser.add_argument('--grad_clipping', type=float, default=10)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.002)
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--vb_dropout', action='store_false')
    parser.add_argument('--dropout_p', type=float, default=0.3)
    parser.add_argument('--dropout_emb', type=float, default=0.4)
    parser.add_argument('--dropout_w', type=float, default=0.05)
    parser.add_argument('--dw_type', type=int, default=0)
    parser.add_argument('--unk_id', type=int, default=1)
    parser.add_argument('--dataset_include_ratio', default=-1.0, type=float,
        help='ratio to include other datasets (paper formulation). override batches_mix_ratio.')
    parser.add_argument('--uncertainty_loss', action='store_true', help='Uncertainty-based loss using Kendall et al., 2017.')

    # scheduler
    parser.add_argument('--no_lr_scheduler', dest='have_lr_scheduler', action='store_false')
    parser.add_argument('--multi_step_lr', type=str, default='10,20,30')
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--scheduler_type', type=str, default='ms', help='ms/rop/exp')
    parser.add_argument('--fix_embeddings', action='store_true', help='if true, `tune_partial` will be ignored.')
    parser.add_argument('--tune_partial', type=int, default=1000, help='finetune top-x embeddings (including <PAD>, <UNK>).')
    parser.add_argument('--model_dir', default='%s/'% base_dir)
    parser.add_argument('--resume_dir', default=None, type=str, help='model_dir to resume.')
    parser.add_argument('--resume', default='best_checkpoint.pt',type=str,help='model name to resume.')
    parser.add_argument('--resume_last_epoch', action='store_true', help='Restore the last previously stored model in model_dir. Will override resume_dir options.')
    parser.add_argument('--new_random_state', action='store_true', help='Use new random states for resume.')
    parser.add_argument('--not_resume_optimizer',type=str2bool, default=False, help='Do not restore state of optimizer.')
    parser.add_argument('--new_scheduler', action='store_true',help='use a new scheduler, multi step 2,4,6.')
    parser.add_argument('-ro', '--resume_options', action='store_true',
                    help='use previous model options, ignore the cli and defaults.')

    parser.add_argument('--seed', type=int, default=2018,
                        help='random seed for data shuffling, embedding init, etc.')
    return parser

def set_args():
    parser = argparse.ArgumentParser()
    parser = data_config(parser)
    parser = model_config(parser)
    parser = train_config(parser)
    args = parser.parse_args()
    return args
