#!/usr/bin/env bash
############################### 
# Download data resources
# by yichongx@cs.cmu.edu
############################### 

# Default behavior is to create a parallel folder for data
DATA_DIR=../data
echo $DATA_DIR
mkdir $DATA_DIR
mkdir $DATA_DIR/squad
mkdir $DATA_DIR/newsqa
mkdir $DATA_DIR/marco
mkdir $DATA_DIR/mtmrc
mkdir $DATA_DIR/mtmrc/mt_snm


# Download SQuAD data
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O $DATA_DIR/squad/train.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O $DATA_DIR/squad/dev.json

# Download NewsQA data
# The data is pre-processed in the same way following AMANDA: https://github.com/nusnlp/amanda/tree/master/NEWSQA
wget http://www.cs.cmu.edu/~yichongx/newsqa/train.json -O $DATA_DIR/newsqa/train.json
wget http://www.cs.cmu.edu/~yichongx/newsqa/dev.json -O $DATA_DIR/newsqa/dev.json

# Download MARCO data and score files
wget http://www.cs.cmu.edu/~yichongx/marco.tar -O $DATA_DIR/marco.tar
tar -xvf $DATA_DIR/marco.tar -C $DATA_DIR/

# Download GloVe
wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O $DATA_DIR/glove.840B.300d.zip
unzip $DATA_DIR/glove.840B.300d.zip -d $DATA_DIR
rm $DATA_DIR/glove.840B.300d.zip

# Download CoVe
wget https://s3.amazonaws.com/research.metamind.io/cove/wmtlstm-b142a7f2.pth -O $DATA_DIR/MT-LSTM.pt

# Download Allennlp ELMo
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5 -O $DATA_DIR/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json -O $DATA_DIR/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json

#Download score file
wget http://www.cs.cmu.edu/~yichongx/nm_lm_scores.json -O $DATA_DIR/mtmrc/mt_snm/nm_lm_scores.json

NER and POS tags
wget http://www.cs.cmu.edu/~yichongx/vocab_ner.pick -O $DATA_DIR/mtmrc/vocab_ner.pick
wget http://www.cs.cmu.edu/~yichongx/vocab_tag.pick -O $DATA_DIR/mtmrc/vocab_tag.pick


# prepare data for training on SQuAD+NewsQA+MARCO
python prepro.py --datasets squad,newsqa,marco --output_path $DATA_DIR/mtmrc/mt_snm

