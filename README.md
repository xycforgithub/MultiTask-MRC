# Multi-task Learning with Sample Re-weighting for Machine Reading Comprehension

This PyTorch package implements the Multi-Task Stochastic Answer Network (MT-SAN) for Machine Reading Comprehension, as described in:

Yichong Xu, Xiaodong Liu, Yelong Shen, Jingjing Liu and Jianfeng Gao<br/>
Multi-task Learning with Sample Re-weighting for Machine Reading Comprehension</br>
North American Chapter of the Association for Computational Linguistics (NAACL), 2019<br/>
[arXiv version](https://arxiv.org/abs/1809.06963)

Please cite the above paper if you use this code. 

## Quickstart 

### Setup Environment
1. python3.6
2. install requirements:
   > pip install -r requirements.txt

### Train a SAN Model
1. prepare data
   > ./prepare_data.sh
2. train a model: See example codes in run.sh

## Notes and Acknowledgments
The code is developed based on the original SAN code: https://github.com/kevinduh/san_mrc

by
yichongx@cs.cmu.edu




